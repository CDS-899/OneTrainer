import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modules.trainer.extension.FlowModelAdapter import FlowModelAdapter, create_flow_model_adapter
from modules.trainer.extension.noise_bands import NoiseBandLoss
from modules.trainer.extension.TrainingExtension import TrainingExtension
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.tqdm_util import tqdm
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.AspectBucketing import AspectBucketing

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from modules.trainer.GenericTrainer import GenericTrainer

CACHE_DIR_NAME = "context_distillation"
CACHE_VERSION = "cd-v1"

@dataclass(frozen=True)
class PromptPair:
    student: str
    teacher: str
    ar: tuple[float, float] | None  # (width, height), None = square


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def parse_ar(value: str) -> tuple[float, float]:
    """'3:4' -> (3.0, 4.0) as width:height."""
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError(f"invalid ar '{value}', expected width:height like \"3:4\"")
    width, height = float(parts[0]), float(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid ar '{value}', both sides must be positive")
    return width, height


def load_prompt_pairs(path: str) -> list[PromptPair]:
    """
    Reads prompt pairs from a .jsonl file, one object per line:
        {"student": "...", "teacher": "...", "ar": "3:4"}
    "ar" (width:height) is optional, without it the pair is trained square. "teacher" may be a JSON object
    (structured prompts), it is passed to the model as compact JSON. A pair with the same prompt twice keeps
    that prompt like the base model.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"context distillation prompts file not found: '{path}'")

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({e})") from e
            student = _as_text(obj.get("student"))
            teacher = _as_text(obj.get("teacher"))
            if not student or not teacher:
                raise ValueError(f"{path}:{line_number}: needs non-empty \"student\" and \"teacher\"")
            ar = parse_ar(obj["ar"]) if obj.get("ar") else None
            pairs.append(PromptPair(student, teacher, ar))

    if not pairs:
        raise ValueError(f"no prompt pairs in '{path}'")
    return pairs


def _quantize(value: float, quantization: int) -> int:
    return max(quantization, round(value / quantization) * quantization)


def bucket_resolutions(target: int, quantization: int) -> list[tuple[int, int]]:
    """OneTrainer's aspect buckets (height, width) for one target resolution, as mgds AspectBucketing builds them."""
    resolutions = set()
    for h, w in AspectBucketing.all_possible_input_aspects:
        for bh, bw in ((h, w), (w, h)):
            scale = target / math.sqrt(bh * bw)
            resolutions.add((_quantize(bh * scale, quantization), _quantize(bw * scale, quantization)))
    return sorted(resolutions)


def resolve_bucket(pair: PromptPair, resolution: str, quantization: int, seed: int) -> tuple[int, int]:
    """
    The (height, width) a pair is trained at. resolution uses the training resolution format:
      "768x512"   fixed size (width x height) for every pair, "ar" is ignored
      "512"       square, or the OneTrainer bucket nearest to the pair's "ar"
      "512,1024"  as above, each pair gets one of the targets (fixed per pair)
    """
    resolution = resolution.strip()
    if "x" in resolution and "," not in resolution:
        width, height = (int(v) for v in resolution.split("x"))
        return _quantize(height, quantization), _quantize(width, quantization)

    targets = [int(v.strip()) for v in resolution.split(",") if v.strip()]
    target = targets[0] if len(targets) == 1 else \
        random.Random(f"{seed}:{pair.student}:{pair.teacher}").choice(targets)
    if pair.ar is None:
        size = _quantize(target, quantization)
        return size, size

    aspect = pair.ar[1] / pair.ar[0]  # height / width
    return min(bucket_resolutions(target, quantization), key=lambda hw: abs(hw[0] / hw[1] - aspect))


class ContextDistillation(TrainingExtension):
    """
    The student (base model + LoRA) sees the student prompt, the teacher (same model, LoRA disabled) sees the
    teacher prompt, both at the same noisy latent. The student learns to match the teacher's velocity:

        loss = || v_student(x_t, sigma, student) - v_teacher(x_t, sigma, teacher) ||^2

    x_t is the pair's cached teacher latent (generated once from the teacher prompt, never decoded), re-noised
    with fresh noise at a noise level from the configured timestep distribution. An epoch trains every pair
    once, in batches of the training batch size grouped by bucket.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.cd = config.context_distillation

        if config.training_method != TrainingMethod.LORA:
            raise NotImplementedError(
                "context distillation needs LoRA training: the teacher is the base model with the LoRA switched off"
            )

        self.pairs = load_prompt_pairs(self.cd.prompts_path)
        self.batch_size = max(1, config.batch_size)

        self.adapter: FlowModelAdapter | None = None
        self.generator: torch.Generator | None = None
        self.cache_dir = os.path.join(config.cache_dir, CACHE_DIR_NAME)
        self.entry_paths: list[str] = []
        self.buckets: list[tuple[int, int]] = []

        self._epoch = 0
        self._queue: list[list[int]] = []
        self.band_loss = NoiseBandLoss("context_distillation", "loss/context_distillation")

    def is_standalone(self) -> bool:
        return self.cd.standalone

    def standalone_epoch_length(self) -> int:
        return len(self._epoch_batches(0))

    # ---------------------------------------------------------------- cache

    def _settings_key(self) -> str:
        config = self.config
        return json.dumps([
            CACHE_VERSION, str(config.model_type), config.base_model_name,
            str(config.transformer.weight_dtype), str(config.text_encoder.weight_dtype), str(config.train_dtype),
            self.cd.teacher_steps, self.cd.teacher_cfg, self.cd.negative_prompt, self.cd.seed,
        ])

    def _entry_key(self, settings_key: str, pair: PromptPair, bucket: tuple[int, int]) -> str:
        data = json.dumps([settings_key, pair.student, pair.teacher, bucket], ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def on_train_start(self, trainer: "GenericTrainer"):
        model = trainer.model
        self.adapter = create_flow_model_adapter(model, trainer.model_setup, self.config, trainer.train_device)
        self.generator = torch.Generator(device=trainer.train_device)

        if self.config.clear_cache_before_training and self.config.latent_caching and os.path.isdir(self.cache_dir):
            print(f"Clearing context distillation cache {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        resolution = self.cd.resolution or self.config.resolution
        quantization = self.adapter.resolution_quantization()
        settings_key = self._settings_key()
        self.buckets = [resolve_bucket(p, resolution, quantization, self.cd.seed) for p in self.pairs]
        keys = [self._entry_key(settings_key, p, b) for p, b in zip(self.pairs, self.buckets, strict=True)]
        self.entry_paths = [os.path.join(self.cache_dir, f"{key}.pt") for key in keys]

        missing = [i for i, path in enumerate(self.entry_paths) if not os.path.isfile(path)]
        print(f"context distillation: {len(self.pairs)} prompt pairs in {len(set(self.buckets))} buckets, "
              f"{len(self.pairs) - len(missing)} cached, {len(missing)} to generate")
        if missing:
            self._build_cache(trainer, missing, keys)

        if not self._epoch_batches(0):
            raise ValueError(
                f"context distillation: no bucket has at least batch size ({self.batch_size}) pairs, nothing to "
                f"train. Bucket sizes: {self._bucket_sizes()}"
            )

    def _bucket_sizes(self) -> dict[str, int]:
        sizes = {}
        for h, w in self.buckets:
            sizes[f"{w}x{h}"] = sizes.get(f"{w}x{h}", 0) + 1
        return sizes

    @torch.no_grad()
    def _build_cache(self, trainer: "GenericTrainer", missing: list[int], keys: list[str]):
        model = trainer.model
        dtype = model.train_dtype.torch_dtype()

        # 1. text embeddings, all with the text encoder on the GPU once
        trainer.callbacks.on_update_status(f"Context distillation: encoding {len(missing)} prompt pairs")
        model.materialize_only("text_encoder")
        encoded = {}
        for i in tqdm(missing, desc="context distillation: encoding prompts"):
            pair = self.pairs[i]
            embeddings = self.adapter.encode_prompts([pair.student, pair.teacher]).to(device="cpu", dtype=dtype)
            encoded[i] = embeddings.split(1)
        negative = self.adapter.encode_prompts([self.cd.negative_prompt]).to(device=trainer.train_device)

        # 2. teacher latents, with the LoRA switched off
        trainer.model_setup.setup_train_device(model, self.config)
        model.transformer.eval()
        for n, i in enumerate(tqdm(missing, desc="context distillation: teacher latents")):
            trainer.callbacks.on_update_status(f"Context distillation: teacher latents ({n + 1}/{len(missing)})")
            student, teacher = encoded.pop(i)
            height, width = self.buckets[i]
            self.generator.manual_seed((int(keys[i][:12], 16) + self.cd.seed) % (2 ** 63))
            with trainer.model_setup.prior_model(model, self.config):
                latent = self.adapter.generate(
                    conditioning=teacher.to(trainer.train_device),
                    negative_conditioning=negative,
                    cfg=self.cd.teacher_cfg,
                    steps=self.cd.teacher_steps,
                    height=height,
                    width=width,
                    generator=self.generator,
                )
            entry = {"student": student, "teacher": teacher, "latent": latent.to(device="cpu", dtype=torch.float32)}
            # write to a temporary name first, so an interrupted run never leaves a broken entry behind
            tmp_path = self.entry_paths[i] + ".tmp"
            torch.save(entry, tmp_path)
            os.replace(tmp_path, self.entry_paths[i])

        trainer.model_setup.setup_train_device(model, self.config)

    # ---------------------------------------------------------------- epochs

    def _epoch_batches(self, epoch: int) -> list[list[int]]:
        """Shuffled batches of pair indices; each batch is one bucket, incomplete batches are dropped (as for images)."""
        rand = random.Random(f"{self.cd.seed}:{epoch}")
        by_bucket: dict[tuple[int, int], list[int]] = {}
        for i, bucket in enumerate(self.buckets):
            by_bucket.setdefault(bucket, []).append(i)
        batches = []
        for bucket in sorted(by_bucket):
            indices = by_bucket[bucket]
            rand.shuffle(indices)
            batches.extend(indices[start:start + self.batch_size]
                           for start in range(0, len(indices) - self.batch_size + 1, self.batch_size))
        rand.shuffle(batches)
        return batches

    def on_epoch_start(self, trainer: "GenericTrainer", train_progress: TrainProgress):
        if self.is_standalone():
            # follow the trainer's epochs, so a resumed run continues inside the same epoch order
            self._epoch = train_progress.epoch
            self._queue = self._epoch_batches(self._epoch)[train_progress.epoch_step:]

    def _next_batch(self) -> list[int]:
        if not self._queue:
            # mixed with image training: the pairs run through their own epochs
            self._queue = self._epoch_batches(self._epoch)
            self._epoch += 1
        return self._queue.pop(0)

    def _load_batch(self, indices: list[int], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        entries = [torch.load(self.entry_paths[i], map_location="cpu", weights_only=True) for i in indices]
        latents = torch.cat([e["latent"] for e in entries]).to(device)
        student = torch.cat([e["student"] for e in entries]).to(device)
        teacher = torch.cat([e["teacher"] for e in entries]).to(device)
        return latents, student, teacher

    # ---------------------------------------------------------------- training

    def current_weight(self, global_step: int) -> float:
        stop = self.cd.stop_after_steps
        if stop > 0 and global_step >= stop:
            return 0.0
        weight = self.cd.loss_weight
        if self.cd.decay_to_zero and stop > 0:
            weight *= 1.0 - global_step / stop
        return weight

    def compute_loss(
            self,
            trainer: "GenericTrainer",
            batch: dict | None,
            train_progress: TrainProgress,
    ) -> Tensor | None:
        weight = self.current_weight(train_progress.global_step)
        if weight <= 0.0:
            return None

        device = trainer.train_device
        clean, student, teacher = self._load_batch(self._next_batch(), device)

        self.generator.manual_seed(self.cd.seed * 1_000_003 + train_progress.global_step)
        noise = torch.randn(clean.shape, generator=self.generator, device=device, dtype=torch.float32)
        sigma = self.adapter.sample_sigma(clean, self.generator)
        sigma_b = sigma.view(-1, *([1] * (clean.dim() - 1)))
        noisy = (1.0 - sigma_b) * clean + sigma_b * noise

        with torch.no_grad(), trainer.model_setup.prior_model(trainer.model, self.config):
            target = self.adapter.velocity(noisy, sigma, teacher).float()

        predicted = self.adapter.velocity(noisy, sigma, student)
        per_sample = F.mse_loss(predicted.float(), target, reduction="none").mean(dim=list(range(1, target.dim())))
        self.band_loss.record(per_sample, sigma)
        return per_sample.mean() * weight

    def report_to_tensorboard(self, tensorboard: SummaryWriter, global_step: int):
        self.band_loss.report(tensorboard, global_step)
        tensorboard.add_scalar("context_distillation/weight", self.current_weight(global_step), global_step)
