import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modules.trainer.extension.ContextDistillation import _as_text, parse_ar, resolve_bucket_for
from modules.trainer.extension.FlowModelAdapter import FlowModelAdapter, create_flow_model_adapter
from modules.trainer.extension.noise_bands import NoiseBandLoss
from modules.trainer.extension.TrainingExtension import TrainingExtension
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.tqdm_util import tqdm
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from modules.trainer.GenericTrainer import GenericTrainer

CACHE_VERSION = "on-policy-v1"


@dataclass(frozen=True)
class DistillationPrompt:
    prompt: str
    ar: tuple[float, float] | None  # (width, height), None = square


def load_prompts(path: str, name: str) -> list[DistillationPrompt]:
    """
    Reads prompts from a .jsonl file, one object per line:
        {"prompt": "...", "ar": "3:4"}
    "ar" (width:height) is optional, without it the prompt is trained square. "prompt" may be a JSON object
    (structured prompts), it is passed to the model as compact JSON.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{name}: prompts file not found: '{path}'")
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({e})") from e
            prompt = _as_text(obj.get("prompt"))
            if not prompt:
                raise ValueError(f"{path}:{line_number}: needs a non-empty \"prompt\"")
            prompts.append(DistillationPrompt(prompt, parse_ar(obj["ar"]) if obj.get("ar") else None))
    if not prompts:
        raise ValueError(f"{name}: no prompts in '{path}'")
    return prompts


def velocity_distance(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-sample RMS distance sqrt(mean((a - b)^2) + eps), as in the SMD reference code (beta = 1)."""
    return ((a.float() - b.float()).pow(2).flatten(1).mean(dim=1) + eps).sqrt()


class OnPolicyDistillation(TrainingExtension):
    """
    Prompt-only, on-policy velocity distillation (the core of SMD, arXiv 2610.18472, without its Pivot).

    The student (base model + LoRA) samples from noise with `steps` Euler steps on the model's own schedule. For each
    of its steps it makes one velocity prediction v at the step start. The teacher (the same model with the LoRA off,
    with CFG against the negative prompt) is queried at random points along that step, at the states the student's
    own prediction leads to, and v learns to match the teacher there:

        x_tau = x - tau * (sigma_i - sigma_j) * v,   s = sigma_i - tau * (sigma_i - sigma_j)
        loss  = d(v, teacher_cfg(x_tau, s)) - repulsion_weight * d(v, teacher_negative(x_tau, s))

    d is the per-sample RMS distance. One training step trains one student step: a batch of prompts runs through
    all `steps` steps in consecutive training steps, continuing from where the previous step's prediction led
    (detached), then the next batch starts from fresh noise.

    Training is standalone (no image data set). An epoch runs every prompt through its trajectory once.
    """

    def __init__(
            self,
            config: TrainConfig,
            name: str,
            block,
            steps: int,
            teacher_cfg: float,
            repulsion_weight: float,
            queries: int,
    ):
        self.config = config
        self.name = name
        self.block = block
        self.steps = steps
        self.teacher_cfg = teacher_cfg
        self.repulsion_weight = repulsion_weight
        self.queries = queries

        if config.training_method != TrainingMethod.LORA:
            raise NotImplementedError(f"{name} needs LoRA training: the teacher is the base model with the LoRA off")
        if steps < 1 or queries < 1:
            raise ValueError(f"{name}: steps and queries must be at least 1")
        if not 0.0 <= repulsion_weight < 1.0:
            raise ValueError(f"{name}: repulsion_weight must be in [0, 1)")

        self.prompts = load_prompts(block.prompts_path, name)
        self.batch_size = max(1, config.batch_size)
        self.cache_dir = os.path.join(config.cache_dir, name)

        self.adapter: FlowModelAdapter | None = None
        self.generator: torch.Generator | None = None
        self.schedule: Tensor | None = None
        self.entry_paths: list[str] = []
        self.negative: Tensor | None = None
        self.buckets: list[tuple[int, int]] = []

        self._epoch = 0
        self._queue: list[list[int]] = []
        self._trajectory: dict | None = None

        self.band_loss = NoiseBandLoss(name, f"loss/{name}")
        self._repulsion_sum = 0.0
        self._repulsion_count = 0

    def is_standalone(self) -> bool:
        return True

    def standalone_epoch_length(self) -> int:
        return len(self._epoch_batches(0)) * self.steps

    # ---------------------------------------------------------------- cache

    def _key(self, *parts) -> str:
        config = self.config
        data = json.dumps([
            CACHE_VERSION, str(config.model_type), config.base_model_name,
            str(config.text_encoder.weight_dtype), str(config.train_dtype), *parts,
        ], ensure_ascii=False)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def on_train_start(self, trainer: "GenericTrainer"):
        model = trainer.model
        self.adapter = create_flow_model_adapter(model, trainer.model_setup, self.config, trainer.train_device)
        self.generator = torch.Generator(device=trainer.train_device)
        self.schedule = self.adapter.sampling_sigmas(self.steps)

        if self.config.clear_cache_before_training and self.config.latent_caching and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        resolution = self.block.resolution or self.config.resolution
        quantization = self.adapter.resolution_quantization()
        self.buckets = [resolve_bucket_for(p.prompt, p.ar, resolution, quantization, self.block.seed)
                        for p in self.prompts]
        self.entry_paths = [os.path.join(self.cache_dir, f"{self._key(p.prompt)}.pt") for p in self.prompts]
        negative_path = os.path.join(self.cache_dir, f"negative_{self._key('negative', self.block.negative_prompt)}.pt")

        missing = [i for i, path in enumerate(self.entry_paths) if not os.path.isfile(path)]
        if missing or not os.path.isfile(negative_path):
            self._encode(trainer, missing, negative_path)
        self.negative = torch.load(negative_path, map_location="cpu", weights_only=True).to(trainer.train_device)

        if not self._epoch_batches(0):
            raise ValueError(f"{self.name}: no bucket has at least batch size ({self.batch_size}) prompts")
        print(f"{self.name}: {len(self.prompts)} prompts in {len(set(self.buckets))} buckets, "
              f"{self.steps} student steps, teacher CFG {self.teacher_cfg}")

    @torch.no_grad()
    def _encode(self, trainer: "GenericTrainer", missing: list[int], negative_path: str):
        model = trainer.model
        dtype = model.train_dtype.torch_dtype()
        trainer.callbacks.on_update_status(f"{self.name}: encoding {len(missing)} prompts")
        model.materialize_only("text_encoder")
        for i in tqdm(missing, desc=f"{self.name}: encoding prompts"):
            embedding = self.adapter.encode_prompts([self.prompts[i].prompt]).to(device="cpu", dtype=dtype)
            torch.save(embedding, self.entry_paths[i] + ".tmp")
            os.replace(self.entry_paths[i] + ".tmp", self.entry_paths[i])
        negative = self.adapter.encode_prompts([self.block.negative_prompt]).to(device="cpu", dtype=dtype)
        torch.save(negative, negative_path + ".tmp")
        os.replace(negative_path + ".tmp", negative_path)
        trainer.model_setup.setup_train_device(model, self.config)

    # ---------------------------------------------------------------- epochs

    def _epoch_batches(self, epoch: int) -> list[list[int]]:
        """Shuffled batches of prompt indices; each batch is one bucket, incomplete batches are dropped."""
        rand = random.Random(f"{self.block.seed}:{epoch}")
        by_bucket: dict[tuple[int, int], list[int]] = {}
        buckets = self.buckets or [(0, 0)] * len(self.prompts)
        for i, bucket in enumerate(buckets):
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
        # follow the trainer's epochs; a resumed run restarts the interrupted trajectory's batch from noise
        self._epoch = train_progress.epoch
        self._queue = self._epoch_batches(self._epoch)[train_progress.epoch_step // self.steps:]
        self._trajectory = None

    def _next_batch(self) -> list[int]:
        if not self._queue:
            self._epoch += 1
            self._queue = self._epoch_batches(self._epoch)
        return self._queue.pop(0)

    def _start_trajectory(self, device: torch.device, global_step: int) -> dict:
        indices = self._next_batch()
        conditioning = torch.cat([
            torch.load(self.entry_paths[i], map_location="cpu", weights_only=True) for i in indices
        ]).to(device)
        height, width = self.buckets[indices[0]]
        self.generator.manual_seed(self.block.seed * 1_000_003 + global_step)
        latents = torch.randn(self.adapter.latent_shape(len(indices), height, width),
                              generator=self.generator, device=device, dtype=torch.float32)
        return {'conditioning': conditioning, 'latents': latents, 'step': 0}

    # ---------------------------------------------------------------- training

    @torch.no_grad()
    def _teacher(self, trainer: "GenericTrainer", latents: Tensor, sigma: Tensor,
                 conditioning: Tensor) -> tuple[Tensor, Tensor | None]:
        """(CFG target, negative reference or None)."""
        need_negative = self.teacher_cfg != 1.0 or self.repulsion_weight > 0
        with trainer.model_setup.prior_model(trainer.model, self.config):
            conditional = self.adapter.velocity(latents, sigma, conditioning).float()
            if not need_negative:
                return conditional, None
            negative = self.adapter.velocity(latents, sigma, self.negative.expand_as(conditioning)).float()
        return negative + self.teacher_cfg * (conditional - negative), negative

    def compute_loss(
            self,
            trainer: "GenericTrainer",
            batch: dict | None,
            train_progress: TrainProgress,
    ) -> Tensor | None:
        device = trainer.train_device
        if self._trajectory is None:
            self._trajectory = self._start_trajectory(device, train_progress.global_step)
        trajectory = self._trajectory
        x, conditioning, i = trajectory['latents'], trajectory['conditioning'], trajectory['step']
        batch_size = x.shape[0]

        sigma_i, sigma_j = self.schedule[i], self.schedule[i + 1]
        delta = sigma_i - sigma_j
        velocity = self.adapter.velocity(x, sigma_i.expand(batch_size), conditioning)
        detached = velocity.detach().float()

        def b(t: Tensor) -> Tensor:
            return t.view(-1, *([1] * (x.dim() - 1)))

        loss = 0.0
        imitation_total = torch.zeros(batch_size, device=device)
        for _ in range(self.queries):
            tau = torch.rand(batch_size, generator=self.generator, device=self.generator.device).to(device)
            query = x - b(tau * delta) * detached
            target, negative = self._teacher(trainer, query, sigma_i - tau * delta, conditioning)
            imitation = velocity_distance(velocity, target)
            term = imitation
            if self.repulsion_weight > 0:
                repulsion = velocity_distance(velocity, negative)
                term = term - self.repulsion_weight * repulsion
                self._repulsion_sum += float(repulsion.detach().mean())
                self._repulsion_count += 1
            loss = loss + term.mean()
            imitation_total += imitation.detach()

        self.band_loss.record(imitation_total / self.queries, sigma_i.expand(batch_size))

        # continue the student's own trajectory
        trajectory['latents'] = (x - delta * detached).detach()
        trajectory['step'] = i + 1
        if trajectory['step'] >= self.steps:
            self._trajectory = None

        return loss / self.queries

    def report_to_tensorboard(self, tensorboard: SummaryWriter, global_step: int):
        self.band_loss.report(tensorboard, global_step)
        if self._repulsion_count > 0:
            tensorboard.add_scalar(f"{self.name}/repulsion_distance", self._repulsion_sum / self._repulsion_count,
                                   global_step)
            self._repulsion_sum, self._repulsion_count = 0.0, 0


class SmdDistillation(OnPolicyDistillation):
    """
    SMD distillation (no Pivot): the student samples at CFG 1 in student_steps steps. With few steps it distills steps
    and CFG at once; with the normal step count it is CFG-only distillation.
    """

    def __init__(self, config: TrainConfig):
        block = config.smd_distillation
        super().__init__(config, "smd_distillation", block, block.student_steps, block.teacher_cfg,
                         block.repulsion_weight, block.queries)
