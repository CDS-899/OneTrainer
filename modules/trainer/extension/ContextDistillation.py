import json
import os
import random
from typing import TYPE_CHECKING

from modules.trainer.extension.FlowModelAdapter import FlowModelAdapter, create_flow_model_adapter
from modules.trainer.extension.TrainingExtension import TrainingExtension
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.tqdm_util import tqdm
from modules.util.TrainProgress import TrainProgress

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from modules.trainer.GenericTrainer import GenericTrainer


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


SUBJECT_PLACEHOLDER = "{subject}"


def _as_list(value) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def _expand_object(obj: dict, base_dir: str) -> list[tuple[str, str]]:
    """
    Turns one entry of the prompts file into (short, dense) pairs.

    Plain pair:      {"short": "...", "dense": "..."}
    Preservation:    {"preserve": "a man reading a book"}   -> short == dense, keeps this prompt like the base model
    Trigger concept: {"trigger": "Taras Shevchenko",
                      "description": "a bald man with a long drooping mustache, ..."   (or a list of descriptions),
                      "contexts": ["{subject} reading a book in a garden", ...],      (and/or "contexts_file")
                      "generic": "an old man"}                                         (optional, or a list)
        For every context and description:  short = context with {subject} -> trigger,
                                             dense = context with {subject} -> description.
        With "generic", every context also yields a preservation pair with {subject} -> generic, so the
        LoRA learns that only the trigger changes and the generic words keep their base meaning.
    """
    pairs = []

    if "trigger" in obj:
        trigger = _as_text(obj["trigger"])
        descriptions = [_as_text(d) for d in _as_list(obj.get("description", obj.get("descriptions")))]
        contexts = [_as_text(c) for c in _as_list(obj.get("contexts"))]
        if obj.get("contexts_file"):
            contexts_file = obj["contexts_file"]
            if not os.path.isabs(contexts_file):
                contexts_file = os.path.join(base_dir, contexts_file)
            contexts += _read_lines(contexts_file)
        if not contexts:
            contexts = [SUBJECT_PLACEHOLDER]
        if not trigger or not descriptions:
            raise ValueError(f"trigger entry needs a non-empty \"trigger\" and \"description\": {obj}")

        for context in contexts:
            if SUBJECT_PLACEHOLDER not in context:
                raise ValueError(f"context without {SUBJECT_PLACEHOLDER}: '{context}'")
            short = context.replace(SUBJECT_PLACEHOLDER, trigger)
            pairs.extend((short, context.replace(SUBJECT_PLACEHOLDER, description)) for description in descriptions)
            for generic in _as_list(obj.get("generic")):
                generic_prompt = context.replace(SUBJECT_PLACEHOLDER, _as_text(generic))
                pairs.append((generic_prompt, generic_prompt))
        return pairs

    if "preserve" in obj:
        return [(p, p) for p in (_as_text(v) for v in _as_list(obj["preserve"])) if p]

    short = _as_text(obj.get("short", obj.get("short_prompt", obj.get("prompt"))))
    dense = _as_text(obj.get("dense", obj.get("dense_prompt")))
    if short and dense:
        pairs.append((short, dense))
    return pairs


def load_prompt_pairs(path: str) -> list[tuple[str, str]]:
    """
    Reads (short, dense) prompt pairs from a .jsonl file (one object per line) or a .json file
    (a list of objects). See _expand_object() for the entry types. A dense prompt can itself be a
    JSON object, it is serialized as compact JSON.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"context distillation prompts file not found: '{path}'")

    with open(path, encoding="utf-8") as f:
        if path.lower().endswith(".jsonl"):
            objects = []
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    objects.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_number}: invalid JSON ({e})") from e
        else:
            objects = json.load(f)

    base_dir = os.path.dirname(os.path.abspath(path))
    pairs = []
    for obj in objects:
        pairs.extend(_expand_object(obj, base_dir))

    if not pairs:
        raise ValueError(f"no usable prompt pairs in '{path}'")
    return pairs


def parse_resolution(resolution: str, quantization: int) -> tuple[int, int]:
    """'512' -> (512, 512), '768x512' -> (height 512, width 768) as WxH, '512,768' -> first entry."""
    first = resolution.split(",")[0].strip()
    if "x" in first:
        width, height = (int(v) for v in first.split("x"))
    else:
        width = height = int(first)
    height = max(quantization, height // quantization * quantization)
    width = max(quantization, width // quantization * quantization)
    return height, width


class TeacherLatentPool:
    """Fixed-size FIFO pool of clean teacher latents, each tagged with the prompt pair it was generated from."""

    def __init__(self, size: int):
        self.size = max(1, size)
        self.latents: list[Tensor] = []
        self.pair_indices: list[int] = []
        self.next_replace = 0

    def __len__(self):
        return len(self.latents)

    def is_full(self) -> bool:
        return len(self.latents) >= self.size

    def add(self, latent: Tensor, pair_index: int):
        if not self.is_full():
            self.latents.append(latent)
            self.pair_indices.append(pair_index)
        else:
            # replace the oldest entry
            self.latents[self.next_replace] = latent
            self.pair_indices[self.next_replace] = pair_index
            self.next_replace = (self.next_replace + 1) % self.size

    def sample(self, count: int, rng: random.Random) -> tuple[Tensor, list[int]]:
        indices = [rng.randrange(len(self.latents)) for _ in range(count)]
        latents = torch.cat([self.latents[i] for i in indices], dim=0)
        return latents, [self.pair_indices[i] for i in indices]


class ContextDistillation(TrainingExtension):
    """
    The student (base model + LoRA) sees the short prompt, the teacher (same model, LoRA disabled) sees the
    dense prompt, both at the same noisy latent. The student learns to match the teacher's velocity:

        loss = || v_student(x_t, sigma, short) - v_teacher(x_t, sigma, dense) ||^2

    x_t comes from a pool of teacher latents (generated from the dense prompts during training, never decoded),
    re-noised with fresh noise at a noise level drawn from the configured timestep distribution. The pool only
    decides *where* the student is trained; the target is always the teacher's live prediction.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.cd = config.context_distillation

        if config.training_method != TrainingMethod.LORA:
            raise NotImplementedError(
                "context distillation needs LoRA training: the teacher is the base model with the LoRA switched off"
            )

        self.pairs = load_prompt_pairs(self.cd.prompts_path)
        preserved = sum(short == dense for short, dense in self.pairs)
        print(f"context distillation: {len(self.pairs)} prompt pairs ({preserved} preservation pairs)")
        self.rng = random.Random(self.cd.seed)

        self.adapter: FlowModelAdapter | None = None
        self.generator: torch.Generator | None = None
        self.short_conditioning: list[Tensor] = []
        self.dense_conditioning: list[Tensor] = []
        self.negative_conditioning: Tensor | None = None
        self.pool = TeacherLatentPool(self.cd.pool_size)
        self.height = 0
        self.width = 0

        self._last_refresh_step = -1
        self._loss_sum = 0.0
        self._loss_count = 0
        self._generated = 0

    def is_standalone(self) -> bool:
        return self.cd.standalone

    def standalone_epoch_length(self) -> int:
        return max(1, self.cd.standalone_epoch_length)

    # ---------------------------------------------------------------- setup

    def on_train_start(self, trainer: "GenericTrainer"):
        model = trainer.model
        self.adapter = create_flow_model_adapter(model, trainer.model_setup, self.config, trainer.train_device)
        self.height, self.width = parse_resolution(
            self.cd.resolution or self.config.resolution, self.adapter.resolution_quantization()
        )
        self.generator = torch.Generator(device=trainer.train_device)
        self.generator.manual_seed(self.cd.seed)

        self._encode_prompts(trainer)

        trainer.model_setup.setup_train_device(model, self.config)
        model.transformer.eval()  # no dropout etc. while generating teacher latents
        for i in tqdm(range(self.pool.size), desc="context distillation: teacher latents"):
            trainer.callbacks.on_update_status(f"Context distillation: generating teacher latents ({i + 1}/{self.pool.size})")
            self._generate_teacher_latent(trainer)
        trainer.model_setup.setup_train_device(model, self.config)

    @torch.no_grad()
    def _encode_prompts(self, trainer: "GenericTrainer"):
        trainer.callbacks.on_update_status(f"Context distillation: encoding {len(self.pairs)} prompt pairs")
        trainer.model.materialize_only("text_encoder")

        chunk = 8
        dtype = trainer.model.train_dtype.torch_dtype()
        for start in tqdm(range(0, len(self.pairs), chunk), desc="context distillation: encoding prompts"):
            pairs = self.pairs[start:start + chunk]
            # kept in RAM in the training dtype; moved to the GPU per step
            short = self.adapter.encode_prompts([s for s, _ in pairs]).to(device="cpu", dtype=dtype)
            dense = self.adapter.encode_prompts([d for _, d in pairs]).to(device="cpu", dtype=dtype)
            self.short_conditioning.extend(short.split(1))
            self.dense_conditioning.extend(dense.split(1))
        self.negative_conditioning = self.adapter.encode_prompts([self.cd.negative_prompt]).to(trainer.train_device)

    @torch.no_grad()
    def _generate_teacher_latent(self, trainer: "GenericTrainer"):
        pair_index = self.rng.randrange(len(self.pairs))
        dense = self.dense_conditioning[pair_index].to(trainer.train_device)
        with trainer.model_setup.prior_model(trainer.model, self.config):
            latent = self.adapter.generate(
                conditioning=dense,
                negative_conditioning=self.negative_conditioning,
                cfg=self.cd.teacher_cfg,
                steps=self.cd.teacher_steps,
                height=self.height,
                width=self.width,
                generator=self.generator,
            )
        self.pool.add(latent.detach(), pair_index)
        self._generated += 1

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

        step = train_progress.global_step
        if self.cd.pool_refresh_every > 0 and step > 0 and step % self.cd.pool_refresh_every == 0 \
                and step != self._last_refresh_step:
            self._last_refresh_step = step
            was_training = trainer.model.transformer.training
            trainer.model.transformer.eval()
            self._generate_teacher_latent(trainer)
            trainer.model.transformer.train(was_training)

        device = trainer.train_device
        clean, pair_indices = self.pool.sample(self.cd.batch_size, self.rng)
        noise = torch.randn(clean.shape, generator=self.generator, device=device, dtype=torch.float32)
        sigma = self.adapter.sample_sigma(clean, self.generator)
        sigma_b = sigma.view(-1, *([1] * (clean.dim() - 1)))
        noisy = (1.0 - sigma_b) * clean + sigma_b * noise

        short = torch.cat([self.short_conditioning[i] for i in pair_indices]).to(device)
        dense = torch.cat([self.dense_conditioning[i] for i in pair_indices]).to(device)

        with torch.no_grad(), trainer.model_setup.prior_model(trainer.model, self.config):
            target = self.adapter.velocity(noisy, sigma, dense).float()
            if self.cd.target_cfg != 1.0:
                negative = self.negative_conditioning.expand(len(pair_indices), *self.negative_conditioning.shape[1:])
                uncond = self.adapter.velocity(noisy, sigma, negative).float()
                target = uncond + self.cd.target_cfg * (target - uncond)

        predicted = self.adapter.velocity(noisy, sigma, short)
        loss = F.mse_loss(predicted.float(), target)

        self._loss_sum += loss.detach()
        self._loss_count += 1
        return loss * weight

    def report_to_tensorboard(self, tensorboard: SummaryWriter, global_step: int):
        if self._loss_count > 0:
            tensorboard.add_scalar("loss/context_distillation", float(self._loss_sum) / self._loss_count, global_step)
            self._loss_sum = 0.0
            self._loss_count = 0
        tensorboard.add_scalar("context_distillation/weight", self.current_weight(global_step), global_step)
        tensorboard.add_scalar("context_distillation/teacher_latents_generated", self._generated, global_step)
