import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from typing import TYPE_CHECKING

from modules.trainer.extension.FlowModelAdapter import FlowModelAdapter, create_flow_model_adapter
from modules.trainer.extension.noise_bands import NoiseBandLoss
from modules.trainer.extension.TrainingExtension import TrainingExtension
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from modules.trainer.GenericTrainer import GenericTrainer

CACHE_DIR_NAME = "soar"
CACHE_VERSION = "soar-v1"

# correction points are never placed below this noise level: their target divides by it
MIN_CORRECTION_SIGMA = 0.02


def correction_point(
        clean: Tensor,
        noise: Tensor,
        rolled_out: Tensor,
        rollout_sigma: Tensor,
        sigma: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    A SOAR correction point at noise level sigma (>= rollout_sigma): the rolled-out latent, re-noised toward the same
    noise, and the velocity that leads from it straight to the clean latent.

        z = (1 - a) * rolled_out + a * noise,   a = (sigma - rollout_sigma) / (1 - rollout_sigma)
        v = (z - clean) / sigma

    If the rollout landed exactly on the training trajectory, z = (1 - sigma) * clean + sigma * noise and v is the
    normal target noise - clean.
    """
    def b(x: Tensor) -> Tensor:
        return x.view(-1, *([1] * (clean.dim() - 1)))

    alpha = ((sigma - rollout_sigma) / (1.0 - rollout_sigma).clamp_min(1e-8)).clamp(0.0, 1.0)
    latent = (1.0 - b(alpha)) * rolled_out + b(alpha) * noise
    target = (latent - clean) / b(sigma)
    return latent, target


class Soar(TrainingExtension):
    """
    SOAR (Tencent HY-SOAR, arXiv 2604.12617), as an addition to normal image training.

    Normal training only shows the model noisy versions of real images, but at sampling time it continues from its
    own slightly wrong states. On SOAR steps, after the normal prediction:
      1. rollout: from the step's noisy latent, the model takes one sampling step itself, with CFG against the empty
         prompt (the conditional half reuses the normal prediction, so only the empty-prompt pass is extra)
      2. N correction points: that state, re-noised toward the same noise at random levels up to pure noise
      3. each point is trained toward the velocity that leads straight back to the real image

    The loss of a SOAR step is (normal + weight * sum(points)) / (1 + weight * N), as in the paper. Each point is
    back-propagated on its own, so the memory peak stays that of a normal step.

    Unlike the paper, samples below min_noise (and rollouts ending at zero noise) get normal training only: there the
    correction target divides a small texture error by a tiny noise level, which with small batches caused loss
    spikes and a mesh texture in samples.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.soar = config.soar
        if self.soar.correction_points < 1:
            raise ValueError("soar.correction_points must be at least 1")
        if self.soar.rollout_steps < 1:
            raise ValueError("soar.rollout_steps must be at least 1")
        if self.soar.weight <= 0:
            raise ValueError("soar.weight must be positive")
        if not 0.0 <= self.soar.min_noise < 1.0:
            raise ValueError("soar.min_noise must be in [0, 1)")
        if config.train_text_encoder_or_embedding():
            raise NotImplementedError("SOAR needs a frozen text encoder (the correction points reuse its output)")

        self.adapter: FlowModelAdapter | None = None
        self.empty_conditioning: Tensor | None = None
        self.schedule: Tensor | None = None
        self.generator: torch.Generator | None = None
        self.cache_dir = os.path.join(config.cache_dir, CACHE_DIR_NAME)

        self._pending: dict | None = None
        self.band_loss = NoiseBandLoss("soar", "loss/soar")
        self._rollout_error = 0.0
        self._rollout_count = 0

    # ---------------------------------------------------------------- setup

    def _empty_prompt_path(self) -> str:
        config = self.config
        key = json.dumps([
            CACHE_VERSION, str(config.model_type), config.base_model_name,
            str(config.text_encoder.weight_dtype), str(config.train_dtype),
        ])
        return os.path.join(self.cache_dir, f"empty_prompt_{hashlib.sha256(key.encode()).hexdigest()[:16]}.pt")

    def on_train_start(self, trainer: "GenericTrainer"):
        model = trainer.model
        self.adapter = create_flow_model_adapter(model, trainer.model_setup, self.config, trainer.train_device)
        self.schedule = self.adapter.sampling_sigmas(self.soar.rollout_steps)
        self.generator = torch.Generator(device=trainer.train_device)

        if self.config.clear_cache_before_training and self.config.latent_caching and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        path = self._empty_prompt_path()
        if not os.path.isfile(path):
            trainer.callbacks.on_update_status("SOAR: encoding the empty prompt")
            model.materialize_only("text_encoder")
            with torch.no_grad():
                empty = self.adapter.encode_prompts([""]).to(device="cpu", dtype=model.train_dtype.torch_dtype())
            torch.save(empty, path + ".tmp")
            os.replace(path + ".tmp", path)
            trainer.model_setup.setup_train_device(model, self.config)
        self.empty_conditioning = torch.load(path, map_location="cpu", weights_only=True).to(trainer.train_device)

    # ---------------------------------------------------------------- training

    def _update_step(self, train_progress: TrainProgress) -> int:
        return train_progress.global_step // max(1, self.config.gradient_accumulation_steps)

    def fires(self, train_progress: TrainProgress) -> bool:
        return self._update_step(train_progress) % max(1, self.soar.every_n_steps) == 0

    def _normalizer(self, active_fraction: float) -> float:
        # paper: (sum(normal) + weight * sum(points)) / (B + weight * points), with points counted over active samples
        return 1.0 + self.soar.weight * self.soar.correction_points * active_fraction

    def adjust_image_loss(
            self,
            trainer: "GenericTrainer",
            batch: dict,
            model_output_data: dict,
            loss: Tensor,
            train_progress: TrainProgress,
    ) -> Tensor:
        self._pending = None
        if not self.fires(train_progress):
            return loss
        if 'prior_target' in model_output_data or model_output_data.get('loss_type') != 'target':
            # prior-prediction samples replace the target, so the real image can not be recovered from it
            return loss

        # SOAR only for samples starting at min_noise or above, and whose rollout step does not end at zero noise
        # (there the correction target would divide the error of the model's own clean image by a tiny noise level)
        timestep = model_output_data['timestep'].detach()
        sigma = self.adapter.sigma_from_timestep(timestep)
        rollout_sigma = self.adapter.next_sampling_sigma(sigma, self.schedule)
        active = (sigma >= self.soar.min_noise) & (rollout_sigma > 0)
        if not bool(active.any()):
            return loss
        active_fraction = float(active.float().mean())

        # only detached tensors are kept: the correction points run after this loss has been back-propagated
        self._pending = {
            'batch': batch,
            'timestep': timestep,
            'sigma': sigma,
            'rollout_sigma': rollout_sigma,
            'active': active,
            'active_fraction': active_fraction,
            'predicted': model_output_data['predicted'].detach(),
            'target': model_output_data['target'].detach(),
            'train_progress': train_progress,
        }
        return loss / self._normalizer(active_fraction)

    @torch.no_grad()
    def _rollout(self, trainer: "GenericTrainer", pending: dict) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        adapter = self.adapter
        batch = pending['batch']
        clean = adapter.clean_latents(batch).float()
        noise = pending['target'].float() + clean
        timestep = pending['timestep']
        sigma = pending['sigma'].to(clean.device)
        b = sigma.view(-1, *([1] * (clean.dim() - 1)))
        noisy = (1.0 - b) * clean + b * noise

        velocity = pending['predicted'].float()
        if self.soar.rollout_cfg != 1.0:
            empty = self.empty_conditioning.expand(clean.shape[0], *self.empty_conditioning.shape[1:])
            uncond = adapter.velocity_at_timestep(noisy, timestep, empty).float()
            velocity = uncond + self.soar.rollout_cfg * (velocity - uncond)

        rollout_sigma = pending['rollout_sigma'].to(clean.device)
        rb = rollout_sigma.view_as(b)
        rolled_out = noisy + (rb - b) * velocity

        # how far the model's own step lands from the training trajectory (the error SOAR corrects)
        on_trajectory = (1.0 - rb) * clean + rb * noise
        active = pending['active'].to(clean.device)
        self._rollout_error += float((rolled_out - on_trajectory)[active].pow(2).mean().sqrt())
        self._rollout_count += 1
        return clean, noise, rolled_out, rollout_sigma, sigma

    def _correction_losses(self, trainer: "GenericTrainer", pending: dict) -> Iterator[Tensor]:
        adapter = self.adapter
        batch, train_progress = pending['batch'], pending['train_progress']
        clean, noise, rolled_out, rollout_sigma, _ = self._rollout(trainer, pending)
        with torch.no_grad():
            conditioning = adapter.batch_conditioning(batch, train_progress)

        # inactive samples keep their place in the batch (same shapes for compiled graphs) with zero loss weight
        active = pending['active'].to(clean.device)
        point_batch = dict(batch)
        point_batch['loss_weight'] = batch['loss_weight'] * active.to(batch['loss_weight'].dtype)

        self.generator.manual_seed(train_progress.global_step * 1_000_003 + 7)
        low = rollout_sigma.clamp_min(MIN_CORRECTION_SIGMA)
        point_weight = self.soar.weight / self._normalizer(pending['active_fraction'])
        for _ in range(self.soar.correction_points):
            with torch.no_grad():
                u = torch.rand(clean.shape[0], generator=self.generator, device=self.generator.device)
                sigma = low + u.to(low.device) * (1.0 - low)
                # snap to a discrete training timestep, so the model sees it exactly as in normal training
                timestep = adapter.timestep_from_sigma(sigma)
                sigma = adapter.sigma_from_timestep(timestep).to(clean.device)
                latent, target = correction_point(clean, noise, rolled_out, rollout_sigma, sigma)

            predicted = adapter.velocity_at_timestep(latent, timestep, conditioning)
            data = {'loss_type': 'target', 'timestep': timestep, 'predicted': predicted, 'target': target}
            # OneTrainer's own loss settings (MSE/MAE/huber strengths, masks, loss weights) apply as for the image loss
            loss = trainer.model_setup.calculate_loss(trainer.model, point_batch, data, self.config)

            with torch.no_grad():
                per_sample = (predicted.float() - target).pow(2).mean(dim=list(range(1, target.dim())))
                self.band_loss.record(per_sample[active], sigma[active])
            yield loss * point_weight

    def compute_loss(
            self,
            trainer: "GenericTrainer",
            batch: dict | None,
            train_progress: TrainProgress,
    ) -> Iterator[Tensor] | None:
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        return self._correction_losses(trainer, pending)

    def report_to_tensorboard(self, tensorboard: SummaryWriter, global_step: int):
        self.band_loss.report(tensorboard, global_step)
        if self._rollout_count > 0:
            tensorboard.add_scalar("soar/rollout_error", self._rollout_error / self._rollout_count, global_step)
            self._rollout_error, self._rollout_count = 0.0, 0
