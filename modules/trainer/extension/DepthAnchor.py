import os
from typing import TYPE_CHECKING

from modules.trainer.extension.FlowModelAdapter import FlowModelAdapter, create_flow_model_adapter
from modules.trainer.extension.noise_bands import NoiseBandLoss
from modules.trainer.extension.perceptual.depth_models import create_depth_estimator
from modules.trainer.extension.TrainingExtension import TrainingExtension
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from modules.trainer.GenericTrainer import GenericTrainer


def normalize_depth(depth: Tensor, eps: float = 1e-6) -> Tensor:
    """Per image: subtract the median, divide by the mean absolute deviation (scale and shift invariant)."""
    flat = depth.flatten(1)
    median = flat.median(dim=1, keepdim=True).values
    scale = (flat - median).abs().mean(dim=1, keepdim=True).clamp_min(eps)
    return ((flat - median) / scale).view_as(depth)


def depth_anchor_loss(predicted: Tensor, target: Tensor, gradient_weight: float, scales: int = 4) -> Tensor:
    """
    Per-sample loss between two depth maps (B, H, W): scale-and-shift-invariant L1 plus a multi-scale gradient
    matching term (as in MiDaS). Both maps are normalized on their own, so only their shape is compared.
    """
    residual = normalize_depth(predicted.float()) - normalize_depth(target.float())
    loss = residual.abs().mean(dim=(1, 2))
    if gradient_weight > 0:
        gradient_loss = torch.zeros_like(loss)
        r = residual.unsqueeze(1)
        for scale in range(scales):
            if scale > 0:
                if min(r.shape[-2:]) < 4:
                    break
                r = F.avg_pool2d(r, 2)
            gradient_loss = gradient_loss \
                + (r[..., :, 1:] - r[..., :, :-1]).abs().mean(dim=(1, 2, 3)) \
                + (r[..., 1:, :] - r[..., :-1, :]).abs().mean(dim=(1, 2, 3))
        loss = loss + gradient_weight * gradient_loss / scales
    return loss


def _to_display(depth: Tensor) -> Tensor:
    d = depth.float()
    d = (d - d.min()) / (d.max() - d.min()).clamp_min(1e-6)
    return d.unsqueeze(0).expand(3, -1, -1)


class DepthAnchor(TrainingExtension):
    """
    Adds a depth-structure loss to normal image training:

        x0_pred = predicted clean latent of the training step (no extra model forward needed)
        loss   += weight * D(depth(decode(x0_pred)), depth(decode(x0)))

    decode() is a tiny latent decoder, depth() a frozen depth estimator. The target side runs without gradients.
    Gradients flow through the estimator and the decoder into the LoRA.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.da = config.depth_anchor
        if self.da.loss_split and self.da.every_n_steps < 2:
            raise ValueError("depth_anchor.loss_split needs every_n_steps >= 2 (2 = alternate diffusion / anchor steps)")
        if not 0.0 <= self.da.min_noise < self.da.max_noise <= 1.0:
            raise ValueError("depth_anchor needs 0 <= min_noise < max_noise <= 1")

        self.adapter: FlowModelAdapter | None = None
        self.decoder: nn.Module | None = None
        self.depth: nn.Module | None = None

        self.band_loss = NoiseBandLoss("depth_anchor", "loss/depth_anchor")
        self._preview: tuple[Tensor, float] | None = None

    def on_train_start(self, trainer: "GenericTrainer"):
        self.adapter = create_flow_model_adapter(trainer.model, trainer.model_setup, self.config, trainer.train_device)
        dtype = trainer.model.train_dtype.torch_dtype()
        model_dir = os.path.join(self.config.cache_dir, "perceptual_models")
        trainer.callbacks.on_update_status("Depth anchor: loading the tiny decoder and the depth model")
        self.decoder = self.adapter.create_tiny_decoder(self.da.decoder_path, model_dir) \
            .to(device=trainer.train_device, dtype=dtype)
        self.depth = create_depth_estimator(self.da.depth_model, self.da.depth_resolution, trainer.train_device, dtype)

    def _update_step(self, train_progress: TrainProgress) -> int:
        return train_progress.global_step // max(1, self.config.gradient_accumulation_steps)

    def fires(self, train_progress: TrainProgress) -> bool:
        n = max(1, self.da.every_n_steps)
        # with loss_split, update step 0 is a diffusion step and the anchor takes every n-th step after it
        offset = 1 if self.da.loss_split else 0
        return (self._update_step(train_progress) - offset) % n == 0 and self._update_step(train_progress) >= offset

    def adjust_image_loss(
            self,
            trainer: "GenericTrainer",
            batch: dict,
            model_output_data: dict,
            loss: Tensor,
            train_progress: TrainProgress,
    ) -> Tensor:
        if self.da.weight <= 0 or not self.fires(train_progress):
            return loss
        if 'prior_target' in model_output_data or model_output_data.get('loss_type') != 'target':
            # prior-prediction samples replace the target, so x0 can not be recovered from it
            return loss

        predicted, clean, sigma = self.adapter.predicted_clean_latents(batch, model_output_data)
        in_window = (sigma >= self.da.min_noise) & (sigma <= self.da.max_noise)
        if not bool(in_window.any()):
            return loss
        index = in_window.nonzero().squeeze(1)
        predicted, clean, sigma = predicted[index], clean[index], sigma[index]

        dtype = trainer.model.train_dtype.torch_dtype()
        with trainer.model.autocast_context:
            with torch.no_grad():
                target_image = self.decoder(clean.to(dtype))
                target_depth = self.depth(target_image)
            predicted_image = self.decoder(predicted.to(dtype))
            predicted_depth = self.depth(predicted_image)

        per_sample = depth_anchor_loss(predicted_depth, target_depth, self.da.gradient_weight)
        self.band_loss.record(per_sample, sigma)
        if self.da.preview_every > 0 and self._update_step(train_progress) % self.da.preview_every == 0:
            self._store_preview(target_image, target_depth, predicted_image, predicted_depth, float(sigma[0]))

        # samples outside the noise window contribute nothing, so scale by the fraction inside
        anchor = per_sample.sum() / in_window.numel() * self.da.weight
        if self.da.loss_split:
            return anchor
        return loss + anchor

    @torch.no_grad()
    def _store_preview(self, image: Tensor, depth: Tensor, predicted_image: Tensor, predicted_depth: Tensor,
                       sigma: float):
        size = image.shape[-2:]

        def resize(d: Tensor) -> Tensor:
            return F.interpolate(d[None, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0]

        tiles = [image[0].float().clamp(0, 1), _to_display(resize(depth[0])),
                 predicted_image[0].float().clamp(0, 1), _to_display(resize(predicted_depth[0]))]
        self._preview = (torch.cat(tiles, dim=2).cpu(), sigma)

    def report_to_tensorboard(self, tensorboard: SummaryWriter, global_step: int):
        self.band_loss.report(tensorboard, global_step)
        if self._preview is not None:
            image, sigma = self._preview
            tensorboard.add_image("depth_anchor/preview", image, global_step)
            tensorboard.add_scalar("depth_anchor/preview_sigma", sigma, global_step)
            self._preview = None
