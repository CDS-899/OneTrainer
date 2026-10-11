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


def align_depth(predicted: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """
    Per image, the least-squares scale s and shift t with s * predicted + t ~ target (MiDaS). Only the predicted map
    is fitted onto the target, so a flat (blurry) prediction gets s ~ 0 instead of being blown up.
    """
    p, g = predicted.flatten(1), target.flatten(1)
    mean_p, mean_g = p.mean(1), g.mean(1)
    var_p = (p * p).mean(1) - mean_p * mean_p
    cov = (p * g).mean(1) - mean_p * mean_g
    s = cov / var_p.clamp_min(1e-6)
    return s, mean_g - s * mean_p


def depth_anchor_loss(
        predicted: Tensor,
        target: Tensor,
        ssi_weight: float = 0.0,
        gradient_weight: float = 1.0,
        scales: int = 6,
) -> Tensor:
    """
    Per-sample loss between two depth maps (B, H, W), as in ai-toolkit-perceptual / MiDaS:
      - ssi:      L1 between the aligned prediction and the target (s, t fitted with gradients)
      - gradient: multi-scale L1 of the depth-gradient difference, on the prediction aligned with detached s, t
    The target is measured in its own units, so the loss scale does not depend on how flat the prediction is.
    """
    predicted, target = predicted.float(), target.float()
    s, t = align_depth(predicted, target)
    loss = predicted.new_zeros(predicted.shape[0])
    if ssi_weight > 0:
        aligned = s.view(-1, 1, 1) * predicted + t.view(-1, 1, 1)
        loss = loss + ssi_weight * (aligned - target).abs().mean(dim=(1, 2))
    if gradient_weight > 0:
        r = (s.detach().view(-1, 1, 1) * predicted + t.detach().view(-1, 1, 1) - target).unsqueeze(1)
        gradient_loss = torch.zeros_like(loss)
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
        self._last_preview_step: int | None = None
        self._push_ratio: float | None = None

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

        per_sample = depth_anchor_loss(predicted_depth, target_depth, self.da.ssi_weight,
                                       self.da.gradient_weight, self.da.gradient_scales)
        self.band_loss.record(per_sample, sigma)

        # samples outside the noise window contribute nothing, so scale by the fraction inside
        anchor = per_sample.sum() / in_window.numel() * self.da.weight

        if self._preview_due(train_progress):
            self._last_preview_step = self._update_step(train_progress)
            self._store_preview(target_image, target_depth, predicted_image, predicted_depth, float(sigma[0]))
            self._push_ratio = self._measure_push_ratio(loss, anchor, model_output_data)

        if self.da.loss_split:
            return anchor
        return loss + anchor

    @staticmethod
    def _measure_push_ratio(loss: Tensor, anchor: Tensor, model_output_data: dict) -> float | None:
        """
        How hard the anchor pushes compared to the diffusion loss: the ratio of their gradient norms at the model's
        output. Everything behind the output (the transformer and the LoRA) is shared by both losses, so this is a
        good proxy for their ratio at the LoRA weights, without an extra backward pass through the transformer.
        """
        predicted = model_output_data['predicted']
        if not (loss.requires_grad and anchor.requires_grad and predicted.requires_grad):
            return None
        g_loss, = torch.autograd.grad(loss, predicted, retain_graph=True, allow_unused=True)
        g_anchor, = torch.autograd.grad(anchor, predicted, retain_graph=True, allow_unused=True)
        if g_loss is None or g_anchor is None:
            return None
        return float(g_anchor.float().norm() / g_loss.float().norm().clamp_min(1e-12))

    def _preview_due(self, train_progress: TrainProgress) -> bool:
        # the first anchored step, then the first anchored step at least preview_every steps after the last preview
        # (the anchor may not run on every step, so a fixed "step % preview_every" could never match)
        if self.da.preview_every <= 0:
            return False
        return self._last_preview_step is None \
            or self._update_step(train_progress) - self._last_preview_step >= self.da.preview_every

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
        if self._push_ratio is not None:
            tensorboard.add_scalar("depth_anchor/push_ratio", self._push_ratio, global_step)
            self._push_ratio = None
