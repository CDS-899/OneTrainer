import copy
import os
import urllib.request
from abc import ABCMeta, abstractmethod
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType

import torch
from torch import Tensor

import numpy as np


class FlowModelAdapter(metaclass=ABCMeta):
    """
    The few model-specific operations the distillation extensions need, for rectified-flow models
    (x_t = (1 - sigma) * x_0 + sigma * noise, the model predicts v = noise - x_0).
    All latents are in the model's scaled latent space. sigma is continuous in [0, 1].

    Adding a model = implementing encode_prompts(), velocity() and latent_shape().
    """

    def __init__(self, model: BaseModel, model_setup: BaseModelSetup, config: TrainConfig, train_device: torch.device):
        self.model = model
        self.model_setup = model_setup
        self.config = config
        self.train_device = train_device

    @abstractmethod
    def encode_prompts(self, prompts: list[str]) -> Tensor:
        """Text conditioning for a list of prompts. The text encoder is materialized by the caller."""

    @abstractmethod
    def velocity(self, latents: Tensor, sigma: Tensor, conditioning: Tensor) -> Tensor:
        """Model prediction v for scaled latents at noise level sigma (shape (B,))."""

    @abstractmethod
    def latent_shape(self, batch_size: int, height: int, width: int) -> tuple[int, ...]:
        """Shape of a scaled latent for an image of height x width pixels."""

    @abstractmethod
    def timestep_shift(self, latents: Tensor) -> float:
        """The timestep shift normal training would use for latents of this shape."""

    def resolution_quantization(self) -> int:
        return 64

    # ---------------------------------------------------------------- training-step predictions

    def clean_latents(self, batch: dict) -> Tensor:
        """The batch's clean latents in the scaled latent space the model predicts in."""
        return self.model.scale_latents(batch['latent_image'])

    def sigma_from_timestep(self, timestep: Tensor) -> Tensor:
        # same mapping as ModelSetupFlowMatchingMixin._add_noise_discrete
        return (timestep.float() + 1.0) / self.model.noise_scheduler.config['num_train_timesteps']

    def predicted_clean_latents(self, batch: dict, model_output_data: dict) -> tuple[Tensor, Tensor, Tensor]:
        """
        (predicted x_0, clean x_0, sigma) for a normal training step, from predict()'s outputs alone:
        x_t = (1 - sigma) x_0 + sigma * noise and target = noise - x_0, so x_0_pred = x_0 + sigma * (target - predicted).
        """
        clean = self.clean_latents(batch).float()
        sigma = self.sigma_from_timestep(model_output_data['timestep']).to(clean.device)
        sigma_b = sigma.view(-1, *([1] * (clean.dim() - 1)))
        predicted = clean + sigma_b * (model_output_data['target'].float() - model_output_data['predicted'].float())
        return predicted, clean, sigma

    def velocity_at_timestep(self, latents: Tensor, timestep: Tensor, conditioning: Tensor) -> Tensor:
        """
        Model prediction v at a discrete training timestep (as drawn by predict()), passed to the model exactly the
        way predict() passes it, so extra forward passes match the training step's own.
        """
        return self.velocity(latents, self.sigma_from_timestep(timestep), conditioning)

    def timestep_from_sigma(self, sigma: Tensor) -> Tensor:
        """The discrete training timestep closest to sigma (inverse of sigma_from_timestep)."""
        num_train_timesteps = self.model.noise_scheduler.config['num_train_timesteps']
        return (sigma.float() * num_train_timesteps).round().long().sub(1).clamp(0, num_train_timesteps - 1)

    def batch_conditioning(self, batch: dict, train_progress) -> Tensor:
        """
        The text conditioning predict() used for this batch, including the same text encoder dropout.
        Needs the text encoder frozen (cached embeddings or no gradients through the text encoder).
        """
        raise NotImplementedError(f"batch_conditioning is not implemented for {self.config.model_type}")

    def next_sampling_sigma(self, sigma: Tensor, schedule: Tensor) -> Tensor:
        """
        The noise level one sampling step after sigma on a descending schedule from sampling_sigmas() (ending in 0).
        sigma between two schedule points moves by the same fraction of the following step.
        """
        sigma = sigma.float()
        ascending = schedule.flip(0).to(device=sigma.device, dtype=torch.float32)
        last = schedule.shape[0] - 1
        # fractional position p of sigma in the descending schedule: schedule[p] == sigma
        upper = torch.searchsorted(ascending, sigma.clamp(0.0, float(ascending[-1]))).clamp(1, last)
        lo, hi = ascending[upper - 1], ascending[upper]
        frac = ((sigma - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
        position = (last - upper).float() + (1.0 - frac)
        # one step further
        nxt = position + 1.0
        index = nxt.floor().long().clamp(0, last)
        index_next = (index + 1).clamp(max=last)
        g = (nxt - nxt.floor()).clamp(0.0, 1.0)
        desc = schedule.to(device=sigma.device, dtype=torch.float32)
        result = desc[index] * (1.0 - g) + desc[index_next] * g
        return torch.where(nxt >= last, torch.zeros_like(result), result)

    # ---------------------------------------------------------------- tiny decoder

    def tiny_decoder_name(self) -> str:
        raise NotImplementedError(f"no tiny decoder for {self.config.model_type}")

    def create_tiny_decoder(self, decoder_path: str, cache_dir: str) -> "TinyDecoder":
        name = self.tiny_decoder_name()
        if not decoder_path:
            decoder_path = os.path.join(cache_dir, f"{name}.safetensors")
            if not os.path.isfile(decoder_path):
                os.makedirs(cache_dir, exist_ok=True)
                url = TINY_DECODER_URLS[name]
                print(f"Downloading tiny decoder {url}")
                urllib.request.urlretrieve(url, decoder_path + ".tmp")
                os.replace(decoder_path + ".tmp", decoder_path)
        return TinyDecoder(decoder_path, name)

    def sample_sigma(self, latents: Tensor, generator: torch.Generator) -> Tensor:
        """Noise levels drawn from the same timestep distribution as normal training."""
        num_train_timesteps = self.model.noise_scheduler.config['num_train_timesteps']
        timestep = self.model_setup._get_timestep_discrete(
            num_train_timesteps,
            False,
            generator,
            latents.shape[0],
            self.config,
            shift=self.timestep_shift(latents) if self.config.dynamic_timestep_shifting else self.config.timestep_shift,
        )
        # same mapping as ModelSetupFlowMatchingMixin._add_noise_discrete
        return (timestep.float() + 1.0) / num_train_timesteps

    def sampling_shift(self) -> float:
        """The static timestep shift of the model's inference schedule."""
        return float(self.model.noise_scheduler.shift)

    def set_sampling_shift(self, shift: float):
        """Changes the static timestep shift of the model's inference schedule (used by its sampler)."""
        scheduler = self.model.noise_scheduler
        if scheduler.config.get("use_dynamic_shifting", False):
            raise ValueError("a fixed sampling shift needs a scheduler without dynamic shifting")
        if hasattr(scheduler, "set_shift"):
            scheduler.set_shift(shift)
        else:
            scheduler._shift = shift

    def sampling_sigmas(self, steps: int) -> Tensor:
        """Noise levels of the model's own inference schedule (including the final 0), as used by its sampler."""
        scheduler = copy.deepcopy(self.model.noise_scheduler)
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), device=self.train_device)
        return scheduler.sigmas.to(device=self.train_device, dtype=torch.float32)

    @torch.no_grad()
    def generate(
            self,
            conditioning: Tensor,
            negative_conditioning: Tensor | None,
            cfg: float,
            steps: int,
            height: int,
            width: int,
            generator: torch.Generator,
    ) -> Tensor:
        """Euler sampling with the current model state. Returns a clean scaled latent (batch size of conditioning)."""
        batch_size = conditioning.shape[0]
        latents = torch.randn(
            self.latent_shape(batch_size, height, width),
            generator=generator, device=self.train_device, dtype=torch.float32,
        )
        sigmas = self.sampling_sigmas(steps)
        use_cfg = cfg != 1.0 and negative_conditioning is not None

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i].expand(batch_size)
            if use_cfg:
                v = self.velocity(
                    torch.cat([latents, latents]),
                    torch.cat([sigma, sigma]),
                    torch.cat([conditioning, negative_conditioning.expand_as(conditioning)]),
                ).float()
                v_cond, v_uncond = v.chunk(2)
                v = v_uncond + cfg * (v_cond - v_uncond)
            else:
                v = self.velocity(latents, sigma, conditioning).float()
            latents = latents + (sigmas[i + 1] - sigmas[i]) * v

        return latents


TINY_DECODER_URLS = {
    "taew2_1": "https://github.com/madebyollin/taehv/raw/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taew2_1.safetensors",
}


class TinyDecoder(torch.nn.Module):
    """madebyollin's TAEHV decoder: scaled latents (B, C, 1, h, w) -> RGB images in [0, 1] (B, 3, H, W)."""

    def __init__(self, path: str, arch_name: str):
        super().__init__()
        from modules.trainer.extension.perceptual.taehv import TAEHV

        # single images: no temporal upscaling, so one latent frame decodes to exactly one image
        self.taehv = TAEHV(checkpoint_path=path, arch_name=arch_name, decoder_time_upscale=(False, False))
        del self.taehv.encoder
        self.requires_grad_(False)
        self.eval()

    def forward(self, latents: Tensor) -> Tensor:
        frames = self.taehv.decode_video(latents.transpose(1, 2), parallel=True, show_progress_bar=False)
        return frames[:, 0]


class AnimaFlowAdapter(FlowModelAdapter):

    def tiny_decoder_name(self) -> str:
        # Anima uses the Qwen-Image VAE, which shares the Wan 2.1 latent space
        return "taew2_1"

    def encode_prompts(self, prompts: list[str]) -> Tensor:
        return self.model.encode_text(
            train_device=self.train_device,
            batch_size=len(prompts),
            text=prompts,
        )

    def _transformer(self, latents: Tensor, model_timestep: Tensor, conditioning: Tensor) -> Tensor:
        model = self.model
        dtype = model.train_dtype.torch_dtype()
        # CosmosTransformer3DModel takes padding_mask in pixel space and repeats it per batch item itself
        padding_mask = latents.new_zeros(1, 1, latents.shape[-2] * 8, latents.shape[-1] * 8, dtype=dtype)
        with model.autocast_context:
            return model.transformer(
                hidden_states=latents.to(dtype=dtype),
                timestep=model_timestep,
                encoder_hidden_states=conditioning.to(device=latents.device, dtype=dtype),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

    def velocity(self, latents: Tensor, sigma: Tensor, conditioning: Tensor) -> Tensor:
        return self._transformer(latents, sigma.to(device=latents.device, dtype=torch.float32), conditioning)

    def velocity_at_timestep(self, latents: Tensor, timestep: Tensor, conditioning: Tensor) -> Tensor:
        # as in BaseAnimaSetup.predict
        return self._transformer(latents, timestep.to(device=latents.device) / 1000, conditioning)

    def batch_conditioning(self, batch: dict, train_progress) -> Tensor:
        # same call and the same seeded dropout as BaseAnimaSetup.predict
        config = self.config
        batch_seed = train_progress.global_step * multi.world_size() + multi.rank()
        with self.model.autocast_context:
            return self.model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=Random(batch_seed),
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch['text_encoder_hidden_state'] \
                    if 'text_encoder_hidden_state' in batch and not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

    def latent_shape(self, batch_size: int, height: int, width: int) -> tuple[int, ...]:
        # 5D video-style latents with a single frame
        return batch_size, 16, 1, height // 8, width // 8

    def timestep_shift(self, latents: Tensor) -> float:
        return self.model.calculate_timestep_shift(latents.shape[-2], latents.shape[-1])


_ADAPTERS: dict[ModelType, type[FlowModelAdapter]] = {
    ModelType.ANIMA: AnimaFlowAdapter,
}


def create_flow_model_adapter(
        model: BaseModel,
        model_setup: BaseModelSetup,
        config: TrainConfig,
        train_device: torch.device,
) -> FlowModelAdapter:
    cls = _ADAPTERS.get(config.model_type)
    if cls is None:
        supported = ", ".join(str(t) for t in _ADAPTERS)
        raise NotImplementedError(f"Distillation is not implemented for {config.model_type} yet (supported: {supported})")
    return cls(model, model_setup, config, train_device)
