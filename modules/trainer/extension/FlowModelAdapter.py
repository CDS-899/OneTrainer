import copy
from abc import ABCMeta, abstractmethod

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


class AnimaFlowAdapter(FlowModelAdapter):

    def encode_prompts(self, prompts: list[str]) -> Tensor:
        return self.model.encode_text(
            train_device=self.train_device,
            batch_size=len(prompts),
            text=prompts,
        )

    def velocity(self, latents: Tensor, sigma: Tensor, conditioning: Tensor) -> Tensor:
        model = self.model
        dtype = model.train_dtype.torch_dtype()
        # CosmosTransformer3DModel takes padding_mask in pixel space and repeats it per batch item itself
        padding_mask = latents.new_zeros(1, 1, latents.shape[-2] * 8, latents.shape[-1] * 8, dtype=dtype)
        with model.autocast_context:
            return model.transformer(
                hidden_states=latents.to(dtype=dtype),
                timestep=sigma.to(device=latents.device, dtype=torch.float32),
                encoder_hidden_states=conditioning.to(device=latents.device, dtype=dtype),
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

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
