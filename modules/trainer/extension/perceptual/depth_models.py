"""
Frozen depth estimators used as perceptual anchors.

Every estimator maps RGB images in [0, 1] (B, 3, H, W) to a relative depth-like map (B, h, w), differentiably.
Only the anchor's own comparison matters (predicted image vs. training image through the same estimator), so the
maps do not need to be metric, and depth vs. inverse depth does not matter either.
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14

# name -> (family, Hugging Face repo, extra)
DEPTH_MODELS = {
    "depth-anything-v2-small": ("v2", "depth-anything/Depth-Anything-V2-Small-hf", None),  # Apache-2.0
    "depth-anything-v2-base": ("v2", "depth-anything/Depth-Anything-V2-Base-hf", None),  # CC-BY-NC-4.0
    "depth-anything-v2-large": ("v2", "depth-anything/Depth-Anything-V2-Large-hf", None),  # CC-BY-NC-4.0
    "da3-small": ("v3", "depth-anything/DA3-SMALL", "da3-small"),  # Apache-2.0
    "da3-base": ("v3", "depth-anything/DA3-BASE", "da3-base"),  # Apache-2.0
    "da3-mono-large": ("v3", "depth-anything/DA3MONO-LARGE", "da3mono-large"),  # Apache-2.0
}

DA3_INSTALL_HINT = (
    "Depth Anything 3 is an optional dependency. Install it without its (heavy) dependencies:\n"
    "  pip install --no-deps \"git+https://github.com/ByteDance-Seed/Depth-Anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4\" addict einops"
)


def resize_for_patches(images: Tensor, resolution: int) -> Tensor:
    """Resize so the longer side is ~resolution and both sides are multiples of the ViT patch size."""
    h, w = images.shape[-2:]
    scale = resolution / max(h, w)
    new_h = max(PATCH_SIZE, round(h * scale / PATCH_SIZE) * PATCH_SIZE)
    new_w = max(PATCH_SIZE, round(w * scale / PATCH_SIZE) * PATCH_SIZE)
    if (new_h, new_w) == (h, w):
        return images
    return F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=False)


class DepthEstimator(nn.Module):
    def __init__(self, network: nn.Module, family: str, resolution: int):
        super().__init__()
        self.network = network
        self.family = family
        self.resolution = resolution
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: Tensor) -> Tensor:
        x = resize_for_patches(images, self.resolution)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        if self.family == "v2":
            return self.network(pixel_values=x).predicted_depth
        # Depth Anything 3 takes (B, views, 3, H, W)
        return self.network(x.unsqueeze(1))["depth"][:, 0]


def _load_v2(repo: str) -> nn.Module:
    from transformers import DepthAnythingForDepthEstimation
    return DepthAnythingForDepthEstimation.from_pretrained(repo)


def _load_v3(repo: str, registry_name: str) -> nn.Module:
    try:
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
    except ImportError as e:
        raise ImportError(DA3_INSTALL_HINT) from e
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    network = create_object(load_config(MODEL_REGISTRY[registry_name]))
    state_dict = load_file(hf_hub_download(repo, "model.safetensors"))
    # the hub checkpoint wraps the network as `model.` inside DepthAnything3
    if all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{repo}: weights do not match the network "
                           f"(missing {len(missing)}, unexpected {len(unexpected)}, e.g. {(missing + unexpected)[:3]})")
    return network


def create_depth_estimator(name: str, resolution: int, device: torch.device, dtype: torch.dtype) -> DepthEstimator:
    if name not in DEPTH_MODELS:
        raise ValueError(f"unknown depth model '{name}', choose one of: {', '.join(DEPTH_MODELS)}")
    family, repo, registry_name = DEPTH_MODELS[name]
    network = _load_v2(repo) if family == "v2" else _load_v3(repo, registry_name)
    estimator = DepthEstimator(network, family, resolution)
    estimator.requires_grad_(False)
    estimator.eval()
    return estimator.to(device=device, dtype=dtype)
