from modules.util.config.BaseConfig import BaseConfig


class DepthAnchorConfig(BaseConfig):
    """
    Depth anchor: during image training, the model's predicted clean image and the training image both go through
    a tiny latent decoder and a frozen depth estimator; the difference of the two depth maps is added to the loss.
    It teaches 3D structure without caring about colors, textures or lighting.
    """

    enabled: bool
    weight: float
    depth_model: str
    depth_resolution: int
    min_noise: float
    max_noise: float
    every_n_steps: int
    loss_split: bool
    gradient_weight: float
    decoder_path: str
    preview_every: int

    @staticmethod
    def default_values() -> 'DepthAnchorConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("enabled", False, bool, False))
        data.append(("weight", 0.1, float, False))
        # depth-anything-v2-small | depth-anything-v2-base | depth-anything-v2-large | da3-small | da3-base | da3-mono-large
        data.append(("depth_model", "depth-anything-v2-small", str, False))
        # longer side of the depth model input (rounded to a multiple of 14)
        data.append(("depth_resolution", 518, int, False))
        # the anchor only applies to samples with a noise level (sigma) in [min_noise, max_noise]
        data.append(("min_noise", 0.0, float, False))
        data.append(("max_noise", 1.0, float, False))
        # apply the anchor on every n-th optimizer step
        data.append(("every_n_steps", 1, int, False))
        # steps with the anchor use only the anchor loss (needs every_n_steps >= 2, e.g. 2 = alternate)
        data.append(("loss_split", False, bool, False))
        # weight of the multi-scale depth gradient term relative to the depth term
        data.append(("gradient_weight", 0.5, float, False))
        # tiny decoder weights. Empty = download the model's default decoder
        data.append(("decoder_path", "", str, False))
        # write a TensorBoard preview (image | depth | predicted image | predicted depth) every n steps, 0 = never
        data.append(("preview_every", 100, int, False))

        return DepthAnchorConfig(data)
