from modules.util.config.BaseConfig import BaseConfig


class ContextDistillationConfig(BaseConfig):
    """
    Image-free context distillation: the student (base + LoRA) sees a short prompt and learns to predict
    what the frozen base model (LoRA off) predicts for the paired dense prompt.

    Training points come from a pool of teacher latents, generated from the dense prompts during training
    and re-noised at random noise levels, so no images are needed.
    """

    enabled: bool
    prompts_path: str
    loss_weight: float
    batch_size: int
    standalone: bool
    standalone_epoch_length: int
    stop_after_steps: int
    decay_to_zero: bool
    resolution: str
    pool_size: int
    pool_refresh_every: int
    teacher_steps: int
    teacher_cfg: float
    negative_prompt: str
    target_cfg: float
    seed: int

    @staticmethod
    def default_values() -> 'ContextDistillationConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("enabled", False, bool, False))
        # .json (list of objects) or .jsonl (one object per line), each {"short": "...", "dense": "..."}
        data.append(("prompts_path", "", str, False))
        data.append(("loss_weight", 1.0, float, False))
        # number of distillation samples per training step
        data.append(("batch_size", 1, int, False))
        # train only on prompt pairs, without an image data set
        data.append(("standalone", False, bool, False))
        # steps per epoch in standalone mode
        data.append(("standalone_epoch_length", 100, int, False))
        # stop distilling after this many training steps (0 = never). Useful as a warm-start before image training.
        data.append(("stop_after_steps", 0, int, False))
        # linearly fade loss_weight to 0 at stop_after_steps
        data.append(("decay_to_zero", False, bool, False))
        # resolution of the teacher latents, e.g. "512" or "768x512". Empty = first entry of the training resolution.
        data.append(("resolution", "", str, False))
        # number of teacher latents kept in memory
        data.append(("pool_size", 16, int, False))
        # replace the oldest teacher latent every N training steps (0 = never refresh)
        data.append(("pool_refresh_every", 8, int, False))
        # sampling steps used to generate a teacher latent
        data.append(("teacher_steps", 20, int, False))
        # CFG used to generate teacher latents (only decides where training happens, not the target)
        data.append(("teacher_cfg", 4.0, float, False))
        data.append(("negative_prompt", "", str, False))
        # CFG of the teacher target. 1.0 = plain conditional prediction (recommended)
        data.append(("target_cfg", 1.0, float, False))
        data.append(("seed", 42, int, False))

        return ContextDistillationConfig(data)
