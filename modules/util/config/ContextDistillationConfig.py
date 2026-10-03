from modules.util.config.BaseConfig import BaseConfig


class ContextDistillationConfig(BaseConfig):
    """
    Image-free context distillation: the student (base + LoRA) sees the "student" prompt and learns to predict
    what the teacher (the same model, LoRA off) predicts for the paired "teacher" prompt.

    The training points are teacher latents generated once per pair and cached (like text embeddings), then
    re-noised at a random timestep every time the pair is trained. Batch size, epochs and the timestep
    distribution come from the main training settings.
    """

    enabled: bool
    prompts_path: str
    loss_weight: float
    standalone: bool
    stop_after_steps: int
    decay_to_zero: bool
    resolution: str
    teacher_steps: int
    teacher_cfg: float
    negative_prompt: str
    seed: int

    @staticmethod
    def default_values() -> 'ContextDistillationConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("enabled", False, bool, False))
        # .jsonl, one {"student": "...", "teacher": "...", "ar": "3:4"} per line ("ar" optional, default square)
        data.append(("prompts_path", "", str, False))
        data.append(("loss_weight", 1.0, float, False))
        # train only on the prompt pairs, without an image data set
        data.append(("standalone", False, bool, False))
        # stop distilling after this many training steps (0 = never). Useful as a warm-start before image training.
        data.append(("stop_after_steps", 0, int, False))
        # linearly fade loss_weight to 0 at stop_after_steps
        data.append(("decay_to_zero", False, bool, False))
        # same format as the training resolution ("512", "512,1024" or "768x512"). Empty = the training resolution.
        data.append(("resolution", "", str, False))
        # sampling steps and CFG used to generate the cached teacher latents (set them like your normal sampling)
        data.append(("teacher_steps", 20, int, False))
        data.append(("teacher_cfg", 4.0, float, False))
        data.append(("negative_prompt", "", str, False))
        data.append(("seed", 42, int, False))

        return ContextDistillationConfig(data)
