from modules.util.config.BaseConfig import BaseConfig


class CfgDistillationConfig(BaseConfig):
    """
    CFG distillation, prompts only: the student (base + LoRA) keeps the step count and learns to sample without CFG,
    matching the teacher (the same model, LoRA off, at teacher_cfg) along its own trajectory (same on-policy core as
    SMD distillation). Standalone training, no image data set.
    """

    enabled: bool
    prompts_path: str
    resolution: str
    steps: int
    teacher_cfg: float
    queries: int
    negative_prompt: str
    seed: int

    @staticmethod
    def default_values() -> 'CfgDistillationConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("enabled", False, bool, False))
        # .jsonl, one {"prompt": "...", "ar": "3:4"} per line ("ar" optional, default square)
        data.append(("prompts_path", "", str, False))
        # same format as the training resolution ("512", "512,1024" or "768x512"). Empty = the training resolution.
        data.append(("resolution", "", str, False))
        # sampling steps of the student (and of the teacher it imitates): set them like your normal sampling
        data.append(("steps", 20, int, False))
        # CFG of the teacher target, against negative_prompt: set it like your normal sampling
        data.append(("teacher_cfg", 4.5, float, False))
        # teacher queries per student step
        data.append(("queries", 1, int, False))
        data.append(("negative_prompt", "", str, False))
        data.append(("seed", 42, int, False))

        return CfgDistillationConfig(data)
