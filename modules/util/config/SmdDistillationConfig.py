from modules.util.config.BaseConfig import BaseConfig


class SmdDistillationConfig(BaseConfig):
    """
    SMD few-step distillation (Simplified Matching Distillation, arXiv 2610.18472, without its Pivot), prompts only:
    the student (base + LoRA) samples in student_steps steps at CFG 1 and learns to match the teacher (the same model,
    LoRA off, at teacher_cfg) along its own trajectory. Standalone training, no image data set.
    """

    enabled: bool
    prompts_path: str
    resolution: str
    student_steps: int
    teacher_cfg: float
    repulsion_weight: float
    queries: int
    negative_prompt: str
    seed: int

    @staticmethod
    def default_values() -> 'SmdDistillationConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("enabled", False, bool, False))
        # .jsonl, one {"prompt": "...", "ar": "3:4"} per line ("ar" optional, default square)
        data.append(("prompts_path", "", str, False))
        # same format as the training resolution ("512", "512,1024" or "768x512"). Empty = the training resolution.
        data.append(("resolution", "", str, False))
        # sampling steps of the distilled student (Euler, the model's own schedule, CFG 1)
        data.append(("student_steps", 4, int, False))
        # CFG of the teacher target, against negative_prompt
        data.append(("teacher_cfg", 5.0, float, False))
        # push the student away from the teacher's negative-prompt prediction (paper: 0.2)
        data.append(("repulsion_weight", 0.2, float, False))
        # teacher queries per student step (paper: 2)
        data.append(("queries", 2, int, False))
        data.append(("negative_prompt", "", str, False))
        data.append(("seed", 42, int, False))

        return SmdDistillationConfig(data)
