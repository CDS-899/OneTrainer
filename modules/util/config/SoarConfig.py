from modules.util.config.BaseConfig import BaseConfig


class SoarConfig(BaseConfig):
    """
    SOAR (Self-Correction for Optimal Alignment and Refinement, Tencent HY-SOAR): normal image training plus
    correction points. From the step's noisy image the model takes one sampling step itself (with CFG), the result is
    re-noised toward the same noise, and the model learns to head from there straight back to the real image.
    """

    enabled: bool
    correction_points: int
    weight: float
    rollout_cfg: float
    rollout_steps: int
    every_n_steps: int
    min_noise: float

    @staticmethod
    def default_values() -> 'SoarConfig':
        data = []

        # name, default value, data type, nullable
        data.append(("enabled", False, bool, False))
        # correction points per step, each one an extra forward and backward pass (the paper uses 6)
        data.append(("correction_points", 1, int, False))
        # weight of each correction point relative to the normal loss (the paper uses 1)
        data.append(("weight", 1.0, float, False))
        # CFG of the rollout step, with the empty prompt as negative. 1 = no CFG (no extra pass)
        data.append(("rollout_cfg", 4.5, float, False))
        # the rollout takes one step of a sampling schedule with this many steps
        data.append(("rollout_steps", 20, int, False))
        # apply SOAR on every n-th optimizer step, normal training on the others
        data.append(("every_n_steps", 1, int, False))
        # SOAR only for samples whose training noise level is at least this; the others get normal training
        data.append(("min_noise", 0.2, float, False))

        return SoarConfig(data)
