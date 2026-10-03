from modules.trainer.extension.TrainingExtension import TrainingExtension
from modules.util.config.TrainConfig import TrainConfig


def create_training_extensions(config: TrainConfig) -> list[TrainingExtension]:
    extensions: list[TrainingExtension] = []

    if config.context_distillation.enabled:
        from modules.trainer.extension.ContextDistillation import ContextDistillation
        extensions.append(ContextDistillation(config))

    if config.depth_anchor.enabled:
        from modules.trainer.extension.DepthAnchor import DepthAnchor
        extensions.append(DepthAnchor(config))

    if sum(ext.is_standalone() for ext in extensions) > 1:
        raise ValueError("Only one training extension can run in standalone mode")

    return extensions
