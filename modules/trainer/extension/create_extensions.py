from modules.trainer.extension.TrainingExtension import TrainingExtension
from modules.util.config.TrainConfig import TrainConfig


def create_training_extensions(config: TrainConfig) -> list[TrainingExtension]:
    extensions: list[TrainingExtension] = []

    if sum(ext.is_standalone() for ext in extensions) > 1:
        raise ValueError("Only one training extension can run in standalone mode")

    return extensions
