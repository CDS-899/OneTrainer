from typing import TYPE_CHECKING

from modules.util.TrainProgress import TrainProgress

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from modules.trainer.GenericTrainer import GenericTrainer


class TrainingExtension:
    """
    An optional piece of training logic that plugs into GenericTrainer without copying its loop.

    GenericTrainer keeps ownership of everything else: sampling, saving, backups, the optimizer step,
    fused back pass, grad scaling, EMA, lr scheduling. An extension only contributes extra loss terms
    (and, in standalone mode, the epoch length instead of an image data set).

    Each loss returned by compute_loss() is back-propagated on its own right after it is returned, so
    the activations of the image loss are already freed when the extension runs its forward pass.
    """

    def is_standalone(self) -> bool:
        """True if training runs without an image data set, driven only by this extension."""
        return False

    def standalone_epoch_length(self) -> int:
        return 0

    def on_train_start(self, trainer: "GenericTrainer"):
        """Called once at the start of GenericTrainer.train(), before the first epoch."""

    def on_epoch_start(self, trainer: "GenericTrainer", train_progress: TrainProgress):
        """Called at the start of every epoch, before its first step."""

    def compute_loss(
            self,
            trainer: "GenericTrainer",
            batch: dict | None,
            train_progress: TrainProgress,
    ) -> Tensor | None:
        """
        Returns an extra loss for this step, or None to skip.
        batch is None in standalone mode. The returned loss is divided by the gradient accumulation
        steps by the trainer, like the image loss.
        """
        return None

    def report_to_tensorboard(self, tensorboard: SummaryWriter, global_step: int):
        """Called on optimizer update steps (master process only)."""
