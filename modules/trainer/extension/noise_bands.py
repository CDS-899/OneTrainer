from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

# noise level (sigma) ranges for per-band losses: high noise decides composition / what is in the image,
# low noise only refines details
NOISE_BANDS = (("high", 2 / 3, 1.0), ("mid", 1 / 3, 2 / 3), ("low", 0.0, 1 / 3))


class NoiseBandLoss:
    """Accumulates a per-sample loss by noise band between TensorBoard reports."""

    def __init__(self, name: str, total_tag: str):
        self.name = name
        self.total_tag = total_tag
        self._loss = {band: 0.0 for band, _, _ in NOISE_BANDS}
        self._count = {band: 0 for band, _, _ in NOISE_BANDS}

    def record(self, per_sample: Tensor, sigma: Tensor):
        per_sample, sigma = per_sample.detach(), sigma.detach()
        for band, low, high in NOISE_BANDS:
            in_band = (sigma >= low) & (sigma < high) if high < 1.0 else (sigma >= low)
            self._loss[band] = self._loss[band] + per_sample[in_band].sum()
            self._count[band] += int(in_band.sum())

    def count(self) -> int:
        return sum(self._count.values())

    def report(self, tensorboard: SummaryWriter, global_step: int):
        total_loss, total_count = 0.0, 0
        for band, _, _ in NOISE_BANDS:
            count = self._count[band]
            if count > 0:
                band_loss = float(self._loss[band])
                tensorboard.add_scalar(f"{self.name}/loss_{band}_noise", band_loss / count, global_step)
                total_loss += band_loss
                total_count += count
            self._loss[band] = 0.0
            self._count[band] = 0
        if total_count > 0:
            tensorboard.add_scalar(self.total_tag, total_loss / total_count, global_step)
