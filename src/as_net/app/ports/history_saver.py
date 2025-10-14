from abc import ABC, abstractmethod


class IHistorySaver(ABC):
    """Interface for a training history saver."""

    @abstractmethod
    def save_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """Saves the history for a single epoch."""
        raise NotImplementedError
