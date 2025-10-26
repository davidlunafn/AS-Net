from abc import ABC, abstractmethod
from typing import Any

from as_net.domain.models.as_net import ASNetConfig


class ICheckpointSaver(ABC):
    """Interface for a checkpoint saver."""

    @abstractmethod
    def save_checkpoint(self, model: Any, optimizer: Any, epoch: int, loss: float, config: ASNetConfig):
        """Saves a model checkpoint."""
        raise NotImplementedError
