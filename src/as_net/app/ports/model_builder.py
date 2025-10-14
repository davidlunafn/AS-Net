from abc import ABC, abstractmethod
from typing import Any

from as_net.domain.models.as_net import ASNetConfig


class IModelBuilder(ABC):
    """Interface for a model builder."""

    @abstractmethod
    def build(self, config: ASNetConfig) -> Any:
        """Builds a model from a configuration."""
        raise NotImplementedError
