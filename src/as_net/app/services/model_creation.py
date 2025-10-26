from typing import Any

from as_net.app.ports.model_builder import IModelBuilder
from as_net.domain.models.as_net import ASNetConfig


class ModelCreationService:
    """Service for creating models."""

    def __init__(self, model_builder: IModelBuilder):
        self.model_builder = model_builder

    def create_model(self, config: ASNetConfig) -> Any:
        """Creates a model from a configuration."""
        return self.model_builder.build(config)
