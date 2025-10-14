import os
from typing import Any

import torch

from as_net.app.ports.checkpoint_saver import ICheckpointSaver
from as_net.config import MODELS_PATH
from as_net.domain.models.as_net import ASNetConfig
from as_net.logger import logger


class CheckpointSaver(ICheckpointSaver):
    """PyTorch implementation of the ICheckpointSaver port."""

    def __init__(self, models_path: str = MODELS_PATH):
        self.models_path = models_path
        os.makedirs(self.models_path, exist_ok=True)

    def save_checkpoint(self, model: Any, optimizer: Any, epoch: int, loss: float, config: ASNetConfig):
        """Saves a model checkpoint."""

        checkpoint_path = os.path.join(self.models_path, f"checkpoint_epoch_{epoch+1}.pth")

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": config,
        }

        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
