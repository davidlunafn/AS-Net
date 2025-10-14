import csv
import os

from as_net.app.ports.history_saver import IHistorySaver
from as_net.config import MODELS_PATH
from as_net.logger import logger


class HistorySaver(IHistorySaver):
    """Saves training history to a CSV file."""

    def __init__(self, history_file_path: str = os.path.join(MODELS_PATH, "training_history.csv")):
        self.history_file_path = history_file_path
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.history_file_path), exist_ok=True)
        # Create the file and write the header if it doesn't exist
        if not os.path.exists(self.history_file_path):
            with open(self.history_file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss"])

    def save_epoch(self, epoch: int, train_loss: float, val_loss: float):
        with open(self.history_file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])
        logger.info(f"Saved epoch {epoch + 1} history to {self.history_file_path}")
