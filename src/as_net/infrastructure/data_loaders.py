import os

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Any, Tuple

from as_net.app.ports.data_loader import IDataLoader
from as_net.config import PROCESSED_DATA_PATH


class AudioDataset(Dataset):
    """PyTorch Dataset for loading the audio data."""

    def __init__(self, labels_file: str, data_path: str):
        # The labels file is expected to be in the root of the data_path
        labels_df = pd.read_csv(os.path.join(data_path, labels_file))
        # Get unique file paths, as one file can have multiple call entries
        self.unique_files = labels_df.iloc[:, 0].unique()
        self.data_path = data_path

    def __len__(self) -> int:
        return len(self.unique_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get the unique file path for this index
        mixed_file_path = self.unique_files[idx]

        # Derive source and noise file paths
        source_file_path = mixed_file_path.replace("_mixed.wav", "_source.wav")
        noise_file_path = mixed_file_path.replace("_mixed.wav", "_noise.wav")

        # Load audio files
        mixed_audio, sr = sf.read(mixed_file_path, dtype="float32")
        source_audio, _ = sf.read(source_file_path, dtype="float32")
        noise_audio, _ = sf.read(noise_file_path, dtype="float32")

        return (
            torch.from_numpy(mixed_audio),
            torch.from_numpy(source_audio),
            torch.from_numpy(noise_audio),
        )


class AudioDataLoader(IDataLoader):
    """PyTorch implementation of the IDataLoader port."""

    def __init__(
        self, data_path: str = PROCESSED_DATA_PATH, batch_size: int = 32, val_split: float = 0.2, num_workers: int = 4
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        dataset = AudioDataset(labels_file="labels.csv", data_path=self.data_path)

        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size

        # For reproducibility, we can set a manual seed for the generator
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    def load_train_data(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def load_val_data(self) -> Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
