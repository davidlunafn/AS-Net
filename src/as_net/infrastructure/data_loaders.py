import os

import numpy as np

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Any, Tuple

from as_net.app.ports.data_loader import IDataLoader
from as_net.config import PROCESSED_DATA_PATH


class AudioDataset(Dataset):
    """PyTorch Dataset for loading the audio data."""

    def __init__(self, files: list):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        mixed_file_path = self.files[idx]

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
            mixed_file_path,
        )


class AudioDataLoader(IDataLoader):
    """PyTorch implementation of the IDataLoader port."""

    def __init__(
        self,
        data_path: str = PROCESSED_DATA_PATH,
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 4,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        labels_df = pd.read_csv(os.path.join(self.data_path, "labels.csv"))
        all_files = labels_df.iloc[:, 0].unique()

        # Create a dictionary to group files by base sample index
        samples = {}
        for f in all_files:
            base_name = "_".join(os.path.basename(f).split("_")[:2])  # e.g. sample_0
            if base_name not in samples:
                samples[base_name] = []
            samples[base_name].append(f)

        sample_keys = list(samples.keys())

        # Shuffle and split the base samples for reproducibility
        np.random.seed(42)
        np.random.shuffle(sample_keys)

        val_size = int(val_split * len(sample_keys))
        test_size = int(test_split * len(sample_keys))

        val_keys = sample_keys[:val_size]
        test_keys = sample_keys[val_size : val_size + test_size]
        train_keys = sample_keys[val_size + test_size :]

        # Create file lists for train, val, and test
        train_files = [f for key in train_keys for f in samples[key]]
        val_files = [f for key in val_keys for f in samples[key]]
        test_files = [f for key in test_keys for f in samples[key]]

        self.train_dataset = AudioDataset(files=train_files)
        self.val_dataset = AudioDataset(files=val_files)
        self.test_dataset = AudioDataset(files=test_files)

    def load_train_data(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def load_val_data(self) -> Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def load_test_data(self) -> Any:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
