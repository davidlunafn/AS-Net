import csv
import json
import os
import random
from multiprocessing import Manager, Pool, Process
from typing import List

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from as_net.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from as_net.domain.models.audio import AudioSignal
from as_net.domain.models.bioacoustic import BioacousticSource, Call, MixedSignal
from as_net.domain.services.mixing import MixingService
from as_net.logger import logger


class DataGenerationService:
    """Service for generating the dataset."""

    def __init__(self, mixing_service: MixingService):
        self.mixing_service = mixing_service

    def generate(
        self, num_samples: int, snr_levels: List[float], output_path: str = PROCESSED_DATA_PATH
    ):
        """Generates the dataset."""

        logger.info("Starting data generation...")
        os.makedirs(output_path, exist_ok=True)

        # Create a dictionary of unique vocalizations and their file paths
        logger.info("Creating a dictionary of unique vocalizations and their file paths...")
        vocalizations = {}
        for p in Path(os.path.join(RAW_DATA_PATH, "WAV")).rglob("*.[Ww][Aa][Vv]"):
            if not p.name.startswith("._"):
                vocalization_id = p.name.split("_")[0]
                if vocalization_id not in vocalizations:
                    vocalizations[vocalization_id] = []
                vocalizations[vocalization_id].append(p)
        logger.info(f"Found {len(vocalizations)} unique vocalizations.")

        # Create a manager and a queue
        with Manager() as manager:
            labels_queue = manager.Queue()

            # Create a writer process
            logger.info("Starting writer process...")
            writer_process = Process(
                target=self._write_labels,
                args=(labels_queue, os.path.join(output_path, "labels.csv")),
            )
            writer_process.start()

            # Create a pool of worker processes
            logger.info("Starting worker processes...")
            with Pool() as pool:
                pool.starmap(
                    self._generate_sample,
                    [
                        (i, snr_levels, output_path, labels_queue, vocalizations)
                        for i in range(num_samples)
                    ],
                )

            # Terminate the writer process
            labels_queue.put(None)
            writer_process.join()
            logger.info("Writer process finished.")

        logger.info("Data generation finished.")

    def _write_labels(self, queue, labels_file: str):
        """Writes the labels to the CSV file."""
        logger.info("Writer process started.")
        with open(labels_file, "w", newline="") as f:
            logger.info(f"Writing labels to {labels_file}")
            writer = csv.writer(f)
            writer.writerow(["sample_file", "start_time", "end_time", "lower_freq", "upper_freq", "json_file"])
            while True:
                item = queue.get()
                if item is None:
                    break
                writer.writerow(item)
        logger.info("Finished writing labels.")

    def _generate_sample(
        self, sample_index: int, snr_levels: List[float], output_path: str, queue, vocalizations: dict
    ):
        """Generates a single sample."""
        logger.info(f"Generating sample {sample_index + 1}...")

        # Create a 20-second audio sample
        sample_duration = 20
        sample_rate = 22050
        sample = np.zeros(sample_duration * sample_rate)

        # Select a random number of calls to add to the sample
        num_calls = random.randint(1, 5)

        # Add the calls to the sample
        calls = []
        for _ in range(num_calls):
            # Select a random unique vocalization
            vocalization_id = random.choice(list(vocalizations.keys()))

            # Get a random file for the selected vocalization
            wav_file = random.choice(vocalizations[vocalization_id])

            # Read the audio file
            audio, _ = librosa.load(str(wav_file), sr=sample_rate)

            # Select a random position to insert the call
            start_position = random.randint(0, len(sample) - len(audio))

            # Add the call to the sample
            sample[start_position : start_position + len(audio)] += audio

            # Get the call information from the JSON file
            json_file = str(wav_file).replace(".wav", ".JSON").replace("WAV", "JSON")
            with open(json_file, "r") as f:
                json_data = json.load(f)
                onsets = json_data["onsets"]
                offsets = json_data["offsets"]
                lower_freq = json_data["lower_freq"]
                upper_freq = json_data["upper_freq"]

            # Create the calls
            for onset, offset in zip(onsets, offsets):
                call = Call(
                    start_time=start_position / sample_rate + onset,
                    end_time=start_position / sample_rate + offset,
                    lower_freq=lower_freq,
                    upper_freq=upper_freq,
                    json_file=json_file,
                )
                calls.append(call)

        # Create the bioacoustic source
        bioacoustic_source = BioacousticSource(
            audio=AudioSignal(samples=sample, sample_rate=sample_rate), calls=calls
        )

        # Generate the noise
        noise = self._generate_pink_noise(len(sample))

        # Mix the signals
        for snr in snr_levels:
            mixed_signal = self.mixing_service.mix(
                bioacoustic_source=bioacoustic_source,
                noise=AudioSignal(samples=noise, sample_rate=sample_rate),
                snr=snr,
            )

            # Save the signals
            output_dir = os.path.join(output_path, f"{snr}dB")
            os.makedirs(output_dir, exist_ok=True)

            # Define file paths
            mixed_output_file = os.path.join(output_dir, f"sample_{sample_index}_mixed.wav")
            source_output_file = os.path.join(output_dir, f"sample_{sample_index}_source.wav")
            noise_output_file = os.path.join(output_dir, f"sample_{sample_index}_noise.wav")

            # Save the mixed signal
            sf.write(mixed_output_file, mixed_signal.audio.samples, mixed_signal.audio.sample_rate)

            # Save the clean bioacoustic source
            sf.write(
                source_output_file,
                mixed_signal.bioacoustic_source.audio.samples,
                mixed_signal.bioacoustic_source.audio.sample_rate,
            )

            # Save the noise
            sf.write(noise_output_file, mixed_signal.noise.samples, mixed_signal.noise.sample_rate)

            # Put the labels in the queue
            for call in mixed_signal.bioacoustic_source.calls:
                queue.put(
                    [
                        mixed_output_file,
                        call.start_time,
                        call.end_time,
                        call.lower_freq,
                        call.upper_freq,
                        call.json_file,
                    ]
                )

            logger.info(f"Generated sample {sample_index + 1} with SNR {snr}dB")

    def _generate_pink_noise(self, num_samples: int) -> np.ndarray:
        """Generates pink noise using the Voss-McCartney algorithm."""
        # The number of octaves
        num_octaves = int(np.log2(num_samples))
        # The number of rows for the white noise
        num_rows = num_octaves + 1
        # The white noise
        white_noise = np.random.randn(num_rows, num_samples)
        # The pink noise
        pink_noise = np.zeros(num_samples)
        # The amplitudes for each octave
        amplitudes = 1 / (2 ** np.arange(num_rows))
        # Sum the octaves
        for i in range(num_rows):
            # Get the white noise for this octave
            octave_white_noise = white_noise[i]
            # Resample the white noise to the correct length
            resampled_white_noise = np.interp(
                np.linspace(0, 1, num_samples),
                np.linspace(0, 1, len(octave_white_noise)),
                octave_white_noise,
            )
            # Add the resampled white noise to the pink noise
            pink_noise += resampled_white_noise * amplitudes[i]

        # Normalize
        pink_noise /= np.sqrt(np.mean(pink_noise ** 2))

        return pink_noise
