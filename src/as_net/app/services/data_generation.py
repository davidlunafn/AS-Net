import csv
import json
import os
import random
from multiprocessing import Manager, Pool, Process
from typing import List

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

        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Create a manager and a queue
        with Manager() as manager:
            labels_queue = manager.Queue()

            # Create a writer process
            writer_process = Process(
                target=self._write_labels,
                args=(labels_queue, os.path.join(output_path, "labels.csv")),
            )
            writer_process.start()

            # Create a pool of worker processes
            with Pool() as pool:
                pool.starmap(
                    self._generate_sample,
                    [
                        (i, snr_levels, output_path, labels_queue)
                        for i in range(num_samples)
                    ],
                )

            # Terminate the writer process
            labels_queue.put(None)
            writer_process.join()

    def _write_labels(self, queue: Manager().Queue, labels_file: str):
        """Writes the labels to the CSV file."""
        with open(labels_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_file", "start_time", "end_time", "lower_freq", "upper_freq", "json_file"])
            while True:
                item = queue.get()
                if item is None:
                    break
                writer.writerow(item)

    def _generate_sample(
        self, sample_index: int, snr_levels: List[float], output_path: str, queue: Manager().Queue
    ):
        """Generates a single sample."""

        # Get the list of audio files
        wav_files = librosa.util.find_files(os.path.join(RAW_DATA_PATH, "WAV"))

        # Create a 20-second audio sample
        sample_duration = 20
        sample_rate = 22050
        sample = np.zeros(sample_duration * sample_rate)

        # Select a random number of calls to add to the sample
        num_calls = random.randint(1, 5)

        # Add the calls to the sample
        calls = []
        for _ in range(num_calls):
            # Select a random audio file
            wav_file = random.choice(wav_files)

            # Read the audio file
            audio, _ = librosa.load(wav_file, sr=sample_rate)

            # Select a random position to insert the call
            start_position = random.randint(0, len(sample) - len(audio))

            # Add the call to the sample
            sample[start_position : start_position + len(audio)] += audio

            # Get the call information from the JSON file
            json_file = wav_file.replace(".wav", ".JSON").replace("WAV", "JSON")
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

            # Save the mixed signal
            output_dir = os.path.join(output_path, f"{snr}dB")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"sample_{sample_index}.wav")
            sf.write(output_file, mixed_signal.audio.samples, mixed_signal.audio.sample_rate)

            # Put the labels in the queue
            for call in mixed_signal.bioacoustic_source.calls:
                queue.put(
                    [output_file, call.start_time, call.end_time, call.lower_freq, call.upper_freq, call.json_file]
                )

            logger.info(f"Generated sample {sample_index + 1} with SNR {snr}dB")

    def _generate_pink_noise(self, num_samples: int) -> np.ndarray:
        """Generates pink noise using the Voss-McCartney algorithm."""
        num_octaves = int(np.log2(num_samples))
        pink_noise = np.zeros(num_samples)
        for i in range(num_octaves):
            amplitude = 1 / (2 ** i)
            frequency = 2 ** i
            white_noise = np.random.randn(num_samples // frequency) * amplitude
            pink_noise += np.repeat(white_noise, frequency)

        # Normalize
        pink_noise /= np.sqrt(np.mean(pink_noise ** 2))

        return pink_noise
