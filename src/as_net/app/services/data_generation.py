import csv
import itertools
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

    def _noise_chunk_generator(self, noise_files: List[Path], target_sr: int):
        """
        A generator that yields 20-second noise chunks.
        It cycles through the noise files, processes one file at a time,
        and yields its chunks.
        """
        shuffled_files = random.sample(noise_files, len(noise_files))
        for noise_file in itertools.cycle(shuffled_files):
            try:
                audio, sr = librosa.load(str(noise_file), sr=None)
                if sr != target_sr:
                    audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

                chunk_duration = 20
                chunk_samples = chunk_duration * target_sr
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i : i + chunk_samples]
                    if len(chunk) < chunk_samples:
                        num_repeats = int(np.ceil(chunk_samples / len(chunk)))
                        tiled_chunk = np.tile(chunk, num_repeats)
                        padded_chunk = tiled_chunk[:chunk_samples]
                        yield padded_chunk
                    else:
                        yield chunk
            except Exception as e:
                logger.error(f"Could not process noise file {noise_file}, skipping: {e}")
                continue

    def generate_with_real_noise(
        self,
        bird_calls_dir: str,
        noise_dir: str,
        output_path: str,
        target_sr: int,
        num_samples: int,
        snr_levels: List[float],
    ):
        """
        Generates a dataset of a specific size (`num_samples`) using real noise.
        """
        logger.info(f"Starting generation of {num_samples} samples with real noise...")
        output_path = os.path.join(output_path, "test_rain")

        logger.info(f"Output directory set to: {os.path.abspath(output_path)}")

        # Check for write permissions on the parent directory
        parent_dir = Path(output_path).parent
        if not os.access(parent_dir, os.W_OK):
            logger.error(f"No write permissions for the output directory: {parent_dir}. Aborting.")
            return

        os.makedirs(output_path, exist_ok=True)

        bird_call_files = list(Path(bird_calls_dir).rglob("*.[Ww][Aa][Vv]"))
        bird_call_files = [f for f in bird_call_files if not f.name.startswith("._")]

        noise_files = list(Path(noise_dir).rglob("*.[Ww][Aa][Vv]"))
        noise_files = [f for f in noise_files if not f.name.startswith("._")]

        logger.info(f"Found {len(bird_call_files)} bird call files.")
        logger.info(f"Found {len(noise_files)} noise files.")

        if not noise_files:
            logger.error("No noise files found. Aborting.")
            return

        total_possible_chunks = len(noise_files) * 3  # Approximation
        if num_samples > total_possible_chunks:
            logger.warning(
                f"Requested {num_samples} samples, which is more than the estimated {total_possible_chunks} unique chunks available. "
                "Noise files will be repeated."
            )

        with Manager() as manager:
            labels_queue = manager.Queue()
            labels_file = os.path.join(output_path, "labels.csv")
            writer_process = Process(target=self._write_labels, args=(labels_queue, labels_file))
            writer_process.start()

            noise_generator = self._noise_chunk_generator(noise_files, target_sr)

            for i in range(num_samples):
                noise_chunk = next(noise_generator)
                self._generate_one_sample(
                    sample_index=i,
                    noise_chunk=noise_chunk,
                    snr_levels=snr_levels,
                    output_path=output_path,
                    bird_call_files=bird_call_files,
                    target_sr=target_sr,
                    queue=labels_queue,
                )

            labels_queue.put(None)
            writer_process.join()

        logger.info("Data generation with real noise finished.")

    def _generate_one_sample(
        self,
        sample_index: int,
        noise_chunk: np.ndarray,
        snr_levels: List[float],
        output_path: str,
        bird_call_files: List[Path],
        target_sr: int,
        queue,
    ):
        """
        Generates a single mixed sample using a provided noise chunk.
        """
        logger.info(f"Generating sample {sample_index + 1}...")

        # Create a new bioacoustic source for each sample
        sample_duration = 20
        bio_sample = np.zeros(sample_duration * target_sr)
        num_calls = random.randint(1, 5)
        calls = []

        for _ in range(num_calls):
            wav_file = random.choice(bird_call_files)
            try:
                call_audio, call_sr = librosa.load(str(wav_file), sr=None)
                if call_sr != target_sr:
                    call_audio = librosa.resample(y=call_audio, orig_sr=call_sr, target_sr=target_sr)

                start_position = random.randint(0, len(bio_sample) - len(call_audio))
                bio_sample[start_position : start_position + len(call_audio)] += call_audio

                json_file = str(wav_file).replace(".wav", ".JSON").replace("WAV", "JSON")
                with open(json_file, "r") as f:
                    json_data = json.load(f)
                for onset, offset in zip(json_data.get("onsets", []), json_data.get("offsets", [])):
                    calls.append(
                        Call(
                            start_time=start_position / target_sr + onset,
                            end_time=start_position / target_sr + offset,
                            lower_freq=json_data.get("lower_freq"),
                            upper_freq=json_data.get("upper_freq"),
                            json_file=str(json_file),
                        )
                    )
            except Exception as e:
                logger.error(f"Could not process bird call file {wav_file}: {e}")
                continue

        bioacoustic_source = BioacousticSource(
            audio=AudioSignal(samples=bio_sample, sample_rate=target_sr), calls=calls
        )

        # Mix with the provided noise chunk
        for snr in snr_levels:
            mixed_signal = self.mixing_service.mix(
                bioacoustic_source=bioacoustic_source,
                noise=AudioSignal(samples=noise_chunk, sample_rate=target_sr),
                snr=snr,
            )

            # Save the signals
            output_dir = os.path.join(output_path, f"{snr}dB")
            for sub_dir in ["mixed", "bio", "noise"]:
                os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)

            mixed_output_file = os.path.join(output_dir, "mixed", f"sample_{sample_index}_mixed.wav")
            source_output_file = os.path.join(output_dir, "bio", f"sample_{sample_index}_source.wav")
            noise_output_file = os.path.join(output_dir, "noise", f"sample_{sample_index}_noise.wav")

            sf.write(mixed_output_file, mixed_signal.audio.samples, mixed_signal.audio.sample_rate)
            sf.write(
                source_output_file,
                mixed_signal.bioacoustic_source.audio.samples,
                mixed_signal.bioacoustic_source.audio.sample_rate,
            )
            sf.write(noise_output_file, mixed_signal.noise.samples, mixed_signal.noise.sample_rate)

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
            logger.info(f"Saved sample {sample_index + 1} with SNR {snr}dB")




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
