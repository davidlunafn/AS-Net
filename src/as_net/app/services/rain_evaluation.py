import os
import re
import glob
import shutil
import tempfile
from typing import Any, Dict

import pandas as pd
import soundfile as sf
import torch
import librosa
import numpy as np
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from mir_eval.separation import bss_eval_sources
from tqdm import tqdm
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

from as_net.logger import logger

class RainEvaluationService:
    """Service for evaluating the model on the real rain test set."""

    def __init__(self, model: Any, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config.get("device", "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.si_sdr = ScaleInvariantSignalDistortionRatio().to(self.device)
        self.birdnet_analyzer = Analyzer()

    def evaluate(self, output_csv_path: str, save_audio_path: str | None = None):
        """Evaluates the model on the rain test set and saves the results."""
        logger.info("Starting evaluation on rain test set...")
        if save_audio_path:
            logger.info(f"Separated audio will be saved to {save_audio_path}")
            os.makedirs(save_audio_path, exist_ok=True)

        test_files = self._get_test_files()
        if not test_files:
            logger.warning("No test files found. Skipping evaluation.")
            return

        results: Dict[str, list] = {
            "snr_level": [], "filename": [], "si_sdr_mixture": [],
            "si_sdr_separated": [], "si_sdr_improvement": [], "sdr_mixture": [],
            "sdr_separated": [], "sdr_improvement": [], "sir_mixture": [],
            "sir_separated": [], "sir_improvement": [], "f1_clean": [],
            "f1_mixed": [], "f1_separated": [],
        }

        temp_dir = tempfile.mkdtemp()
        try:
            with torch.no_grad():
                for file_path in tqdm(test_files, desc="Evaluating Rain Set"):
                    try:
                        snr_level, clean_signal, noise_signal, mixed_signal = self._load_audio_files(file_path)

                        mixed_dir, mixed_file_name = os.path.split(file_path)
                        base_dir = os.path.dirname(mixed_dir)
                        base_name = mixed_file_name.replace("_mixed.wav", "")
                        clean_name = f"{base_name}_source.wav"
                        clean_path = os.path.join(base_dir, "bio", clean_name)

                        separated_bio, sdr_metrics = self._get_objective_metrics(clean_signal, noise_signal, mixed_signal)
                        f1_scores = self._get_functional_metrics(clean_path, file_path, separated_bio, temp_dir)

                        if save_audio_path:
                            output_dir = os.path.join(save_audio_path, f"{snr_level}dB")
                            os.makedirs(output_dir, exist_ok=True)
                            output_filename = f"{base_name}_separated.wav"
                            output_filepath = os.path.join(output_dir, output_filename)
                            sf.write(output_filepath, separated_bio, self.config["audio"]["sample_rate"])

                        results["snr_level"].append(snr_level)
                        results["filename"].append(os.path.basename(file_path))
                        for key, value in sdr_metrics.items():
                            results[key].append(value)
                        for key, value in f1_scores.items():
                            results[key].append(value)

                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
        finally:
            shutil.rmtree(temp_dir)

        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Rain evaluation results saved to {output_csv_path}")

        avg_results = df.groupby("snr_level")[list(df.columns)[2:]].mean()
        logger.info("\nAverage results per SNR level:\n" + avg_results.to_string())

    def _get_test_files(self) -> list[str]:
        """Finds all the mixed audio files in the rain test set."""
        search_path = os.path.join(self.config["evaluation"]["rain_test_dir"], "**", "mixed", "*.wav")
        return glob.glob(search_path, recursive=True)

    def _load_audio_files(self, mixed_path: str):
        """Loads the mixed, clean, and noise audio files for a given mixed path."""
        mixed_dir, mixed_file_name = os.path.split(mixed_path)
        base_dir = os.path.dirname(mixed_dir)
        base_name = mixed_file_name.replace("_mixed.wav", "")
        clean_name = f"{base_name}_source.wav"
        noise_name = f"{base_name}_noise.wav"
        clean_path = os.path.join(base_dir, "bio", clean_name)
        noise_path = os.path.join(base_dir, "noise", noise_name)

        if not os.path.exists(clean_path) or not os.path.exists(noise_path):
            raise FileNotFoundError(f"Clean or noise file not found for {mixed_path}. Looked for {clean_path} and {noise_path}")

        target_sr = self.config["audio"]["sample_rate"]
        mixed_signal, sr = librosa.load(mixed_path, sr=None)
        if sr != target_sr:
            mixed_signal = librosa.resample(mixed_signal, orig_sr=sr, target_sr=target_sr)
        clean_signal, sr = librosa.load(clean_path, sr=None)
        if sr != target_sr:
            clean_signal = librosa.resample(clean_signal, orig_sr=sr, target_sr=target_sr)
        noise_signal, sr = librosa.load(noise_path, sr=None)
        if sr != target_sr:
            noise_signal = librosa.resample(noise_signal, orig_sr=sr, target_sr=target_sr)

        snr_match = re.search(r"/(-?\d+)dB/", mixed_path)
        snr_level = int(snr_match.group(1)) if snr_match else -1
        return snr_level, clean_signal, noise_signal, mixed_signal

    def _get_objective_metrics(self, clean_signal: np.ndarray, noise_signal: np.ndarray, mixed_signal: np.ndarray):
        """Performs separation and calculates objective metrics."""
        epsilon = 1e-8
        mixture_tensor = torch.from_numpy(mixed_signal).float().unsqueeze(0).to(self.device)
        source_tensor = torch.from_numpy(clean_signal).float().unsqueeze(0).to(self.device)
        est_source1, est_source2 = self.model(mixture_tensor)
        sdr_perm1 = self.si_sdr(est_source1, source_tensor)
        sdr_perm2 = self.si_sdr(est_source2, source_tensor)
        if sdr_perm1 >= sdr_perm2:
            separated_bio_tensor = est_source1
        else:
            separated_bio_tensor = est_source2
        separated_bio = separated_bio_tensor.squeeze().cpu().numpy()

        power_clean = np.sum(clean_signal**2) + epsilon
        power_noise = np.sum(noise_signal**2) + epsilon
        sdr_mix = 10 * np.log10(power_clean / power_noise)
        sir_mix = sdr_mix

        reference_sources = np.array([clean_signal, noise_signal])
        estimated_sources_separated = np.array([separated_bio, mixed_signal - separated_bio])
        sdr_sep, sir_sep, _, _ = bss_eval_sources(reference_sources, estimated_sources_separated, compute_permutation=False)

        si_sdr_mixture = self.si_sdr(mixture_tensor, source_tensor).item()
        si_sdr_separated = self.si_sdr(separated_bio_tensor, source_tensor).item()

        return separated_bio, {
            "si_sdr_mixture": si_sdr_mixture, "si_sdr_separated": si_sdr_separated,
            "si_sdr_improvement": si_sdr_separated - si_sdr_mixture, "sdr_mixture": sdr_mix,
            "sdr_separated": sdr_sep[0], "sdr_improvement": sdr_sep[0] - sdr_mix,
            "sir_mixture": sir_mix, "sir_separated": sir_sep[0], "sir_improvement": sir_sep[0] - sir_mix,
        }

    def _get_functional_metrics(self, clean_path: str, mixed_path: str, separated_signal: np.ndarray, temp_dir: str) -> Dict[str, float]:
        """Runs BirdNET and calculates F1 score."""
        target_species = self.config.get("functional_evaluation", {}).get("target_species", "Great Tit")
        confidence_threshold = self.config.get("functional_evaluation", {}).get("confidence_threshold", 0.5)

        separated_path = os.path.join(temp_dir, "temp_separated.wav")
        sf.write(separated_path, separated_signal, self.config["audio"]["sample_rate"])

        paths_to_analyze = {"clean": clean_path, "mixed": mixed_path, "separated": separated_path}
        f1_scores = {}

        for key, path in paths_to_analyze.items():
            try:
                recording = Recording(analyzer=self.birdnet_analyzer, path=path, min_conf=confidence_threshold)
                recording.analyze()
                detections = recording.detections
                true_positive = 0
                for detection in detections:
                    if detection['common_name'] == target_species:
                        true_positive = 1
                        break
                if true_positive == 1:
                    f1_scores[f"f1_{key}"] = 1.0
                else:
                    f1_scores[f"f1_{key}"] = 0.0
            except Exception as e:
                logger.error(f"BirdNET analysis failed for {path}: {e}")
                f1_scores[f"f1_{key}"] = -1.0
        return f1_scores