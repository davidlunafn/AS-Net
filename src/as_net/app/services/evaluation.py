import os
import re
from typing import Any, Dict

import pandas as pd
import soundfile as sf
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from tqdm import tqdm

from as_net.app.ports.data_loader import IDataLoader
from as_net.logger import logger


class EvaluationService:
    """Service for evaluating the model."""

    def __init__(self, data_loader: IDataLoader):
        self.data_loader = data_loader

    def evaluate(
        self, model: Any, device: str, output_path: str, save_audio_path: str = None
    ) -> None:
        """Evaluates the model on the test set and saves the results."""

        logger.info("Starting evaluation...")
        if save_audio_path:
            logger.info(f"Separated audio will be saved to {save_audio_path}")
            os.makedirs(save_audio_path, exist_ok=True)

        test_loader = self.data_loader.load_test_data()
        model.to(device)
        model.eval()

        si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
        results: Dict[str, list] = {
            "snr_level": [],
            "si_sdr_mixture": [],
            "si_sdr_separated": [],
            "filename": [],
        }

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                mixture, source, _, file_paths = batch  # We only need the clean source (source1)
                mixture, source = (
                    mixture.to(device),
                    source.to(device),
                )

                est_source1, est_source2 = model(mixture)

                # Permutation Invariant Training (PIT) style evaluation
                sdr_perm1 = si_sdr(est_source1, source)
                sdr_perm2 = si_sdr(est_source2, source)

                # Choose the best permutation
                if sdr_perm1 >= sdr_perm2:
                    best_est_source = est_source1
                    sdr_val = sdr_perm1.item()
                else:
                    best_est_source = est_source2
                    sdr_val = sdr_perm2.item()

                # Calculate SI-SDR of the original mixture
                sdr_mixture_val = si_sdr(mixture, source).item()

                # Store results for each item in the batch
                for i in range(len(file_paths)):
                    snr_match = re.search(r"/(-?\d+)dB/", file_paths[i])
                    snr_level = int(snr_match.group(1)) if snr_match else -1

                    results["snr_level"].append(snr_level)
                    results["si_sdr_separated"].append(sdr_val)
                    results["si_sdr_mixture"].append(sdr_mixture_val)
                    results["filename"].append(os.path.basename(file_paths[i]))

                    # Save the separated audio if requested
                    if save_audio_path:
                        output_filename = os.path.basename(file_paths[i]).replace(
                            "_mixed.wav", "_separated.wav"
                        )
                        output_filepath = os.path.join(save_audio_path, output_filename)
                        sf.write(
                            output_filepath,
                            best_est_source[i].cpu().numpy(),
                            22050,  # Assuming a fixed sample rate
                        )

        # Create and save the results dataframe
        df = pd.DataFrame(results)
        df["si_sdr_improvement"] = df["si_sdr_separated"] - df["si_sdr_mixture"]

        avg_results = df.groupby("snr_level")[["si_sdr_mixture", "si_sdr_separated", "si_sdr_improvement"]].mean()
        logger.info("\nEvaluation results (averages):\n" + avg_results.to_string())

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Full evaluation results saved to {output_path}")