import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

from as_net.logger import logger


class PlottingService:
    """Service for creating publication-quality plots."""

    def __init__(self):
        sns.set_theme(style="whitegrid", palette="viridis")
        self.output_dir = "models/plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_learning_curves(self, history_file: str) -> None:
        """Plots and saves the training and validation learning curves."""
        logger.info(f"Plotting learning curves from {history_file}")
        history_df = pd.read_csv(history_file)

        plt.figure(figsize=(12, 6))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Training Loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
        plt.title("Learning Curves", fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("SI-SDR Loss", fontsize=12)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "learning_curves.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved learning curves plot to {output_path}")
        plt.close()

    def plot_sdr_improvement(self, results_file: str) -> None:
        """Plots the SI-SDR improvement as a bar chart."""
        logger.info(f"Plotting SI-SDR improvement from {results_file}")
        results_df = pd.read_csv(results_file)
        avg_results = results_df.groupby("snr_level")["si_sdr_improvement"].mean().reset_index()

        plt.figure(figsize=(12, 6))
        sns.barplot(x="snr_level", y="si_sdr_improvement", data=avg_results)
        plt.title("Average SI-SDR Improvement vs. Input SNR", fontsize=16)
        plt.xlabel("Input SNR Level (dB)", fontsize=12)
        plt.ylabel("SI-SDR Improvement (dB)", fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "sdr_improvement_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved SI-SDR improvement plot to {output_path}")
        plt.close()

    def plot_evaluation_distribution(self, results_file: str) -> None:
        """Plots the distribution of evaluation metrics as a box plot."""
        logger.info(f"Plotting evaluation distribution from {results_file}")
        results_df = pd.read_csv(results_file)

        plt.figure(figsize=(12, 6))
        sns.boxplot(x="snr_level", y="si_sdr_separated", data=results_df)
        plt.title("Distribution of Separated SI-SDR vs. Input SNR", fontsize=16)
        plt.xlabel("Input SNR Level (dB)", fontsize=12)
        plt.ylabel("Output SI-SDR (dB)", fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "evaluation_distribution_box.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved evaluation distribution box plot to {output_path}")
        plt.close()

    def _plot_spectrogram(self, ax, y, sr, title):
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log", ax=ax)
        ax.set_title(title, fontsize=14)
        return img

    def plot_error_spectrogram(self, clean_file: str, separated_file: str) -> None:
        """Plots the spectrogram of the residual error."""
        logger.info(f"Plotting error spectrogram between {clean_file} and {separated_file}")
        y_clean, sr = sf.read(clean_file)
        y_separated, _ = sf.read(separated_file)

        # Ensure same length
        min_len = min(len(y_clean), len(y_separated))
        y_clean = y_clean[:min_len]
        y_separated = y_separated[:min_len]

        error_signal = y_clean - y_separated

        fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True, sharey=True)
        self._plot_spectrogram(axes[0], y_clean, sr, "Clean Source")
        self._plot_spectrogram(axes[1], y_separated, sr, "Separated Source")
        img = self._plot_spectrogram(axes[2], error_signal, sr, "Residual Error")
        fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.8)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "error_spectrogram.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved error spectrogram plot to {output_path}")
        plt.close()