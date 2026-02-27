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

    def __init__(self, config: dict):
        sns.set_theme(style="whitegrid", palette="viridis")
        self.output_dir = "models/plots"
        self.config = config
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_learning_curves(self, history_file: str) -> None:
        """Plots training and validation learning curves with the best model highlighted."""
        logger.info(f"Plotting learning curves from {history_file}")
        history_df = pd.read_csv(history_file)

        # --- Data Preparation (No Smoothing) ---
        history_df.rename(columns={'train_loss': 'Training Loss', 'val_loss': 'Validation Loss'}, inplace=True)
        plot_df = history_df.melt(
            id_vars=['epoch'], 
            value_vars=['Training Loss', 'Validation Loss'], 
            var_name='Loss Type', 
            value_name='Loss'
        )

        # --- Find Best Epoch ---
        best_epoch_idx = history_df['Validation Loss'].idxmin()
        best_epoch = history_df.loc[best_epoch_idx, 'epoch']
        best_loss = history_df.loc[best_epoch_idx, 'Validation Loss']

        # --- Plotting ---
        plt.figure(figsize=(12, 8))
        ax = sns.lineplot(
            data=plot_df, 
            x='epoch', 
            y='Loss', 
            hue='Loss Type', 
            style='Loss Type', 
            linewidth=1.5, # Thinner line for noisy data
            alpha=0.8 # Slight transparency
        )

        # --- Highlight Best Model ---
        ax.axvline(x=best_epoch, color='grey', linestyle='--', linewidth=1.5)
        ax.axhline(y=best_loss, color='grey', linestyle='--', linewidth=1.5)
        ax.plot(best_epoch, best_loss, marker='*', markersize=15, color='gold', markeredgecolor='black', label=f'Best Model (Epoch {best_epoch})')
        ax.annotate(f'Best Loss: {best_loss:.4f}', 
                    xy=(best_epoch, best_loss), 
                    xytext=(best_epoch + 2, best_loss * 0.9), # Offset text slightly
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    fontsize=16, backgroundcolor='w')

        # --- Aesthetics ---
        ax.set_title("Model Convergence", fontsize=30, weight='bold', pad=20)
        ax.set_xlabel("Epoch", fontsize=22, weight='bold')
        ax.set_ylabel("SI-SDR Loss", fontsize=22, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, which="both", ls="--", c='0.85')
        sns.despine()
        plt.legend(title='', fontsize=16)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "learning_curves.png")
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved enhanced learning curves plot to {output_path}")
        plt.close()

    def plot_sdr_improvement(self, results_file: str) -> None:
        """Plots the SI-SDR improvement as a bar chart."""
        with plt.style.context('seaborn-v0_8-paper'):
            logger.info(f"Plotting SI-SDR improvement from {results_file}")
            results_df = pd.read_csv(results_file)
            avg_results = results_df.groupby("snr_level")["si_sdr_improvement"].mean().reset_index()

            plt.figure(figsize=(10, 6))
            bar_plot = sns.barplot(x="snr_level", y="si_sdr_improvement", data=avg_results, palette="viridis")
            
            # Add value labels on top of each bar
            for index, row in avg_results.iterrows():
                bar_plot.text(index, row.si_sdr_improvement + 0.1, f'{row.si_sdr_improvement:.2f} dB', 
                              color='black', ha="center", fontsize=10)
            
            plt.title("Average SI-SDR Improvement vs. Input SNR", fontsize=16, weight='bold')
            plt.xlabel("Input SNR Level (dB)", fontsize=12)
            plt.ylabel("SI-SDR Improvement (dB)", fontsize=12)
            plt.ylim(0, avg_results.si_sdr_improvement.max() * 1.1)
            plt.tight_layout()

            output_path = os.path.join(self.output_dir, "sdr_improvement_bar.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved SI-SDR improvement plot to {output_path}")
            plt.close()

    def plot_evaluation_distribution(self, results_file: str) -> None:
        """Plots the distribution of evaluation metrics as a grouped box plot (before vs. after)."""
        with plt.style.context('seaborn-v0_8-paper'):
            logger.info(f"Plotting before-and-after evaluation distribution from {results_file}")
            results_df = pd.read_csv(results_file)

            # Prepare dataframe for grouped boxplot
            df_melted = results_df.melt(id_vars=['snr_level'], 
                                        value_vars=['si_sdr_mixture', 'si_sdr_separated'], 
                                        var_name='Signal Type', 
                                        value_name='SI-SDR (dB)')
            
            df_melted['Signal Type'] = df_melted['Signal Type'].map({
                'si_sdr_mixture': 'Mixture (Input)',
                'si_sdr_separated': 'Separated (Output)'
            })

            plt.figure(figsize=(12, 7))
            sns.boxplot(x="snr_level", y="SI-SDR (dB)", hue="Signal Type", data=df_melted, palette="viridis")
            
            plt.title("Input vs. Output SI-SDR Distribution by SNR Level", fontsize=16, weight='bold')
            plt.xlabel("Input SNR Level (dB)", fontsize=12)
            plt.ylabel("SI-SDR (dB)", fontsize=12)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend(title='Signal Type', fontsize=10)
            plt.tight_layout()

            output_path = os.path.join(self.output_dir, "evaluation_distribution_box.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved evaluation distribution box plot to {output_path}")
            plt.close()

    def plot_sdr_sir_improvement(self, results_file: str) -> None:
        """Plots the SDR and SIR improvement as a grouped bar chart."""
        logger.info(f"Plotting SDR and SIR improvement from {results_file}")
        results_df = pd.read_csv(results_file)
        
        # Ensure required columns exist
        if "sdr_improvement" not in results_df.columns or "sir_improvement" not in results_df.columns:
            logger.warning("SDR/SIR improvement columns not found in results file. Skipping plot.")
            return

        avg_results = results_df.groupby("snr_level")[["sdr_improvement", "sir_improvement"]].mean().reset_index()
        
        # Melt the dataframe for grouped bar plot
        df_melted = avg_results.melt(id_vars='snr_level', var_name='Metric', value_name='Improvement (dB)')
        df_melted["Metric"] = df_melted["Metric"].str.replace("_improvement", "").str.upper()

        plt.figure(figsize=(14, 7))
        sns.barplot(x="snr_level", y="Improvement (dB)", hue="Metric", data=df_melted)
        plt.title("Average SDR & SIR Improvement vs. Input SNR", fontsize=16)
        plt.xlabel("Input SNR Level (dB)", fontsize=12)
        plt.ylabel("Improvement (dB)", fontsize=12)
        plt.legend(title="Metric")
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "sdr_sir_improvement_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved SDR/SIR improvement plot to {output_path}")
        plt.close()

    def plot_f1_scores(self, results_file: str) -> None:
        """Plots the F1 scores for clean, mixed, and separated audio with variance."""
        logger.info(f"Plotting F1 scores with variance from {results_file}")
        results_df = pd.read_csv(results_file)

        # Ensure required columns exist
        f1_cols = ["f1_clean", "f1_mixed", "f1_separated"]
        if not all(col in results_df.columns for col in f1_cols):
            logger.warning("F1 score columns not found in results file. Skipping plot.")
            return

        # Calculate mean and std dev
        stats_df = results_df.groupby("snr_level")[f1_cols].agg(['mean', 'std']).reset_index()
        stats_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in stats_df.columns.values]

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(14, 9))
        
        # Define colors and markers for consistency
        colors = {'f1_clean': '#2ca02c', 'f1_mixed': '#d62728', 'f1_separated': '#1f77b4'} # Green, Red, Blue
        markers = {'f1_clean': 's', 'f1_mixed': 'o', 'f1_separated': '^'}
        labels = {'f1_clean': 'Clean', 'f1_mixed': 'Mixed', 'f1_separated': 'Separated (AS-Net)'}

        for col in f1_cols:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            
            # Line plot for the mean
            plt.plot(stats_df['snr_level'], stats_df[mean_col], marker=markers[col], linestyle='-', label=labels[col], color=colors[col], markersize=8)

        # Aesthetics
        plt.title("F1 Score for BirdNET Detection vs. Input SNR", fontsize=24, weight='bold', pad=20)
        plt.xlabel("Input SNR Level (dB)", fontsize=20, weight='bold')
        plt.ylabel("F1 Score", fontsize=20, weight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim(0, 1.05)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        
        legend = plt.legend(title="Audio Type", fontsize=14, title_fontsize=16)
        legend.get_frame().set_alpha(0.8)
        
        sns.despine()
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "f1_score_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved F1 score comparison plot to {output_path}")
        plt.close()

    def plot_spectrogram_grid(self, results_file: str) -> None:
        """Plots a 7x3 grid of spectrograms for clean, mixed, and separated audio across SNR levels, using a random sample for each SNR."""
        logger.info(f"Generating final spectrogram grid...")
        results_df = pd.read_csv(results_file)
        snr_levels = sorted(results_df["snr_level"].unique())

        fig, axes = plt.subplots(len(snr_levels), 3, figsize=(20, 30), sharex=True, sharey=True)

        for i, snr in enumerate(snr_levels):
            snr_samples_df = results_df[results_df["snr_level"] == snr]
            random_sample_filename = snr_samples_df.sample(1)["filename"].iloc[0]
            sample_base_name = random_sample_filename.replace("_mixed.wav", "")
            logger.info(f"Plotting sample '{sample_base_name}' for SNR {snr}dB")

            rain_test_dir = self.config["evaluation"]["rain_test_dir"]
            separated_audio_dir = self.config["evaluation"]["save_separated_path"]
            clean_path = os.path.join(rain_test_dir, f"{snr}dB", "bio", f"{sample_base_name}_source.wav")
            mixed_path = os.path.join(rain_test_dir, f"{snr}dB", "mixed", f"{sample_base_name}_mixed.wav")
            separated_path = os.path.join(separated_audio_dir, f"{snr}dB", f"{sample_base_name}_separated.wav")

            # --- Plotting ---
            y_clean, sr = librosa.load(clean_path, sr=None)
            self._plot_spectrogram(axes[i, 0], y_clean, sr, "")
            y_mixed, _ = librosa.load(mixed_path, sr=sr)
            self._plot_spectrogram(axes[i, 1], y_mixed, sr, "")
            y_separated, _ = librosa.load(separated_path, sr=sr)
            img = self._plot_spectrogram(axes[i, 2], y_separated, sr, "")

            # --- Set Titles and Labels ---
            if i == 0:
                axes[i, 0].set_title("Clean", fontsize=26, weight='bold')
                axes[i, 1].set_title("Mixed", fontsize=26, weight='bold')
                axes[i, 2].set_title("Separated (AS-Net)", fontsize=26, weight='bold')
            axes[i, 0].set_ylabel(f"{snr} dB", fontsize=24, weight='bold')

        # --- Final Aesthetic Tweaks ---
        for i in range(len(snr_levels)):
            for j in range(3):
                axes[i, j].tick_params(axis='both', which='major', labelsize=16) # Larger tick numbers
                axes[i, j].xaxis.label.set_size(20) # Larger x-axis label
                axes[i, j].yaxis.label.set_size(20) # Larger y-axis label
                if i < len(snr_levels) - 1:
                    axes[i, j].set_xlabel('')
                if j > 0:
                    axes[i, j].set_ylabel('')
        
        axes[-1, 0].set_xticks([0, 5, 10, 15, 20])
        axes[-1, 1].set_xticks([0, 5, 10, 15, 20])
        axes[-1, 2].set_xticks([0, 5, 10, 15, 20])
        for i in range(len(snr_levels)):
            # Set custom ticks on the linear scale
            tick_locs = [2000, 4000, 6000, 8000]
            axes[i, 0].set_yticks(tick_locs)
            axes[i, 0].set_ylim(0, 10000)
            axes[i, 0].yaxis.labelpad = 15

        # Adjust layout and spacing
        fig.subplots_adjust(bottom=0.15, wspace=0.05, hspace=0.05) # Compact spacing
        cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.015])
        cbar = fig.colorbar(img, cax=cbar_ax, orientation='horizontal', format="%+2.0f dB")
        cbar.set_label('Power/Frequency (dB/Hz)', size=20)
        cbar.ax.tick_params(labelsize=16)

        output_path = os.path.join(self.output_dir, "spectrogram_grid.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved final spectrogram grid to {output_path}")
        plt.close()

    def _plot_spectrogram(self, ax, y, sr, title):
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="linear", ax=ax)
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