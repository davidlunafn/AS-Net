import argparse
import sys
import os

import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from as_net.app.services.data_generation import DataGenerationService
from as_net.app.services.evaluation import EvaluationService
from as_net.app.services.plotting import PlottingService
from as_net.domain.services.mixing import MixingService
from as_net.app.services.model_creation import ModelCreationService
from as_net.app.services.training import TrainingService
from as_net.domain.models.as_net import (
    ASNetConfig,
    DecoderConfig,
    EncoderConfig,
    SeparationModuleConfig,
)
from as_net.infrastructure.checkpoint_savers import CheckpointSaver
from as_net.infrastructure.data_loaders import AudioDataLoader
from as_net.infrastructure.history_savers import HistorySaver
from as_net.infrastructure.models.as_net_torch import ASNetTorchBuilder
from as_net.logger import logger


def generate_data(args):
    """Generates the dataset."""
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    data_gen_config = config_dict.get("data_generation", {}).get("synthetic", {})

    num_samples = args.num_samples if args.num_samples is not None else data_gen_config.get("num_samples", 100)
    snr_levels = args.snr_levels if args.snr_levels is not None else data_gen_config.get("snr_levels", [-15, -10, -5, 0, 5, 10, 15])
    output_path = args.output_path if args.output_path is not None else data_gen_config.get("output_path", "data/processed")

    mixing_service = MixingService()
    data_generation_service = DataGenerationService(mixing_service)
    data_generation_service.generate(
        num_samples=num_samples, snr_levels=snr_levels, output_path=output_path
    )


def generate_rain_data(args):
    """Generates the test dataset with real rain noise."""
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    data_gen_config = config_dict.get("data_generation", {}).get("test_rain", {})

    bird_calls_dir = args.bird_calls_dir if args.bird_calls_dir is not None else data_gen_config.get("bird_calls_dir")
    rain_audio_dir = args.rain_audio_dir if args.rain_audio_dir is not None else data_gen_config.get("rain_audio_dir")
    output_dir = args.output_dir if args.output_dir is not None else data_gen_config.get("output_dir")
    target_sr = args.target_sr if args.target_sr is not None else data_gen_config.get("target_sr", 22050)
    num_samples = args.num_samples if args.num_samples is not None else data_gen_config.get("num_samples", 100)
    snr_levels = args.snr_levels if args.snr_levels is not None else data_gen_config.get("snr_levels", [-15, -10, -5, 0, 5, 10, 15])

    mixing_service = MixingService()
    data_generation_service = DataGenerationService(mixing_service)
    data_generation_service.generate_with_real_noise(
        bird_calls_dir=bird_calls_dir,
        noise_dir=rain_audio_dir,
        output_path=output_dir,
        target_sr=target_sr,
        num_samples=num_samples,
        snr_levels=snr_levels,
    )


def train_model(args):
    """Trains the model."""
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    train_config = config_dict.get("training", {})
    num_epochs = args.num_epochs if args.num_epochs is not None else train_config.get("epochs", 100)
    batch_size = args.batch_size if args.batch_size is not None else train_config.get("batch_size", 16)
    learning_rate = args.lr if args.lr is not None else train_config.get("learning_rate", 1e-4)
    num_workers = args.num_workers if args.num_workers is not None else train_config.get("num_workers", 4)
    accumulation_steps = (
        args.accumulation_steps
        if args.accumulation_steps is not None
        else train_config.get("accumulation_steps", 1)
    )
    steps_per_epoch = (
        args.steps_per_epoch if args.steps_per_epoch is not None else train_config.get("steps_per_epoch")
    )
    validation_steps = (
        args.validation_steps
        if args.validation_steps is not None
        else train_config.get("validation_steps")
    )
    early_stopping_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else train_config.get("early_stopping_patience")
    )

    if args.dropout_rate is not None:
        config_dict["separation"]["dropout_rate"] = args.dropout_rate

    config = ASNetConfig(
        encoder=EncoderConfig(**config_dict["encoder"]),
        decoder=DecoderConfig(**config_dict["decoder"]),
        separation=SeparationModuleConfig(
            **config_dict["separation"], tcn_blocks=[]
        ),
    )

    data_loader = AudioDataLoader(batch_size=batch_size, num_workers=num_workers)
    checkpoint_saver = CheckpointSaver()
    history_saver = HistorySaver()
    model_builder = ASNetTorchBuilder()
    model_creation_service = ModelCreationService(model_builder=model_builder)
    training_service = TrainingService(
        data_loader=data_loader,
        checkpoint_saver=checkpoint_saver,
        history_saver=history_saver,
    )

    model = model_creation_service.create_model(config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.5)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device.upper()}")

    training_service.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        config=config,
        accumulation_steps=accumulation_steps,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        early_stopping_patience=early_stopping_patience,
    )


def evaluate_model(args):
    """Evaluates the model."""
    logger.info(f"Loading checkpoint from {args.checkpoint}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device.upper()}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model_builder = ASNetTorchBuilder()
    model = model_builder.build(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded successfully.")

    data_loader = AudioDataLoader(
        data_path=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers
    )
    evaluation_service = EvaluationService(data_loader=data_loader)
    evaluation_service.evaluate(
        model=model, device=device, output_path=args.output, save_audio_path=args.save_audio_path
    )


def plot_results(args):
    """Plots various results."""
    plotting_service = PlottingService()
    if args.type == "all":
        logger.info("Generating all summary plots...")
        plotting_service.plot_learning_curves(args.history_file)
        plotting_service.plot_sdr_improvement(args.results_file)
        plotting_service.plot_evaluation_distribution(args.results_file)
        logger.info("Finished generating all summary plots.")
    elif args.type == "learning-curves":
        plotting_service.plot_learning_curves(args.history_file)
    elif args.type == "sdr-improvement":
        plotting_service.plot_sdr_improvement(args.results_file)
    elif args.type == "eval-distribution":
        plotting_service.plot_evaluation_distribution(args.results_file)
    elif args.type == "error-spectrogram":
        plotting_service.plot_error_spectrogram(args.clean_file, args.separated_file)


def main():
    """Main function for the AS-Net project."""
    parser = argparse.ArgumentParser(description="AS-Net: A Deep Learning Framework for Bioacoustic Source Separation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate the dataset.")
    generate_parser.add_argument("--num-samples", type=int, help="Number of samples to generate.")
    generate_parser.add_argument("--snr-levels", type=int, nargs="+", help="SNR levels to use.")
    generate_parser.add_argument("--output-path", type=str, help="Path to save the generated data.")
    generate_parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    generate_parser.set_defaults(func=generate_data)

    # Generate rain test data command
    generate_rain_parser = subparsers.add_parser("generate-rain-test", help="Generate the test dataset with real rain noise.")
    generate_rain_parser.add_argument("--bird-calls-dir", type=str, help="Directory with the clean bird vocalizations.")
    generate_rain_parser.add_argument("--rain-audio-dir", type=str, help="Directory with the rain recordings.")
    generate_rain_parser.add_argument("--output-dir", type=str, help="Directory to save the generated dataset.")
    generate_rain_parser.add_argument("--target-sr", type=int, help="Target sample rate.")
    generate_rain_parser.add_argument("--num-samples", type=int, help="Number of samples to generate.")
    generate_rain_parser.add_argument("--snr-levels", type=int, nargs="+", help="SNR levels to use.")
    generate_rain_parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    generate_rain_parser.set_defaults(func=generate_rain_data)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model.")
    train_parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs to train for (overrides config).")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training (overrides config).")
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    train_parser.add_argument("--num-workers", type=int, default=None, help="Number of workers for the DataLoader (overrides config).")
    train_parser.add_argument("--accumulation-steps", type=int, default=None, help="Number of steps for gradient accumulation (overrides config).")
    train_parser.add_argument("--steps-per-epoch", type=int, default=None, help="Number of steps per epoch (overrides config).")
    train_parser.add_argument("--config", type=str, default="config.yaml", help="Path to the model configuration file.")
    train_parser.add_argument("--dropout-rate", type=float, default=None, help="Dropout rate for regularization (overrides config file).")
    train_parser.add_argument("--validation-steps", type=int, default=None, help="Number of steps for validation (overrides config).")
    train_parser.add_argument("--early-stopping-patience", type=int, default=None, help="Number of epochs to wait before early stopping (overrides config).")
    train_parser.set_defaults(func=train_model)

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model.")
    evaluate_parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint to evaluate.")
    evaluate_parser.add_argument("--data-path", type=str, default="/Volumes/SSD DL/osfstorage-archive/data/test_data/", help="Path to the test data to evaluate.")
    evaluate_parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation.")
    evaluate_parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for the DataLoader.")
    evaluate_parser.add_argument("--output", type=str, default="models/evaluation_results.csv", help="Path to save the evaluation results CSV.")
    evaluate_parser.add_argument("--save-audio-path", type=str, default="data/separated_audio/", help="Path to save separated audio files for analysis.")
    evaluate_parser.set_defaults(func=evaluate_model)

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots for the paper.")
    plot_subparsers = plot_parser.add_subparsers(dest="type", required=True)

    # 'all' subcommand
    all_parser = plot_subparsers.add_parser("all", help="Generate all standard plots at once.")
    all_parser.add_argument("--history-file", type=str, default="models/training_history.csv", help="Path to the training history CSV file.")
    all_parser.add_argument("--results-file", type=str, default="models/evaluation_results.csv", help="Path to the evaluation results CSV file.")
    all_parser.set_defaults(func=plot_results)

    # Individual plot subcommands
    lc_parser = plot_subparsers.add_parser("learning-curves", help="Plot training and validation learning curves.")
    lc_parser.add_argument("--history-file", type=str, default="models/training_history.csv", help="Path to the training history CSV file.")
    lc_parser.set_defaults(func=plot_results)

    sdr_parser = plot_subparsers.add_parser("sdr-improvement", help="Plot SI-SDR improvement.")
    sdr_parser.add_argument("--results-file", type=str, default="models/evaluation_results.csv", help="Path to the evaluation results CSV file.")
    sdr_parser.set_defaults(func=plot_results)

    ed_parser = plot_subparsers.add_parser("eval-distribution", help="Plot distribution of evaluation metrics.")
    ed_parser.add_argument("--results-file", type=str, default="models/evaluation_results.csv", help="Path to the evaluation results CSV file.")
    ed_parser.set_defaults(func=plot_results)

    err_parser = plot_subparsers.add_parser("error-spectrogram", help="Plot the spectrogram of the residual error.")
    err_parser.add_argument("--clean-file", type=str, required=True, help="Path to the original clean audio file.")
    err_parser.add_argument("--separated-file", type=str, required=True, help="Path to the model-separated audio file.")
    err_parser.set_defaults(func=plot_results)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()