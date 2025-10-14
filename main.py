import argparse
import sys
import os

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from as_net.app.services.data_generation import DataGenerationService
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

    # Create the services
    mixing_service = MixingService()
    data_generation_service = DataGenerationService(mixing_service)

    # Generate the data
    data_generation_service.generate(num_samples=args.num_samples, snr_levels=args.snr_levels)


def train_model(args):
    """Trains the model."""

    # 1. Configuration
    # TODO: This should be loaded from a file (e.g., YAML)
    config = ASNetConfig(
        encoder=EncoderConfig(kernel_size=16, stride=8, out_channels=256),
        decoder=DecoderConfig(kernel_size=16, stride=8, in_channels=256),
        separation=SeparationModuleConfig(
            num_blocks=8, tcn_blocks=[]
        ),  # TCN blocks are created inside the model for now
    )

    # 2. Instantiate Adapters and Services
    data_loader = AudioDataLoader(batch_size=args.batch_size, num_workers=args.num_workers)
    checkpoint_saver = CheckpointSaver()
    history_saver = HistorySaver()
    model_builder = ASNetTorchBuilder()
    model_creation_service = ModelCreationService(model_builder=model_builder)
    training_service = TrainingService(
        data_loader=data_loader,
        checkpoint_saver=checkpoint_saver,
        history_saver=history_saver,
    )

    # 3. Create Model and Optimizer
    model = model_creation_service.create_model(config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Start Training
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
        num_epochs=args.num_epochs,
        device=device,
        config=config,
        accumulation_steps=args.accumulation_steps,
        steps_per_epoch=args.steps_per_epoch,
    )


def evaluate_model(args):
    """Evaluates the model."""
    # TODO: Implement the evaluation logic
    print("Evaluating the model...")


def main():
    """Main function for the AS-Net project."""

    # Create the argument parser
    parser = argparse.ArgumentParser(description="AS-Net: A Deep Learning Framework for Bioacoustic Source Separation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create the parser for the "generate" command
    generate_parser = subparsers.add_parser("generate", help="Generate the dataset.")
    generate_parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate.")
    generate_parser.add_argument("--snr-levels", type=int, nargs="+", default=[-5, 0, 5, 10, 15], help="SNR levels to use.")
    generate_parser.set_defaults(func=generate_data)

    # Create the parser for the "train" command
    train_parser = subparsers.add_parser("train", help="Train the model.")
    train_parser.add_argument("--num-epochs", type=int, default=100, help="Number of epochs to train for.")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train_parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for the DataLoader."
    )
    train_parser.add_argument(
        "--accumulation-steps", type=int, default=1, help="Number of steps for gradient accumulation."
    )
    train_parser.add_argument(
        "--steps-per-epoch", type=int, default=None, help="Number of steps per epoch. Runs a full epoch if not set."
    )
    train_parser.set_defaults(func=train_model)

    # Create the parser for the "evaluate" command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model.")
    evaluate_parser.set_defaults(func=evaluate_model)

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function
    args.func(args)


if __name__ == "__main__":
    main()
