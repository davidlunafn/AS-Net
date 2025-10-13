import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from as_net.app.services.data_generation import DataGenerationService
from as_net.domain.services.mixing import MixingService


def generate_data(args):
    """Generates the dataset."""

    # Create the services
    mixing_service = MixingService()
    data_generation_service = DataGenerationService(mixing_service)

    # Generate the data
    data_generation_service.generate(num_samples=args.num_samples, snr_levels=args.snr_levels)


def train_model(args):
    """Trains the model."""
    # TODO: Implement the training logic
    print("Training the model...")


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
