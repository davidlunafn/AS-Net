# Project Overview

This project, "AS-Net", is a research initiative focused on developing a deep learning framework for bioacoustic source separation. The primary goal is to separate bird vocalizations from mixed-source soundscapes, a critical task in computational ecoacoustics for biodiversity monitoring. The project is detailed in a comprehensive research proposal located in `docs/planning.md`.

The proposed solution is a novel deep neural network architecture named "Avian Separation Network (AS-Net)". This model is designed to be an end-to-end, time-domain, fully-convolutional network inspired by architectures like Conv-TasNet, but specifically adapted for the unique spectro-temporal characteristics of avian bioacoustics.

The project will use a synthetic dataset of bird vocalizations mixed with pink noise to train and evaluate the model. The evaluation will be a dual-approach, using both objective signal-quality metrics (SDR, SIR) and a functional evaluation based on the performance of a downstream bird sound classifier (BirdNET).

## Building and Running

The project is a Python-based deep learning project. The core dependencies are PyTorch and librosa.

**TODO:** Add specific commands for building, running, and testing the project once the codebase is more mature. This will likely include:

*   `pip install -r requirements.txt` to install dependencies.
*   `python train.py` to train the model.
*   `python evaluate.py` to evaluate the model.

## Development Conventions

*   **Code Style:** The project will follow standard Python coding conventions (PEP 8).
*   **Testing:** The project will include a suite of tests to ensure the correctness of the data generation, model architecture, and evaluation pipeline.
*   **Versioning:** The project will use Git for version control.
