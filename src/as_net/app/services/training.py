from typing import Any

import torch
from tqdm import tqdm

from as_net.app.ports.checkpoint_saver import ICheckpointSaver
from as_net.app.ports.data_loader import IDataLoader
from as_net.app.ports.history_saver import IHistorySaver
from as_net.domain.models.as_net import ASNetConfig
from as_net.logger import logger


def si_sdr_loss(estimation: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Calculates the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.
    The loss is the negative SDR.

    Args:
        estimation (torch.Tensor): Estimated signal (batch_size, num_samples).
        target (torch.Tensor): Target signal (batch_size, num_samples).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: The SI-SDR loss (a single value).
    """
    # Ensure target and estimate have the same shape
    if target.shape != estimation.shape:
        raise ValueError(f"Target and estimate should have the same shape, but got {target.shape} and {estimation.shape}")

    # Subtract mean to make it zero-mean
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimation = estimation - torch.mean(estimation, dim=-1, keepdim=True)

    # <s_hat, s>
    dot_product = torch.sum(estimation * target, dim=-1, keepdim=True)
    # ||s||^2
    target_norm_squared = torch.sum(target**2, dim=-1, keepdim=True)

    # s_target = (<s_hat, s> * s) / ||s||^2
    s_target = (dot_product * target) / (target_norm_squared + epsilon)

    # e_noise = s_hat - s_target
    e_noise = estimation - s_target

    # SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    sdr = 10 * torch.log10(torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + epsilon) + epsilon)

    # The loss is the negative SDR, and we want to average over the batch
    return -torch.mean(sdr)


class TrainingService:
    """Service for training the model."""

    def __init__(
        self,
        data_loader: IDataLoader,
        checkpoint_saver: ICheckpointSaver,
        history_saver: IHistorySaver,
    ):
        self.data_loader = data_loader
        self.checkpoint_saver = checkpoint_saver
        self.history_saver = history_saver

    def train(
        self,
        model: Any,
        optimizer: Any,
        num_epochs: int,
        device: str,
        config: ASNetConfig,
        accumulation_steps: int = 1,
        steps_per_epoch: int = None,
    ):
        """Trains the model."""

        logger.info("Starting training...")
        if accumulation_steps > 1:
            logger.info(f"Using gradient accumulation with {accumulation_steps} steps.")
        if steps_per_epoch is not None:
            logger.info(f"Running for {steps_per_epoch} steps per epoch.")

        train_loader = self.data_loader.load_train_data()
        val_loader = self.data_loader.load_val_data()

        model.to(device)

        for epoch in range(num_epochs):
            # Training loop
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            num_steps_to_run = steps_per_epoch if steps_per_epoch is not None else len(train_loader)
            train_iterator = iter(train_loader)

            with tqdm(range(num_steps_to_run), desc=f"Epoch {epoch + 1}/{num_epochs} [Train]") as pbar:
                for i in pbar:
                    try:
                        batch = next(train_iterator)
                    except StopIteration:
                        logger.warning("DataLoader exhausted before reaching steps_per_epoch. Ending epoch early.")
                        num_steps_to_run = i  # Adjust for correct average loss calculation
                        break

                    # Assuming batch is a tuple of (mixture, source1, source2)
                    mixture, source1, source2 = batch
                    mixture, source1, source2 = (
                        mixture.to(device),
                        source1.to(device),
                        source2.to(device),
                    )

                    est_source1, est_source2 = model(mixture)

                    # The loss needs to handle permutation invariance (PIT)
                    loss1 = si_sdr_loss(est_source1, source1) + si_sdr_loss(est_source2, source2)
                    loss2 = si_sdr_loss(est_source1, source2) + si_sdr_loss(est_source2, source1)

                    loss, _ = torch.min(torch.stack([loss1, loss2]), dim=0)
                    loss = loss / accumulation_steps
                    loss.backward()

                    # Update weights on accumulation step or last step of the epoch
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == num_steps_to_run:
                        optimizer.step()
                        optimizer.zero_grad()

                    train_loss += loss.item() * accumulation_steps
                    pbar.set_postfix({"loss": loss.item() * accumulation_steps})

            if num_steps_to_run > 0:
                avg_train_loss = train_loss / num_steps_to_run
            else:
                avg_train_loss = 0.0
            logger.info(f"Epoch {epoch + 1} - Training loss: {avg_train_loss:.4f}")

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]") as pbar:
                    for batch in pbar:
                        mixture, source1, source2 = batch
                        mixture, source1, source2 = (
                            mixture.to(device),
                            source1.to(device),
                            source2.to(device),
                        )

                        est_source1, est_source2 = model(mixture)

                        loss1 = si_sdr_loss(est_source1, source1) + si_sdr_loss(est_source2, source2)
                        loss2 = si_sdr_loss(est_source1, source2) + si_sdr_loss(est_source2, source1)
                        loss, _ = torch.min(torch.stack([loss1, loss2]), dim=0)

                        val_loss += loss.item()
                        pbar.set_postfix({"loss": loss.item()})

            if len(val_loader) > 0:
                avg_val_loss = val_loss / len(val_loader)
            else:
                avg_val_loss = 0.0
            logger.info(f"Epoch {epoch + 1} - Validation loss: {avg_val_loss:.4f}")

            # Save checkpoint and history
            self.checkpoint_saver.save_checkpoint(model, optimizer, epoch, avg_val_loss, config)
            self.history_saver.save_epoch(epoch, avg_train_loss, avg_val_loss)

        logger.info("Training finished.")
