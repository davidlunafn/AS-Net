import torch
import torch.nn as nn
from typing import Any, List

from as_net.app.ports.model_builder import IModelBuilder
from as_net.domain.models.as_net import ASNetConfig, TCNBlockConfig


class Encoder(nn.Module):
    """Encoder for the AS-Net model. It's a 1D convolutional layer."""

    def __init__(self, kernel_size: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1d = nn.Conv1d(
            1, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected a 2D tensor, but got {x.dim()}D")
        # Add channel dimension
        x = x.unsqueeze(1)
        encoded = self.conv1d(x)
        return self.relu(encoded)


class Decoder(nn.Module):
    """Decoder for the AS-Net model. It's a 1D transposed convolutional layer."""

    def __init__(self, in_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, 1, kernel_size=kernel_size, stride=stride, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D tensor, but got {x.dim()}D")
        decoded = self.conv_transpose1d(x)
        # remove channel dimension
        return decoded.squeeze(1)


class TCNBlock(nn.Module):
    """A single block of a Temporal Convolutional Network."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, stride: int, dropout_rate: float
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # Use LayerNorm instead of BatchNorm for better generalization with small batch sizes
        self.norm1 = nn.GroupNorm(1, out_channels)  # GroupNorm with 1 group = LayerNorm
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dconv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=out_channels,
            padding=(dilation * (kernel_size - 1)) // 2,
        )
        self.norm2 = nn.GroupNorm(1, out_channels)  # GroupNorm with 1 group = LayerNorm
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(in_channels, in_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.prelu1(out)
        out = self.dropout1(out)
        out = self.dconv(out)
        out = self.norm2(out)
        out = self.prelu2(out)
        out = self.dropout2(out)
        out = self.conv2(out)
        if self.residual_conv:
            residual = self.residual_conv(residual)
        return residual + out


class SeparationModule(nn.Module):
    """The separation module of AS-Net, composed of a stack of TCN blocks."""

    def __init__(self, in_channels: int, num_blocks: int, tcn_kernel_size: int, dropout_rate: float):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(
                TCNBlock(
                    in_channels,
                    in_channels,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    stride=1,
                    dropout_rate=dropout_rate,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ASNet(nn.Module):
    """Avian Separation Network (AS-Net) model."""

    def __init__(self, config: ASNetConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.encoder.kernel_size, config.encoder.out_channels, config.encoder.stride)

        # TODO: Make TCN kernel size configurable
        self.separation = SeparationModule(
            config.encoder.out_channels,
            config.separation.num_blocks,
            tcn_kernel_size=3,
            dropout_rate=config.separation.dropout_rate,
        )

        self.mask_estimation = nn.Conv1d(
            config.encoder.out_channels, config.encoder.out_channels * 2, kernel_size=1
        )  # 2 sources

        self.decoder = Decoder(config.decoder.in_channels, config.decoder.kernel_size, config.decoder.stride)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        input_length = x.shape[-1]
        encoded_features = self.encoder(x)
        separated_features = self.separation(encoded_features)

        masks = self.mask_estimation(separated_features)
        masks = nn.functional.sigmoid(masks)

        num_channels = self.config.decoder.in_channels
        mask_source1 = masks[:, :num_channels, :]
        mask_source2 = masks[:, num_channels:, :]

        features_source1 = encoded_features * mask_source1
        features_source2 = encoded_features * mask_source2

        est_source1 = self.decoder(features_source1)
        est_source2 = self.decoder(features_source2)

        # Pad to match input length
        output_length = est_source1.shape[-1]
        padding = input_length - output_length
        if padding > 0:
            est_source1 = nn.functional.pad(est_source1, (0, padding))
            est_source2 = nn.functional.pad(est_source2, (0, padding))
        else:
            est_source1 = est_source1[..., :input_length]
            est_source2 = est_source2[..., :input_length]

        return est_source1, est_source2


class ASNetTorchBuilder(IModelBuilder):
    """Builds an AS-Net torch model from a configuration."""

    def build(self, config: ASNetConfig) -> ASNet:
        return ASNet(config)
