from dataclasses import dataclass, field
from typing import List


@dataclass
class EncoderConfig:
    """Configuration for the Encoder."""

    kernel_size: int
    stride: int
    out_channels: int


@dataclass
class DecoderConfig:
    """Configuration for the Decoder."""

    kernel_size: int
    stride: int
    in_channels: int


@dataclass
class TCNBlockConfig:
    """Configuration for a single TCN block."""

    in_channels: int
    out_channels: int
    kernel_size: int
    dilation: int
    stride: int = 1


@dataclass
class SeparationModuleConfig:
    """Configuration for the Separation Module."""

    num_blocks: int
    tcn_blocks: List[TCNBlockConfig] = field(default_factory=list)


@dataclass
class ASNetConfig:
    """Configuration for the AS-Net model."""

    encoder: EncoderConfig
    decoder: DecoderConfig
    separation: SeparationModuleConfig
