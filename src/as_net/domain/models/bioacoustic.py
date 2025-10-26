from dataclasses import dataclass, field
from typing import List

from as_net.domain.models.audio import AudioSignal


@dataclass
class Call:
    """Represents a single bird call in an audio signal."""
    start_time: float
    end_time: float
    lower_freq: float
    upper_freq: float
    json_file: str


@dataclass
class BioacousticSource:
    """Represents a bioacoustic source, such as a bird vocalization."""
    audio: AudioSignal
    calls: List[Call] = field(default_factory=list)


@dataclass
class MixedSignal:
    """Represents a mixed audio signal, containing a bioacoustic source and noise."""
    audio: AudioSignal
    bioacoustic_source: BioacousticSource
    noise: AudioSignal
    snr: float
