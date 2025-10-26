from dataclasses import dataclass
import numpy as np

@dataclass
class AudioSignal:
    """Represents an audio signal."""
    samples: np.ndarray
    sample_rate: int
