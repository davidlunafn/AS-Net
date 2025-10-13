import numpy as np

from as_net.domain.models.audio import AudioSignal
from as_net.domain.models.bioacoustic import BioacousticSource, MixedSignal


class MixingService:
    """Service for mixing audio signals."""

    def mix(
        self, bioacoustic_source: BioacousticSource, noise: AudioSignal, snr: float
    ) -> MixedSignal:
        """Mixes a bioacoustic source with noise at a given SNR."""

        # Calculate the power of the bioacoustic source and the noise
        bioacoustic_power = np.mean(bioacoustic_source.audio.samples ** 2)
        noise_power = np.mean(noise.samples ** 2)

        # Calculate the scaling factor for the noise
        scaling_factor = np.sqrt(bioacoustic_power / (10 ** (snr / 10) * noise_power))

        # Scale the noise
        scaled_noise_samples = noise.samples * scaling_factor

        # Mix the signals
        mixed_samples = bioacoustic_source.audio.samples + scaled_noise_samples

        # Create the mixed signal
        mixed_signal = MixedSignal(
            audio=AudioSignal(samples=mixed_samples, sample_rate=bioacoustic_source.audio.sample_rate),
            bioacoustic_source=bioacoustic_source,
            noise=AudioSignal(samples=scaled_noise_samples, sample_rate=noise.sample_rate),
            snr=snr,
        )

        return mixed_signal
