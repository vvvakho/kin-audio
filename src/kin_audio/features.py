from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Performance:
    f0_hz: np.ndarray
    loudness: np.ndarray
    voiced: np.ndarray
    hop_size: int
    sample_rate: int

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_size


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, center, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index + 0.5 * (left - right) / denominator)


def extract_performance(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_size: int = 1024,
    hop_size: int = 160,
    fmin: float = 55.0,
    fmax: float = 1200.0,
    voicing_threshold: float = 0.28,
) -> Performance:
    """Extract monophonic pitch, loudness, and voicing with autocorrelation.

    This deliberately small baseline is deterministic and dependency-light. It is suitable
    for clean monophonic recordings; production experiments can replace it while preserving
    the Performance contract.
    """
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError("audio must be mono")
    if signal.size == 0:
        raise ValueError("audio must not be empty")
    if not 0 < fmin < fmax < sample_rate / 2:
        raise ValueError("invalid pitch range")

    frame_count = max(1, 1 + int(np.ceil(max(0, signal.size - frame_size) / hop_size)))
    padded_size = (frame_count - 1) * hop_size + frame_size
    padded = np.pad(signal, (0, max(0, padded_size - signal.size)))
    window = np.hanning(frame_size).astype(np.float32)
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frame_size - 2, int(sample_rate / fmin))

    f0 = np.zeros(frame_count, dtype=np.float32)
    loudness = np.zeros(frame_count, dtype=np.float32)
    voiced = np.zeros(frame_count, dtype=np.float32)

    for frame_index in range(frame_count):
        start = frame_index * hop_size
        frame = padded[start : start + frame_size]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-10))
        loudness[frame_index] = rms
        if rms < 1e-4:
            continue
        centered = (frame - frame.mean()) * window
        correlation = np.correlate(centered, centered, mode="full")[frame_size - 1 :]
        zero_lag = float(correlation[0])
        if zero_lag <= 1e-10:
            continue
        candidates = correlation[min_lag : max_lag + 1] / zero_lag
        relative_index = int(np.argmax(candidates))
        confidence = float(candidates[relative_index])
        if confidence < voicing_threshold:
            continue
        lag_index = min_lag + relative_index
        refined_lag = _parabolic_peak(correlation, lag_index)
        f0[frame_index] = sample_rate / max(refined_lag, 1.0)
        voiced[frame_index] = 1.0

    return Performance(
        f0_hz=f0,
        loudness=loudness,
        voiced=voiced,
        hop_size=hop_size,
        sample_rate=sample_rate,
    )


def model_features(performance: Performance) -> np.ndarray:
    """Convert physical controls into stable neural-network inputs."""
    safe_f0 = np.maximum(performance.f0_hz, 1.0)
    pitch = np.log2(safe_f0 / 440.0) * performance.voiced
    loudness_db = 20.0 * np.log10(np.maximum(performance.loudness, 1e-5))
    scaled_loudness = np.clip((loudness_db + 80.0) / 80.0, 0.0, 1.0)
    return np.stack((pitch, scaled_loudness, performance.voiced), axis=-1).astype(np.float32)
