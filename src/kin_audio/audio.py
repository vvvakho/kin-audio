from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Load a file as mono float32 and resample when necessary."""
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_rate == sample_rate:
        return mono
    if mono.size == 0:
        raise ValueError(f"empty audio file: {path}")
    duration = mono.size / source_rate
    target_size = max(1, round(duration * sample_rate))
    source_positions = np.arange(mono.size, dtype=np.float64) / source_rate
    target_positions = np.arange(target_size, dtype=np.float64) / sample_rate
    return np.interp(target_positions, source_positions, mono).astype(np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono float audio without normalizing away its dynamics."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(output, clipped, sample_rate, subtype="PCM_24")
