import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import save_audio


class PerformanceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, manifest: str | Path) -> None:
        manifest_path = Path(manifest)
        root = manifest_path.parent
        self.records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
        if not self.records:
            raise ValueError(f"manifest has no records: {manifest}")
        self.root = root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        controls = np.load(self.root / record["controls"])
        import soundfile as sf

        target, sample_rate = sf.read(self.root / record["target"], dtype="float32")
        if sample_rate != record["sample_rate"]:
            raise ValueError(f"sample-rate mismatch in {record['target']}")
        return {
            "f0_hz": torch.from_numpy(controls["f0_hz"].astype(np.float32)),
            "loudness": torch.from_numpy(controls["loudness"].astype(np.float32)),
            "voiced": torch.from_numpy(controls["voiced"].astype(np.float32)),
            "target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
        }


def generate_smoke_dataset(
    output_dir: str | Path,
    *,
    examples: int = 48,
    sample_rate: int = 16_000,
    hop_size: int = 160,
    duration_seconds: float = 1.6,
    seed: int = 20261006,
) -> Path:
    """Generate a deterministic, license-clean paired corpus for pipeline validation.

    The source is a plucked harmonic body and the target is a sustained two-formant body.
    Both are rendered from identical note, vibrato, timing, and loudness controls.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frames = round(duration_seconds * sample_rate / hop_size)
    sample_count = round(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for index in range(examples):
        f0, loudness, voiced = _performance_controls(rng, frames, sample_rate, hop_size)
        source = _render_body(
            f0,
            loudness,
            voiced,
            sample_count,
            sample_rate,
            body="pluck",
        )
        target = _render_body(
            f0,
            loudness,
            voiced,
            sample_count,
            sample_rate,
            body="glass_choir",
        )
        item_dir = root / f"{index:05d}"
        item_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            item_dir / "controls.npz",
            f0_hz=f0,
            loudness=loudness,
            voiced=voiced,
        )
        save_audio(item_dir / "source.wav", source, sample_rate)
        save_audio(item_dir / "target.wav", target, sample_rate)
        records.append(
            {
                "id": f"smoke-{index:05d}",
                "controls": f"{index:05d}/controls.npz",
                "source": f"{index:05d}/source.wav",
                "target": f"{index:05d}/target.wav",
                "sample_rate": sample_rate,
                "license": "CC0-1.0",
                "generator": "kin-audio/synthetic-v1",
            }
        )

    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    metadata = {
        "name": "kin-smoke",
        "purpose": "pipeline validation only",
        "examples": examples,
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "duration_seconds": duration_seconds,
        "seed": seed,
        "license": "CC0-1.0",
    }
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest


def _performance_controls(
    rng: np.random.Generator,
    frames: int,
    sample_rate: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0 = np.zeros(frames, dtype=np.float32)
    loudness = np.zeros(frames, dtype=np.float32)
    voiced = np.zeros(frames, dtype=np.float32)
    cursor = 0
    while cursor < frames:
        rest = int(rng.integers(2, 9))
        cursor += rest
        if cursor >= frames:
            break
        note_frames = min(int(rng.integers(18, 43)), frames - cursor)
        midi = int(rng.integers(48, 77))
        base_frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        time = np.arange(note_frames, dtype=np.float32) * hop_size / sample_rate
        vibrato_rate = float(rng.uniform(4.2, 6.3))
        vibrato_depth = float(rng.uniform(0.0, 0.22))
        pitch_curve = base_frequency * 2.0 ** (
            vibrato_depth * np.sin(2.0 * np.pi * vibrato_rate * time) / 12.0
        )
        attack = np.minimum(1.0, np.arange(note_frames) / max(2.0, note_frames * 0.12))
        release = np.minimum(1.0, np.arange(note_frames)[::-1] / max(2.0, note_frames * 0.18))
        envelope = np.minimum(attack, release)
        level = float(rng.uniform(0.14, 0.32))
        section = slice(cursor, cursor + note_frames)
        f0[section] = pitch_curve
        loudness[section] = level * envelope
        voiced[section] = 1.0
        cursor += note_frames
    return f0, loudness, voiced


def _render_body(
    f0: np.ndarray,
    loudness: np.ndarray,
    voiced: np.ndarray,
    sample_count: int,
    sample_rate: int,
    *,
    body: str,
    harmonics: int = 48,
) -> np.ndarray:
    positions = np.linspace(0, len(f0) - 1, sample_count)
    frame_positions = np.arange(len(f0))
    sample_f0 = np.interp(positions, frame_positions, f0)
    sample_loudness = np.interp(positions, frame_positions, loudness)
    sample_voiced = np.interp(positions, frame_positions, voiced)
    phase = np.cumsum(2.0 * np.pi * sample_f0 / sample_rate)
    numbers = np.arange(1, harmonics + 1, dtype=np.float32)
    frequencies = sample_f0[:, None] * numbers[None, :]

    if body == "pluck":
        weights = 1.0 / numbers[None, :] ** 1.15
        weights = np.broadcast_to(weights, frequencies.shape).copy()
    elif body == "glass_choir":
        formant_a = np.exp(-0.5 * ((frequencies - 720.0) / 230.0) ** 2)
        formant_b = 0.7 * np.exp(-0.5 * ((frequencies - 1450.0) / 420.0) ** 2)
        shimmer = 0.13 / np.sqrt(numbers[None, :])
        weights = formant_a + formant_b + shimmer
    else:
        raise ValueError(f"unknown body: {body}")

    weights *= frequencies < sample_rate / 2.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    partials = np.sin(phase[:, None] * numbers[None, :]) * weights
    audio = partials.sum(axis=1) * sample_loudness * sample_voiced
    return np.tanh(audio * 2.0).astype(np.float32)
