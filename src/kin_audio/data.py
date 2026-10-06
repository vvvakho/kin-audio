import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .audio import save_audio

REFERENCE_TARGET_BODIES = ("glass_choir", "warm_strings", "bright_brass")


def _load_records(manifest: str | Path) -> tuple[Path, list[dict[str, object]]]:
    manifest_path = Path(manifest)
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if not records:
        raise ValueError(f"manifest has no records: {manifest}")
    return manifest_path.parent, records


def _read_mono(path: Path, expected_sample_rate: int) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != expected_sample_rate:
        raise ValueError(f"sample-rate mismatch in {path}")
    if audio.ndim != 1:
        raise ValueError(f"expected mono audio in {path}")
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


class PerformanceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, manifest: str | Path) -> None:
        self.root, self.records = _load_records(manifest)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        controls = np.load(self.root / str(record["controls"]))
        sample_rate = int(record["sample_rate"])
        return {
            "f0_hz": torch.from_numpy(controls["f0_hz"].astype(np.float32)),
            "loudness": torch.from_numpy(controls["loudness"].astype(np.float32)),
            "voiced": torch.from_numpy(controls["voiced"].astype(np.float32)),
            "source": _read_mono(self.root / str(record["source"]), sample_rate),
            "target": _read_mono(self.root / str(record["target"]), sample_rate),
        }


class PairedAudioDataset(Dataset[dict[str, torch.Tensor]]):
    """Aligned source/target audio pairs for mono and polyphonic translation."""

    def __init__(self, manifest: str | Path) -> None:
        self.root, self.records = _load_records(manifest)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        sample_rate = int(record["sample_rate"])
        return {
            "source": _read_mono(self.root / str(record["source"]), sample_rate),
            "target": _read_mono(self.root / str(record["target"]), sample_rate),
            "voices": torch.tensor(int(record.get("voices", 1))),
        }


class ReferenceAudioDataset(Dataset[dict[str, torch.Tensor]]):
    """Performance, body-reference, and target triples for conditioned transfer."""

    def __init__(self, manifest: str | Path) -> None:
        self.root, self.records = _load_records(manifest)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        sample_rate = int(record["sample_rate"])
        return {
            "source": _read_mono(self.root / str(record["source"]), sample_rate),
            "reference": _read_mono(self.root / str(record["reference"]), sample_rate),
            "target": _read_mono(self.root / str(record["target"]), sample_rate),
            "voices": torch.tensor(int(record.get("voices", 1))),
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


def generate_polyphonic_smoke_dataset(
    output_dir: str | Path,
    *,
    examples: int = 48,
    sample_rate: int = 16_000,
    hop_size: int = 160,
    duration_seconds: float = 1.6,
    seed: int = 20261006,
) -> Path:
    """Generate aligned mono and polyphonic pairs for audio-to-audio validation."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frames = round(duration_seconds * sample_rate / hop_size)
    sample_count = round(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for index in range(examples):
        voice_count = 1 if index % 4 == 0 else int(rng.integers(2, 5))
        f0, loudness, voiced = _chord_controls(
            rng,
            frames,
            sample_rate,
            hop_size,
            voice_count,
        )
        source = _render_polyphonic_body(
            f0,
            loudness,
            voiced,
            sample_count,
            sample_rate,
            body="pluck",
        )
        target = _render_polyphonic_body(
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
                "id": f"poly-smoke-{index:05d}",
                "controls": f"{index:05d}/controls.npz",
                "source": f"{index:05d}/source.wav",
                "target": f"{index:05d}/target.wav",
                "sample_rate": sample_rate,
                "license": "CC0-1.0",
                "generator": "kin-audio/polyphonic-synthetic-v1",
                "content_type": "monophonic" if voice_count == 1 else "polyphonic",
                "voices": voice_count,
            }
        )

    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    metadata = {
        "name": "kin-polyphonic-smoke",
        "purpose": "paired audio-to-audio pipeline validation only",
        "examples": examples,
        "monophonic_examples": sum(record["voices"] == 1 for record in records),
        "polyphonic_examples": sum(record["voices"] > 1 for record in records),
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "duration_seconds": duration_seconds,
        "seed": seed,
        "license": "CC0-1.0",
    }
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest


def generate_reference_smoke_dataset(
    output_dir: str | Path,
    *,
    examples: int = 48,
    sample_rate: int = 16_000,
    hop_size: int = 160,
    duration_seconds: float = 1.6,
    seed: int = 20261006,
) -> Path:
    """Generate source, independent body-reference, and target triples."""
    if examples < 2 or examples % 2:
        raise ValueError("reference corpus needs an even number of at least two performances")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frames = round(duration_seconds * sample_rate / hop_size)
    sample_count = round(duration_seconds * sample_rate)
    rng = np.random.default_rng(seed)
    performances: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    voice_counts: list[int] = []

    for index in range(examples):
        voice_count = 1 if index % 4 == 0 else int(rng.integers(2, 5))
        performances.append(
            _chord_controls(rng, frames, sample_rate, hop_size, voice_count)
        )
        voice_counts.append(voice_count)

    for index, (f0, loudness, voiced) in enumerate(performances):
        item_dir = root / f"{index:05d}"
        item_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            item_dir / "controls.npz",
            f0_hz=f0,
            loudness=loudness,
            voiced=voiced,
        )
        source = _render_polyphonic_body(
            f0,
            loudness,
            voiced,
            sample_count,
            sample_rate,
            body="pluck",
        )
        save_audio(item_dir / "source.wav", source, sample_rate)
        for body in REFERENCE_TARGET_BODIES:
            target = _render_polyphonic_body(
                f0,
                loudness,
                voiced,
                sample_count,
                sample_rate,
                body=body,
            )
            save_audio(item_dir / f"target-{body}.wav", target, sample_rate)

    records: list[dict[str, object]] = []
    for index in range(examples):
        reference_index = index + 1 if index % 2 == 0 else index - 1
        for body_index, body in enumerate(REFERENCE_TARGET_BODIES):
            records.append(
                {
                    "id": f"reference-smoke-{index:05d}-{body}",
                    "controls": f"{index:05d}/controls.npz",
                    "source": f"{index:05d}/source.wav",
                    "reference": f"{reference_index:05d}/target-{body}.wav",
                    "target": f"{index:05d}/target-{body}.wav",
                    "split_group": f"pair-{index // 2:05d}",
                    "target_body": body,
                    "target_body_id": body_index,
                    "sample_rate": sample_rate,
                    "license": "CC0-1.0",
                    "generator": "kin-audio/reference-synthetic-v1",
                    "content_type": (
                        "monophonic" if voice_counts[index] == 1 else "polyphonic"
                    ),
                    "voices": voice_counts[index],
                }
            )

    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    metadata = {
        "name": "kin-reference-smoke",
        "purpose": "reference-conditioned pipeline validation only",
        "performances": examples,
        "records": len(records),
        "target_bodies": list(REFERENCE_TARGET_BODIES),
        "monophonic_performances": sum(voices == 1 for voices in voice_counts),
        "polyphonic_performances": sum(voices > 1 for voices in voice_counts),
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "duration_seconds": duration_seconds,
        "seed": seed,
        "license": "CC0-1.0",
    }
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return manifest


def _chord_controls(
    rng: np.random.Generator,
    frames: int,
    sample_rate: int,
    hop_size: int,
    voice_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0 = np.zeros((voice_count, frames), dtype=np.float32)
    loudness = np.zeros((voice_count, frames), dtype=np.float32)
    voiced = np.zeros((voice_count, frames), dtype=np.float32)
    chord_shapes = ((0, 4, 7, 11), (0, 3, 7, 10), (0, 5, 7, 10), (0, 3, 7, 11))
    cursor = 0
    while cursor < frames:
        chord_frames = min(int(rng.integers(28, 53)), frames - cursor)
        sounding_frames = max(1, chord_frames - int(rng.integers(2, 6)))
        root = int(rng.integers(43, 61))
        intervals = chord_shapes[int(rng.integers(0, len(chord_shapes)))]
        time = np.arange(sounding_frames, dtype=np.float32) * hop_size / sample_rate
        attack = np.minimum(1.0, np.arange(sounding_frames) / max(2.0, sounding_frames * 0.1))
        release = np.minimum(
            1.0,
            np.arange(sounding_frames)[::-1] / max(2.0, sounding_frames * 0.2),
        )
        envelope = np.minimum(attack, release)
        section = slice(cursor, cursor + sounding_frames)
        for voice in range(voice_count):
            midi = root + intervals[voice]
            base_frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            vibrato = 2.0 ** (
                0.08
                * np.sin(2.0 * np.pi * (4.5 + 0.35 * voice) * time + voice)
                / 12.0
            )
            f0[voice, section] = base_frequency * vibrato
            loudness[voice, section] = float(rng.uniform(0.08, 0.18)) * envelope
            voiced[voice, section] = 1.0
        cursor += chord_frames
    return f0, loudness, voiced


def _render_polyphonic_body(
    f0: np.ndarray,
    loudness: np.ndarray,
    voiced: np.ndarray,
    sample_count: int,
    sample_rate: int,
    *,
    body: str,
) -> np.ndarray:
    voices = [
        _render_body(
            f0[index],
            loudness[index],
            voiced[index],
            sample_count,
            sample_rate,
            body=body,
        )
        for index in range(f0.shape[0])
    ]
    mixture = np.stack(voices).sum(axis=0) / np.sqrt(len(voices))
    return np.tanh(mixture).astype(np.float32)


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
    elif body == "warm_strings":
        tilt = 1.0 / numbers[None, :] ** 1.55
        body_resonance = 0.8 * np.exp(-0.5 * ((frequencies - 520.0) / 310.0) ** 2)
        weights = tilt + body_resonance
    elif body == "bright_brass":
        tilt = 1.0 / numbers[None, :] ** 0.72
        bell = 1.1 * np.exp(-0.5 * ((frequencies - 1250.0) / 650.0) ** 2)
        weights = tilt + bell
    else:
        raise ValueError(f"unknown body: {body}")

    weights *= frequencies < sample_rate / 2.0
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    partials = np.sin(phase[:, None] * numbers[None, :]) * weights
    audio = partials.sum(axis=1) * sample_loudness * sample_voiced
    return np.tanh(audio * 2.0).astype(np.float32)
