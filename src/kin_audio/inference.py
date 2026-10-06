import json
from pathlib import Path

import numpy as np
import torch

from .audio import load_audio, save_audio
from .features import extract_performance
from .model import HarmonicRenderer
from .polyphonic import PolyphonicSpectralUNet
from .train import resolve_device


def render_file(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "auto",
) -> Path:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = HarmonicRenderer.from_checkpoint(checkpoint).to(device).eval()
    audio = load_audio(input_path, model.config.sample_rate)
    performance = extract_performance(
        audio,
        model.config.sample_rate,
        hop_size=model.config.hop_size,
    )
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(performance.f0_hz).unsqueeze(0).to(device),
            torch.from_numpy(performance.loudness).unsqueeze(0).to(device),
            torch.from_numpy(performance.voiced).unsqueeze(0).to(device),
            len(audio),
        )[0]
    output = Path(output_path)
    save_audio(output, prediction.cpu().numpy(), model.config.sample_rate)
    return output


def render_polyphonic_file(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    device_name: str = "auto",
) -> Path:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = PolyphonicSpectralUNet.from_checkpoint(checkpoint).to(device).eval()
    audio = load_audio(input_path, model.config.sample_rate)
    with torch.no_grad():
        prediction = model(torch.from_numpy(audio).unsqueeze(0).to(device))[0]
    output = Path(output_path)
    save_audio(output, prediction.cpu().numpy(), model.config.sample_rate)
    return output


def evaluate_preservation(
    source_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    sample_rate: int = 16_000,
    hop_size: int = 160,
) -> dict[str, float | int | None]:
    source = load_audio(source_path, sample_rate)
    rendered = load_audio(output_path, sample_rate)
    source_performance = extract_performance(source, sample_rate, hop_size=hop_size)
    output_performance = extract_performance(rendered, sample_rate, hop_size=hop_size)
    frames = min(len(source_performance.f0_hz), len(output_performance.f0_hz))
    source_voiced = source_performance.voiced[:frames] > 0.5
    output_voiced = output_performance.voiced[:frames] > 0.5
    jointly_voiced = source_voiced & output_voiced
    union_voiced = source_voiced | output_voiced

    if jointly_voiced.any():
        cents = 1200.0 * np.log2(
            output_performance.f0_hz[:frames][jointly_voiced]
            / source_performance.f0_hz[:frames][jointly_voiced]
        )
        median_absolute_cents = float(np.median(np.abs(cents)))
        pitch_within_50_cents = float(np.mean(np.abs(cents) <= 50.0))
    else:
        median_absolute_cents = None
        pitch_within_50_cents = 0.0

    report: dict[str, float | int | None] = {
        "frames": frames,
        "jointly_voiced_frames": int(jointly_voiced.sum()),
        "median_absolute_pitch_error_cents": median_absolute_cents,
        "pitch_frames_within_50_cents": pitch_within_50_cents,
        "voicing_iou": float(jointly_voiced.sum() / max(1, union_voiced.sum())),
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
