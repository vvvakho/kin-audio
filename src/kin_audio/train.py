import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from .audio import save_audio
from .data import PerformanceDataset
from .model import HarmonicRenderer, RendererConfig, multi_resolution_stft_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 8
    batch_size: int = 4
    learning_rate: float = 3e-4
    validation_fraction: float = 0.2
    seed: int = 20261006
    device: str = "auto"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    renderer_config: RendererConfig | None = None,
    train_config: TrainConfig | None = None,
) -> dict[str, object]:
    renderer_config = renderer_config or RendererConfig()
    train_config = train_config or TrainConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    dataset = PerformanceDataset(manifest)
    validation_size = max(1, round(len(dataset) * train_config.validation_fraction))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError("dataset needs at least two examples")
    generator = torch.Generator().manual_seed(train_config.seed)
    training_set, validation_set = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )
    training_loader = DataLoader(
        training_set,
        batch_size=train_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    device = resolve_device(train_config.device)
    model = HarmonicRenderer(renderer_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    best_validation = float("inf")
    metrics_path = output / "metrics.jsonl"
    started = time.time()

    configuration = {
        "renderer": asdict(renderer_config),
        "training": asdict(train_config),
        "manifest": str(Path(manifest).resolve()),
        "training_examples": training_size,
        "validation_examples": validation_size,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")

    for epoch in range(1, train_config.epochs + 1):
        training_loss = _run_epoch(model, training_loader, device, optimizer)
        validation_loss = _run_epoch(model, validation_loader, device, optimizer=None)
        metric = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "elapsed_seconds": time.time() - started,
        }
        with metrics_path.open("a") as metrics_file:
            metrics_file.write(json.dumps(metric, sort_keys=True) + "\n")
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(model.checkpoint(), output / "checkpoint.pt")

    checkpoint = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
    best_model = HarmonicRenderer.from_checkpoint(checkpoint).to(device).eval()
    comparison = _render_validation_example(best_model, validation_set, device, output)
    summary: dict[str, object] = {
        "status": "completed",
        "best_validation_loss": best_validation,
        "epochs": train_config.epochs,
        "duration_seconds": time.time() - started,
        "device": str(device),
        "parameter_count": configuration["parameter_count"],
        "checkpoint": "checkpoint.pt",
        "comparison": comparison,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _run_epoch(
    model: HarmonicRenderer,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    examples = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            f0 = batch["f0_hz"].to(device)
            loudness = batch["loudness"].to(device)
            voiced = batch["voiced"].to(device)
            target = batch["target"].to(device)
            prediction = model(f0, loudness, voiced, target.shape[1])
            spectral = multi_resolution_stft_loss(prediction, target)
            waveform = nn.functional.l1_loss(prediction, target)
            loss = spectral + 0.5 * waveform
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = target.shape[0]
            total_loss += float(loss.detach()) * batch_size
            examples += batch_size
    return total_loss / examples


def _render_validation_example(
    model: HarmonicRenderer,
    validation_set: Subset,
    device: torch.device,
    output: Path,
) -> dict[str, str]:
    item = validation_set[0]
    with torch.no_grad():
        prediction = model(
            item["f0_hz"].unsqueeze(0).to(device),
            item["loudness"].unsqueeze(0).to(device),
            item["voiced"].unsqueeze(0).to(device),
            item["target"].numel(),
        )[0]
    prediction_path = output / "validation_prediction.wav"
    target_path = output / "validation_target.wav"
    save_audio(prediction_path, prediction.cpu().numpy(), model.config.sample_rate)
    save_audio(target_path, item["target"].numpy(), model.config.sample_rate)
    comparison = {
        "name": "Held-out synthetic target reconstruction",
        "prediction": prediction_path.name,
        "target": target_path.name,
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison
