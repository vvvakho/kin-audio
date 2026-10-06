import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .audio import save_audio
from .data import ReferenceAudioDataset
from .model import multi_resolution_stft_loss
from .reference import ReferenceConditionedUNet, ReferenceConfig
from .train import TrainConfig, resolve_device


def train_reference_model(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    model_config: ReferenceConfig | None = None,
    train_config: TrainConfig | None = None,
) -> dict[str, object]:
    model_config = model_config or ReferenceConfig()
    train_config = train_config or TrainConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    dataset = ReferenceAudioDataset(manifest)
    training_set, validation_set = _split_by_group(
        dataset,
        validation_fraction=train_config.validation_fraction,
        seed=train_config.seed,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
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
    model = ReferenceConditionedUNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    best_validation = float("inf")
    metrics_path = output / "metrics.jsonl"
    started = time.time()
    configuration = {
        "model_type": "reference-conditioned-spectral-unet",
        "model": asdict(model_config),
        "training": asdict(train_config),
        "manifest": str(Path(manifest).resolve()),
        "training_examples": len(training_set),
        "validation_examples": len(validation_set),
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
    best_model = ReferenceConditionedUNet.from_checkpoint(checkpoint).to(device).eval()
    comparison = _render_validation_example(best_model, validation_set, device, output)
    presets = _export_reference_presets(dataset, output)
    summary: dict[str, object] = {
        "status": "completed",
        "model_type": "reference-conditioned-spectral-unet",
        "best_validation_loss": best_validation,
        "epochs": train_config.epochs,
        "duration_seconds": time.time() - started,
        "device": str(device),
        "parameter_count": configuration["parameter_count"],
        "checkpoint": "checkpoint.pt",
        "comparison": comparison,
        "reference_presets": presets,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _split_by_group(
    dataset: ReferenceAudioDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    groups = sorted({str(record["split_group"]) for record in dataset.records})
    random.Random(seed).shuffle(groups)
    validation_group_count = max(1, round(len(groups) * validation_fraction))
    if validation_group_count >= len(groups):
        raise ValueError("reference dataset needs at least two split groups")
    validation_groups = set(groups[:validation_group_count])
    training_indices = [
        index
        for index, record in enumerate(dataset.records)
        if str(record["split_group"]) not in validation_groups
    ]
    validation_indices = [
        index
        for index, record in enumerate(dataset.records)
        if str(record["split_group"]) in validation_groups
    ]
    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def _run_epoch(
    model: ReferenceConditionedUNet,
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
            source = batch["source"].to(device)
            reference = batch["reference"].to(device)
            target = batch["target"].to(device)
            strength = torch.ones(source.shape[0], device=device)
            prediction = model(source, reference, strength)
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
    model: ReferenceConditionedUNet,
    validation_set: Subset,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    example_index = next(
        (
            index
            for index in range(len(validation_set))
            if int(validation_set[index]["voices"]) > 1
        ),
        0,
    )
    item = validation_set[example_index]
    record_index = validation_set.indices[example_index]
    record = validation_set.dataset.records[record_index]
    strength = torch.ones(1, device=device)
    with torch.no_grad():
        prediction = model(
            item["source"].unsqueeze(0).to(device),
            item["reference"].unsqueeze(0).to(device),
            strength,
        )[0]
    paths = {
        "source": output / "validation_source.wav",
        "reference": output / "validation_reference.wav",
        "prediction": output / "validation_prediction.wav",
        "target": output / "validation_target.wav",
    }
    save_audio(paths["source"], item["source"].numpy(), model.config.sample_rate)
    save_audio(paths["reference"], item["reference"].numpy(), model.config.sample_rate)
    save_audio(paths["prediction"], prediction.cpu().numpy(), model.config.sample_rate)
    save_audio(paths["target"], item["target"].numpy(), model.config.sample_rate)
    voices = int(item["voices"])
    comparison: dict[str, object] = {
        "name": "Held-out reference-conditioned polyphonic transfer",
        "content_type": "polyphonic" if voices > 1 else "monophonic",
        "voices": voices,
        "target_body": str(record["target_body"]),
        **{name: path.name for name, path in paths.items()},
    }
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return comparison


def _export_reference_presets(
    dataset: ReferenceAudioDataset,
    output: Path,
) -> dict[str, str]:
    presets: dict[str, str] = {}
    for record in dataset.records:
        body = str(record["target_body"])
        if body in presets:
            continue
        destination = output / f"reference_{body}.wav"
        shutil.copyfile(dataset.root / str(record["reference"]), destination)
        presets[body] = destination.name
    return presets
