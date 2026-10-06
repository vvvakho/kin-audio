import json
from pathlib import Path

import torch

from kin_audio.audio import save_audio
from kin_audio.dashboard import _render_prompt
from kin_audio.data import ReferenceAudioDataset, generate_reference_smoke_dataset
from kin_audio.reference import ReferenceConditionedUNet, ReferenceConfig
from kin_audio.train_reference import _split_by_group


def test_reference_corpus_has_independent_multi_body_prompts(tmp_path) -> None:
    manifest = generate_reference_smoke_dataset(tmp_path, examples=4)
    metadata = json.loads((tmp_path / "dataset.json").read_text())
    dataset = ReferenceAudioDataset(manifest)

    assert metadata["records"] == 12
    assert metadata["target_bodies"] == [
        "glass_choir",
        "warm_strings",
        "bright_brass",
    ]
    assert len(dataset) == 12
    assert dataset.records[0]["source"] != dataset.records[0]["reference"]
    assert dataset[0]["source"].shape == dataset[0]["reference"].shape


def test_group_split_keeps_performance_and_references_together(tmp_path) -> None:
    manifest = generate_reference_smoke_dataset(tmp_path, examples=8)
    dataset = ReferenceAudioDataset(manifest)
    training, validation = _split_by_group(dataset, validation_fraction=0.25, seed=7)
    training_groups = {
        dataset.records[index]["split_group"] for index in training.indices
    }
    validation_groups = {
        dataset.records[index]["split_group"] for index in validation.indices
    }

    assert training_groups.isdisjoint(validation_groups)
    assert len(training) + len(validation) == len(dataset)


def test_reference_model_supports_polyphony_style_and_exact_dry_control() -> None:
    sample_rate = 16_000
    time = torch.arange(4_096) / sample_rate
    source = (
        torch.sin(2.0 * torch.pi * 220.0 * time)
        + torch.sin(2.0 * torch.pi * 277.18 * time)
        + torch.sin(2.0 * torch.pi * 329.63 * time)
    ).unsqueeze(0) / 6.0
    dark_reference = torch.sin(2.0 * torch.pi * 110.0 * time).unsqueeze(0) * 0.2
    bright_reference = torch.sin(2.0 * torch.pi * 1760.0 * time).unsqueeze(0) * 0.2
    model = ReferenceConditionedUNet(
        ReferenceConfig(sample_rate=sample_rate, base_channels=8, style_dim=16)
    )

    dry = model(source, dark_reference, torch.tensor([0.0]))
    dark = model(source, dark_reference, torch.tensor([1.0]))
    bright = model(source, bright_reference, torch.tensor([1.0]))

    assert torch.equal(dry, source)
    assert dark.shape == source.shape
    assert torch.isfinite(dark).all()
    assert not torch.allclose(dark, bright)


def test_dashboard_renders_with_reference_and_strength(tmp_path) -> None:
    run = tmp_path / "runs" / "reference"
    run.mkdir(parents=True)
    model = ReferenceConditionedUNet(
        ReferenceConfig(base_channels=8, style_dim=16)
    )
    torch.save(model.checkpoint(), run / "checkpoint.pt")
    (run / "summary.json").write_text(
        json.dumps({"model_type": "reference-conditioned-spectral-unet"})
    )
    time = torch.arange(4_096) / model.config.sample_rate
    source = torch.sin(2.0 * torch.pi * 220.0 * time) * 0.2
    reference = torch.sin(2.0 * torch.pi * 880.0 * time) * 0.2
    source_path = tmp_path / "source.wav"
    reference_path = tmp_path / "reference.wav"
    save_audio(source_path, source.numpy(), model.config.sample_rate)
    save_audio(reference_path, reference.numpy(), model.config.sample_rate)

    output_path, status = _render_prompt(
        tmp_path,
        "reference",
        str(source_path),
        str(reference_path),
        0.6,
    )

    assert output_path is not None
    assert Path(output_path).exists()
    assert "60% strength" in status
    Path(output_path).unlink()
