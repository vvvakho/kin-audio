import json

import torch

from kin_audio.data import PairedAudioDataset, generate_polyphonic_smoke_dataset
from kin_audio.polyphonic import PolyphonicConfig, PolyphonicSpectralUNet


def test_polyphonic_renderer_accepts_chords_and_gradients() -> None:
    sample_rate = 16_000
    time = torch.arange(4_096) / sample_rate
    chord = sum(
        torch.sin(2.0 * torch.pi * frequency * time)
        for frequency in (220.0, 277.18, 329.63)
    ) / 6.0
    source = torch.stack((chord, chord * 0.7))
    model = PolyphonicSpectralUNet(
        PolyphonicConfig(sample_rate=sample_rate, base_channels=8)
    )

    output = model(source)

    assert output.shape == source.shape
    assert torch.isfinite(output).all()
    assert torch.allclose(output, source, atol=2e-5)
    output.square().mean().backward()
    assert model.output_projection.weight.grad is not None


def test_polyphonic_smoke_corpus_contains_mono_and_chords(tmp_path) -> None:
    manifest = generate_polyphonic_smoke_dataset(tmp_path, examples=8)
    metadata = json.loads((tmp_path / "dataset.json").read_text())
    dataset = PairedAudioDataset(manifest)

    assert metadata["monophonic_examples"] == 2
    assert metadata["polyphonic_examples"] == 6
    assert len(dataset) == 8
    assert dataset[0]["source"].shape == dataset[0]["target"].shape
