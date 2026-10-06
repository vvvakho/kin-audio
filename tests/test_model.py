import torch

from kin_audio.model import HarmonicRenderer, RendererConfig, multi_resolution_stft_loss


def test_renderer_preserves_shape_and_gradients() -> None:
    config = RendererConfig(sample_rate=16_000, hop_size=160, harmonics=12, hidden_size=32)
    model = HarmonicRenderer(config)
    frames = 20
    f0 = torch.full((2, frames), 220.0)
    loudness = torch.full((2, frames), 0.2)
    voiced = torch.ones((2, frames))

    output = model(f0, loudness, voiced, sample_count=3_200)
    assert output.shape == (2, 3_200)
    assert torch.isfinite(output).all()
    assert output.abs().max() <= 1.0

    loss = output.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_silence_controls_produce_silence() -> None:
    model = HarmonicRenderer(RendererConfig(harmonics=8, hidden_size=16))
    frames = 10
    output = model(
        torch.zeros((1, frames)),
        torch.zeros((1, frames)),
        torch.zeros((1, frames)),
        sample_count=1_600,
    )
    assert torch.count_nonzero(output) == 0


def test_spectral_loss_is_zero_for_identical_audio() -> None:
    signal = torch.randn(2, 2_048) * 0.1
    loss = multi_resolution_stft_loss(signal, signal)
    assert float(loss) == 0.0
