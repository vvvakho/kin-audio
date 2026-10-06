from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RendererConfig:
    sample_rate: int = 16_000
    hop_size: int = 160
    harmonics: int = 48
    hidden_size: int = 128
    recurrent_layers: int = 1


class HarmonicRenderer(nn.Module):
    """A compact DDSP-style monophonic target-body renderer.

    Pitch and loudness are explicit controls. The network learns a time-varying harmonic
    distribution and gain, which makes pitch preservation structural rather than incidental.
    """

    def __init__(self, config: RendererConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(3, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.SiLU(),
        )
        self.recurrent = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            num_layers=config.recurrent_layers,
            batch_first=True,
        )
        self.control_projection = nn.Linear(config.hidden_size, config.harmonics + 1)

    def forward(
        self,
        f0_hz: Tensor,
        loudness: Tensor,
        voiced: Tensor,
        sample_count: int,
    ) -> Tensor:
        if f0_hz.ndim != 2 or loudness.shape != f0_hz.shape or voiced.shape != f0_hz.shape:
            raise ValueError("f0_hz, loudness, and voiced must have shape [batch, frames]")
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")

        safe_f0 = f0_hz.clamp_min(1.0)
        pitch = torch.log2(safe_f0 / 440.0) * voiced
        loudness_db = 20.0 * torch.log10(loudness.clamp_min(1e-5))
        scaled_loudness = ((loudness_db + 80.0) / 80.0).clamp(0.0, 1.0)
        features = torch.stack((pitch, scaled_loudness, voiced), dim=-1)

        hidden = self.input_projection(features)
        hidden, _ = self.recurrent(hidden)
        controls = self.control_projection(hidden)
        harmonic_distribution = torch.softmax(controls[..., : self.config.harmonics], dim=-1)
        gain = 2.0 * torch.sigmoid(controls[..., -1]) * loudness

        frame_f0 = f0_hz * voiced
        sample_f0 = _upsample(frame_f0.unsqueeze(1), sample_count).squeeze(1)
        sample_voiced = _upsample(voiced.unsqueeze(1), sample_count).squeeze(1).clamp(0.0, 1.0)
        sample_gain = _upsample(gain.unsqueeze(1), sample_count).squeeze(1)
        sample_distribution = _upsample(
            harmonic_distribution.transpose(1, 2), sample_count
        ).transpose(1, 2)

        fundamental_phase = torch.cumsum(
            2.0 * torch.pi * sample_f0 / self.config.sample_rate,
            dim=1,
        )
        harmonic_numbers = torch.arange(
            1,
            self.config.harmonics + 1,
            device=f0_hz.device,
            dtype=f0_hz.dtype,
        )
        phases = fundamental_phase.unsqueeze(-1) * harmonic_numbers
        frequencies = sample_f0.unsqueeze(-1) * harmonic_numbers
        anti_alias = frequencies < (self.config.sample_rate / 2.0)
        partials = torch.sin(phases) * sample_distribution * anti_alias
        audio = partials.sum(dim=-1) * sample_gain * sample_voiced
        return torch.tanh(audio)

    def checkpoint(self) -> dict[str, object]:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, object]) -> "HarmonicRenderer":
        config = RendererConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model


def _upsample(controls: Tensor, sample_count: int) -> Tensor:
    return F.interpolate(controls, size=sample_count, mode="linear", align_corners=True)


def multi_resolution_stft_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Perceptual spectral loss over complementary time/frequency resolutions."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    losses: list[Tensor] = []
    for fft_size in (256, 512, 1024):
        hop = fft_size // 4
        window = torch.hann_window(fft_size, device=prediction.device, dtype=prediction.dtype)
        predicted_spectrum = torch.stft(
            prediction,
            n_fft=fft_size,
            hop_length=hop,
            window=window,
            return_complex=True,
        ).abs()
        target_spectrum = torch.stft(
            target,
            n_fft=fft_size,
            hop_length=hop,
            window=window,
            return_complex=True,
        ).abs()
        linear = (predicted_spectrum - target_spectrum).abs().mean()
        logarithmic = (
            torch.log(predicted_spectrum + 1e-5) - torch.log(target_spectrum + 1e-5)
        ).abs().mean()
        losses.append(linear + logarithmic)
    return torch.stack(losses).mean()
