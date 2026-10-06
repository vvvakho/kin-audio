from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .polyphonic import _ConvBlock, _pad_to_multiple


@dataclass(frozen=True)
class ReferenceConfig:
    sample_rate: int = 16_000
    n_fft: int = 512
    hop_size: int = 128
    base_channels: int = 16
    style_dim: int = 64


class ReferenceConditionedUNet(nn.Module):
    """Polyphonic re-instrumentation conditioned by target-body reference audio."""

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.base_channels
        self.style_encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, channels * 2), channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(8, channels * 4), channels * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 4, config.style_dim),
            nn.SiLU(),
        )
        self.style_affine = nn.Linear(config.style_dim, channels * 8)

        self.encoder_one = _ConvBlock(3, channels)
        self.down_one = nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1)
        self.encoder_two = _ConvBlock(channels * 2, channels * 2)
        self.down_two = nn.Conv2d(channels * 2, channels * 4, kernel_size=4, stride=2, padding=1)
        self.bottleneck = _ConvBlock(channels * 4, channels * 4)
        self.up_two = nn.ConvTranspose2d(
            channels * 4,
            channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.decoder_two = _ConvBlock(channels * 4, channels * 2)
        self.up_one = nn.ConvTranspose2d(
            channels * 2,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.decoder_one = _ConvBlock(channels * 2, channels)
        self.output_projection = nn.Conv2d(channels, 2, kernel_size=1)

    def forward(
        self,
        source: Tensor,
        reference: Tensor,
        transformation_strength: Tensor,
    ) -> Tensor:
        if source.ndim != 2 or reference.ndim != 2:
            raise ValueError("source and reference must have shape [batch, samples]")
        if source.shape[0] != reference.shape[0]:
            raise ValueError("source and reference batch sizes must match")
        strength = transformation_strength.to(device=source.device, dtype=source.dtype)
        if strength.ndim == 0:
            strength = strength.expand(source.shape[0])
        if strength.shape != (source.shape[0],):
            raise ValueError("transformation_strength must be scalar or have shape [batch]")
        strength = strength.clamp(0.0, 1.0)

        window = torch.hann_window(
            self.config.n_fft,
            device=source.device,
            dtype=source.dtype,
        )
        source_spectrum = torch.stft(
            source,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_size,
            window=window,
            return_complex=True,
        )
        components = torch.stack((source_spectrum.real, source_spectrum.imag), dim=1)
        scale = components.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
        normalized = components / scale
        frequency_position = torch.linspace(
            -1.0,
            1.0,
            normalized.shape[-2],
            device=source.device,
            dtype=source.dtype,
        ).view(1, 1, -1, 1)
        normalized = torch.cat(
            (
                normalized,
                frequency_position.expand(source.shape[0], -1, -1, normalized.shape[-1]),
            ),
            dim=1,
        )
        padded, original_shape = _pad_to_multiple(normalized, multiple=4)

        reference_spectrum = torch.stft(
            reference,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_size,
            window=window,
            return_complex=True,
        ).abs()
        envelope_gain = _spectral_envelope_gain(
            source_spectrum,
            reference_spectrum,
            strength,
        )
        conditioned_components = components * envelope_gain[:, None, :, None]
        reference_features = torch.log1p(reference_spectrum).unsqueeze(1)
        feature_mean = reference_features.mean(dim=(2, 3), keepdim=True)
        feature_scale = reference_features.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        style = self.style_encoder((reference_features - feature_mean) / feature_scale)

        first = self.encoder_one(padded)
        second = self.encoder_two(self.down_one(first))
        latent = self.bottleneck(self.down_two(second))
        style_scale, style_shift = self.style_affine(style).chunk(2, dim=1)
        latent = latent * (1.0 + 0.1 * torch.tanh(style_scale[:, :, None, None]))
        latent = latent + style_shift[:, :, None, None]
        decoded_two = self.up_two(latent)
        decoded_two = self.decoder_two(torch.cat((decoded_two, second), dim=1))
        decoded_one = self.up_one(decoded_two)
        decoded_one = self.decoder_one(torch.cat((decoded_one, first), dim=1))
        residual = self.output_projection(decoded_one)
        residual = residual[..., : original_shape[0], : original_shape[1]]
        residual = residual * strength[:, None, None, None]

        predicted_components = conditioned_components + residual * scale
        predicted_spectrum = torch.complex(
            predicted_components[:, 0],
            predicted_components[:, 1]
        )
        transformed = torch.istft(
            predicted_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_size,
            window=window,
            length=source.shape[1],
        ).clamp(-1.0, 1.0)
        return torch.where(strength[:, None] == 0.0, source, transformed)

    def checkpoint(self) -> dict[str, object]:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, object]) -> "ReferenceConditionedUNet":
        model = cls(ReferenceConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model


def _spectral_envelope_gain(
    source_spectrum: Tensor,
    reference_magnitude: Tensor,
    strength: Tensor,
) -> Tensor:
    source_envelope = torch.log(source_spectrum.abs().mean(dim=-1) + 1e-4)
    reference_envelope = torch.log(reference_magnitude.mean(dim=-1) + 1e-4)
    log_gain = reference_envelope - source_envelope
    log_gain = F.avg_pool1d(
        log_gain.unsqueeze(1),
        kernel_size=31,
        stride=1,
        padding=15,
    ).squeeze(1)
    log_gain = log_gain - log_gain.mean(dim=1, keepdim=True)
    log_gain = log_gain.clamp(-1.4, 1.4)
    return torch.exp(log_gain * strength[:, None])
