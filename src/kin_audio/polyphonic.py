from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PolyphonicConfig:
    sample_rate: int = 16_000
    n_fft: int = 512
    hop_size: int = 128
    base_channels: int = 24
    frequency_position: bool = True


class _ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        while output_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class PolyphonicSpectralUNet(nn.Module):
    """Paired audio-to-audio baseline that retains the complete polyphonic spectrum.

    The network predicts a complex STFT residual. Unlike the harmonic renderer, it does not reduce
    the input to one pitch track, so chords, overlapping notes, transients, and monophonic material
    share the same model path.
    """

    def __init__(self, config: PolyphonicConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.base_channels
        input_channels = 3 if config.frequency_position else 2
        self.encoder_one = _ConvBlock(input_channels, channels)
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

    def forward(self, source: Tensor) -> Tensor:
        if source.ndim != 2:
            raise ValueError("source must have shape [batch, samples]")
        window = torch.hann_window(
            self.config.n_fft,
            device=source.device,
            dtype=source.dtype,
        )
        spectrum = torch.stft(
            source,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_size,
            window=window,
            return_complex=True,
        )
        components = torch.stack((spectrum.real, spectrum.imag), dim=1)
        scale = components.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
        normalized = components / scale
        if self.config.frequency_position:
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

        first = self.encoder_one(padded)
        second = self.encoder_two(self.down_one(first))
        latent = self.bottleneck(self.down_two(second))
        decoded_two = self.up_two(latent)
        decoded_two = self.decoder_two(torch.cat((decoded_two, second), dim=1))
        decoded_one = self.up_one(decoded_two)
        decoded_one = self.decoder_one(torch.cat((decoded_one, first), dim=1))
        residual = self.output_projection(decoded_one)
        residual = residual[..., : original_shape[0], : original_shape[1]]

        predicted_components = components + residual * scale
        predicted_spectrum = torch.complex(
            predicted_components[:, 0],
            predicted_components[:, 1],
        )
        return torch.istft(
            predicted_spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_size,
            window=window,
            length=source.shape[1],
        ).clamp(-1.0, 1.0)

    def checkpoint(self) -> dict[str, object]:
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, object]) -> "PolyphonicSpectralUNet":
        config_values = dict(checkpoint["config"])
        config_values.setdefault("frequency_position", False)
        model = cls(PolyphonicConfig(**config_values))
        model.load_state_dict(checkpoint["state_dict"])
        return model


def _pad_to_multiple(inputs: Tensor, multiple: int) -> tuple[Tensor, tuple[int, int]]:
    frequency, time = inputs.shape[-2:]
    frequency_padding = (-frequency) % multiple
    time_padding = (-time) % multiple
    return F.pad(inputs, (0, time_padding, 0, frequency_padding)), (frequency, time)
