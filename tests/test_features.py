import numpy as np

from kin_audio.features import extract_performance


def test_extracts_clean_monophonic_pitch() -> None:
    sample_rate = 16_000
    frequency = 220.0
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2.0 * np.pi * frequency * time)

    performance = extract_performance(audio, sample_rate)
    detected = performance.f0_hz[performance.voiced > 0.5]

    assert detected.size > 80
    cents = 1200.0 * np.log2(detected / frequency)
    assert float(np.median(np.abs(cents))) < 5.0


def test_rejects_empty_audio() -> None:
    try:
        extract_performance(np.array([], dtype=np.float32), 16_000)
    except ValueError as error:
        assert "must not be empty" in str(error)
    else:
        raise AssertionError("empty audio should be rejected")
