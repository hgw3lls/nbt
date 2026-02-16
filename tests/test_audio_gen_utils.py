import wave
from pathlib import Path

import numpy as np

from neural_bending_toolkit.models.audio_gen import (
    AudioGenAdapter,
    AudioGenerationResult,
)


def test_proxy_generation_has_expected_shape() -> None:
    adapter = AudioGenAdapter(backend="auto")

    out = adapter.generate("test prompt", duration_s=1.0, seed=1)

    assert out.waveform.ndim == 1
    assert out.sample_rate == 32000
    assert out.metadata["backend"] in {"proxy", "musicgen"}


def test_save_wav_writes_mono_pcm(tmp_path: Path) -> None:
    waveform = np.linspace(-0.5, 0.5, 320, dtype=np.float32)
    result = AudioGenerationResult(
        sample_rate=16000,
        waveform=waveform,
        metadata={},
    )

    out = AudioGenAdapter.save_wav(result, tmp_path / "clip.wav")

    with wave.open(str(out), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == 16000
