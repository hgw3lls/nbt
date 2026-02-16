"""Audio generation adapter with optional MusicGen backend and modular fallback."""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AudioGenerationResult:
    """Container for generated audio waveform and metadata."""

    sample_rate: int
    waveform: np.ndarray
    metadata: dict[str, Any]


class AudioGenAdapter:
    """Adapter for AudioCraft/MusicGen, with a deterministic proxy fallback backend."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        model_name: str = "facebook/musicgen-small",
        sample_rate: int = 32000,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.sample_rate = sample_rate
        self._musicgen = None
        self._backend_name = "proxy"

        if backend in {"auto", "musicgen"}:
            try:
                from audiocraft.models import MusicGen

                self._musicgen = MusicGen.get_pretrained(model_name)
                self._backend_name = "musicgen"
            except Exception as err:
                if backend == "musicgen":
                    raise RuntimeError(
                        "Requested MusicGen backend but AudioCraft/MusicGen "
                        "is unavailable."
                    ) from err

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def generate(
        self,
        prompt: str,
        *,
        duration_s: float = 3.0,
        conditioning_scale: float = 1.0,
        attention_scale: float = 1.0,
        seed: int = 0,
    ) -> AudioGenerationResult:
        """Generate a short audio clip from prompt using available backend."""
        if self._backend_name == "musicgen":
            return self._generate_musicgen(
                prompt,
                duration_s=duration_s,
                conditioning_scale=conditioning_scale,
                attention_scale=attention_scale,
                seed=seed,
            )

        return self._generate_proxy(
            prompt,
            duration_s=duration_s,
            conditioning_scale=conditioning_scale,
            attention_scale=attention_scale,
            seed=seed,
        )

    def _generate_musicgen(
        self,
        prompt: str,
        *,
        duration_s: float,
        conditioning_scale: float,
        attention_scale: float,
        seed: int,
    ) -> AudioGenerationResult:
        # AudioCraft APIs may differ by version; keep best-effort compatibility.
        import torch

        model = self._musicgen
        model.set_generation_params(duration=duration_s)
        try:
            model.set_generation_params(
                duration=duration_s,
                cfg_coef=conditioning_scale,
            )
        except Exception:
            pass

        torch.manual_seed(seed)
        wav = model.generate([prompt])
        arr = wav[0].detach().cpu().numpy()
        if arr.ndim == 2:
            arr = np.mean(arr, axis=0)

        return AudioGenerationResult(
            sample_rate=self.sample_rate,
            waveform=arr.astype(np.float32),
            metadata={
                "backend": "musicgen",
                "prompt": prompt,
                "duration_s": duration_s,
                "conditioning_scale": conditioning_scale,
                "attention_scale": attention_scale,
                "limitations": (
                    "attention_scale may be unsupported by installed backend"
                ),
            },
        )

    def _generate_proxy(
        self,
        prompt: str,
        *,
        duration_s: float,
        conditioning_scale: float,
        attention_scale: float,
        seed: int,
    ) -> AudioGenerationResult:
        """Proxy drift generator when MusicGen backend is unavailable."""
        n_samples = int(self.sample_rate * duration_s)
        t = np.linspace(0.0, duration_s, n_samples, endpoint=False)

        prompt_hash = abs(hash(prompt)) % 1000
        base_freq = 110.0 + (prompt_hash % 330)

        rng = np.random.default_rng(seed)
        harmonic = np.sin(2.0 * math.pi * base_freq * t)
        overtones = 0.5 * np.sin(2.0 * math.pi * (base_freq * 2.0) * t)
        noise = rng.normal(0.0, 0.03, size=n_samples)

        drifted = (harmonic + overtones) * conditioning_scale
        drifted = drifted * attention_scale + noise

        max_amp = max(np.max(np.abs(drifted)), 1e-8)
        waveform = (drifted / max_amp).astype(np.float32)

        return AudioGenerationResult(
            sample_rate=self.sample_rate,
            waveform=waveform,
            metadata={
                "backend": "proxy",
                "prompt": prompt,
                "duration_s": duration_s,
                "conditioning_scale": conditioning_scale,
                "attention_scale": attention_scale,
                "limitations": (
                    "AudioCraft/MusicGen unavailable; using prompt-conditioned "
                    "synthetic proxy drift generator (not true attention-head "
                    "intervention)."
                ),
            },
        )

    @staticmethod
    def save_wav(result: AudioGenerationResult, path: str | Path) -> Path:
        """Save mono float waveform to 16-bit PCM WAV artifact."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        clipped = np.clip(result.waveform, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)

        with wave.open(str(out), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(result.sample_rate)
            f.writeframes(struct.pack("<" + "h" * len(pcm), *pcm.tolist()))

        return out
