# Audio Adapter (MusicGen/Proxy)

`models/audio_gen.py` provides an audio generation adapter:

- uses AudioCraft/MusicGen when available
- otherwise falls back to a modular prompt-conditioned proxy generator

## Install

```bash
pip install -e .[dev,audio]
```

## Generate clips

Use `AudioGenAdapter.generate(prompt, ...)` to produce short clips and `save_wav(...)` to store WAV artifacts.

## Limitations

If MusicGen is unavailable, the adapter explicitly marks limitations in metadata and uses proxy drift behavior (not true attention-head intervention).
