"""Audio drift experiment with real or proxy intervention behavior."""

from __future__ import annotations

from pydantic import Field

from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


class AudioInterHeadDriftConfig(ExperimentSettings):
    """Config for attention/conditioning drift experiment in audio generation."""

    prompt: str = "A minimalist ambient synth line"
    duration_s: float = Field(default=3.0, ge=0.5, le=15.0)
    baseline_conditioning_scale: float = Field(default=1.0, ge=0.1, le=3.0)
    drift_conditioning_scale: float = Field(default=1.4, ge=0.1, le=4.0)
    baseline_attention_scale: float = Field(default=1.0, ge=0.1, le=3.0)
    drift_attention_scale: float = Field(default=1.6, ge=0.1, le=4.0)
    model_name: str = "facebook/musicgen-small"
    seed: int = 13


class AudioInterHeadDrift(Experiment):
    """Perturb attention/conditioning when possible; else run marked proxy drift."""

    name = "audio-inter-head-drift"
    config_model = AudioInterHeadDriftConfig

    def _adapter(self):
        from neural_bending_toolkit.models.audio_gen import AudioGenAdapter

        return AudioGenAdapter(backend="auto", model_name=self.config.model_name)

    def run(self, context: RunContext) -> None:
        adapter = self._adapter()

        baseline = adapter.generate(
            self.config.prompt,
            duration_s=self.config.duration_s,
            conditioning_scale=self.config.baseline_conditioning_scale,
            attention_scale=self.config.baseline_attention_scale,
            seed=self.config.seed,
        )
        drifted = adapter.generate(
            self.config.prompt,
            duration_s=self.config.duration_s,
            conditioning_scale=self.config.drift_conditioning_scale,
            attention_scale=self.config.drift_attention_scale,
            seed=self.config.seed,
        )

        baseline_path = adapter.save_wav(
            baseline,
            context.artifacts_dir / "audio_baseline.wav",
        )
        drifted_path = adapter.save_wav(
            drifted,
            context.artifacts_dir / "audio_drift.wav",
        )

        amp_delta = float(abs(drifted.waveform).mean() - abs(baseline.waveform).mean())
        context.log_metric(
            step=1,
            metric_name="mean_abs_amplitude_delta",
            value=amp_delta,
            metadata={"backend": adapter.backend_name},
        )

        limitations = drifted.metadata.get("limitations")
        if limitations:
            context.log_event(
                "Audio drift limitation",
                backend=adapter.backend_name,
                limitations=limitations,
            )

        context.log_event(
            "Saved audio drift artifacts",
            backend=adapter.backend_name,
            baseline=str(baseline_path),
            drifted=str(drifted_path),
        )
