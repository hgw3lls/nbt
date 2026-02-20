import sys
import types

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

import numpy as np

from neural_bending_toolkit.experiments.embedding_contamination_diffusion import (
    EmbeddingContaminationDiffusion,
    EmbeddingContaminationDiffusionConfig,
    blend_embeddings,
)


def test_blend_embeddings_weighted_average() -> None:
    base = np.array([[[1.0, 2.0]]])
    contaminant = np.array([[[3.0, 6.0]]])

    mixed = blend_embeddings(base, contaminant, alpha=0.25)

    assert np.allclose(mixed, np.array([[[1.5, 3.0]]]))


def test_diffusion_config_backwards_compatible_without_bends() -> None:
    config = EmbeddingContaminationDiffusionConfig.model_validate(
        {
            "model_id": "hf-internal-testing/tiny-stable-diffusion-pipe",
            "base_prompt": "a clean laboratory bench",
            "contaminant_prompt": "a chaotic graffiti wall",
            "contamination_alpha": 0.25,
            "num_inference_steps": 10,
            "guidance_scale": 7.0,
            "seed": 7,
        }
    )

    assert config.bends is None


class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def to(self, _device: str):
        return self


class _FakeTorch:
    @staticmethod
    def from_numpy(arr: np.ndarray) -> _FakeTensor:
        return _FakeTensor(arr)


class _FakeImage:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def __array__(self, dtype=None):
        return self._arr.astype(dtype) if dtype else self._arr


class _FakeOutput:
    def __init__(self) -> None:
        self.images = [_FakeImage(np.zeros((4, 4, 3), dtype=np.uint8))]
        self.attention_heatmaps = {}


class _FakeAdapter:
    def __init__(self) -> None:
        self._torch = _FakeTorch()
        self.device = "cpu"
        self.calls: list[dict[str, object]] = []

    def _encode_prompt(self, _prompt: str) -> _FakeTensor:
        return _FakeTensor(np.ones((1, 2, 3), dtype=np.float32))

    def generate(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return _FakeOutput()

    def save_artifacts(self, _output, _artifacts_dir, *, prefix: str):
        return [f"{prefix}_artifact"]


def test_diffusion_experiment_passes_compiled_bend_hook(monkeypatch, tmp_path) -> None:
    config = EmbeddingContaminationDiffusionConfig.model_validate(
        {
            "bends": [
                {
                    "name": "gate",
                    "site": {
                        "kind": "diffusion.cross_attention",
                        "allow_all_layers": True,
                    },
                    "actuator": {"type": "attention_head_gate", "params": {}},
                    "schedule": {"mode": "constant", "strength": 1.0},
                }
            ]
        }
    )
    experiment = EmbeddingContaminationDiffusion(config)
    fake_adapter = _FakeAdapter()

    monkeypatch.setattr(experiment, "_load_adapter", lambda: fake_adapter)
    def _fake_compiler(_plan, tracer=None):
        if tracer is None:
            return None
        return lambda _payload: None

    monkeypatch.setattr(
        "neural_bending_toolkit.experiments.embedding_contamination_diffusion.compile_diffusion_cross_attention_hook",
        _fake_compiler,
    )

    class _FakeContext:
        def __init__(self) -> None:
            self.artifacts_dir = tmp_path

        def pre_intervention_snapshot(self, **_kwargs):
            return None

        def post_intervention_snapshot(self, **_kwargs):
            return None

        def log_metric(self, **_kwargs):
            return None

        def log_event(self, *_args, **_kwargs):
            return None

    experiment.run(_FakeContext())

    assert len(fake_adapter.calls) == 2
    assert fake_adapter.calls[0].get("cross_attention_hook") is None
    assert callable(fake_adapter.calls[1].get("cross_attention_hook"))
