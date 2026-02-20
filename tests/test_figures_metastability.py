import sys
import types


class _FakePyplot(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("matplotlib.pyplot")

    def figure(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return None

    def xlabel(self, *_args, **_kwargs):
        return None

    def ylabel(self, *_args, **_kwargs):
        return None

    def title(self, *_args, **_kwargs):
        return None

    def legend(self, *_args, **_kwargs):
        return None

    def tight_layout(self):
        return None

    def savefig(self, path):
        path.write_bytes(b"fake-png")

    def close(self):
        return None


def test_plot_entropy_over_steps_writes_png(tmp_path, monkeypatch) -> None:
    fake_pyplot = _FakePyplot()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    from neural_bending_toolkit.analysis.figures_metastability import (
        plot_entropy_over_steps,
    )

    out = plot_entropy_over_steps(
        {
            "baseline": [1.0, 0.9, 1.1],
            "shock": [1.0, 0.5, 0.7],
            "shock_counter": [1.0, 0.7, 0.95],
        },
        tmp_path / "comparisons" / "entropy_over_steps.png",
    )

    assert out.exists()
