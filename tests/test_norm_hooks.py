from __future__ import annotations

from types import SimpleNamespace

import pytest

from neural_bending_toolkit.models.diffusion_diffusers import DiffusersStableDiffusionAdapter

torch = pytest.importorskip("torch")


class _FakeUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=1, num_channels=4)
        self.attn_processors = {"down.0.attn": object()}

    def set_attn_processor(self, _processor_map: dict[str, object]) -> None:
        return None


class _FakePipe:
    def __init__(self, unet: _FakeUNet) -> None:
        self.unet = unet
        self.tokenizer = SimpleNamespace(model_max_length=1)
        self.text_encoder = lambda _input_ids: (torch.ones((1, 1, 4), dtype=torch.float32),)

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        callback = kwargs["callback_on_step_end"]
        callback(None, 3, None, {})
        x = torch.ones((1, 4, 2, 2), dtype=torch.float32)
        _ = self.unet.norm(x)
        return SimpleNamespace(images=[])


def _build_adapter() -> DiffusersStableDiffusionAdapter:
    adapter = DiffusersStableDiffusionAdapter.__new__(DiffusersStableDiffusionAdapter)
    adapter._torch = torch
    adapter.device = "cpu"
    adapter._pipe = _FakePipe(_FakeUNet())
    adapter.current_step = 0
    adapter._encode_prompt = lambda _prompt: torch.ones((1, 1, 4), dtype=torch.float32)
    return adapter


def test_norm_hook_called_for_groupnorm_module() -> None:
    adapter = _build_adapter()
    calls: list[tuple[str, int, str]] = []

    def norm_hook(x: torch.Tensor, ctx: object) -> torch.Tensor:
        calls.append((ctx.layer_name, ctx.step, ctx.metadata["hook_kind"]))
        return x

    _ = adapter.generate("prompt", norm_hook=norm_hook)

    assert calls
    assert any(layer_name == "norm" for layer_name, _, _ in calls)
    assert any(step == 3 for _, step, _ in calls)


def test_norm_hook_perturbation_preserves_shape_dtype_and_device() -> None:
    adapter = _build_adapter()
    seen_before: list[torch.Tensor] = []
    seen_after: list[torch.Tensor] = []

    def norm_hook(x: torch.Tensor, _ctx: object) -> torch.Tensor:
        seen_before.append(x.detach().clone())
        updated = x + 0.25
        seen_after.append(updated.detach().clone())
        return updated

    _ = adapter.generate("prompt", norm_hook=norm_hook)

    assert seen_before
    assert seen_after
    before = seen_before[0]
    after = seen_after[0]
    assert before.shape == after.shape
    assert before.dtype == after.dtype
    assert before.device == after.device
    assert not torch.allclose(before.mean(), after.mean())
