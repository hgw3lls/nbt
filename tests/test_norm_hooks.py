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


class _AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _ResidualBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnets = torch.nn.ModuleList([_AddOne()])
        self.transformer_blocks = torch.nn.ModuleList([_AddOne()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.resnets:
            x = module(x)
        for module in self.transformer_blocks:
            x = module(x)
        return x


class _FakeResidualUNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = torch.nn.ModuleList([_ResidualBlock()])
        self.attn_processors = {"down.0.attn": object()}

    def set_attn_processor(self, _processor_map: dict[str, object]) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.down_blocks:
            x = block(x)
        return x


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


class _FakeResidualPipe:
    def __init__(self, unet: _FakeResidualUNet) -> None:
        self.unet = unet
        self.tokenizer = SimpleNamespace(model_max_length=1)
        self.text_encoder = lambda _input_ids: (torch.ones((1, 1, 4), dtype=torch.float32),)
        self.outputs: list[torch.Tensor] = []

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        callback = kwargs["callback_on_step_end"]
        x = torch.ones((1, 4, 2, 2), dtype=torch.float32)
        callback(None, 0, None, {})
        self.outputs.append(self.unet(x))
        callback(None, 1, None, {})
        self.outputs.append(self.unet(x))
        return SimpleNamespace(images=[])


def _build_adapter() -> DiffusersStableDiffusionAdapter:
    adapter = DiffusersStableDiffusionAdapter.__new__(DiffusersStableDiffusionAdapter)
    adapter._torch = torch
    adapter.device = "cpu"
    adapter._pipe = _FakePipe(_FakeUNet())
    adapter.current_step = 0
    adapter._encode_prompt = lambda _prompt: torch.ones((1, 1, 4), dtype=torch.float32)
    return adapter


def _build_residual_adapter() -> tuple[DiffusersStableDiffusionAdapter, _FakeResidualPipe]:
    adapter = DiffusersStableDiffusionAdapter.__new__(DiffusersStableDiffusionAdapter)
    adapter._torch = torch
    adapter.device = "cpu"
    pipe = _FakeResidualPipe(_FakeResidualUNet())
    adapter._pipe = pipe
    adapter.current_step = 0
    adapter._encode_prompt = lambda _prompt: torch.ones((1, 1, 4), dtype=torch.float32)
    return adapter, pipe


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


def test_residual_hook_receives_previous_step_cache() -> None:
    adapter, _pipe = _build_residual_adapter()
    previous_by_step: dict[tuple[str, int], torch.Tensor | None] = {}

    def residual_hook(x: torch.Tensor, ctx: object) -> torch.Tensor:
        prev = ctx.cache["residual_echo"].get(ctx.layer_name)
        previous_by_step[(ctx.layer_name, ctx.step)] = None if prev is None else prev.clone()
        return x

    _ = adapter.generate("prompt", residual_hook=residual_hook)

    first_step_prev = previous_by_step[("down_blocks.0.resnets.0", 0)]
    second_step_prev = previous_by_step[("down_blocks.0.resnets.0", 1)]
    assert first_step_prev is None
    assert second_step_prev is not None
    assert torch.allclose(second_step_prev, torch.full_like(second_step_prev, 2.0))


def test_residual_echo_hook_deterministically_changes_second_step_output() -> None:
    baseline_adapter, baseline_pipe = _build_residual_adapter()
    _ = baseline_adapter.generate("prompt")

    echo_adapter, echo_pipe = _build_residual_adapter()

    def residual_hook(x: torch.Tensor, ctx: object) -> torch.Tensor:
        prev = ctx.cache["residual_echo"].get(ctx.layer_name)
        if prev is None:
            return x
        return x + 0.5 * prev

    _ = echo_adapter.generate("prompt", residual_hook=residual_hook)

    baseline_second = baseline_pipe.outputs[1]
    echoed_second = echo_pipe.outputs[1]
    assert torch.allclose(baseline_second, torch.full_like(baseline_second, 3.0))
    assert torch.allclose(echoed_second, torch.full_like(echoed_second, 5.5))
