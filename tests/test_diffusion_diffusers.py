from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neural_bending_toolkit.models.diffusion_diffusers import (
    HookedCrossAttentionProcessor,
    _ensure_torch_xpu_namespace,
)


def test_ensure_torch_xpu_namespace_adds_stub_when_missing() -> None:
    torch_module = SimpleNamespace()

    _ensure_torch_xpu_namespace(torch_module)

    assert hasattr(torch_module, "xpu")
    assert torch_module.xpu.is_available() is False
    assert torch_module.xpu.empty_cache() is None


def test_ensure_torch_xpu_namespace_keeps_existing_namespace() -> None:
    existing_xpu = object()
    torch_module = SimpleNamespace(xpu=existing_xpu)

    _ensure_torch_xpu_namespace(torch_module)

    assert torch_module.xpu is existing_xpu


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def view(self, *shape: int) -> "_FakeTensor":
        return _FakeTensor(self.array.reshape(shape))

    def transpose(self, *dims: int) -> "_FakeTensor":
        if len(dims) == 2:
            return _FakeTensor(np.swapaxes(self.array, dims[0], dims[1]))
        return _FakeTensor(np.transpose(self.array, dims))

    def detach(self) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self

    def mean(self, dim: int) -> "_FakeTensor":
        return _FakeTensor(self.array.mean(axis=dim))

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array

    def __matmul__(self, other: "_FakeTensor") -> "_FakeTensor":
        return _FakeTensor(self.array @ other.array)

    def __add__(self, other: "_FakeTensor") -> "_FakeTensor":
        return _FakeTensor(self.array + other.array)

    def __truediv__(self, value: float) -> "_FakeTensor":
        return _FakeTensor(self.array / value)

    def __bool__(self) -> bool:
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")


class _FakeAttention:
    spatial_norm = None
    group_norm = None
    norm_cross = False
    residual_connection = False
    rescale_output_factor = 1.0

    to_out = [staticmethod(lambda x: x), staticmethod(lambda x: x)]

    @staticmethod
    def prepare_attention_mask(*_args: object, **_kwargs: object) -> None:
        return None

    @staticmethod
    def to_q(hidden_states: _FakeTensor) -> _FakeTensor:
        return hidden_states

    @staticmethod
    def to_k(hidden_states: _FakeTensor) -> _FakeTensor:
        return hidden_states

    @staticmethod
    def to_v(hidden_states: _FakeTensor) -> _FakeTensor:
        return hidden_states

    @staticmethod
    def head_to_batch_dim(tensor: _FakeTensor) -> _FakeTensor:
        return tensor

    @staticmethod
    def batch_to_head_dim(tensor: _FakeTensor) -> _FakeTensor:
        return tensor

    @staticmethod
    def get_attention_scores(
        query: _FakeTensor,
        key: _FakeTensor,
        _attention_mask: None,
    ) -> _FakeTensor:
        scores = query @ key.transpose(1, 2)
        return scores / max(query.shape[-1], 1)


def test_hooked_cross_attention_processor_accepts_tensor_encoder_states() -> None:
    processor = HookedCrossAttentionProcessor(
        layer_name="layer.0",
        get_step=lambda: 0,
        hook=None,
        heatmaps={},
    )
    attn = _FakeAttention()
    hidden_states = _FakeTensor(np.ones((1, 2, 3), dtype=np.float32))
    encoder_hidden_states = _FakeTensor(np.ones((1, 2, 3), dtype=np.float32))

    output = processor(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)

    assert output.shape == hidden_states.shape


def test_hooked_cross_attention_processor_rejects_invalid_hook_return() -> None:
    processor = HookedCrossAttentionProcessor(
        layer_name="layer.0",
        get_step=lambda: 0,
        hook=lambda _payload: _FakeTensor(np.ones((1, 1, 1), dtype=np.float32)),
        heatmaps={},
    )
    attn = _FakeAttention()
    hidden_states = _FakeTensor(np.ones((1, 2, 3), dtype=np.float32))

    with pytest.raises(TypeError, match="cross_attention_hook must return"):
        processor(attn, hidden_states, encoder_hidden_states=hidden_states)
