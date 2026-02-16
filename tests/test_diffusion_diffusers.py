from types import SimpleNamespace

from neural_bending_toolkit.models.diffusion_diffusers import _ensure_torch_xpu_namespace


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
