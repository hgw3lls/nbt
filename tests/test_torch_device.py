import warnings

from neural_bending_toolkit.models.torch_device import normalize_torch_device


def test_normalize_torch_device_keeps_cpu_without_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalized = normalize_torch_device("cpu")

    assert normalized == "cpu"
    assert not caught


def test_normalize_torch_device_falls_back_to_cpu_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalized = normalize_torch_device("cuda")

    assert normalized == "cpu"
    assert len(caught) == 1
    assert "falling back to CPU" in str(caught[0].message)
