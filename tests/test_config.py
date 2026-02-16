from pathlib import Path

import pytest

from neural_bending_toolkit.config import (
    ConfigValidationError,
    load_and_validate_config,
)
from neural_bending_toolkit.experiments.hello import HelloExperimentConfig


def test_valid_config_loads(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: Ada\nrepeats: 2\n", encoding="utf-8")

    config = load_and_validate_config(config_path, HelloExperimentConfig)

    assert config.name == "Ada"
    assert config.repeats == 2


def test_invalid_config_fails_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "name: Ada\nrepeats: 0\nextra_field: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError):
        load_and_validate_config(config_path, HelloExperimentConfig)
