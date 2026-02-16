from pathlib import Path

import pytest
from typer.testing import CliRunner

from neural_bending_toolkit.bends.geopolitical import (
    GeopoliticalBend,
    GeopoliticalConfig,
)
from neural_bending_toolkit.cli import app
from neural_bending_toolkit.models.geopolitical_metadata import (
    get_geopolitical_metadata,
)


def _config() -> GeopoliticalConfig:
    return GeopoliticalConfig(
        model_identifiers=["meta/llama-3.1-70b-instruct"],
        llm_model_identifier="sshleifer/tiny-gpt2",
        diffusion_model_identifier="hf-internal-testing/tiny-stable-diffusion-pipe",
        governance_concepts=["sovereignty"],
        contradictory_prompt_pairs=[("A", "B")],
        justice_attractor_token_sets=[["equity", "rights"]],
        log_level="info",
        save_intermediate_artifacts=True,
        random_seed=1,
    )


def test_geopolitical_class_initialization() -> None:
    exp = GeopoliticalBend(_config())

    assert exp.name == "geopolitical-bend"
    assert exp.config.model_identifiers[0] == "meta/llama-3.1-70b-instruct"


def test_geopolitical_config_validation_rejects_bad_log_level() -> None:
    with pytest.raises(ValueError):
        GeopoliticalConfig(
            model_identifiers=["meta/llama-3.1-70b-instruct"],
            governance_concepts=["x"],
            contradictory_prompt_pairs=[("a", "b")],
            justice_attractor_token_sets=[["j"]],
            log_level="verbose",
        )


def test_metadata_lookup() -> None:
    hit = get_geopolitical_metadata("meta/llama-3.1-70b-instruct")
    miss = get_geopolitical_metadata("unknown/model")

    assert hit is not None
    assert hit["region"]
    assert miss is None


def test_cli_geopolitical_describe_parses() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["geopolitical", "describe"])

    assert result.exit_code == 0
    assert "geopolitical-bend" in result.stdout


def test_cli_geopolitical_run_validates_yaml_extension(tmp_path: Path) -> None:
    runner = CliRunner()
    bad = tmp_path / "config.txt"
    bad.write_text("x: 1", encoding="utf-8")

    result = runner.invoke(app, ["geopolitical", "run", "--config", str(bad)])

    assert result.exit_code != 0
    assert "YAML" in result.stdout
