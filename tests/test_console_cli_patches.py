from typer.testing import CliRunner

from neural_bending_toolkit.cli import app


runner = CliRunner()


def test_console_patches_lists_starter_patch() -> None:
    result = runner.invoke(app, ["console", "patches"])
    assert result.exit_code == 0
    assert "patches/starter_8_bends_ab.json" in result.stdout
