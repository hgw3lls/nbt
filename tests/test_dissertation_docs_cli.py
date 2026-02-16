from pathlib import Path

from typer.testing import CliRunner

from neural_bending_toolkit.cli import app

REQUIRED_README_HEADINGS = [
    "# Neural Bending Toolkit (NBT)",
    "## What is neural bending?",
    "## Toolkit philosophy",
    "## Supported model types",
    "## Installation",
    "## Core concepts",
    "## CLI quickstart",
    "## Reproducibility and run structure",
    "## Dissertation workflows",
]


def test_readme_exists_and_has_required_headings() -> None:
    readme = Path("README.md")
    assert readme.exists()

    content = readme.read_text(encoding="utf-8")
    for heading in REQUIRED_README_HEADINGS:
        assert heading in content


def test_templates_exist() -> None:
    assert Path("templates/theory_memo.md").exists()
    assert Path("templates/figure_caption.md").exists()
    assert Path("templates/methods_appendix.md").exists()


def test_nbt_docs_runs_without_error() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["docs"])

    assert result.exit_code == 0
    assert "documentation guide" in result.stdout.lower()


def test_nbt_init_dissertation_creates_expected_structure(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["init", "dissertation", "--path", str(tmp_path)])

    assert result.exit_code == 0
    dissertation = tmp_path / "dissertation"
    assert dissertation.exists()
    assert (dissertation / "figures").exists()
    assert (dissertation / "tables").exists()
    assert (dissertation / "memos").exists()
    assert (dissertation / "exports").exists()

    readme = dissertation / "README.md"
    assert readme.exists()
    assert "# Dissertation Workspace" in readme.read_text(encoding="utf-8")
