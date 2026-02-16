from pathlib import Path

from neural_bending_toolkit.experiment import RunContext
from neural_bending_toolkit.experiments.dissertation_bends import (
    EmbeddingContaminationFamily,
    EmbeddingContaminationFamilyConfig,
)


def test_family_writes_theory_memo_and_comparison(tmp_path: Path) -> None:
    config = EmbeddingContaminationFamilyConfig()
    exp = EmbeddingContaminationFamily(config)
    context = RunContext(tmp_path)

    exp.run(context)

    assert (tmp_path / "theory_memo.md").exists()
    artifact_dir = tmp_path / "artifacts"
    assert any(
        path.name.endswith("comparison.json") for path in artifact_dir.glob("*.json")
    )
