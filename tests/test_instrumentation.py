import json
from pathlib import Path

import numpy as np

from neural_bending_toolkit.experiment import RunContext


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_metrics_logger_writes_expected_schema(tmp_path: Path) -> None:
    context = RunContext(tmp_path)

    context.log_metric(
        step=1,
        metric_name="loss",
        value=0.42,
        metadata={"split": "train"},
    )

    records = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(records) == 1
    assert records[0]["step"] == 1
    assert records[0]["metric_name"] == "loss"
    assert records[0]["value"] == 0.42
    assert records[0]["metadata"] == {"split": "train"}


def test_snapshot_and_event_logs_are_emitted(tmp_path: Path) -> None:
    context = RunContext(tmp_path)

    context.log_event("hello")
    context.pre_intervention_snapshot("state", {"x": 1})
    context.post_intervention_snapshot("state", {"x": 2})

    text_log = (tmp_path / "events.log").read_text(encoding="utf-8")
    assert "hello" in text_log

    records = _read_jsonl(tmp_path / "events.jsonl")
    events = [record["event"] for record in records]
    assert "event" in events
    assert "pre_intervention_snapshot" in events
    assert "post_intervention_snapshot" in events


def test_artifact_helpers_save_text_and_numpy(tmp_path: Path) -> None:
    context = RunContext(tmp_path)

    text_path = context.save_text_artifact("note.txt", "instrumented")
    array_path = context.save_numpy_artifact("matrix.npy", np.array([[1, 2], [3, 4]]))

    assert text_path.read_text(encoding="utf-8") == "instrumented"
    assert np.array_equal(np.load(array_path), np.array([[1, 2], [3, 4]]))
