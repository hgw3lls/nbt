import sys
import types

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

import json

from neural_bending_toolkit.analysis.bend_tagging import (
    score_and_tag_metastability,
    write_comparison_report,
)


def test_score_and_tag_recoherent_case() -> None:
    result = score_and_tag_metastability(
        recovery_index_value=0.9,
        concentration_collapse_index_value=0.2,
        basin_shift_proxy={"pairwise_mse": {"baseline_vs_shock": 1200.0}},
        output_variance=40.0,
    )

    assert "recoherent" in result.tags
    assert "revelatory" not in result.tags


def test_write_comparison_report(tmp_path) -> None:
    path = write_comparison_report(
        tmp_path,
        comparisons={"k": 1},
        summary={"tags": {"tags": ["recoherent"]}},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["comparisons"]["k"] == 1
    assert payload["summary"]["tags"]["tags"] == ["recoherent"]
