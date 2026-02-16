import pytest

from neural_bending_toolkit.console.runtime import node_specs
from neural_bending_toolkit.console.schema import validate_patch_graph


def test_patch_validation_accepts_good_patch() -> None:
    patch = {
        "nodes": [
            {
                "id": "prompt",
                "type": "PromptSourceNode",
                "params": {"text": "hello world"},
                "ui": {"x": 0, "y": 0},
                "enabled": True,
            },
            {
                "id": "gen",
                "type": "DummyTextGenNode",
                "params": {},
                "ui": {"x": 1, "y": 1},
                "enabled": True,
            },
        ],
        "edges": [
            {
                "id": "edge_1",
                "from_node": "prompt",
                "from_port": "text",
                "to_node": "gen",
                "to_port": "prompt",
            }
        ],
        "globals": {},
    }

    graph = validate_patch_graph(patch, node_specs=node_specs())
    assert len(graph.nodes) == 2


def test_patch_validation_rejects_bad_patch() -> None:
    bad_patch = {
        "nodes": [
            {
                "id": "n1",
                "type": "PromptSourceNode",
                "params": {},
                "ui": {"x": 0, "y": 0},
                "enabled": True,
            },
            {
                "id": "n2",
                "type": "MetricProbeNode",
                "params": {},
                "ui": {"x": 0, "y": 0},
                "enabled": True,
            },
        ],
        "edges": [
            {
                "id": "bad",
                "from_node": "n1",
                "from_port": "missing_port",
                "to_node": "n2",
                "to_port": "text",
            }
        ],
        "globals": {},
    }

    with pytest.raises(ValueError):
        validate_patch_graph(bad_patch, node_specs=node_specs())
