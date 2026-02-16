import asyncio

from neural_bending_toolkit.console.runtime import ConsoleRuntime, node_specs
from neural_bending_toolkit.console.schema import validate_patch_graph


def test_runtime_executes_in_topological_order() -> None:
    patch = {
        "nodes": [
            {
                "id": "prompt",
                "type": "PromptSourceNode",
                "params": {"text": "alpha beta"},
                "ui": {"x": 0, "y": 0},
                "enabled": True,
            },
            {
                "id": "gen",
                "type": "DummyTextGenNode",
                "params": {},
                "ui": {"x": 1, "y": 0},
                "enabled": True,
            },
            {
                "id": "metric",
                "type": "MetricProbeNode",
                "params": {},
                "ui": {"x": 2, "y": 0},
                "enabled": True,
            },
        ],
        "edges": [
            {
                "id": "e1",
                "from_node": "prompt",
                "from_port": "text",
                "to_node": "gen",
                "to_port": "prompt",
            },
            {
                "id": "e2",
                "from_node": "gen",
                "from_port": "text",
                "to_node": "metric",
                "to_port": "text",
            },
        ],
        "globals": {},
    }

    graph = validate_patch_graph(patch, node_specs=node_specs())
    runtime = ConsoleRuntime(graph)
    assert runtime.execution_order == ["prompt", "gen", "metric"]

    asyncio.run(runtime.step_once())
    assert runtime.tick == 1
    assert "text" in runtime._latest_outputs["gen"]
    assert "metric" in runtime._latest_outputs["metric"]
