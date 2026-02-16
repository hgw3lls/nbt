from neural_bending_toolkit.console.recorder import ConsoleRecorder
from neural_bending_toolkit.console.runtime import (
    CompareNode,
    EmbeddingContaminationNode,
    RuntimeContext,
)
from neural_bending_toolkit.console.runtime import node_specs
from neural_bending_toolkit.console.schema import validate_patch_graph


def _ctx(*, token_latch: bool, diffusion_latch: bool) -> RuntimeContext:
    return RuntimeContext(
        tick=1,
        tick_rate=30.0,
        latch_state={
            "token_boundary_latch": token_latch,
            "diffusion_step_latch": diffusion_latch,
        },
        recorder=ConsoleRecorder(),
    )


def test_embedding_contamination_is_latch_safe() -> None:
    node = EmbeddingContaminationNode(
        "emb",
        {"enabled": True, "dry_wet": 1.0},
    )
    no_latch = node.process(_ctx(token_latch=False, diffusion_latch=False), {
        "embedding_a": [1.0, 0.0],
        "embedding_b": [0.0, 1.0],
    })
    assert no_latch == {}

    with_latch = node.process(_ctx(token_latch=True, diffusion_latch=False), {
        "embedding_a": [1.0, 0.0],
        "embedding_b": [0.0, 1.0],
    })
    assert "embedding" in with_latch
    assert len(with_latch["embedding"]) == 2


def test_compare_node_emits_metrics() -> None:
    node = CompareNode("cmp", {"attractor_lexicon": ["care", "justice"]})
    out = node.process(
        _ctx(token_latch=True, diffusion_latch=True),
        {
            "baseline_text": "we cannot help",
            "bent_text": "we care and justice now",
            "baseline_image": "a.png",
            "bent_image": "b.png",
        },
    )
    assert "metric" in out
    assert "divergence_proxy" in out["metric"]
    assert "attractor_density_delta" in out["metric"]
    assert "refusal_delta" in out["metric"]


def test_schema_accepts_real_model_patch_types() -> None:
    patch = {
        "nodes": [
            {
                "id": "prompt",
                "type": "PromptSourceNode",
                "params": {"text": "hello"},
                "ui": {"x": 0, "y": 0},
                "enabled": True,
            },
            {
                "id": "llm",
                "type": "LLMVoiceNode",
                "params": {},
                "ui": {"x": 100, "y": 0},
                "enabled": True,
            },
            {
                "id": "cmp",
                "type": "CompareNode",
                "params": {},
                "ui": {"x": 200, "y": 0},
                "enabled": True,
            },
        ],
        "edges": [
            {
                "id": "e1",
                "from_node": "prompt",
                "from_port": "text",
                "to_node": "llm",
                "to_port": "prompt",
            },
            {
                "id": "e2",
                "from_node": "llm",
                "from_port": "text",
                "to_node": "cmp",
                "to_port": "bent_text",
            },
        ],
        "globals": {},
    }

    graph = validate_patch_graph(patch, node_specs=node_specs())
    assert len(graph.nodes) == 3
