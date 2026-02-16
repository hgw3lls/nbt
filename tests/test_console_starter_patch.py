import json
from pathlib import Path

from neural_bending_toolkit.console.schema import PortType, validate_patch_graph


def _infer_port_type(name: str) -> PortType:
    n = name.lower()
    if "embedding" in n:
        return PortType.EMBEDDING
    if "image" in n:
        return PortType.IMAGE_PATH
    if "metric" in n:
        return PortType.METRIC
    if "trigger" in n:
        return PortType.TRIGGER
    if "cv" in n:
        return PortType.CV
    return PortType.TEXT


def _build_specs(patch: dict) -> dict[str, dict[str, dict[str, PortType]]]:
    specs: dict[str, dict[str, dict[str, PortType]]] = {}
    nodes_by_id = {node["id"]: node for node in patch["nodes"]}
    for edge in patch["edges"]:
        src_type = nodes_by_id[edge["from_node"]]["type"]
        dst_type = nodes_by_id[edge["to_node"]]["type"]
        ptype = _infer_port_type(edge["from_port"])

        src_spec = specs.setdefault(src_type, {"inputs": {}, "outputs": {}})
        dst_spec = specs.setdefault(dst_type, {"inputs": {}, "outputs": {}})

        src_spec["outputs"].setdefault(edge["from_port"], ptype)
        dst_spec["inputs"].setdefault(edge["to_port"], ptype)
    return specs


def test_starter_8_bends_patch_parses_with_schema_validator() -> None:
    patch_path = Path("patches/starter_8_bends_ab.json")
    payload = json.loads(patch_path.read_text(encoding="utf-8"))
    specs = _build_specs(payload)

    graph = validate_patch_graph(payload, node_specs=specs)
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
