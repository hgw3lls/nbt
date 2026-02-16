"""Patch graph schema and validation for the Neural Bending Console."""

from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class PortType(str, Enum):
    """Supported port payload types."""

    TEXT = "TEXT"
    CV = "CV"
    METRIC = "METRIC"
    EMBEDDING = "EMBEDDING"
    LATENTS = "LATENTS"
    ATTENTION = "ATTENTION"
    RESIDUAL = "RESIDUAL"
    TRIGGER = "TRIGGER"
    IMAGE_PATH = "IMAGE_PATH"
    AUDIO_PATH = "AUDIO_PATH"


class EdgeType(str, Enum):
    """Edge execution semantics."""

    NORMAL = "NORMAL"
    FEEDBACK = "FEEDBACK"


class NodeUI(BaseModel):
    """UI placement metadata."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class Node(BaseModel):
    """Patch node description."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    ui: NodeUI
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("node id cannot be empty")
        return trimmed


class Edge(BaseModel):
    """Patch edge description."""

    model_config = ConfigDict(extra="forbid")

    id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    edge_type: EdgeType = EdgeType.NORMAL
    gated: bool = False


class PatchGraph(BaseModel):
    """Patch graph composed of nodes and directed edges."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)
    globals: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_graph(self, info: ValidationInfo) -> "PatchGraph":
        errors: list[str] = []
        node_ids = [node.id for node in self.nodes]
        duplicates = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
        if duplicates:
            errors.append(f"duplicate node ids: {', '.join(duplicates)}")

        by_id = {node.id: node for node in self.nodes}
        specs = (info.context or {}).get("node_specs", {})

        for edge in self.edges:
            if edge.from_node not in by_id:
                errors.append(
                    f"edge '{edge.id}' references missing from_node '{edge.from_node}'"
                )
            if edge.to_node not in by_id:
                errors.append(
                    f"edge '{edge.id}' references missing to_node '{edge.to_node}'"
                )

            from_node = by_id.get(edge.from_node)
            to_node = by_id.get(edge.to_node)
            if not from_node or not to_node:
                continue

            from_spec = specs.get(from_node.type)
            to_spec = specs.get(to_node.type)
            if not from_spec:
                errors.append(
                    f"edge '{edge.id}' has unknown from_node type '{from_node.type}'"
                )
                continue
            if not to_spec:
                errors.append(
                    f"edge '{edge.id}' has unknown to_node type '{to_node.type}'"
                )
                continue

            out_ports = from_spec.get("outputs", {})
            in_ports = to_spec.get("inputs", {})
            if edge.from_port not in out_ports:
                errors.append(
                    f"edge '{edge.id}' unknown output port '{edge.from_port}' on node '{from_node.id}'"
                )
                continue
            is_param_mod = edge.to_port.startswith('param:')
            if edge.to_port not in in_ports and not is_param_mod:
                errors.append(
                    f"edge '{edge.id}' unknown input port '{edge.to_port}' on node '{to_node.id}'"
                )
                continue

            if is_param_mod:
                if out_ports[edge.from_port] != PortType.CV:
                    errors.append(
                        f"edge '{edge.id}' incompatible types: {from_node.id}.{edge.from_port} must be CV for param modulation"
                    )
                continue

            if out_ports[edge.from_port] != in_ports[edge.to_port]:
                errors.append(
                    "edge "
                    f"'{edge.id}' incompatible types: "
                    f"{from_node.id}.{edge.from_port}({out_ports[edge.from_port].value}) -> "
                    f"{to_node.id}.{edge.to_port}({in_ports[edge.to_port].value})"
                )

        if self._has_disallowed_cycle():
            errors.append(
                "graph contains a cycle that is not explicitly FEEDBACK and gated"
            )

        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _has_disallowed_cycle(self) -> bool:
        """Detect cycles in non-feedback/gated execution graph."""

        allowed_feedback = {
            edge.id
            for edge in self.edges
            if edge.edge_type == EdgeType.FEEDBACK and edge.gated
        }

        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree: dict[str, int] = {node.id: 0 for node in self.nodes}

        for edge in self.edges:
            if edge.id in allowed_feedback:
                continue
            if edge.from_node in indegree and edge.to_node in indegree:
                if edge.to_node not in adjacency[edge.from_node]:
                    adjacency[edge.from_node].add(edge.to_node)
                    indegree[edge.to_node] += 1

        queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
        visited = 0
        while queue:
            node_id = queue.popleft()
            visited += 1
            for nxt in adjacency[node_id]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        return visited != len(indegree)


def validate_patch_graph(data: dict[str, Any], *, node_specs: dict[str, Any]) -> PatchGraph:
    """Validate raw patch payload against schema and node port contracts."""

    return PatchGraph.model_validate(data, context={"node_specs": node_specs})
