"""Console graph runtime and built-in nodes."""

from __future__ import annotations

import asyncio
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from neural_bending_toolkit.console.adapters import (
    LLMToken,
    adapter_hash,
    make_diffusion_adapter,
    make_llm_adapter,
)
from neural_bending_toolkit.console.recorder import ConsoleRecorder
from neural_bending_toolkit.console.schema import EdgeType, PatchGraph, PortType

PortDecl = dict[str, dict[str, PortType]]


@dataclass
class RuntimeContext:
    tick: int
    tick_rate: float
    latch_state: dict[str, bool]
    recorder: ConsoleRecorder


class BaseNode:
    def __init__(self, node_id: str, params: dict[str, Any]) -> None:
        self.node_id = node_id
        self.params = params

    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {"inputs": {}, "outputs": {}}

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        return {}

    def on_patch_update(self, params: dict[str, Any]) -> None:
        self.params = params

    def modulated_numeric(
        self,
        name: str,
        inputs: dict[str, Any],
        default: float,
        *,
        clamp: tuple[float, float] | None = None,
    ) -> float:
        base = float(self.params.get(name, default))
        cv = inputs.get(f"param:{name}", 0.0)
        try:
            cv_val = float(cv)
        except (TypeError, ValueError):
            cv_val = 0.0
        att = float(self.params.get(f"{name}_cv_att", 1.0))
        offset = float(self.params.get(f"{name}_cv_offset", 0.0))
        value = base + cv_val * att + offset
        if clamp:
            value = min(clamp[1], max(clamp[0], value))
        return value


class PromptSourceNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {"inputs": {}, "outputs": {"text": PortType.TEXT}}

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        del ctx, inputs
        return {"text": self.params.get("text", "")}


class CV_LFONode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {"inputs": {}, "outputs": {"cv": PortType.CV}}

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        del inputs
        waveform = str(self.params.get("waveform", "sine")).lower()
        freq = self.modulated_numeric("frequency_hz", {}, 1.0, clamp=(0.01, 64.0))
        phase = (ctx.tick / ctx.tick_rate) * freq
        if waveform == "saw":
            value = (phase % 1.0) * 2 - 1
        elif waveform == "random":
            rng = random.Random(ctx.tick + hash(self.node_id))
            value = rng.uniform(-1.0, 1.0)
        else:
            value = math.sin(2 * math.pi * phase)
        return {"cv": value}


class CV_StepSequencerNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {"inputs": {"trigger": PortType.TRIGGER}, "outputs": {"cv": PortType.CV}}

    def __init__(self, node_id: str, params: dict[str, Any]) -> None:
        super().__init__(node_id, params)
        self.step_index = 0

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        del ctx
        sequence = self.params.get("sequence", [0.0, 0.25, 0.5, 0.75]) or [0.0]
        if inputs.get("trigger"):
            self.step_index = (self.step_index + 1) % len(sequence)
        return {"cv": float(sequence[self.step_index])}


class DummyTextGenNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {"inputs": {"prompt": PortType.TEXT}, "outputs": {"text": PortType.TEXT}}

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        prompt = str(inputs.get("prompt") or self.params.get("seed_text", "hello"))
        tokens = prompt.split() or ["hello"]
        return {"text": f"{tokens[ctx.tick % len(tokens)]} "}


class MetricProbeNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {"inputs": {"text": PortType.TEXT}, "outputs": {"metric": PortType.METRIC}}

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        del ctx
        text = str(inputs.get("text", ""))
        length = len(text)
        entropy_proxy = (len(set(text)) / max(1, length))
        return {"metric": {"text_length": length, "entropy_proxy": entropy_proxy}}


class RecorderNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {"inputs": {"trigger": PortType.TRIGGER}, "outputs": {}}

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs.get("trigger"):
            ctx.recorder.save_take(str(ctx.tick))
        return {}


class LLMVoiceNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {
                "prompt": PortType.TEXT,
                "temp_cv": PortType.CV,
                "top_p_cv": PortType.CV,
                "top_k_cv": PortType.CV,
                "param:temperature": PortType.CV,
                "param:top_p": PortType.CV,
                "param:top_k": PortType.CV,
            },
            "outputs": {"text": PortType.TEXT, "metric": PortType.METRIC},
        }

    def __init__(self, node_id: str, params: dict[str, Any]) -> None:
        super().__init__(node_id, params)
        self.adapter = make_llm_adapter(params)
        self._last_prompt = ""
        self._stream: list[LLMToken] = []
        self._idx = 0

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        if not ctx.latch_state.get("token_boundary_latch", False):
            return {}

        prompt = str(inputs.get("prompt") or self.params.get("prompt", ""))
        if not prompt:
            return {}

        if prompt != self._last_prompt or self._idx >= len(self._stream):
            temperature = self.modulated_numeric("temperature", inputs, 0.8, clamp=(0.05, 3.0)) + float(inputs.get("temp_cv", 0.0))
            top_p = self.modulated_numeric("top_p", inputs, 0.9, clamp=(0.05, 1.0)) + float(inputs.get("top_p_cv", 0.0))
            top_k = int(self.modulated_numeric("top_k", inputs, 40.0, clamp=(1.0, 200.0)) + float(inputs.get("top_k_cv", 0.0) * 10))
            max_new_tokens = int(self.params.get("max_new_tokens", 24))
            self._stream = self.adapter.stream_generate(
                prompt,
                temperature=max(0.05, temperature),
                top_p=min(1.0, max(0.05, top_p)),
                top_k=max(1, top_k),
                max_new_tokens=max_new_tokens,
            )
            self._last_prompt = prompt
            self._idx = 0

        if self._idx >= len(self._stream):
            return {}

        token = self._stream[self._idx]
        self._idx += 1
        entropy_proxy = min(1.0, abs(token.logprob_proxy))
        return {
            "text": token.text,
            "metric": {
                "token_entropy_proxy": entropy_proxy,
                "token_logprob_proxy": token.logprob_proxy,
            },
        }


class DiffusionVoiceNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {
                "prompt": PortType.TEXT,
                "guidance_cv": PortType.CV,
                "embedding": PortType.EMBEDDING,
                "param:guidance_scale": PortType.CV,
            },
            "outputs": {"image_path": PortType.IMAGE_PATH, "metric": PortType.METRIC},
        }

    def __init__(self, node_id: str, params: dict[str, Any]) -> None:
        super().__init__(node_id, params)
        self.adapter = make_diffusion_adapter(params)
        self._step = 0
        self._prompt = ""
        self._steps_total = int(self.params.get("num_steps", 20))
        self._done = False

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        prompt = str(inputs.get("prompt") or self.params.get("prompt", ""))
        if not prompt:
            return {}

        if prompt != self._prompt:
            self._prompt = prompt
            self._step = 0
            self._done = False

        if self._done or not ctx.latch_state.get("diffusion_step_latch", False):
            return {}

        self._step += 1
        metric = {"diffusion_step": self._step, "diffusion_steps_total": self._steps_total}
        if self._step < self._steps_total:
            return {"metric": metric}

        guidance_scale = self.modulated_numeric("guidance_scale", inputs, 7.5, clamp=(0.0, 25.0)) + float(inputs.get("guidance_cv", 0.0))
        embedding = inputs.get("embedding") if isinstance(inputs.get("embedding"), list) else None
        run_dir = ctx.recorder.run_dir or Path("runs")
        img_path = run_dir / "outputs" / f"diff_{self.node_id}_{adapter_hash(prompt)}.png"
        generated = self.adapter.generate_image(
            prompt=prompt,
            guidance_scale=guidance_scale,
            embedding=embedding,
            output_path=img_path,
        )
        self._done = True
        metric.update({k: v for k, v in generated.items() if k != "image_path"})
        return {"image_path": generated["image_path"], "metric": metric}


def _enabled(params: dict[str, Any]) -> bool:
    return bool(params.get("enabled", True))


def _dry_wet(node: BaseNode, inputs: dict[str, Any]) -> float:
    return node.modulated_numeric("dry_wet", inputs, 1.0, clamp=(0.0, 1.0))


def _to_embedding(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        chars = value.encode("utf-8")[:16]
        return [(b / 255.0) * 2 - 1 for b in chars] or [0.0]
    return [0.0]


class EmbeddingContaminationNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {
                "embedding_a": PortType.EMBEDDING,
                "embedding_b": PortType.EMBEDDING,
                "param:dry_wet": PortType.CV,
            },
            "outputs": {"embedding": PortType.EMBEDDING},
        }

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        if not ctx.latch_state.get("token_boundary_latch", False):
            return {}
        emb_a = _to_embedding(inputs.get("embedding_a") or self.params.get("concept_a", ""))
        emb_b = _to_embedding(inputs.get("embedding_b") or self.params.get("concept_b", ""))
        if not _enabled(self.params):
            return {"embedding": emb_a}
        wet = _dry_wet(self, inputs)
        size = max(len(emb_a), len(emb_b))
        return {
            "embedding": [
                (1 - wet) * emb_a[i % len(emb_a)] + wet * ((emb_a[i % len(emb_a)] + emb_b[i % len(emb_b)]) / 2.0)
                for i in range(size)
            ]
        }


class StratigraphySamplerNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {"cv": PortType.CV, "param:dry_wet": PortType.CV},
            "outputs": {"cv": PortType.CV},
        }

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        del ctx
        cv = float(inputs.get("cv", 0.0))
        if not _enabled(self.params):
            return {"cv": cv}
        wet = _dry_wet(self, inputs)
        return {
            "cv": {
                "anti_top_p": max(0.0, min(1.0, (1 - ((cv + 1) / 2)) * wet)),
                "low_prob_force": max(0.0, min(1.0, abs(cv) * wet)),
            }
        }


class GovernanceDissonanceNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {
                "text": PortType.TEXT,
                "embedding_a": PortType.EMBEDDING,
                "embedding_b": PortType.EMBEDDING,
                "cv": PortType.CV,
                "param:dry_wet": PortType.CV,
            },
            "outputs": {"text": PortType.TEXT, "embedding": PortType.EMBEDDING, "cv": PortType.CV},
        }

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        if not _enabled(self.params):
            return {"text": str(inputs.get("text", "")), "embedding": inputs.get("embedding_a") or [], "cv": inputs.get("cv", 0.0)}

        wet = _dry_wet(self, inputs)
        contradiction = str(self.params.get("inject", "yet also its opposite is true"))
        text = str(inputs.get("text", ""))
        text_out = text
        if text and ctx.latch_state.get("token_boundary_latch", False):
            text_out = f"{text} {contradiction[: max(4, int(len(contradiction) * wet))]}"

        emb_a = _to_embedding(inputs.get("embedding_a") or self.params.get("concept_a", ""))
        emb_b = _to_embedding(inputs.get("embedding_b") or self.params.get("concept_b", ""))
        phase = math.sin(ctx.tick / max(1, int(self.params.get("window", 8))))
        osc = (phase * 0.5 + 0.5) * wet
        emb = [(1 - osc) * emb_a[i % len(emb_a)] + osc * emb_b[i % len(emb_b)] for i in range(max(len(emb_a), len(emb_b)))]
        cv = float(inputs.get("cv", 0.0))
        return {"text": text_out, "embedding": emb, "cv": cv + (phase * wet * 0.2)}


class JusticeReweightingNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {
                "text": PortType.TEXT,
                "embedding": PortType.EMBEDDING,
                "param:dry_wet": PortType.CV,
            },
            "outputs": {"text": PortType.TEXT, "embedding": PortType.EMBEDDING, "metric": PortType.METRIC},
        }

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        del ctx
        text = str(inputs.get("text", ""))
        embedding = _to_embedding(inputs.get("embedding") or [])
        lexicon = list(self.params.get("attractor_lexicon", ["care", "justice", "repair"]))
        if not _enabled(self.params):
            density = sum(text.lower().count(word.lower()) for word in lexicon)
            return {"text": text, "embedding": embedding, "metric": {"attractor_density": float(density)}}

        wet = _dry_wet(self, inputs)
        inject_count = max(1, int(wet * 2))
        out_text = f"{text} {' '.join(lexicon[:inject_count])}".strip()
        shifted = [v + wet * 0.05 for v in embedding] if embedding else [wet * 0.05]
        density = sum(out_text.lower().count(word.lower()) for word in lexicon) / max(1, len(out_text.split()))
        return {"text": out_text, "embedding": shifted, "metric": {"attractor_density": density}}


class CompareNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {
                "baseline_text": PortType.TEXT,
                "bent_text": PortType.TEXT,
                "baseline_image": PortType.IMAGE_PATH,
                "bent_image": PortType.IMAGE_PATH,
            },
            "outputs": {"metric": PortType.METRIC},
        }

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        del ctx
        b_txt = str(inputs.get("baseline_text", ""))
        x_txt = str(inputs.get("bent_text", ""))
        divergence = abs(len(b_txt) - len(x_txt)) / max(1, len(b_txt) + len(x_txt))
        lexicon = list(self.params.get("attractor_lexicon", ["care", "justice", "repair"]))
        base_density = sum(b_txt.lower().count(w) for w in lexicon) / max(1, len(b_txt.split()) or 1)
        bent_density = sum(x_txt.lower().count(w) for w in lexicon) / max(1, len(x_txt.split()) or 1)
        refusal_words = {"cannot", "won't", "unable", "refuse"}
        base_refusal = sum(1 for w in b_txt.lower().split() if w in refusal_words)
        bent_refusal = sum(1 for w in x_txt.lower().split() if w in refusal_words)
        img_delta = float(str(inputs.get("baseline_image", "")) != str(inputs.get("bent_image", "")))
        return {
            "metric": {
                "divergence_proxy": divergence + 0.1 * img_delta,
                "attractor_density_delta": bent_density - base_density,
                "refusal_delta": float(bent_refusal - base_refusal),
            }
        }


class MixerNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        inputs: dict[str, PortType] = {"analysis_bus": PortType.METRIC}
        for i in range(1, 5):
            inputs[f"ch{i}_text"] = PortType.TEXT
            inputs[f"ch{i}_image"] = PortType.IMAGE_PATH
            inputs[f"ch{i}_metric"] = PortType.METRIC
            inputs[f"param:ch{i}_volume"] = PortType.CV
        return {"inputs": inputs, "outputs": {"text": PortType.TEXT, "image_path": PortType.IMAGE_PATH, "metric": PortType.METRIC}}

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        del ctx
        solo_channels = [i for i in range(1, 5) if bool(self.params.get(f"ch{i}_solo", False))]
        text_parts: list[str] = []
        chosen_image = ""
        channel_levels: dict[str, float] = {}
        analysis_energy = 0.0

        for i in range(1, 5):
            if bool(self.params.get(f"ch{i}_mute", False)):
                continue
            if solo_channels and i not in solo_channels:
                continue
            vol = self.modulated_numeric(f"ch{i}_volume", inputs, 1.0, clamp=(0.0, 2.0))
            channel_levels[f"ch{i}"] = vol
            txt = str(inputs.get(f"ch{i}_text", "")).strip()
            if txt:
                text_parts.append(f"[ch{i}@{vol:.2f}] {txt}")
            img = str(inputs.get(f"ch{i}_image", "")).strip()
            if img and not chosen_image:
                chosen_image = img
            if bool(self.params.get(f"ch{i}_send_analysis", True)):
                analysis_energy += vol

        metric = {"analysis_send_energy": analysis_energy, **channel_levels}
        return {"text": " | ".join(text_parts), "image_path": chosen_image, "metric": metric}


class FeedbackBusNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {"text": PortType.TEXT, "cv": PortType.CV},
            "outputs": {"text": PortType.TEXT, "metric": PortType.METRIC},
        }

    def __init__(self, node_id: str, params: dict[str, Any]) -> None:
        super().__init__(node_id, params)
        self._buffer = ""

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        if not ctx.latch_state.get("token_boundary_latch", False):
            return {}
        gate_threshold = self.modulated_numeric("gate_threshold", inputs, 0.8, clamp=(0.0, 2.0))
        feedback_cv = abs(float(inputs.get("cv", 0.0)))
        incoming = str(inputs.get("text", ""))
        if feedback_cv > gate_threshold:
            return {"metric": {"feedback_gated": 1.0, "feedback_cv": feedback_cv}}

        max_tokens = int(self.params.get("max_feedback_tokens", 12))
        merged = f"{incoming} {self._buffer}".strip()
        tokens = merged.split()[-max_tokens:]
        self._buffer = " ".join(tokens)
        return {"text": self._buffer, "metric": {"feedback_gated": 0.0, "feedback_cv": feedback_cv}}


class SovereignSwitchboardNode(BaseNode):
    @classmethod
    def declare_ports(cls) -> PortDecl:
        return {
            "inputs": {"prompt": PortType.TEXT},
            "outputs": {
                "refusal_delta_cv": PortType.CV,
                "framing_delta_cv": PortType.CV,
                "ontology_distance_delta_cv": PortType.CV,
                "metric": PortType.METRIC,
            },
        }

    def __init__(self, node_id: str, params: dict[str, Any]) -> None:
        super().__init__(node_id, params)
        self._cache_prompt = ""
        self._cache_output: dict[str, Any] = {}

    def process(self, ctx: RuntimeContext, inputs: dict[str, Any]) -> dict[str, Any]:
        if not ctx.latch_state.get("token_boundary_latch", False):
            return {}

        prompt = str(inputs.get("prompt") or self.params.get("prompt", ""))
        if not prompt:
            return {}
        if prompt == self._cache_prompt:
            return self._cache_output

        voices = list(
            self.params.get(
                "voices",
                [
                    {"name": "us_voice", "region": "NA"},
                    {"name": "eu_voice", "region": "EU"},
                    {"name": "cn_voice", "region": "APAC"},
                ],
            )
        )
        outputs: list[str] = []
        for voice in voices:
            adapter = make_llm_adapter({**self.params, **voice})
            stream = adapter.stream_generate(
                prompt,
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                max_new_tokens=8,
            )
            outputs.append("".join(token.text for token in stream))

        refusal_words = {"cannot", "won't", "unable", "refuse"}
        refusal_rates = [sum(1 for w in out.lower().split() if w in refusal_words) for out in outputs]
        framing_scores = [len(out.split()) for out in outputs]

        token_sets = [set(out.lower().split()) for out in outputs if out]
        if len(token_sets) >= 2:
            a, b = token_sets[0], token_sets[1]
            ontology_dist = 1 - (len(a.intersection(b)) / max(1, len(a.union(b))))
        else:
            ontology_dist = 0.0

        refusal_delta = float(max(refusal_rates) - min(refusal_rates))
        framing_delta = float(max(framing_scores) - min(framing_scores)) / max(1.0, float(max(framing_scores) or 1.0))

        self._cache_prompt = prompt
        self._cache_output = {
            "refusal_delta_cv": refusal_delta,
            "framing_delta_cv": framing_delta,
            "ontology_distance_delta_cv": ontology_dist,
            "metric": {
                "refusal_delta": refusal_delta,
                "framing_delta": framing_delta,
                "ontology_distance_delta": ontology_dist,
            },
        }
        return self._cache_output


NODE_TYPES: dict[str, type[BaseNode]] = {
    "PromptSourceNode": PromptSourceNode,
    "CV_LFONode": CV_LFONode,
    "CV_StepSequencerNode": CV_StepSequencerNode,
    "DummyTextGenNode": DummyTextGenNode,
    "MetricProbeNode": MetricProbeNode,
    "RecorderNode": RecorderNode,
    "LLMVoiceNode": LLMVoiceNode,
    "DiffusionVoiceNode": DiffusionVoiceNode,
    "EmbeddingContaminationNode": EmbeddingContaminationNode,
    "StratigraphySamplerNode": StratigraphySamplerNode,
    "GovernanceDissonanceNode": GovernanceDissonanceNode,
    "JusticeReweightingNode": JusticeReweightingNode,
    "CompareNode": CompareNode,
    "MixerNode": MixerNode,
    "FeedbackBusNode": FeedbackBusNode,
    "SovereignSwitchboardNode": SovereignSwitchboardNode,
}


def node_specs() -> dict[str, PortDecl]:
    return {name: cls.declare_ports() for name, cls in NODE_TYPES.items()}


class ConsoleRuntime:
    def __init__(
        self,
        patch: PatchGraph,
        *,
        tick_rate: float = 30.0,
        event_cb: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.patch = patch
        self.tick_rate = tick_rate
        self.event_cb = event_cb
        self.tick = 0
        self.running = False
        self._task: asyncio.Task[None] | None = None
        self._latest_outputs: dict[str, dict[str, Any]] = defaultdict(dict)
        self.recorder = ConsoleRecorder()
        self.nodes = {node.id: NODE_TYPES[node.type](node.id, node.params) for node in patch.nodes}
        self.execution_order = self._topological_order()

    def _topological_order(self) -> list[str]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree: dict[str, int] = {node.id: 0 for node in self.patch.nodes}
        for edge in self.patch.edges:
            if edge.edge_type == EdgeType.FEEDBACK and edge.gated:
                continue
            if edge.to_node not in adjacency[edge.from_node]:
                adjacency[edge.from_node].add(edge.to_node)
                indegree[edge.to_node] += 1
        queue = deque([n for n, deg in indegree.items() if deg == 0])
        order: list[str] = []
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for nxt in adjacency[node_id]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(indegree):
            raise ValueError("unable to determine topological order due to cycle")
        return order

    async def start(self, options: dict[str, Any] | None = None) -> None:
        if self.running:
            return
        options = options or {}
        self.tick_rate = float(options.get("tick_rate", self.tick_rate))
        self.running = True
        self.tick = 0
        self.recorder.start(self.patch.model_dump(mode="json"), options)
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self._task:
            await self._task

    async def _run_loop(self) -> None:
        while self.running:
            await self.step_once()
            await asyncio.sleep(max(0.0, 1.0 / self.tick_rate))

    async def step_once(self) -> None:
        self.tick += 1
        latch_state = {"token_boundary_latch": True, "diffusion_step_latch": self.tick % 5 == 0}
        ctx = RuntimeContext(self.tick, self.tick_rate, latch_state, self.recorder)

        for node_id in self.execution_order:
            node = self.nodes[node_id]
            patch_node = next(n for n in self.patch.nodes if n.id == node_id)
            if not patch_node.enabled:
                continue
            outputs = node.process(ctx, self._collect_inputs(node_id))
            self._latest_outputs[node_id] = outputs

            if outputs.get("text") is not None:
                text_chunk = str(outputs["text"])
                self.recorder.record_text(text_chunk)
                await self._emit({"type": "TEXT_UPDATE", "ts": datetime.now(timezone.utc).isoformat(), "text_chunk": text_chunk, "channel": node_id})

            if outputs.get("image_path") is not None:
                await self._emit({"type": "IMAGE_UPDATE", "ts": datetime.now(timezone.utc).isoformat(), "image_path": str(outputs["image_path"]), "channel": node_id})

            if outputs.get("metric") is not None:
                payload = {"type": "METRIC_UPDATE", "ts": datetime.now(timezone.utc).isoformat(), "metrics": outputs["metric"]}
                self.recorder.record_metric(payload)
                await self._emit(payload)

        await self._emit({"type": "RUNTIME_STATUS", "running": self.running, "tick": self.tick, "latch_state": latch_state})

    def _collect_inputs(self, node_id: str) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for edge in self.patch.edges:
            if edge.to_node != node_id:
                continue
            source_outputs = self._latest_outputs.get(edge.from_node, {})
            if edge.from_port in source_outputs:
                inputs[edge.to_port] = source_outputs[edge.from_port]
        return inputs

    async def _emit(self, payload: dict[str, Any]) -> None:
        if self.event_cb:
            result = self.event_cb(payload)
            if asyncio.iscoroutine(result):
                await result

    def update_param(self, node_id: str, param: str, value: Any) -> None:
        for node in self.patch.nodes:
            if node.id == node_id:
                node.params[param] = value
                self.nodes[node_id].on_patch_update(node.params)
                return
        raise KeyError(f"unknown node_id '{node_id}'")
