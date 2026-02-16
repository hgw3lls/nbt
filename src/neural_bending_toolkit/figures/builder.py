"""Figure building from run artifacts and strict YAML specs."""

from __future__ import annotations

import csv
import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from neural_bending_toolkit.figures.specs import FigureSpec, PlotType, load_figure_spec


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_run_dirs(patterns: list[str], base: Path | None = None) -> list[Path]:
    resolved: list[Path] = []
    root = base or Path.cwd()
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if not matches:
            candidate = (root / pattern).resolve()
            if candidate.exists():
                matches = [candidate]
        for path in matches:
            if path.is_dir():
                resolved.append(path.resolve())
    unique = sorted(set(resolved))
    if not unique:
        raise ValueError("No run directories matched input_run_dirs patterns.")
    return unique


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _save_figure(fig: Any, stem: Path, fmt_png: bool, fmt_pdf: bool) -> list[Path]:
    outputs: list[Path] = []
    if fmt_png:
        png = stem.with_suffix(".png")
        fig.savefig(png)
        outputs.append(png)
    if fmt_pdf:
        pdf = stem.with_suffix(".pdf")
        fig.savefig(pdf)
        outputs.append(pdf)
    return outputs


def _plot_embedding_similarity_heatmap(
    ax: Any, run_dirs: list[Path], inputs: dict[str, Any]
) -> None:
    csv_pattern = str(
        inputs.get(
            "similarity_csv",
            "artifacts/geopolitical/phase_1_ontology_mapping/*_similarity.csv",
        )
    )
    for run_dir in run_dirs:
        candidates = sorted(run_dir.glob(csv_pattern))
        if candidates:
            csv_path = candidates[0]
            break
    else:
        raise ValueError("No similarity CSV found for embedding heatmap.")

    labels: list[str] = []
    values: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            labels.append(row[0])
            values.append([float(x) for x in row[1:]])

    matrix = np.asarray(values, dtype=np.float64)
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_embedding_umap_scatter(
    ax: Any, run_dirs: list[Path], inputs: dict[str, Any]
) -> None:
    npy_pattern = str(
        inputs.get(
            "umap_npy", "artifacts/geopolitical/phase_1_ontology_mapping/*_umap.npy"
        )
    )
    for run_dir in run_dirs:
        candidates = sorted(run_dir.glob(npy_pattern))
        if candidates:
            arr = np.load(candidates[0])
            break
    else:
        raise ValueError("No UMAP .npy found for scatter plot.")

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("UMAP array must be 2D with at least 2 columns.")
    ax.scatter(arr[:, 0], arr[:, 1])
    for idx in range(arr.shape[0]):
        ax.annotate(str(idx), (arr[idx, 0], arr[idx, 1]))


def _entropy(values: np.ndarray, eps: float = 1e-12) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = np.clip(v, eps, None)
    v = v / np.sum(v)
    return float(-np.sum(v * np.log(v)))


def _plot_attention_entropy_timeseries(
    ax: Any, run_dirs: list[Path], inputs: dict[str, Any]
) -> None:
    event_pattern = str(inputs.get("events_jsonl", "events.jsonl"))
    entropies: list[float] = []
    for run_dir in run_dirs:
        path = run_dir / event_pattern
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            payload = event.get("payload", {})
            attn = payload.get("attention") if isinstance(payload, dict) else None
            if attn is not None:
                arr = np.asarray(attn, dtype=np.float64).ravel()
                if arr.size > 0:
                    entropies.append(_entropy(arr))
    if not entropies:
        raise ValueError(
            "No attention payloads found in events for entropy timeseries."
        )
    ax.plot(range(len(entropies)), entropies)
    ax.set_xlabel("Index")
    ax.set_ylabel("Entropy")


def _plot_divergence_bar_chart(
    ax: Any, run_dirs: list[Path], inputs: dict[str, Any]
) -> None:
    metric_names = inputs.get("metric_names", ["distribution_kl", "js_divergence"])
    metrics = _load_metrics(run_dirs[0])
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for row in metrics:
        metric_name = str(row.get("metric_name", ""))
        if metric_name in values:
            values[metric_name].append(float(row.get("value", 0.0)))
    means = [
        float(np.mean(values[name])) if values[name] else 0.0 for name in metric_names
    ]
    ax.bar(metric_names, means)
    ax.set_ylabel("Mean value")


def _plot_refusal_rate_table_to_figure(
    ax: Any, run_dirs: list[Path], inputs: dict[str, Any]
) -> None:
    json_pattern = str(
        inputs.get(
            "phase2_json",
            "artifacts/geopolitical/phase_2_governance_dissonance/governance_dissonance_results.json",
        )
    )
    rows = _read_json(run_dirs[0] / json_pattern, default=[])
    if not isinstance(rows, list):
        rows = []
    flags: list[bool] = []
    for row in rows:
        flags.extend(
            [bool(row.get("refusal_a", False)), bool(row.get("refusal_b", False))]
        )
    refusal_rate = float(np.mean(flags)) if flags else 0.0

    ax.axis("off")
    table = ax.table(
        cellText=[[str(len(rows)), f"{refusal_rate:.4f}"]],
        colLabels=["Prompt Pairs", "Refusal Rate"],
        loc="center",
    )
    table.scale(1.2, 1.6)


def _plot_causal_framing_bar_chart(
    ax: Any, run_dirs: list[Path], inputs: dict[str, Any]
) -> None:
    json_pattern = str(
        inputs.get(
            "phase2_json",
            "artifacts/geopolitical/phase_2_governance_dissonance/governance_dissonance_results.json",
        )
    )
    rows = _read_json(run_dirs[0] / json_pattern, default=[])
    if not isinstance(rows, list) or not rows:
        raise ValueError("No phase 2 rows found for causal framing bar chart.")

    structural_scores: list[float] = []
    individual_scores: list[float] = []
    for row in rows:
        sa = float(row.get("structural_score_a", 0.5))
        sb = float(row.get("structural_score_b", 0.5))
        structural_scores.extend([sa, sb])
        individual_scores.extend([1.0 - sa, 1.0 - sb])

    values = [float(np.mean(structural_scores)), float(np.mean(individual_scores))]
    ax.bar(["structural", "individual"], values)
    ax.set_ylim(0.0, 1.0)


def _plot_attractor_density_comparison(
    ax: Any, run_dirs: list[Path], inputs: dict[str, Any]
) -> None:
    json_pattern = str(
        inputs.get(
            "phase3_json",
            "artifacts/geopolitical/phase_3_justice_attractors/justice_attractor_results.json",
        )
    )
    rows = _read_json(run_dirs[0] / json_pattern, default=[])
    if not isinstance(rows, list) or not rows:
        raise ValueError("No phase 3 rows found for attractor density comparison.")

    baseline = [float(row.get("baseline_density", 0.0)) for row in rows]
    bent = [float(row.get("bent_density", 0.0)) for row in rows]
    x = np.arange(len(rows))
    width = 0.35
    ax.bar(x - width / 2, baseline, width, label="baseline")
    ax.bar(x + width / 2, bent, width, label="bent")
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.legend()


def _read_wav_as_float(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wavf:
        n_frames = wavf.getnframes()
        n_channels = wavf.getnchannels()
        sampwidth = wavf.getsampwidth()
        frames = wavf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is supported for montage waveform plots.")
    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr / 32768.0


def _plot_montage_grid(ax: Any, run_dirs: list[Path], inputs: dict[str, Any]) -> None:
    image_pattern = str(inputs.get("image_glob", "artifacts/**/*.png"))
    wav_pattern = str(inputs.get("audio_glob", "artifacts/**/*.wav"))

    images: list[Path] = []
    wavs: list[Path] = []
    for run_dir in run_dirs:
        images.extend(sorted(run_dir.glob(image_pattern)))
        wavs.extend(sorted(run_dir.glob(wav_pattern)))

    if images:
        max_items = int(inputs.get("max_items", 6))
        selected = images[:max_items]
        cols = int(math.ceil(math.sqrt(len(selected))))
        rows = int(math.ceil(len(selected) / cols))
        ax.axis("off")
        fig = ax.figure
        fig.clear()
        sub_axes = fig.subplots(rows, cols)
        if not isinstance(sub_axes, np.ndarray):
            sub_axes = np.asarray([sub_axes])
        flat_axes = list(sub_axes.ravel())
        for idx, sub_ax in enumerate(flat_axes):
            if idx < len(selected):
                img = plt.imread(selected[idx])
                sub_ax.imshow(img)
                sub_ax.set_title(selected[idx].name)
            sub_ax.axis("off")
        return

    if wavs:
        signal = _read_wav_as_float(wavs[0])
        ax.plot(signal)
        ax.set_title(wavs[0].name)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        return

    raise ValueError("No images or audio files found for montage_grid plot.")


_PLOTTERS = {
    PlotType.EMBEDDING_SIMILARITY_HEATMAP: _plot_embedding_similarity_heatmap,
    PlotType.EMBEDDING_UMAP_SCATTER: _plot_embedding_umap_scatter,
    PlotType.ATTENTION_ENTROPY_TIMESERIES: _plot_attention_entropy_timeseries,
    PlotType.DIVERGENCE_BAR_CHART: _plot_divergence_bar_chart,
    PlotType.REFUSAL_RATE_TABLE_TO_FIGURE: _plot_refusal_rate_table_to_figure,
    PlotType.CAUSAL_FRAMING_BAR_CHART: _plot_causal_framing_bar_chart,
    PlotType.ATTRACTOR_DENSITY_COMPARISON: _plot_attractor_density_comparison,
    PlotType.MONTAGE_GRID: _plot_montage_grid,
}


def _caption_from_spec(spec: FigureSpec, run_dirs: list[Path], repo_root: Path) -> str:
    vars_map = spec.caption_template_variables
    template_path = repo_root / "templates" / "figure_caption.md"
    template_text = ""
    if template_path.exists():
        template_text = template_path.read_text(encoding="utf-8")

    lines = [
        f"# Caption — {spec.figure_id}",
        "",
        f"**Title:** {spec.title}",
        f"**Plot type:** {spec.plot_type.value}",
        "",
        "## Template Variables",
    ]
    if vars_map:
        lines.extend(f"- {k}: {v}" for k, v in vars_map.items())
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Run Sources",
            *[f"- `{path}`" for path in run_dirs],
        ]
    )
    if template_text:
        lines.extend(["", "## Caption Style Guide Template", "", template_text])
    return "\n".join(lines) + "\n"


def _git_commit_hash(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    head_content = head.read_text(encoding="utf-8").strip()
    if head_content.startswith("ref:"):
        ref_path = repo_root / ".git" / head_content.split(" ", 1)[1]
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return head_content


def build_figure_from_spec(spec_path: Path, repo_root: Path | None = None) -> Path:
    """Build a figure from one YAML spec and return its output directory."""
    spec = load_figure_spec(spec_path)
    root = repo_root or _repo_root()
    run_dirs = _resolve_run_dirs(spec.input_run_dirs, base=root)

    out_dir = root / "dissertation" / "figures" / spec.figure_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotter = _PLOTTERS[spec.plot_type]
    plotter(ax, run_dirs, spec.inputs)
    ax.set_title(spec.title)
    fig.tight_layout()

    stem = out_dir / spec.figure_id
    outputs = _save_figure(
        fig,
        stem,
        fmt_png=spec.output_format.png,
        fmt_pdf=spec.output_format.pdf,
    )
    plt.close(fig)

    caption = _caption_from_spec(spec, run_dirs, root)
    (out_dir / "caption.md").write_text(caption, encoding="utf-8")

    provenance = {
        "figure_id": spec.figure_id,
        "spec_path": str(spec_path.resolve()),
        "run_dirs": [str(path) for path in run_dirs],
        "outputs": [str(path) for path in outputs],
        "timestamp_utc": _utc_now(),
        "git_commit": _git_commit_hash(root),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )

    return out_dir


def build_figures_from_run(run_dir: Path, repo_root: Path | None = None) -> list[Path]:
    """Build all figure specs found in run_dir/figure_specs."""
    run_dir = run_dir.resolve()
    specs_dir = run_dir / "figure_specs"
    if not specs_dir.exists():
        raise ValueError(f"No figure_specs directory found in run: {run_dir}")

    spec_files = sorted(list(specs_dir.glob("*.yaml")) + list(specs_dir.glob("*.yml")))
    if not spec_files:
        raise ValueError(f"No figure spec files found in: {specs_dir}")

    outputs = [build_figure_from_spec(path, repo_root=repo_root) for path in spec_files]
    return outputs
