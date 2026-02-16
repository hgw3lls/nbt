import json
from pathlib import Path

import numpy as np

from neural_bending_toolkit.analysis.attention import attention_entropy, js_divergence
from neural_bending_toolkit.analysis.coherence import (
    repetition_score,
    self_consistency_score,
    temporal_reference_stability,
)
from neural_bending_toolkit.analysis.distributions import kl_divergence
from neural_bending_toolkit.analysis.embeddings import compute_pca_projection
from neural_bending_toolkit.analysis.images import perceptual_hash
from neural_bending_toolkit.analysis.report import generate_markdown_report


def test_distribution_and_attention_metrics_basic() -> None:
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.2, 0.3, 0.5])
    assert abs(kl_divergence(p, q)) < 1e-10
    assert abs(js_divergence(p, q)) < 1e-10

    ent = attention_entropy(np.array([[0.5, 0.5]], dtype=np.float64))
    assert ent.shape == (1,)


def test_coherence_proxies() -> None:
    texts = ["In 2024 the model repeats repeats tokens", "In 2024 tokens stabilize"]
    assert 0.0 <= self_consistency_score(texts) <= 1.0
    assert repetition_score(texts[0]) > 0.0
    assert 0.0 <= temporal_reference_stability(texts) <= 1.0


def test_embedding_projection_and_hash() -> None:
    emb = np.random.default_rng(1).normal(size=(10, 6))
    proj = compute_pca_projection(emb)
    assert proj.shape == (10, 2)

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    hsh = perceptual_hash(img)
    assert isinstance(hsh, str)


def test_report_generation(tmp_path: Path) -> None:
    run_dir = tmp_path
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sample_txt = artifacts_dir / "sample.txt"
    sample_txt.write_text("In 2025 a stable output", encoding="utf-8")

    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"value": 0.2, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps({"payload": {}}) + "\n",
        encoding="utf-8",
    )

    out = generate_markdown_report(run_dir)

    assert out.exists()
    assert (artifacts_dir / "analysis" / "summary.json").exists()
