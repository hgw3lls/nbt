"""Analysis utilities for post-run evaluation and report generation."""

from neural_bending_toolkit.analysis.attention import attention_entropy, js_divergence
from neural_bending_toolkit.analysis.coherence import (
    repetition_score,
    self_consistency_score,
    temporal_reference_stability,
)
from neural_bending_toolkit.analysis.distributions import kl_divergence
from neural_bending_toolkit.analysis.embeddings import (
    compute_pca_projection,
    compute_umap_projection,
)
from neural_bending_toolkit.analysis.geopolitical_report import (
    compare_geopolitical_runs,
    generate_geopolitical_report,
)
from neural_bending_toolkit.analysis.geopolitical_utils import (
    attractor_density,
    cosine_similarity_matrix,
    detect_refusal,
    structural_causality_score,
)
from neural_bending_toolkit.analysis.images import (
    image_diversity_lpips_or_hash,
    perceptual_hash,
)
from neural_bending_toolkit.analysis.report import generate_markdown_report

__all__ = [
    "attention_entropy",
    "compare_geopolitical_runs",
    "compute_pca_projection",
    "compute_umap_projection",
    "cosine_similarity_matrix",
    "detect_refusal",
    "generate_geopolitical_report",
    "generate_markdown_report",
    "image_diversity_lpips_or_hash",
    "js_divergence",
    "kl_divergence",
    "attractor_density",
    "perceptual_hash",
    "repetition_score",
    "self_consistency_score",
    "structural_causality_score",
    "temporal_reference_stability",
]
