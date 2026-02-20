"""Analysis utilities for post-run evaluation and report generation."""

from neural_bending_toolkit.analysis.attention import attention_entropy, js_divergence
from neural_bending_toolkit.analysis.bend_classifier import (
    classify_bend,
    write_bend_classification,
)
from neural_bending_toolkit.analysis.bend_tagging import (
    score_and_tag_metastability,
    write_comparison_report,
)
from neural_bending_toolkit.analysis.coherence import (
    repetition_score,
    self_consistency_score,
    temporal_reference_stability,
)
from neural_bending_toolkit.analysis.derived_metrics import (
    compute_derived_metrics,
    robust_median_iqr,
    robust_scale,
    write_derived_metrics,
)
from neural_bending_toolkit.analysis.distributions import kl_divergence
from neural_bending_toolkit.analysis.figures_metastability import plot_entropy_over_steps
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
from neural_bending_toolkit.analysis.metastability import (
    basin_shift_proxy,
    compute_attention_entropy_profile,
    compute_attention_topk_mass_profile,
    concentration_collapse_index,
    recovery_index,
)
from neural_bending_toolkit.analysis.report import (
    generate_html_report,
    generate_markdown_report,
)
from neural_bending_toolkit.analysis.theory_memo_generator import (
    build_theory_memo,
    build_theory_memos_for_runs,
)

__all__ = [
    "attention_entropy",
    "attractor_density",
    "basin_shift_proxy",
    "build_theory_memo",
    "build_theory_memos_for_runs",
    "classify_bend",
    "compare_geopolitical_runs",
    "compute_attention_entropy_profile",
    "compute_attention_topk_mass_profile",
    "compute_derived_metrics",
    "compute_pca_projection",
    "compute_umap_projection",
    "concentration_collapse_index",
    "cosine_similarity_matrix",
    "detect_refusal",
    "generate_geopolitical_report",
    "generate_html_report",
    "generate_markdown_report",
    "image_diversity_lpips_or_hash",
    "js_divergence",
    "kl_divergence",
    "perceptual_hash",
    "plot_entropy_over_steps",
    "recovery_index",
    "repetition_score",
    "robust_median_iqr",
    "robust_scale",
    "self_consistency_score",
    "score_and_tag_metastability",
    "structural_causality_score",
    "temporal_reference_stability",
    "write_bend_classification",
    "write_comparison_report",
    "write_derived_metrics",
]
