from neural_bending_toolkit.registry import ExperimentRegistry


def test_registry_discovers_builtin_experiments() -> None:
    registry = ExperimentRegistry()
    registry.discover_modules()

    experiments = registry.list_experiments()

    assert "hello-experiment" in experiments
    assert "llm-sampling-stratigraphy" in experiments
    assert "embedding-contamination-diffusion" in experiments
    assert "gan-stratigraphy-edges" in experiments
    assert "audio-inter-head-drift" in experiments
    assert "family-embedding-contamination" in experiments
    assert "family-corpus-stratigraphy" in experiments
    assert "family-inter-head-drift" in experiments
    assert "family-governance-dissonance" in experiments
    assert "family-residual-distortion" in experiments
    assert "family-norm-perturbation" in experiments
    assert "family-justice-reweighting" in experiments
    assert "family-justice-attractors" in experiments
    assert "geopolitical-bend" in experiments
    assert "bend-family" not in experiments
