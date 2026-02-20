"""Figure helpers for metastability-focused analyses."""

from __future__ import annotations

from pathlib import Path


def plot_entropy_over_steps(
    entropy_profiles: dict[str, list[float]],
    output_path: Path,
) -> Path:
    """Plot baseline/shock/shock_counter entropy trajectories over denoising steps."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))
    ordered = ["baseline", "shock", "shock_counter"]
    for condition in ordered:
        profile = list(entropy_profiles.get(condition, []))
        if not profile:
            continue
        x = list(range(len(profile)))
        plt.plot(x, profile, label=condition)

    plt.xlabel("Step")
    plt.ylabel("Attention entropy")
    plt.title("Attention entropy over steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
