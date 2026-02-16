"""Explore GAN samples near truncation edges and compare diversity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import Field

from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


def average_pairwise_l2(images: list[np.ndarray]) -> float:
    """Simple diversity metric: mean pairwise L2 between flattened images."""
    if len(images) < 2:
        return 0.0
    flattened = [img.astype(np.float32).reshape(-1) for img in images]
    distances: list[float] = []
    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            distances.append(float(np.linalg.norm(flattened[i] - flattened[j])))
    return float(np.mean(distances)) if distances else 0.0


def make_montage(images: list[np.ndarray], cols: int = 4) -> np.ndarray:
    """Create a grid montage from uint8 HWC images."""
    if not images:
        raise ValueError("No images provided for montage")

    rows = int(np.ceil(len(images) / cols))
    h, w, c = images[0].shape
    canvas = np.zeros((rows * h, cols * w, c), dtype=np.uint8)

    for idx, image in enumerate(images):
        r = idx // cols
        col = idx % cols
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = image

    return canvas


class GANStratigraphyEdgesConfig(ExperimentSettings):
    """Config for GAN edge-truncation sampling experiment."""

    checkpoint_path: str
    latent_dim: int = 512
    n_samples: int = Field(default=8, ge=2, le=64)
    low_truncation: float = Field(default=0.3, ge=0.0, le=2.0)
    high_truncation: float = Field(default=1.4, ge=0.0, le=2.0)
    seed: int = 17


class GANStratigraphyEdges(Experiment):
    """Sample low/high truncation edges, compare diversity, and save montages."""

    name = "gan-stratigraphy-edges"
    config_model = GANStratigraphyEdgesConfig

    def _load_adapter(self):
        from neural_bending_toolkit.models.gan_stylegan import StyleGANAdapter

        return StyleGANAdapter(
            checkpoint_path=Path(self.config.checkpoint_path),
            device="cpu",
            latent_dim=self.config.latent_dim,
        )

    def run(self, context: RunContext) -> None:
        adapter = self._load_adapter()
        z = adapter.sample_latents(self.config.n_samples, seed=self.config.seed)

        low_images_tensor = adapter.generate(
            z,
            truncation_psi=self.config.low_truncation,
        )
        high_images_tensor = adapter.generate(
            z,
            truncation_psi=self.config.high_truncation,
        )

        low_images = adapter.to_uint8_images(low_images_tensor)
        high_images = adapter.to_uint8_images(high_images_tensor)

        low_div = average_pairwise_l2(low_images)
        high_div = average_pairwise_l2(high_images)

        context.log_metric(
            step=1,
            metric_name="diversity_low_truncation",
            value=low_div,
            metadata={"truncation": self.config.low_truncation},
        )
        context.log_metric(
            step=2,
            metric_name="diversity_high_truncation",
            value=high_div,
            metadata={"truncation": self.config.high_truncation},
        )

        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Pillow is required for GAN montage artifacts. "
                "Install with .[images] or .[gan]"
            ) from exc

        low_montage = make_montage(low_images)
        high_montage = make_montage(high_images)

        low_path = context.artifacts_dir / "gan_low_truncation_montage.png"
        high_path = context.artifacts_dir / "gan_high_truncation_montage.png"
        Image.fromarray(low_montage).save(low_path)
        Image.fromarray(high_montage).save(high_path)

        context.log_event(
            "Saved GAN montage artifacts",
            low_truncation=self.config.low_truncation,
            high_truncation=self.config.high_truncation,
            low_diversity=low_div,
            high_diversity=high_div,
        )
