"""StyleGAN2/3 adapter with latent sampling and traversal utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neural_bending_toolkit.models.torch_device import normalize_torch_device


class StyleGANAdapter:
    """Wrapper for loading and running StyleGAN-style generators."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        latent_dim: int = 512,
        truncation_psi: float = 1.0,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required for StyleGANAdapter. "
                "Install with: pip install .[gan]"
            ) from exc

        self._torch = torch
        self.device = normalize_torch_device(device)
        self.latent_dim = latent_dim
        self.truncation_psi = truncation_psi
        self.generator = self._load_generator(Path(checkpoint_path))

    def _load_generator(self, checkpoint_path: Path) -> Any:
        """Load generator from local checkpoint path."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        torch = self._torch
        loaded = torch.load(checkpoint_path, map_location=self.device)

        if hasattr(loaded, "eval"):
            generator = loaded
        elif isinstance(loaded, dict) and "generator" in loaded:
            generator = loaded["generator"]
        elif isinstance(loaded, dict) and "G_ema" in loaded:
            generator = loaded["G_ema"]
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected module or dict with "
                "'generator'/'G_ema'."
            )

        generator = generator.to(self.device)
        generator.eval()
        return generator

    def sample_latents(self, n: int, seed: int | None = None) -> Any:
        """Sample latent vectors z from a standard normal distribution."""
        torch = self._torch
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
            z = torch.randn((n, self.latent_dim), generator=g, device=self.device)
        else:
            z = torch.randn((n, self.latent_dim), device=self.device)
        return z

    @staticmethod
    def linear_interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation between vectors."""
        return (1.0 - t) * a + t * b

    @staticmethod
    def slerp(a: np.ndarray, b: np.ndarray, t: float, eps: float = 1e-8) -> np.ndarray:
        """Spherical interpolation between vectors."""
        a_norm = a / (np.linalg.norm(a) + eps)
        b_norm = b / (np.linalg.norm(b) + eps)
        dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
        theta = np.arccos(dot) * t
        rel = b_norm - a_norm * dot
        rel = rel / (np.linalg.norm(rel) + eps)
        return a_norm * np.cos(theta) + rel * np.sin(theta)

    def traverse_direction(
        self,
        z: Any,
        direction: Any,
        *,
        scale: float,
        toward: bool = True,
    ) -> Any:
        """Move a latent vector toward/away from a target direction."""
        sign = 1.0 if toward else -1.0
        return z + sign * scale * direction

    def generate(self, z: Any, truncation_psi: float | None = None) -> Any:
        """Generate image tensors from latent vectors."""
        psi = self.truncation_psi if truncation_psi is None else truncation_psi
        torch = self._torch
        with torch.no_grad():
            try:
                images = self.generator(z, truncation_psi=psi, noise_mode="const")
            except TypeError:
                images = self.generator(z)
        return images

    @staticmethod
    def to_uint8_images(images: Any) -> list[np.ndarray]:
        """Convert generator output tensor in [-1, 1] to uint8 HWC arrays."""
        arr = images.detach().cpu().numpy()
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        arr = np.transpose(arr, (0, 2, 3, 1))
        return [img for img in arr]
