"""Image diversity proxies (LPIPS optional, perceptual hash fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def perceptual_hash(image: np.ndarray) -> str:
    """Simple difference-hash-like proxy from grayscale image array."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr.astype(np.float32)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return "0"
    h = arr[:8, :9]
    diff = h[:, 1:] > h[:, :-1]
    bits = "".join("1" if b else "0" for b in diff.flatten())
    return hex(int(bits, 2))[2:]


def _hamming(a: str, b: str) -> int:
    max_len = max(len(a), len(b))
    aa = a.zfill(max_len)
    bb = b.zfill(max_len)
    return sum(ch1 != ch2 for ch1, ch2 in zip(aa, bb, strict=False))


def image_diversity_lpips_or_hash(images: list[np.ndarray]) -> dict[str, Any]:
    """Use LPIPS when available, else use perceptual-hash dispersion."""
    if len(images) < 2:
        return {"method": "hash", "score": 0.0}

    try:
        import lpips
        import torch

        model = lpips.LPIPS(net="alex")
        pairs: list[float] = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                a = torch.tensor(images[i]).permute(2, 0, 1).unsqueeze(0).float()
                b = torch.tensor(images[j]).permute(2, 0, 1).unsqueeze(0).float()
                a = a / 127.5 - 1
                b = b / 127.5 - 1
                pairs.append(float(model(a, b).item()))
        return {"method": "lpips", "score": float(np.mean(pairs))}
    except Exception:
        hashes = [perceptual_hash(img) for img in images]
        dists: list[float] = []
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                dists.append(float(_hamming(hashes[i], hashes[j])))
        return {"method": "hash", "score": float(np.mean(dists) if dists else 0.0)}


def load_image_arrays(paths: list[Path]) -> list[np.ndarray]:
    """Load image arrays using Pillow if available."""
    from PIL import Image

    return [np.asarray(Image.open(path).convert("RGB")) for path in paths]
