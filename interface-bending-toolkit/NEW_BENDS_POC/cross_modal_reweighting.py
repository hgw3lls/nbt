"""
Script to sketch a cross-modal reweighting bend for a CLIP-style text-image model.

This script demonstrates how to preferentially surface images from a subaltern
archive for a given text prompt by boosting their similarity scores relative to
images from a dominant archive. It uses OpenCLIP via the `open_clip` package to
load a pretrained text-image model.

Usage example:
    python cross_modal_reweighting.py --text "scientist" --beta 1.8 --top-k 3

If you do not have real subaltern or dominant image folders handy, the script
will generate placeholder images so that it can run end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

import open_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal reweighting demo")
    parser.add_argument(
        "--text",
        default="scientist",
        help="Text prompt to query (e.g., 'scientist', 'family', 'protest').",
    )
    parser.add_argument(
        "--dominant-dir",
        type=Path,
        default=Path("dominant_images"),
        help="Path to folder with dominant or stock imagery.",
    )
    parser.add_argument(
        "--subaltern-dir",
        type=Path,
        default=Path("subaltern_images"),
        help="Path to folder with subaltern/community archives.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.6,
        help="Reweighting multiplier (>1) for subaltern images.",
    )
    parser.add_argument(
        "--dominant-factor",
        type=float,
        default=1.0,
        help="Optional down-weighting factor (<1) for dominant images.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many images to display before/after reweighting.",
    )
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="OpenCLIP model name (e.g., ViT-B-32).",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained tag.",
    )
    return parser.parse_args()


def ensure_placeholder_images(directory: Path, label: str, colors: Iterable[Tuple[int, int, int]]):
    """Create simple placeholder images if the directory is empty.

    These placeholders allow the script to run without real data. Replace with
    actual archives when available.
    """

    directory.mkdir(parents=True, exist_ok=True)
    existing_images = list(directory.glob("*.png"))
    if existing_images:
        return

    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
    except OSError:
        # Optional: if the font is unavailable, Pillow will fall back to default.
        pass

    for idx, color in enumerate(colors):
        img = Image.new("RGB", (320, 320), color=color)
        draw = ImageDraw.Draw(img)
        text = f"{label} {idx + 1}"
        draw.text((20, 140), text, fill=(255, 255, 255), font=font)
        img.save(directory / f"{label.lower()}_{idx + 1}.png")


def load_and_preprocess_images(image_dir: Path, preprocess: T.Compose) -> List[Tuple[str, torch.Tensor]]:
    images: List[Tuple[str, torch.Tensor]] = []
    for image_path in sorted(image_dir.glob("*.png")):
        with Image.open(image_path).convert("RGB") as img:
            images.append((image_path.name, preprocess(img)))
    return images


def encode_images(model, images: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    if not images:
        return torch.empty((0, model.visual.output_dim), device=device)
    batch = torch.stack(images).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        embeddings = model.encode_image(batch)
    return torch.nn.functional.normalize(embeddings, dim=-1)


def encode_text(model, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    tokenized = tokenizer([text]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        text_features = model.encode_text(tokenized)
    return torch.nn.functional.normalize(text_features, dim=-1)


def compute_similarities(text_emb: torch.Tensor, image_embs: torch.Tensor) -> torch.Tensor:
    if image_embs.numel() == 0:
        return torch.empty((0,), device=text_emb.device)
    return (text_emb @ image_embs.T).squeeze(0)


def reweight_scores(
    scores: torch.Tensor,
    is_subaltern: List[bool],
    beta: float,
    dominant_factor: float,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    weights = torch.tensor(
        [beta if sub else dominant_factor for sub in is_subaltern], device=scores.device
    )
    reweighted = scores * weights
    # Renormalize to keep scores in a comparable range.
    reweighted = torch.nn.functional.normalize(reweighted.unsqueeze(0), p=2, dim=1).squeeze(0)
    return reweighted


def top_k(scores: torch.Tensor, names: List[str], k: int) -> List[Tuple[str, float]]:
    if scores.numel() == 0:
        return []
    k = min(k, len(names))
    values, indices = torch.topk(scores, k)
    return [(names[i], float(values[j])) for j, i in enumerate(indices)]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Replace placeholder images with real subaltern archives and dominant sources.
    ensure_placeholder_images(
        args.dominant_dir, "Dominant", colors=[(200, 200, 200), (160, 160, 160), (120, 120, 120)]
    )
    ensure_placeholder_images(
        args.subaltern_dir,
        "Subaltern",
        colors=[(220, 180, 180), (180, 220, 180), (180, 180, 220)],
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    dominant = load_and_preprocess_images(args.dominant_dir, preprocess)
    subaltern = load_and_preprocess_images(args.subaltern_dir, preprocess)

    dominant_names, dominant_tensors = zip(*dominant) if dominant else ([], [])
    subaltern_names, subaltern_tensors = zip(*subaltern) if subaltern else ([], [])

    text_emb = encode_text(model, tokenizer, args.text, device)
    dominant_embs = encode_images(model, list(dominant_tensors), device)
    subaltern_embs = encode_images(model, list(subaltern_tensors), device)

    all_names = list(dominant_names) + list(subaltern_names)
    all_embs = (
        torch.cat([dominant_embs, subaltern_embs], dim=0)
        if dominant_embs.numel() or subaltern_embs.numel()
        else torch.empty((0, model.visual.output_dim), device=device)
    )
    is_subaltern = [False] * len(dominant_names) + [True] * len(subaltern_names)

    base_scores = compute_similarities(text_emb, all_embs)
    print("\n=== Base ranking ===")
    for name, score in top_k(base_scores, all_names, args.top_k):
        print(f"{name}\t{score:.4f}")

    adjusted_scores = reweight_scores(base_scores, is_subaltern, args.beta, args.dominant_factor)
    print("\n=== Reweighted ranking (subaltern boosted) ===")
    for name, score in top_k(adjusted_scores, all_names, args.top_k):
        print(f"{name}\t{score:.4f}")


if __name__ == "__main__":
    main()
