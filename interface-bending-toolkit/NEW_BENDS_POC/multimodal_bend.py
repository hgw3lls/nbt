"""multimodal_bend.py - circuit-bending CLIP alignment.

Simulates subaltern alignment or deliberate mismatch between text and image embeddings
without touching pretrained weights permanently. Uses a direction vector derived from
subaltern phrases to steer text embeddings or rotates pairings to create controlled
misalignment. Logs top-k matches for baseline and bent runs.
"""
import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circuit-bend CLIP text-image alignment")
    parser.add_argument(
        "--text_prompts",
        type=str,
        required=True,
        help="Comma-separated phrases describing intended prompts",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Folder containing images to score against prompts",
    )
    parser.add_argument(
        "--mode",
        choices=["subaltern_align", "mismatch"],
        required=True,
        help="Bend mode: alignment boost or deliberate mismatch",
    )
    parser.add_argument(
        "--subaltern_phrases",
        type=str,
        default="care,repair,community",
        help="Comma-separated phrases acting as alignment boosters",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="How many best matches to show per prompt",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Bend strength scaling toward subaltern direction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable mismatch rotations",
    )
    return parser.parse_args()


def load_images(folder: str) -> List[Tuple[str, Image.Image]]:
    paths = sorted(
        [
            p
            for p in Path(folder).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        ]
    )
    images: List[Tuple[str, Image.Image]] = []
    for p in paths:
        try:
            images.append((p.name, Image.open(p).convert("RGB")))
        except Exception as exc:  # pragma: no cover - best effort loading
            print(f"[warn] could not load {p}: {exc}")
    return images


def normalize_features(feats: torch.Tensor) -> torch.Tensor:
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def compute_direction(
    model: CLIPModel, processor: CLIPProcessor, phrases: Sequence[str], device: torch.device
) -> torch.Tensor:
    if not phrases:
        return torch.zeros(model.config.projection_dim, device=device)
    tokens = processor(text=list(phrases), padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        sub_feats = model.get_text_features(**tokens)
    return normalize_features(sub_feats).mean(dim=0)


def rotate_embeddings(feats: torch.Tensor) -> torch.Tensor:
    if feats.size(0) <= 1:
        return feats
    return torch.roll(feats, shifts=1, dims=0)


def rank_images(
    text_feats: torch.Tensor, image_feats: torch.Tensor, top_k: int
) -> List[List[int]]:
    sims = text_feats @ image_feats.T
    top = torch.topk(sims, k=min(top_k, sims.size(1)), dim=1).indices
    return top.tolist()


def run_alignment(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    text_prompts: List[str],
    images: List[Tuple[str, Image.Image]],
    mode: str,
    subaltern_phrases: List[str],
    alpha: float,
    top_k: int,
) -> Dict[str, List[List[int]]]:
    if not images:
        raise ValueError("No images found for scoring.")

    image_inputs = processor(images=[img for _, img in images], return_tensors="pt").to(device)
    text_inputs = processor(text=text_prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        img_feats = normalize_features(model.get_image_features(**image_inputs))
        text_feats = normalize_features(model.get_text_features(**text_inputs))

    results: Dict[str, List[List[int]]] = {}
    results["baseline"] = rank_images(text_feats, img_feats, top_k)

    if mode == "subaltern_align":
        direction = compute_direction(model, processor, subaltern_phrases, device)
        bent_feats = normalize_features(text_feats + alpha * direction)
        results["bent"] = rank_images(bent_feats, img_feats, top_k)
    else:
        bent_feats = rotate_embeddings(text_feats)
        results["bent"] = rank_images(bent_feats, img_feats, top_k)

    return results


def save_log(
    log_dir: Path,
    text_prompts: List[str],
    image_names: List[str],
    mode: str,
    subaltern_phrases: List[str],
    alpha: float,
    results: Dict[str, List[List[int]]],
):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = log_dir / f"multimodal_bend_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt",
            "mode",
            "subaltern_phrases",
            "alpha",
            "baseline_top",
            "bent_top",
        ])
        for i, prompt in enumerate(text_prompts):
            base = [image_names[idx] for idx in results["baseline"][i]]
            bent = [image_names[idx] for idx in results["bent"][i]]
            writer.writerow([prompt, mode, ";".join(subaltern_phrases), alpha, ";".join(base), ";".join(bent)])
    print(f"[log] saved bend report to {csv_path}")


def print_topk(
    text_prompts: List[str],
    image_names: List[str],
    results: Dict[str, List[List[int]]],
):
    for i, prompt in enumerate(text_prompts):
        print("\n== Prompt ==")
        print(f"{prompt}")
        base = [image_names[idx] for idx in results["baseline"][i]]
        bent = [image_names[idx] for idx in results["bent"][i]]
        print("# FUSION BEND: re-synesthetizing the link between seeing and saying.")
        print("Baseline top-k:", base)
        print("Bent top-k:    ", bent)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    text_prompts = [p.strip() for p in args.text_prompts.split(",") if p.strip()]
    subaltern_phrases = [p.strip() for p in args.subaltern_phrases.split(",") if p.strip()]
    images = load_images(args.image_folder)
    if not text_prompts:
        raise ValueError("No text prompts supplied.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] loading CLIP model on {device}...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    results = run_alignment(
        model=model,
        processor=processor,
        device=device,
        text_prompts=text_prompts,
        images=images,
        mode=args.mode,
        subaltern_phrases=subaltern_phrases,
        alpha=args.alpha,
        top_k=args.top_k,
    )

    image_names = [name for name, _ in images]
    print_topk(text_prompts, image_names, results)

    log_dir = Path("runs") / "multimodal_bend"
    save_log(log_dir, text_prompts, image_names, args.mode, subaltern_phrases, args.alpha, results)


if __name__ == "__main__":
    main()
