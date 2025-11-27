"""modality_parallax.py - simulate modality parallax by rotating text embeddings.

Twists the text embedding space relative to images to expose how fragile the
<<<<<<< ours
<<<<<<< ours
multimodal weld is when geometry is perturbed. Logs baseline vs rotated matches.
=======
multimodal weld is when geometry is perturbed. Logs baseline vs rotated matches
and similarity scores so we can see the weld come apart.
>>>>>>> theirs
=======
multimodal weld is when geometry is perturbed. Logs baseline vs rotated matches
and similarity scores so we can see the weld come apart.
>>>>>>> theirs
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Circuit-bend CLIP alignment via text-space rotation"
    )
    parser.add_argument(
        "--text_prompts",
        type=str,
        required=True,
        help="Comma-separated list of text prompts",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Folder containing .jpg/.png images",
    )
    parser.add_argument(
        "--rotation_angle_deg",
        type=float,
        default=15.0,
        help="Rotation angle in degrees for the parallax bend",
    )
    parser.add_argument(
        "--mix",
        type=float,
        default=1.0,
        help="Interpolation between original and rotated embeddings",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top matches to report per prompt",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP checkpoint to load",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible parallax plane",
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


def normalize(feats: torch.Tensor) -> torch.Tensor:
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def sample_plane(dim: int, device: torch.device, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    u = torch.randn(dim, device=device, generator=generator)
    u = u / u.norm().clamp_min(1e-12)
    v = torch.randn(dim, device=device, generator=generator)
    v = v - (u * v).sum() * u
    if v.norm() < 1e-6:
        v = torch.randn(dim, device=device, generator=generator)
        v = v - (u * v).sum() * u
    v = v / v.norm().clamp_min(1e-12)
    return u, v


def rotate_embeddings(
    text_feats: torch.Tensor, angle_rad: float, mix: float, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # PARALLAX BEND: twist text jack in the patchbay so images and text misalign.
    proj_u = (text_feats * u).sum(dim=-1, keepdim=True)
    proj_v = (text_feats * v).sum(dim=-1, keepdim=True)
    project = proj_u * u + proj_v * v
    orth = text_feats - project

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated_proj = (proj_u * cos_a - proj_v * sin_a) * u + (proj_u * sin_a + proj_v * cos_a) * v
    text_rot = orth + rotated_proj
    mixed = (1 - mix) * text_feats + mix * text_rot
    return normalize(mixed)


<<<<<<< ours
<<<<<<< ours
def rank_images(text_feats: torch.Tensor, image_feats: torch.Tensor, top_k: int) -> List[List[int]]:
    sims = text_feats @ image_feats.T
    k = min(top_k, sims.size(1))
    return torch.topk(sims, k=k, dim=1).indices.tolist()
=======
=======
>>>>>>> theirs
def rank_images(
    text_feats: torch.Tensor, image_feats: torch.Tensor, top_k: int
) -> Tuple[List[List[int]], List[List[float]]]:
    sims = text_feats @ image_feats.T
    k = min(top_k, sims.size(1))
    topk = torch.topk(sims, k=k, dim=1)
    return topk.indices.tolist(), topk.values.tolist()
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs


def run_parallax(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    text_prompts: Sequence[str],
    images: List[Tuple[str, Image.Image]],
    angle_deg: float,
    mix: float,
    top_k: int,
    seed: int,
<<<<<<< ours
<<<<<<< ours
) -> Dict[str, List[List[int]]]:
=======
) -> Dict[str, Dict[str, List[List]]]:
>>>>>>> theirs
=======
) -> Dict[str, Dict[str, List[List]]]:
>>>>>>> theirs
    if not images:
        raise ValueError("No images found for scoring.")

    img_inputs = processor(images=[img for _, img in images], return_tensors="pt").to(device)
    txt_inputs = processor(text=list(text_prompts), padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
<<<<<<< ours
<<<<<<< ours
=======
        model.eval()
>>>>>>> theirs
=======
        model.eval()
>>>>>>> theirs
        image_feats = normalize(model.get_image_features(**img_inputs))
        text_feats = normalize(model.get_text_features(**txt_inputs))

    u, v = sample_plane(text_feats.size(-1), device, seed)
    angle_rad = math.radians(angle_deg)
    rotated_text = rotate_embeddings(text_feats, angle_rad, mix, u, v)

<<<<<<< ours
<<<<<<< ours
    return {
        "baseline": rank_images(text_feats, image_feats, top_k),
        "parallax": rank_images(rotated_text, image_feats, top_k),
=======
=======
>>>>>>> theirs
    base_idx, base_scores = rank_images(text_feats, image_feats, top_k)
    bent_idx, bent_scores = rank_images(rotated_text, image_feats, top_k)

    return {
        "baseline": {"indices": base_idx, "scores": base_scores},
        "parallax": {"indices": bent_idx, "scores": bent_scores},
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
    }


def save_log(
    log_dir: Path,
    text_prompts: Sequence[str],
    image_names: Sequence[str],
    angle_deg: float,
    mix: float,
<<<<<<< ours
<<<<<<< ours
    results: Dict[str, List[List[int]]],
=======
    results: Dict[str, Dict[str, List[List]]],
>>>>>>> theirs
=======
    results: Dict[str, Dict[str, List[List]]],
>>>>>>> theirs
):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "timestamp": timestamp,
        "prompts": list(text_prompts),
        "images": list(image_names),
        "rotation_angle_deg": angle_deg,
        "mix": mix,
<<<<<<< ours
<<<<<<< ours
        "baseline_top": [[image_names[i] for i in row] for row in results["baseline"]],
        "parallax_top": [[image_names[i] for i in row] for row in results["parallax"]],
=======
=======
>>>>>>> theirs
        "baseline_top": [
            {
                "names": [image_names[i] for i in row],
                "scores": results["baseline"]["scores"][idx],
            }
            for idx, row in enumerate(results["baseline"]["indices"])
        ],
        "parallax_top": [
            {
                "names": [image_names[i] for i in row],
                "scores": results["parallax"]["scores"][idx],
            }
            for idx, row in enumerate(results["parallax"]["indices"])
        ],
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
    }
    out_path = log_dir / f"modality_parallax_{timestamp}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[log] saved parallax report to {out_path}")


def print_results(
    text_prompts: Sequence[str],
    image_names: Sequence[str],
<<<<<<< ours
<<<<<<< ours
    results: Dict[str, List[List[int]]],
):
    for idx, prompt in enumerate(text_prompts):
        baseline = [image_names[i] for i in results["baseline"][idx]]
        bent = [image_names[i] for i in results["parallax"][idx]]
        print("\n== Prompt ==")
        print(prompt)
        print("# PARALLAX ERROR: twisting one modality's jack to misalign fusion.")
        print("Baseline top-k:", baseline)
        print("Parallax top-k:", bent)
=======
=======
>>>>>>> theirs
    results: Dict[str, Dict[str, List[List]]],
):
    for idx, prompt in enumerate(text_prompts):
        baseline = [image_names[i] for i in results["baseline"]["indices"][idx]]
        bent = [image_names[i] for i in results["parallax"]["indices"][idx]]
        baseline_scores = results["baseline"]["scores"][idx]
        bent_scores = results["parallax"]["scores"][idx]
        print("\n== Prompt ==")
        print(prompt)
        print("# PARALLAX ERROR: twisting one modality's jack to misalign fusion.")
        print("Baseline top-k (cosine):", list(zip(baseline, [round(s, 4) for s in baseline_scores])))
        print("Parallax top-k (cosine):", list(zip(bent, [round(s, 4) for s in bent_scores])))
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    text_prompts = [p.strip() for p in args.text_prompts.split(",") if p.strip()]
    images = load_images(args.image_folder)
    image_names = [name for name, _ in images]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

    results = run_parallax(
        model=model,
        processor=processor,
        device=device,
        text_prompts=text_prompts,
        images=images,
        angle_deg=args.rotation_angle_deg,
        mix=args.mix,
        top_k=args.top_k,
        seed=args.seed,
    )

    print_results(text_prompts, image_names, results)
    save_log(Path("runs/modality_parallax"), text_prompts, image_names, args.rotation_angle_deg, args.mix, results)


if __name__ == "__main__":
    main()
