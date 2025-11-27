"""Demonstration of a semantic–visual mismatch injection bend for CLIP-style models.

The script builds a tiny synthetic image set and shows how to steer similarity
scores toward counter-hegemonic visual themes for specific semantic terms. It
prints both vanilla CLIP rankings and rankings after the mismatch injection.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor


# Small, explicit mapping from semantic terms to counter-hegemonic visual themes.
SEMANTIC_MISMATCH_MAP: Dict[str, Dict[str, object]] = {
    "authority": {
        "mapped_theme": "community_gathering",
        "default_theme": "police_aesthetic",
        "boost": 0.55,
        "penalty": 0.25,
    },
    "future city": {
        "mapped_theme": "cooperative_garden",
        "default_theme": "high_tech_glow",
        "boost": 0.55,
        "penalty": 0.25,
    },
}


@dataclass
class ImageRecord:
    path: str
    theme: str
    label: str


def ensure_demo_images(base_dir: str) -> List[ImageRecord]:
    """Create a few themed placeholder images so the demo is self contained."""
    os.makedirs(base_dir, exist_ok=True)

    specs: List[Tuple[str, str, str]] = [
        ("community_circle.png", "community_gathering", "Neighbors in a circle"),
        ("grassroots_meeting.png", "community_gathering", "Meeting in a park"),
        ("police_line.png", "police_aesthetic", "Police line"),
        ("boardroom.png", "police_aesthetic", "Boardroom hierarchy"),
        ("coop_garden.png", "cooperative_garden", "Urban cooperative garden"),
        ("mutual_aid_kitchen.png", "cooperative_garden", "Mutual aid kitchen"),
        ("hightech_city.png", "high_tech_glow", "Neon skyline"),
    ]

    records: List[ImageRecord] = []
    for filename, theme, caption in specs:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            image = Image.new("RGB", (320, 200), color=(240, 235, 228))
            drawer = ImageDraw.Draw(image)
            text = f"{theme}\n{caption}"
            drawer.rectangle([(8, 8), (312, 192)], outline=(32, 32, 32), width=3)
            drawer.multiline_text((20, 50), text, fill=(20, 20, 20), spacing=6)
            image.save(path)
        records.append(ImageRecord(path=path, theme=theme, label=caption))
    return records


def load_clip(device: torch.device) -> Tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def encode_images(model: CLIPModel, processor: CLIPProcessor, device: torch.device, records: Sequence[ImageRecord]) -> torch.Tensor:
    images = [Image.open(record.path).convert("RGB") for record in records]
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**{k: v.to(device) for k, v in inputs.items()})
    return image_features / image_features.norm(p=2, dim=-1, keepdim=True)


def encode_texts(model: CLIPModel, processor: CLIPProcessor, device: torch.device, prompts: Sequence[str]) -> torch.Tensor:
    inputs = processor(text=prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**{k: v.to(device) for k, v in inputs.items()})
    return text_features / text_features.norm(p=2, dim=-1, keepdim=True)


def rank_images(image_features: torch.Tensor, text_feature: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    return (logit_scale.exp() * image_features @ text_feature.T).squeeze(-1)


def inject_mismatch_scores(
    base_scores: torch.Tensor,
    records: Sequence[ImageRecord],
    semantic_key: str,
) -> torch.Tensor:
    settings = SEMANTIC_MISMATCH_MAP.get(semantic_key.lower())
    if settings is None:
        return base_scores

    adjusted = base_scores.clone()
    for idx, record in enumerate(records):
        if record.theme == settings["mapped_theme"]:
            adjusted[idx] += float(settings["boost"])
        if record.theme == settings["default_theme"]:
            adjusted[idx] -= float(settings["penalty"])
    return adjusted


def print_top_k(title: str, scores: torch.Tensor, records: Sequence[ImageRecord], k: int = 3) -> None:
    sorted_indices = torch.argsort(scores, descending=True)
    print(f"\n{title}")
    for rank, idx in enumerate(sorted_indices[:k], start=1):
        record = records[idx]
        print(f"  {rank}. {os.path.basename(record.path):<24} score={scores[idx]:.3f} theme={record.theme}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = ensure_demo_images(os.path.join("demo_assets", "semantic_mismatch"))
    model, processor = load_clip(device)

    print(f"Loaded {len(records)} demo images on {device}.")
    image_features = encode_images(model, processor, device, records)

    semantic_prompts = ["authority", "future city"]
    text_features = encode_texts(model, processor, device, semantic_prompts)
    logit_scale = model.logit_scale

    for prompt, text_feature in zip(semantic_prompts, text_features):
        base_scores = rank_images(image_features, text_feature.unsqueeze(0), logit_scale)
        injected_scores = inject_mismatch_scores(base_scores, records, prompt)

        print(f"\n=== Prompt: '{prompt}' ===")
        print_top_k("Vanilla CLIP ranking", base_scores, records)
        print_top_k("Mismatch-injected ranking", injected_scores, records)

    print(
        "\nExtension idea: replace the tiny demo set with a curated photo dataset that tags "
        "themes (e.g., cooperative labor, mutual aid), expose the injection toggle in a "
        "UI search bar, and log human feedback to refine which visual counterpoints best "
        "serve each semantic term."
    )


if __name__ == "__main__":
    main()
