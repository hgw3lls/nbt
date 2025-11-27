"""
Demonstrate an "embedding neighborhood drift" bend on a small transformer
language model using Hugging Face Transformers and PyTorch.

The script:
1. Loads a pretrained masked language model and tokenizer.
2. Defines dominant and marginalized concept clusters.
3. Extracts token embeddings and computes centroids for each cluster.
4. Creates a drifted copy of the embedding matrix that nudges marginalized
   tokens toward the dominant centroid.
5. Compares masked language predictions before and after the drift.

Run directly to see the drift measurements and prediction differences.
"""

import copy
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_word_embedding(
    word: str, tokenizer: AutoTokenizer, embedding_matrix: torch.Tensor
) -> torch.Tensor:
    """Return the average embedding vector for the given word/phrase.

    Tokenization can split a phrase into multiple subword pieces. To represent
    the phrase with a single vector, we average the embeddings for its pieces.
    """

    token_ids = tokenizer(word, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0]
    return embedding_matrix[token_ids].mean(dim=0)


def compute_centroid(
    words: Iterable[str], tokenizer: AutoTokenizer, embedding_matrix: torch.Tensor
) -> torch.Tensor:
    """Compute the centroid vector for a cluster of words."""

    embeddings = [get_word_embedding(w, tokenizer, embedding_matrix) for w in words]
    return torch.stack(embeddings, dim=0).mean(dim=0)


def build_drifted_embedding_matrix(
    original_embeddings: torch.Tensor,
    marginalized_words: Iterable[str],
    tokenizer: AutoTokenizer,
    dominant_centroid: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Create a drifted copy of the embedding matrix.

    Each marginalized word vector is linearly interpolated toward the dominant
    centroid by factor ``alpha``. The function returns the new embedding matrix
    and a mapping from marginalized words to their first token id.
    """

    drifted = original_embeddings.detach().clone()
    marginalized_token_ids: Dict[str, int] = {}

    for word in marginalized_words:
        token_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_id = token_ids[0]
        marginalized_token_ids[word] = token_id

        original_vec = original_embeddings[token_id]
        drifted_vec = (1 - alpha) * original_vec + alpha * dominant_centroid
        drifted[token_id] = drifted_vec

    return drifted, marginalized_token_ids


def masked_top_k(
    model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, prompt: str, k: int = 5
) -> List[Tuple[str, float]]:
    """Return top-k masked-token predictions for the provided prompt."""

    inputs = tokenizer(prompt, return_tensors="pt")
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()
    if mask_positions.numel() == 0:
        raise ValueError("Prompt must include a [MASK] token.")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Assume a single mask token; take the first occurrence.
    batch_idx, mask_idx = mask_positions[0].tolist()
    mask_logits = logits[batch_idx, mask_idx]
    topk = torch.topk(mask_logits, k)

    tokens = tokenizer.convert_ids_to_tokens(topk.indices.tolist())
    probabilities = torch.softmax(topk.values, dim=0).tolist()
    return list(zip(tokens, probabilities))


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine distance (1 - cosine similarity) between two vectors."""

    return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def run_drift_demo(alpha: float = 0.4, top_k: int = 5) -> None:
    """Execute the embedding drift demonstration and print comparisons."""

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Concept clusters.
    dominant_words = ["scientist", "engineer", "executive"]
    marginalized_words = [
        "midwife",
        "care worker",
        "community organizer",
        "indigenous scientist",
    ]

    # Original embeddings and cluster centroids.
    embedding_matrix = base_model.get_input_embeddings().weight
    dominant_centroid = compute_centroid(dominant_words, tokenizer, embedding_matrix)
    marginalized_centroid = compute_centroid(
        marginalized_words, tokenizer, embedding_matrix
    )

    drifted_embeddings, marginalized_token_ids = build_drifted_embedding_matrix(
        embedding_matrix, marginalized_words, tokenizer, dominant_centroid, alpha
    )

    # Create a drifted copy of the model with modified embeddings.
    drifted_model = copy.deepcopy(base_model)
    drifted_model.get_input_embeddings().weight.data = drifted_embeddings

    # Report embedding drift distances.
    print("=== Embedding Drift Distances (original vs. drifted) ===")
    for word, token_id in marginalized_token_ids.items():
        original_vec = embedding_matrix[token_id]
        drifted_vec = drifted_embeddings[token_id]
        distance = cosine_distance(original_vec, drifted_vec)
        print(f"{word:>20s}: cosine distance = {distance:.6f}")

    # Centroid distances for context.
    dominant_marginal_distance = cosine_distance(
        dominant_centroid, marginalized_centroid
    )
    print("\n=== Cluster Centroid Distances ===")
    print(
        f"Dominant vs. marginalized centroids: {dominant_marginal_distance:.6f}"
    )

    # Compare masked-token predictions before and after the drift.
    prompts = ["The scientist is [MASK].", "The expert is [MASK]."]
    for prompt in prompts:
        print(f"\n=== Predictions for: '{prompt}' ===")
        original_preds = masked_top_k(base_model, tokenizer, prompt, k=top_k)
        drifted_preds = masked_top_k(drifted_model, tokenizer, prompt, k=top_k)

        print("-- Original top-k --")
        for token, prob in original_preds:
            print(f"{token:>15s} (p={prob:.4f})")

        print("-- Drifted top-k --")
        for token, prob in drifted_preds:
            print(f"{token:>15s} (p={prob:.4f})")


if __name__ == "__main__":
    run_drift_demo()
