"""
A small demo script that applies a "semantic inversion" bend to token
embeddings in a masked language model. The idea is to partially reflect
selected concept embeddings across their midpoint, shifting how the model
interprets those words without retraining.
"""

import copy
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str = "bert-base-uncased"):
    """
    Load a masked language model and its tokenizer.
    Keeping the call here makes it easy to swap to a different backbone
    if you want to experiment with larger or domain-specific models.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def get_token_ids(tokenizer: AutoTokenizer, concept: str) -> List[int]:
    """
    Map a concept string to token ids.

    Some concepts, such as "community care", break into multiple WordPiece
    tokens. We keep all of them and average their embeddings when computing
    the concept vector. The same averaged vector is then used to reflect each
    token in the set so they stay aligned.
    """

    tokens = tokenizer.tokenize(concept)
    if not tokens:
        raise ValueError(f"Concept '{concept}' produced no tokens")
    return tokenizer.convert_tokens_to_ids(tokens)


def concept_embedding(embedding_matrix: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    """
    Compute a concept embedding by averaging its token vectors.
    """

    vectors = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    return vectors.mean(dim=0)


def apply_semantic_inversion(
    base_embedding: torch.Tensor,
    tokenizer: AutoTokenizer,
    concept_pairs: Iterable[Tuple[str, str]],
    alpha: float = 0.6,
) -> torch.Tensor:
    """
    Create a modified embedding matrix with partially inverted concepts.

    For each pair (a, b), we compute their midpoint m and reflect each embedding
    across m: inv_a = a + alpha * ((2m - a) - a). Setting alpha=1.0 performs a
    full reflection (swapping positions), while alpha=0 leaves embeddings
    unchanged. Intermediate values let you smoothly dial the bend strength.
    """

    if not 0 <= alpha <= 1:
        raise ValueError("alpha should be in [0, 1] to interpolate reflection strength")

    modified = base_embedding.clone()

    for left, right in concept_pairs:
        left_ids = get_token_ids(tokenizer, left)
        right_ids = get_token_ids(tokenizer, right)

        left_vec = concept_embedding(modified, left_ids)
        right_vec = concept_embedding(modified, right_ids)

        midpoint = 0.5 * (left_vec + right_vec)

        # Reflect both concept vectors across the midpoint with partial strength.
        reflected_left = left_vec + alpha * ((2 * midpoint - left_vec) - left_vec)
        reflected_right = right_vec + alpha * ((2 * midpoint - right_vec) - right_vec)

        # Apply the same reflected vector to every token in the concept so they stay aligned.
        for idx in left_ids:
            modified[idx] = reflected_left
        for idx in right_ids:
            modified[idx] = reflected_right

    return modified


def top_k_mask_predictions(model, tokenizer, prompt: str, k: int = 10) -> List[Tuple[str, float]]:
    """
    Run masked prediction on a prompt and return top-k token strings with scores.
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    mask_index = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if mask_index.numel() != 1:
        raise ValueError("Prompt must contain exactly one [MASK] token")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_logits = logits[0, mask_index.item()]
    scores, indices = torch.topk(mask_logits, k)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    probabilities = scores.softmax(dim=0).tolist()
    return list(zip(tokens, probabilities))


def print_comparisons(
    original_preds: Dict[str, List[Tuple[str, float]]],
    inverted_preds: Dict[str, List[Tuple[str, float]]],
    k: int,
) -> None:
    """
    Display side-by-side predictions before and after inversion.
    """

    for prompt in original_preds:
        print("=" * 80)
        print(f"Prompt: {prompt}")
        print("Rank | Original token (prob)        | Inverted token (prob)")
        print("-" * 80)
        for rank in range(k):
            tok_o, prob_o = original_preds[prompt][rank]
            tok_i, prob_i = inverted_preds[prompt][rank]
            print(
                f"{rank+1:>4} | {tok_o:<22} ({prob_o:0.3f}) | "
                f"{tok_i:<22} ({prob_i:0.3f})"
            )
        print()


def main():
    prompts = [
        "The role of the [MASK] is to protect the neighborhood.",
        "When there is a crisis, the [MASK] is called.",
    ]

    concept_pairs = [
        ("police", "community care"),
        ("citizen", "resident"),
    ]

    model_original, tokenizer = load_model_and_tokenizer()

    # Build inverted model by copying weights then swapping the embedding matrix.
    model_inverted = copy.deepcopy(model_original)
    base_embedding = model_original.get_input_embeddings().weight.data
    inverted_embedding = apply_semantic_inversion(
        base_embedding, tokenizer, concept_pairs, alpha=0.6
    )
    model_inverted.get_input_embeddings().weight.data = inverted_embedding

    k = 10
    original_predictions: Dict[str, List[Tuple[str, float]]] = {}
    inverted_predictions: Dict[str, List[Tuple[str, float]]] = {}

    for prompt in prompts:
        original_predictions[prompt] = top_k_mask_predictions(
            model_original, tokenizer, prompt, k=k
        )
        inverted_predictions[prompt] = top_k_mask_predictions(
            model_inverted, tokenizer, prompt, k=k
        )

    print_comparisons(original_predictions, inverted_predictions, k)


if __name__ == "__main__":
    main()
