import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, GPT2LMHeadModel, GPT2TokenizerFast


"""Gradient freeze reroute demo.

This script freezes most of a small language model and only re-opens a narrow
slice of parameters, simulating a bent feedback trace on an otherwise epoxy-locked
circuit. A few gradient-carrying wires rewrite behavior in a hyper-local way.
"""


@dataclass
class TrainingExample:
    input_ids: torch.Tensor
    labels: torch.Tensor


class TinyTextDataset(Dataset):
    """Minimal dataset built from a text corpus for pedagogical fine-tuning."""

    def __init__(self, encodings: List[torch.Tensor]):
        self.examples = encodings

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        ids = self.examples[idx]
        return TrainingExample(input_ids=ids, labels=ids.clone())


def build_dataset(tokenizer: GPT2TokenizerFast, path: str, block_size: int = 64) -> TinyTextDataset:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # Slice the text into small overlapping windows to simulate a corpus.
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    windows = []
    stride = block_size // 2
    for start in range(0, max(tokens.size(0) - 1, 1), stride):
        window = tokens[start : start + block_size]
        if window.numel() < 2:
            continue
        windows.append(window)
    return TinyTextDataset(windows)


def freeze_parameters(model: GPT2LMHeadModel, unfrozen_pattern: str) -> Tuple[int, int, List[str]]:
    total_params = 0
    trainable_params = 0
    unfrozen = []
    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False
    # CIRCUIT NOTE: epoxy over the whole board, then scratch a tiny trace free.
    for name, param in model.named_parameters():
        if unfrozen_pattern in name:
            param.requires_grad = True
            trainable_params += param.numel()
            unfrozen.append(name)
    return trainable_params, total_params, unfrozen


def reroute_train_loop(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    num_steps: int,
) -> List[float]:
    model.train()
    losses = []
    step = 0
    data_iter = iter(dataloader)
    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        optimizer.zero_grad()
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        # FEEDBACK PATCH: gradients only flow through the unfrozen shard.
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        step += 1
    return losses


def generate_text(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, prompt: str, max_new_tokens: int) -> str:
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def log_run(
    log_path: str,
    prompt: str,
    unfrozen: List[str],
    trainable_params: int,
    total_params: int,
    num_steps: int,
    losses: List[float],
    baseline_text: str,
    bent_text: str,
) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    payload = {
        "prompt": prompt,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "unfrozen_parameters": unfrozen,
        "num_steps": num_steps,
        "losses": losses,
        "baseline_text": baseline_text,
        "bent_text": bent_text,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze almost everything, reroute a tiny gradient patch.")
    parser.add_argument("--train_text", type=str, required=True, help="Path to tiny training corpus.")
    parser.add_argument("--num_steps", type=int, default=100, help="Training steps (tiny, pedagogical).")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the unfrozen sliver.")
    parser.add_argument(
        "--unfrozen_pattern", type=str, default="lm_head", help="Substring of parameter names to unfreeze."
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to test generation after bending.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Tokens to generate for evaluation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Baseline model is untouched; training model will be bent.
    baseline_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    bent_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)

    trainable_params, total_params, unfrozen = freeze_parameters(bent_model, args.unfrozen_pattern)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if not unfrozen:
        print("WARNING: No parameters matched unfrozen_pattern; model will remain frozen.")

    dataset = build_dataset(tokenizer, args.train_text)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, bent_model.parameters()), lr=args.lr)

    # BASELINE: before any epoxy scratching, read the unbent continuation.
    baseline_text = generate_text(
        baseline_model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens
    )

    # TRAINING LOOP: only the unfrozen shard gets gradient flow.
    print("Starting bent training loop...")
    losses = reroute_train_loop(bent_model, dataloader, optimizer, device, args.num_steps)

    bent_text = generate_text(bent_model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", "gradient_freeze_reroute")
    log_path = os.path.join(log_dir, f"run-{timestamp}.json")

    log_run(
        log_path,
        args.prompt,
        unfrozen,
        trainable_params,
        total_params,
        args.num_steps,
        losses,
        baseline_text,
        bent_text,
    )

    print("// FEEDBACK PATCH: only a microscopic trace was left conductive.")
    for name in unfrozen:
        print(f"// REROUTED: {name}")

    print("Sample losses (first 5):", losses[:5])
    print("Baseline continuation:\n", baseline_text)
    print("Bent continuation:\n", bent_text)


if __name__ == "__main__":
    main()
