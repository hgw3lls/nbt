"""Neural bending orchestrator to run multiple bend scripts and compile a report.

This script treats each bend as a subprocess call, gathers their latest logs,
computes lightweight drift diagnostics, and writes a consolidated Markdown
report suitable for dissertation figures.
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import textwrap
from typing import Dict, List, Optional, Tuple

# Registry of bend experiments. Comment out entries as needed if a script is absent.
BENDS: List[Dict[str, object]] = [
    {
        "name": "Embedding Drift",
        "script": "embedding_drift.py",
        "description": "Shifts marginalized concepts toward the center of embedding space, revealing how the model's ontology is culturally stratified.",
        "args": ["--source_tokens", "scientist", "--target_tokens", "midwife", "--alpha", "0.4", "--mix", "0.7"],
    },
    {
        "name": "Embedding Inversion",
        "script": "embedding_inversion.py",
        "description": "Reflects embeddings across midpoints to surface hidden dualities in representation space.",
        "args": ["--source_tokens", "order", "--target_tokens", "subversion", "--alpha", "0.5", "--mix", "0.6"],
    },
    {
        "name": "Attention Boost",
        "script": "attention_boost.py",
        "description": "Boosts visibility for selected semantic nodes during decoding to expose attentional favoritism.",
        "args": ["--boost_tokens", "care,collective", "--suppress_tokens", "blame,criminal", "--boost_scale", "4.0", "--suppress_scale", "-3.0", "--mode", "boost"],
    },
    {
        "name": "Divergent Heads",
        "script": "divergent_heads.py",
        "description": "Splits attention preference into disagreeing clusters to surface internal dissent during decoding.",
        "args": ["--boost_tokens", "solidarity,repair", "--suppress_tokens", "punish,isolate", "--mode", "divergent", "--mix", "0.6"],
    },
    {
        "name": "Residual Bottleneck",
        "script": "residual_bottleneck.py",
        "description": "Constrains residual flow through a bottleneck to watch grammar and meaning squeeze through a choke point.",
        "args": ["--mode", "bottleneck", "--bottleneck_dim", "64", "--alpha", "0.5", "--mix", "0.7"],
    },
    {
        "name": "Residual Relational",
        "script": "residual_relational.py",
        "description": "Adds relational direction vectors to the residual stream, nudging narration toward communal ties.",
        "args": ["--mode", "relational", "--relational_tokens", "community,repair,mutual,system", "--alpha", "0.35", "--mix", "0.65"],
    },
    {
        "name": "LayerNorm Variance",
        "script": "layernorm_variance.py",
        "description": "Perturbs perceived norm variance to destabilize default equilibria in decoding.",
        "args": ["--mode", "variance", "--variance_scale", "0.8", "--mix", "0.6"],
    },
    {
        "name": "LayerNorm Recenter",
        "script": "layernorm_recenter.py",
        "description": "Recenters normalization toward care-oriented embeddings to bias what counts as 'normal'.",
        "args": ["--mode", "care_center", "--care_tokens", "care,repair,collective,community", "--care_scale", "2.5", "--mix", "0.65"],
    },
    {
        "name": "Positional Scramble",
        "script": "positional_scramble.py",
        "description": "Scrambles clause order to misclock temporal expectations in the transformer sequence clock.",
        "args": ["--mode", "scramble", "--scramble_prob", "0.35", "--mix", "0.7"],
    },
    {
        "name": "Positional Anchor",
        "script": "positional_anchor.py",
        "description": "Anchors prompts with historical phrases to keep the decoder latched onto justice timelines.",
        "args": ["--mode", "anchor", "--anchor_phrases", "long history of mutual aid", "--mix", "0.6"],
    },
    {
        "name": "Multimodal Align",
        "script": "multimodal_align.py",
        "description": "Bends CLIP alignment toward subaltern phrases to reveal suppressed associations.",
        "args": ["--text_prompts", "community repair", "--image_folder", "images/", "--mode", "subaltern_align", "--top_k", "3"],
        "prompt_arg": "--text_prompts",
    },
    {
        "name": "Multimodal Mismatch",
        "script": "multimodal_mismatch.py",
        "description": "Intentionally crosswires text and image matching to surface structured misunderstandings.",
        "args": ["--text_prompts", "care network", "--image_folder", "images/", "--mode", "mismatch", "--top_k", "3"],
        "prompt_arg": "--text_prompts",
    },
    {
        "name": "Latent Attractor",
        "script": "latent_attractor.py",
        "description": "Injects attractor vectors into hidden states to see where narratives settle under gentle pulls.",
        "args": ["--attractor_tokens", "commons,mutual,care", "--alpha", "0.4", "--mix", "0.7"],
    },
    {
        "name": "Structural Drift",
        "script": "structural_drift.py",
        "description": "Boosts systemic vocabulary while suppressing individualizing terms to bias causal frames.",
        "args": ["--structural_tokens", "history,system,structure,infrastructure,policy,collective", "--individual_tokens", "blame,criminal,bad,personal,individual", "--structural_scale", "3.0", "--individual_scale", "-4.0", "--mix", "0.6"],
    },
    {
        "name": "Attention Head Disalign",
        "script": "attention_head_disalign.py",
        "description": "Blends conflicting head-group biases to simulate patched disagreement in attention routing.",
        "args": ["--focus_tokens_A", "community,care", "--focus_tokens_B", "order,control", "--scale_A", "4.0", "--scale_B", "4.0", "--mix", "0.6"],
    },
    {
        "name": "Middle Layer Shear",
        "script": "middle_layer_shear.py",
        "description": "Shears mid-layer activations to twist conceptual geometry midstream and observe semantic drift.",
        "args": ["--layer_start", "8", "--layer_end", "11", "--shear_strength", "0.15", "--mix", "0.65"],
    },
    {
        "name": "Gradient Freeze Reroute",
        "script": "gradient_freeze_reroute.py",
        "description": "Freezes most parameters while fine-tuning a tiny patch, rerouting learning feedback through a narrow trace.",
        "args": ["--train_text", "data/tiny_corpus.txt", "--num_steps", "60", "--lr", "5e-5", "--unfrozen_pattern", "lm_head", "--prompt", ""],
    },
    {
        "name": "LayerNorm Inversion",
        "script": "layernorm_inversion.py",
        "description": "Swaps early and late LayerNorm gains to invert stabilization roles like flipping gain stages.",
        "args": ["--invert_pairs", "0:11,1:10", "--mix", "0.5"],
    },
    {
        "name": "Synthetic Perplexity Injection",
        "script": "synthetic_perplexity_injection.py",
        "description": "Favors low-probability tokens to expose the model's hidden aesthetic away from semantic likelihood.",
        "args": ["--chaos_scale", "1.5", "--mix", "0.7", "--temperature", "1.0", "--top_k", "50"],
    },
    {
        "name": "KV Memory Drift",
        "script": "kv_memory_drift.py",
        "description": "Adds noise to cached keys more than values to misaddress transformer memory retrieval.",
        "args": ["--key_noise_scale", "0.1", "--value_noise_scale", "0.02"],
    },
    {
        "name": "Modality Parallax",
        "script": "modality_parallax.py",
        "description": "Rotates text embeddings relative to images to illustrate how fragile multimodal alignment is to geometric shifts.",
        "args": ["--text_prompts", "community garden", "--image_folder", "images/", "--rotation_angle_deg", "12.0", "--mix", "0.7", "--top_k", "3"],
        "prompt_arg": "--text_prompts",
    },
]


def find_latest_json(run_dir: str) -> Optional[str]:
    """Return path to the most recently modified JSON file in run_dir, if any."""
    if not os.path.isdir(run_dir):
        return None
    candidates = glob.glob(os.path.join(run_dir, "*.json"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_log_data(log_path: str) -> Dict[str, object]:
    """Load baseline/bent text and any multimodal matches from a log JSON file."""
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}

    baseline = data.get("baseline_text") or data.get("baseline_caption") or ""
    bent = data.get("bent_text") or data.get("bent_caption") or ""
    before_matches = data.get("before_matches") or data.get("baseline_matches")
    after_matches = data.get("after_matches") or data.get("bent_matches")
    return {
        "baseline": baseline,
        "bent": bent,
        "before_matches": before_matches,
        "after_matches": after_matches,
    }


def compute_text_metrics(baseline: str, bent: str) -> Tuple[float, int, int]:
    """Compute a naive token-overlap drift score and token lengths."""
    base_tokens = baseline.split()
    bent_tokens = bent.split()
    if not base_tokens or not bent_tokens:
        return 0.0, len(base_tokens), len(bent_tokens)

    min_len = min(len(base_tokens), len(bent_tokens))
    diff = sum(1 for i in range(min_len) if base_tokens[i] != bent_tokens[i])
    diff += abs(len(base_tokens) - len(bent_tokens))
    drift_score = diff / max(len(base_tokens), len(bent_tokens), 1)
    return drift_score, len(base_tokens), len(bent_tokens)


def truncate_text(text: str, width: int = 600) -> str:
    if not text:
        return ""
    return textwrap.shorten(text, width=width, placeholder="...")


def run_bend(bend: Dict[str, object], prompt: str, max_new_tokens: int, dry_run: bool) -> Dict[str, object]:
    script = str(bend["script"])
    prompt_arg = bend.get("prompt_arg", "--prompt")
    args: List[str] = list(bend.get("args", []))

    cmd: List[str] = ["python", script]
    if prompt_arg:
        if prompt_arg in args:
            try:
                idx = args.index(prompt_arg)
                if idx + 1 < len(args):
                    args[idx + 1] = prompt
                else:
                    args.append(prompt)
            except ValueError:
                args.extend([prompt_arg, prompt])
        else:
            cmd.extend([str(prompt_arg), prompt])
    if "--max_new_tokens" not in args and prompt_arg == "--prompt":
        cmd.extend(["--max_new_tokens", str(max_new_tokens)])
    cmd.extend(args)

    script_exists = os.path.isfile(script)
    run_dir = bend.get("run_dir") or os.path.join("runs", os.path.splitext(os.path.basename(script))[0])

    if dry_run or not script_exists:
        return {
            "command": " ".join(cmd),
            "stdout": "(dry run)" if dry_run else "(script missing, skipped)",
            "stderr": "" if dry_run else "Script not found.",
            "returncode": 0 if dry_run else 1,
            "log_path": None,
            "log_data": {},
            "run_dir": run_dir,
        }

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    latest_log = find_latest_json(run_dir)
    log_data = load_log_data(latest_log) if latest_log else {}

    return {
        "command": " ".join(cmd),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "log_path": latest_log,
        "log_data": log_data,
        "run_dir": run_dir,
    }


def format_multimodal_section(log_data: Dict[str, object]) -> str:
    before = log_data.get("before_matches")
    after = log_data.get("after_matches")
    lines = []
    if before:
        lines.append("Top-3 before: " + ", ".join(map(str, before[:3])))
    if after:
        lines.append("Top-3 after: " + ", ".join(map(str, after[:3])))
    return "\n".join(lines)


def write_report(output_path: str, prompt: str, results: List[Tuple[Dict[str, object], Dict[str, object]]]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = [
        "# Neural Bending Orchestrator Report",
        f"Generated: {now}",
        "",
        f"**Base prompt:** \"{prompt}\"",
        "",
    ]

    for bend, outcome in results:
        log_data = outcome.get("log_data", {})
        baseline = truncate_text(str(log_data.get("baseline", "")))
        bent = truncate_text(str(log_data.get("bent", "")))
        drift, len_base, len_bent = compute_text_metrics(str(log_data.get("baseline", "")), str(log_data.get("bent", "")))
        stdout_lines = (outcome.get("stdout") or "").splitlines()
        stderr_lines = (outcome.get("stderr") or "").splitlines()
        stdout_head = "\n".join(stdout_lines[:5])
        stderr_head = "\n".join(stderr_lines[:5])

        multimodal_notes = format_multimodal_section(log_data)
        script_present = os.path.isfile(str(bend.get("script")))

        lines.extend([
            f"## Bend: {bend['name']}",
            "",
            f"**Script:** `{bend['script']}`  ",
            f"**Description:** {bend['description']}",
            "",
            "**Command executed:**",
            "```bash",
            outcome.get("command", ""),
            "```",
            "",
        ])

        if not script_present:
            lines.append("Script not found on disk; skipped.\n")
            continue

        if multimodal_notes:
            lines.extend([
                "**Multimodal summary:**",
                multimodal_notes,
                "",
            ])
        else:
            lines.extend([
                "**Baseline excerpt (truncated):**",
                "",
                f"> {baseline or '*[no baseline_text found]*'}",
                "",
                "**Bent excerpt (truncated):**",
                "",
                f"> {bent or '*[no bent_text found]*'}",
                "",
                "**Mini-metrics:**",
                "",
                f"* Token overlap drift: {drift:.2f}",
                f"* Baseline length (tokens): {len_base}",
                f"* Bent length (tokens): {len_bent}",
                "",
            ])

        lines.extend([
            "**Notes:**",
            "",
            f"* Log file: `{outcome.get('log_path') or 'not found'}`",
            "* Stdout (first 5 lines):",
            "",
            "```text",
            stdout_head,
            "```",
            "",
            "* Stderr (if any, first 5 lines):",
            "",
            "```text",
            stderr_head or "",
            "```",
            "",
        ])

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple bend scripts and aggregate their logs into a Markdown report.")
    parser.add_argument("--prompt", required=True, help="Base prompt to feed to text-based bend scripts.")
    parser.add_argument("--output", default="reports/bend_report.md", help="Path to write the aggregated Markdown report.")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Max new tokens for text-generating bends.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing bend scripts.")
    args = parser.parse_args()

    results = []
    for bend in BENDS:
        outcome = run_bend(bend, args.prompt, args.max_new_tokens, args.dry_run)
        results.append((bend, outcome))

    write_report(args.output, args.prompt, results)

    if args.dry_run:
        print("Dry run complete. Report outline written to", args.output)
    else:
        print("Bend runs complete. Report written to", args.output)


if __name__ == "__main__":
    main()
