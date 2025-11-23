import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------------------
# Loading runs (compatible with previous tooling)
# -------------------------------------------------------------------

def load_run(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["__file"] = path
        return data
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return None


def load_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not runs_dir.exists():
        print(f"[warn] runs dir {runs_dir} does not exist.", file=sys.stderr)
        return runs
    for p in sorted(runs_dir.glob("*.json")):
        rf = load_run(p)
        if rf:
            runs.append(rf)
    return runs


def filter_runs(
    runs: List[Dict[str, Any]],
    tag: Optional[str] = None,
    command: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in runs:
        meta = r.get("meta", {}) or {}
        if tag and meta.get("tag") != tag:
            continue
        if command and r.get("command") != command:
            continue
        out.append(r)
    return out


# -------------------------------------------------------------------
# Drift computation
# -------------------------------------------------------------------

def collect_outputs(
    runs: List[Dict[str, Any]]
) -> Tuple[List[str], List[Tuple[int, int]], Dict[int, Dict[str, Any]]]:
    """
    Returns:
      texts: list of all outputs (one per iteration)
      index_map: list of (run_index, iteration_index_within_run)
      run_meta: mapping run_index -> {meta}
    """
    texts: List[str] = []
    index_map: List[Tuple[int, int]] = []
    run_meta: Dict[int, Dict[str, Any]] = {}

    for ri, r in enumerate(runs):
        run_meta[ri] = {
            "timestamp": r.get("timestamp", ""),
            "command": r.get("command", ""),
            "meta": r.get("meta", {}) or {},
            "__file": r.get("__file"),
        }
        iters = r.get("iterations", []) or []
        for ii, it in enumerate(iters):
            out = (it.get("output") or "").strip()
            texts.append(out)
            index_map.append((ri, ii))

    return texts, index_map, run_meta


def compute_similarity_matrix(texts: List[str]) -> np.ndarray:
    """
    Returns cosine similarity matrix between all outputs (TF-IDF).
    """
    if not texts:
        return np.zeros((0, 0), dtype=float)

    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    S = cosine_similarity(X)
    return S


def per_run_drift(
    runs: List[Dict[str, Any]],
    texts: List[str],
    index_map: List[Tuple[int, int]],
    S: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Compute per-run drift scores:
      - mean_pairwise_drift (all pairs)
      - mean_sequential_drift (only consecutive iterations)
    """
    # Build mapping from run_index -> list of global indices
    run_to_global: Dict[int, List[int]] = {}
    for global_idx, (ri, _ii) in enumerate(index_map):
        run_to_global.setdefault(ri, []).append(global_idx)

    results: List[Dict[str, Any]] = []

    for ri, r in enumerate(runs):
        global_indices = run_to_global.get(ri, [])
        if not global_indices:
            continue

        # Pairwise drift
        pair_sims: List[float] = []
        for i in range(len(global_indices)):
            for j in range(i + 1, len(global_indices)):
                gi = global_indices[i]
                gj = global_indices[j]
                sim = float(S[gi, gj])
                pair_sims.append(sim)

        mean_pair_sim = float(np.mean(pair_sims)) if pair_sims else None
        mean_pair_drift = 1.0 - mean_pair_sim if mean_pair_sim is not None else None

        # Sequential drift (between iteration n and n+1 inside run)
        # We assume the iterations are logged in order already.
        seq_sims: List[float] = []
        for idx in range(len(global_indices) - 1):
            gi = global_indices[idx]
            gj = global_indices[idx + 1]
            sim = float(S[gi, gj])
            seq_sims.append(sim)

        mean_seq_sim = float(np.mean(seq_sims)) if seq_sims else None
        mean_seq_drift = 1.0 - mean_seq_sim if mean_seq_sim is not None else None

        meta = r.get("meta", {}) or {}
        results.append(
            {
                "run_index": ri,
                "timestamp": r.get("timestamp", ""),
                "command": r.get("command", ""),
                "tag": meta.get("tag", ""),
                "provider": meta.get("provider", ""),
                "model": meta.get("model", ""),
                "n_iterations": len(global_indices),
                "mean_pairwise_similarity": mean_pair_sim,
                "mean_pairwise_drift": mean_pair_drift,
                "mean_sequential_similarity": mean_seq_sim,
                "mean_sequential_drift": mean_seq_drift,
            }
        )

    return results


def global_drift(S: np.ndarray) -> Dict[str, Any]:
    """
    Overall drift over all outputs in the selection.
    """
    n = S.shape[0]
    if n == 0:
        return {
            "global_mean_similarity": None,
            "global_mean_drift": None,
        }
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(float(S[i, j]))
    if not sims:
        return {
            "global_mean_similarity": None,
            "global_mean_drift": None,
        }
    mean_sim = float(np.mean(sims))
    return {
        "global_mean_similarity": mean_sim,
        "global_mean_drift": 1.0 - mean_sim,
    }


# -------------------------------------------------------------------
# Pretty printing + CSV export
# -------------------------------------------------------------------

def print_summary(
    per_run: List[Dict[str, Any]],
    global_stats: Dict[str, Any],
) -> None:
    if not per_run:
        print("No runs / iterations found for drift computation.")
        return

    print("\n=== Semantic Drift Summary (per run) ===\n")
    header = (
        f"{'idx':>3} | {'timestamp':<16} | {'cmd':<14} | {'tag':<14} | "
        f"{'n_it':>4} | {'pair_drift':>10} | {'seq_drift':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in per_run:
        idx = r["run_index"]
        ts = (r["timestamp"] or "")[:16]
        cmd = (r["command"] or "")[:14]
        tag = (r["tag"] or "")[:14]
        n_it = r["n_iterations"]
        pd = r["mean_pairwise_drift"]
        sd = r["mean_sequential_drift"]
        pd_str = f"{pd:.3f}" if pd is not None else "   -   "
        sd_str = f"{sd:.3f}" if sd is not None else "   -   "
        print(f"{idx:>3} | {ts:<16} | {cmd:<14} | {tag:<14} | {n_it:>4} | {pd_str:>10} | {sd_str:>10}")

    print("\n=== Global Drift (all selected outputs) ===\n")
    gms = global_stats["global_mean_similarity"]
    gmd = global_stats["global_mean_drift"]
    print(f"Global mean similarity: {gms:.3f}" if gms is not None else "Global mean similarity: -")
    print(f"Global mean drift:      {gmd:.3f}" if gmd is not None else "Global mean drift:      -")
    print("")


def export_csv(
    out_path: Path,
    per_run: List[Dict[str, Any]],
    global_stats: Dict[str, Any],
) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_index",
            "timestamp",
            "command",
            "tag",
            "provider",
            "model",
            "n_iterations",
            "mean_pairwise_similarity",
            "mean_pairwise_drift",
            "mean_sequential_similarity",
            "mean_sequential_drift",
        ])
        for r in per_run:
            writer.writerow([
                r["run_index"],
                r["timestamp"],
                r["command"],
                r["tag"],
                r["provider"],
                r["model"],
                r["n_iterations"],
                r["mean_pairwise_similarity"],
                r["mean_pairwise_drift"],
                r["mean_sequential_similarity"],
                r["mean_sequential_drift"],
            ])

        # Add a final row for global stats (optional)
        writer.writerow([])
        writer.writerow(["GLOBAL_STATS"])
        writer.writerow(["global_mean_similarity", global_stats["global_mean_similarity"]])
        writer.writerow(["global_mean_drift", global_stats["global_mean_drift"]])

    print(f"[ok] wrote drift CSV to {out_path}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute semantic drift scores from neural bending run logs."
    )
    p.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing JSON run logs (default: runs).",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Filter by tag (meta.tag).",
    )
    p.add_argument(
        "--command-filter",
        default=None,
        help="Filter by command (e.g., entropy-seed).",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to export per-run drift scores as CSV.",
    )
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    all_runs = load_runs(runs_dir)
    runs = filter_runs(all_runs, tag=args.tag, command=args.command_filter)

    if not runs:
        print("No runs found matching filters.")
        return

    texts, index_map, _run_meta = collect_outputs(runs)
    if not texts:
        print("No outputs found in selected runs.")
        return

    S = compute_similarity_matrix(texts)
    per_run = per_run_drift(runs, texts, index_map, S)
    global_stats = global_drift(S)

    print_summary(per_run, global_stats)

    if args.out_csv:
        export_csv(Path(args.out_csv), per_run, global_stats)


if __name__ == "__main__":
    main()

