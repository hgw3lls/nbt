import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- Data Structures --------------------------------------------------------


@dataclass
class RunFile:
    path: Path
    timestamp: str
    command: str
    meta: Dict[str, Any]
    iterations: List[Dict[str, Any]]

    @property
    def tag(self) -> Optional[str]:
        return self.meta.get("tag")

    @property
    def model(self) -> Optional[str]:
        return self.meta.get("model")

    @property
    def provider(self) -> Optional[str]:
        return self.meta.get("provider")

    @property
    def ts_dt(self) -> Optional[datetime]:
        try:
            return datetime.strptime(self.timestamp, "%Y%m%dT%H%M%SZ")
        except Exception:
            return None


# --- Helpers ----------------------------------------------------------------


def load_run_file(path: Path) -> Optional[RunFile]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return None

    timestamp = data.get("timestamp", "")
    command = data.get("command", "")
    meta = data.get("meta", {}) or {}
    iterations = data.get("iterations", []) or []
    return RunFile(path=path, timestamp=timestamp, command=command, meta=meta, iterations=iterations)


def load_runs(
    runs_dir: Path,
    command: Optional[str] = None,
    tag: Optional[str] = None,
) -> List[RunFile]:
    runs: List[RunFile] = []
    if not runs_dir.exists():
        print(f"[warn] runs dir {runs_dir} does not exist.", file=sys.stderr)
        return runs

    for p in sorted(runs_dir.glob("*.json")):
        rf = load_run_file(p)
        if not rf:
            continue
        if command and rf.command != command:
            continue
        if tag and rf.tag != tag:
            continue
        runs.append(rf)

    return runs


# --- Subcommands ------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir, command=args.command_filter, tag=args.tag)
    if not runs:
        print("No runs found matching filters.")
        return

    print(f"Found {len(runs)} run(s) in {runs_dir}:\n")
    for rf in runs:
        tag = rf.tag or "-"
        model = rf.model or "-"
        provider = rf.provider or "-"
        ts = rf.timestamp or "-"
        print(f"{ts} | {rf.command:16} | tag={tag:16} | model={model:18} | provider={provider:8} | {rf.path.name}")


def cmd_export_md(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir, command=args.command_filter, tag=args.tag)
    if not runs:
        print("No runs found to export.")
        return

    out_path = Path(args.out)
    lines: List[str] = []

    lines.append(f"# Run Export (runs_dir = {runs_dir})")
    if args.tag:
        lines.append(f"- Filter: tag = `{args.tag}`")
    if args.command_filter:
        lines.append(f"- Filter: command = `{args.command_filter}`")
    lines.append("")

    for rf in runs:
        lines.append(f"## {rf.timestamp} — `{rf.command}` — `{rf.path.name}`")
        lines.append("")
        lines.append(f"- **Tag**: `{rf.tag or '-'}`")
        lines.append(f"- **Provider**: `{rf.provider or '-'}`")
        lines.append(f"- **Model**: `{rf.model or '-'}`")
        lines.append(f"- **Meta**: ```json")
        lines.append(json.dumps(rf.meta, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

        for it in rf.iterations:
            label_parts = []
            for k in ("step", "run", "case"):
                if k in it:
                    label_parts.append(f"{k}={it[k]}")
            label = ", ".join(label_parts) if label_parts else "iteration"

            lines.append(f"### Iteration ({label})")
            lines.append("")
            prompt = it.get("prompt", "")
            output = it.get("output", "")
            temp = it.get("temperature", None)
            seed = it.get("seed", None)

            if args.include_prompts:
                lines.append("**Prompt:**")
                lines.append("```")
                lines.append(prompt)
                lines.append("```")
                lines.append("")

            if args.include_outputs:
                lines.append("**Output:**")
                lines.append("```")
                lines.append(output)
                lines.append("```")
                lines.append("")

            # small param block
            meta_bits = []
            if temp is not None:
                meta_bits.append(f"temperature={temp}")
            if seed is not None:
                meta_bits.append(f"seed={seed}")
            if meta_bits:
                lines.append("_Params: " + ", ".join(meta_bits) + "_")
                lines.append("")

        lines.append("---")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[ok] wrote {out_path}")


def cmd_export_csv(args: argparse.Namespace) -> None:
    import csv

    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir, command=args.command_filter, tag=args.tag)
    if not runs:
        print("No runs found to export.")
        return

    out_path = Path(args.out)

    # Flatten: each iteration is a row
    fieldnames = [
        "timestamp",
        "file",
        "command",
        "tag",
        "provider",
        "model",
        "iteration_index",
        "iteration_label",
        "temperature",
        "seed",
        "prompt",
        "output",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rf in runs:
            for idx, it in enumerate(rf.iterations):
                label_parts = []
                for k in ("step", "run", "case"):
                    if k in it:
                        label_parts.append(f"{k}={it[k]}")
                label = ", ".join(label_parts)

                row = {
                    "timestamp": rf.timestamp,
                    "file": rf.path.name,
                    "command": rf.command,
                    "tag": rf.tag or "",
                    "provider": rf.provider or "",
                    "model": rf.model or "",
                    "iteration_index": idx,
                    "iteration_label": label,
                    "temperature": it.get("temperature", ""),
                    "seed": it.get("seed", ""),
                    "prompt": it.get("prompt", ""),
                    "output": it.get("output", ""),
                }
                writer.writerow(row)

    print(f"[ok] wrote {out_path}")


def cmd_diagram_prompt(args: argparse.Namespace) -> None:
    """
    Generate a text block you can feed to your diagram generator,
    summarizing all runs for a given tag (or command+tag).
    """
    runs_dir = Path(args.runs_dir)
    runs = load_runs(runs_dir, command=args.command_filter, tag=args.tag)

    if not runs:
        print("No runs found for diagram-prompt.")
        return

    title_parts = []
    if args.tag:
        title_parts.append(f"tag={args.tag}")
    if args.command_filter:
        title_parts.append(f"command={args.command_filter}")
    title_str = ", ".join(title_parts) if title_parts else "all runs"

    print("----- DIAGRAM PROMPT BEGIN -----\n")
    print(f"Create a clean MIT-Press-style technical diagram that visualizes neural bending runs ({title_str}).")
    print("Use monochrome vector lines, rectangular nodes, and clear labels. No decorative elements.")
    print("")
    print("Diagram requirements:")
    print("- Title: \"Interface Bends – Run Atlas\"")
    print("- For each run, draw a vertical strip or lane.")
    print("- At the top of each lane, label: timestamp, command, tag, model, provider.")
    print("- Inside each lane, create a sequence of nodes for each iteration.")
    print("- Each iteration node should include:")
    print("  - iteration label (step/run/case)")
    print("  - temperature")
    print("  - seed")
    print("  - a very short paraphrase of the output (2–5 words).")
    print("- Connect nodes in a lane with arrows to show temporal sequence.")
    print("- Optionally draw faint cross-lane lines to connect iterations that share similar prompts or outputs.")
    print("- Use consistent typography and spacing.")
    print("")
    print("Source summary from logs:")
    print("")

    for rf in runs:
        print(f"- Run: {rf.timestamp} | command={rf.command} | tag={rf.tag or '-'} | model={rf.model or '-'} | provider={rf.provider or '-'}")
        for it in rf.iterations:
            label_parts = []
            for k in ("step", "run", "case"):
                if k in it:
                    label_parts.append(f"{k}={it[k]}")
            label = ", ".join(label_parts) if label_parts else "iteration"
            temp = it.get("temperature", None)
            seed = it.get("seed", None)
            output = (it.get("output", "") or "").strip()
            # crude little 2–5 word paraphrase: first 5 words of output
            words = output.split()
            short = " ".join(words[:5]) if words else ""
            print(f"  • iteration [{label}] temp={temp} seed={seed} → \"{short}\"")

    print("\n----- DIAGRAM PROMPT END -----")


# --- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-processor for neural bending toolkit runs/"
    )
    p.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing JSON run logs (default: runs).",
    )

    subparsers = p.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List runs with basic metadata.")
    p_list.add_argument(
        "--tag",
        default=None,
        help="Filter by tag (meta.tag).",
    )
    p_list.add_argument(
        "--command-filter",
        default=None,
        help="Filter by command name (e.g., entropy-seed).",
    )

    # export-md
    p_md = subparsers.add_parser("export-md", help="Export runs to a Markdown file.")
    p_md.add_argument(
        "--tag",
        default=None,
        help="Filter by tag.",
    )
    p_md.add_argument(
        "--command-filter",
        default=None,
        help="Filter by command.",
    )
    p_md.add_argument(
        "--out",
        required=True,
        help="Output Markdown file path.",
    )
    p_md.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include prompts in the export.",
    )
    p_md.add_argument(
        "--include-outputs",
        action="store_true",
        help="Include outputs in the export.",
    )

    # export-csv
    p_csv = subparsers.add_parser("export-csv", help="Export runs to a CSV file.")
    p_csv.add_argument(
        "--tag",
        default=None,
        help="Filter by tag.",
    )
    p_csv.add_argument(
        "--command-filter",
        default=None,
        help="Filter by command.",
    )
    p_csv.add_argument(
        "--out",
        required=True,
        help="Output CSV file path.",
    )

    # diagram-prompt
    p_diag = subparsers.add_parser(
        "diagram-prompt",
        help="Print a diagram-generation prompt summarizing runs.",
    )
    p_diag.add_argument(
        "--tag",
        default=None,
        help="Filter by tag.",
    )
    p_diag.add_argument(
        "--command-filter",
        default=None,
        help="Filter by command.",
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        cmd_list(args)
    elif args.command == "export-md":
        cmd_export_md(args)
    elif args.command == "export-csv":
        cmd_export_csv(args)
    elif args.command == "diagram-prompt":
        cmd_diagram_prompt(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

