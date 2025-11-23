import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_run(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def load_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for f in sorted(runs_dir.glob("*.json")):
        data = load_run(f)
        if data:
            data["__file"] = f
            out.append(data)
    return out

def filter_runs(runs: List[Dict[str, Any]], tag=None, command=None):
    out = []
    for r in runs:
        meta = r.get("meta", {})
        if tag and meta.get("tag") != tag:
            continue
        if command and r.get("command") != command:
            continue
        out.append(r)
    return out

def first_words(s: str, n=6):
    s = s.strip()
    if not s:
        return ""
    parts = s.split()
    return " ".join(parts[:n])

# ------------------------------------------------------------
# Markdown Builders (Appendix Style)
# ------------------------------------------------------------

def build_header(title: str) -> str:
    return f"# {title}\n\n"

def build_run_block(r: Dict[str, Any]) -> str:
    ts = r.get("timestamp", "")
    cmd = r.get("command", "")
    meta = r.get("meta", {})
    tag = meta.get("tag", "")
    model = meta.get("model", "")
    provider = meta.get("provider", "")
    max_tok = meta.get("max_tokens", "")

    h = []
    h.append(f"## Run {ts}")
    h.append(f"**Command:** `{cmd}`")
    if tag:
        h.append(f"**Tag:** `{tag}`")
    h.append(f"**Model / Provider:** `{model}` / `{provider}`")
    h.append(f"**Max Tokens:** `{max_tok}`")
    h.append("")
    h.append("### Iterations")
    h.append("")

    for it in r.get("iterations", []):
        label = []
        for k in ("step", "run", "case"):
            if k in it:
                label.append(f"{k}={it[k]}")
        label = ", ".join(label) if label else "iteration"

        temp = it.get("temperature", "")
        seed = it.get("seed", "")
        prompt = it.get("prompt", "")
        output = it.get("output", "")

        h.append(f"#### {label}")
        h.append(f"- Temperature: `{temp}`")
        h.append(f"- Seed: `{seed}`")
        h.append("**Prompt**")
        h.append("```")
        h.append(prompt)
        h.append("```")
        h.append("**Output (excerpt)**")
        h.append("```")
        h.append(output)
        h.append("```")
        h.append("")
    h.append("---")
    h.append("")
    return "\n".join(h)

def build_atlas_section(runs: List[Dict[str, Any]], title: str) -> str:
    out = []
    out.append(build_header(title))
    out.append(
        "The following field logs document the operational behavior of the model under a variety\n"
        "of interface-level manipulations. Each run is captured as a discrete experimental lane,\n"
        "showing input conditions, drift behaviors, thermal variance (via temperature), and\n"
        "semantic deformation of outputs. These logs constitute the empirical substrate of\n"
        "interface-level bending."
    )
    out.append("\n---\n")

    for r in runs:
        out.append(build_run_block(r))

    return "\n".join(out)

# ------------------------------------------------------------
# Diagram Prompt Builder
# ------------------------------------------------------------

def build_diagram_prompt(runs: List[Dict[str, Any]], title="Interface Bending Atlas") -> str:
    out = []
    out.append("----- DIAGRAM PROMPT BEGIN -----")
    out.append("")
    out.append(
        f"Create a clean, monochrome, MIT-Press-style technical diagram titled '{title}'.\n"
        "Render a multi-lane run atlas:\n"
        "- Each lane represents one run (from logs).\n"
        "- Label lane headers with timestamp, command, tag, model, and provider.\n"
        "- Inside each lane, draw a linear sequence of nodes for each iteration.\n"
        "- For each node include: iteration label, temperature, seed, and a short 3–6 word paraphrase of the output.\n"
        "- Connect nodes with arrows.\n"
        "- Use consistent spacing, geometric clarity, and academic typography.\n"
    )
    out.append("")
    out.append("Run summary:")
    out.append("")

    for r in runs:
        ts = r.get("timestamp","")
        cmd = r.get("command","")
        tag = r.get("meta",{}).get("tag","")
        model = r.get("meta",{}).get("model","")
        provider = r.get("meta",{}).get("provider","")

        out.append(f"- {ts} | command={cmd} | tag={tag} | model={model} | provider={provider}")
        for it in r.get("iterations",[]):
            label = []
            for k in ("step","run","case"):
                if k in it:
                    label.append(f"{k}={it[k]}")
            label = ", ".join(label) if label else "iteration"
            temp = it.get("temperature","")
            seed = it.get("seed","")
            short = first_words(it.get("output",""), 6)
            out.append(f"  • [{label}]  temp={temp}, seed={seed} → \"{short}\"")

    out.append("")
    out.append("----- DIAGRAM PROMPT END -----")
    return "\n".join(out)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Neural Bending Atlas Builder")
    ap.add_argument("--runs-dir", default="runs", help="Path to runs directory")
    ap.add_argument("--tag", default=None, help="Filter by tag")
    ap.add_argument("--command-filter", default=None, help="Filter by command")
    ap.add_argument("--out", required=True, help="Output markdown file")
    ap.add_argument("--include-diagram-prompt", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    all_runs = load_runs(runs_dir)
    filtered = filter_runs(all_runs, tag=args.tag, command=args.command_filter)

    if not filtered:
        print("No runs found.")
        return

    # main atlas body
    section = build_atlas_section(filtered, "Appendix: Interface-Level Run Atlas")

    if args.include_diagram_prompt:
        section += "\n\n" + build_diagram_prompt(filtered)

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(section, encoding="utf-8")

    print(f"[ok] wrote atlas to {out_path}")

if __name__ == "__main__":
    main()

