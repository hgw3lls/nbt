import json, sys, os
from tabulate import tabulate

def load(path):
    with open(path, "r") as f:
        return json.load(f)

def print_header(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def visualize_masked(log):
    print_header(f"MASKED-LM BEND: {log['bend_name']}")
    for prompt, data in log["per_prompt"].items():
        print(f"\nPrompt: {prompt}")
        diff = data["diff"]
        rows = []
        for c in diff["changes"]:
            rows.append([
                c["rank"],
                c["base_token"], f"{c['base_prob']:.3f}",
                c["bent_token"], f"{c['bent_prob']:.3f}",
                "★" if c["changed"] else ""
            ])
        print(tabulate(rows, headers=["rank","base","p","bent","p","chg"]))
    print("\nRole counts:")
    print(tabulate([
        ["baseline"] + list(log["role_counts"]["baseline"].values()),
        ["bent"] + list(log["role_counts"]["bent"].values()),
    ], headers=["", *log["role_vocab"]]))

def visualize_generation(log):
    print_header(f"GENERATION BEND: {log['bend_name']}")
    toks = log["target_tokens"]

    # aggregate
    base = log["aggregate_counts"]["baseline"]
    bent = log["aggregate_counts"]["bent"]

    rows = []
    for t in toks:
        rows.append([t, base[t], bent[t], bent[t] - base[t]])
    print(tabulate(rows, headers=["token","baseline","bent","Δ"]))

def visualize_latent_modbus(log):
    print_header(f"LATENT MODBUS: {log['bend_name']}")
    print("\nBASELINE TEXT:\n", log["baseline_text"])
    print("\nBENT TEXT:\n", log["bent_text"])

    rep = log["report"]
    print("\nSimilarity Meter:")
    baseline = rep["baseline_mean_similarity"]
    bent = rep["bent_mean_similarity"]
    print(f"Baseline: {baseline:.4f}")
    print(f"Bent:     {bent:.4f}")
    print(f"Δ:        {bent - baseline:+.4f}")

def visualize_latent_midlayer(log):
    print_header(f"MIDLAYER BEND: {log['bend_name']}")
    for p, data in log["per_prompt"].items():
        print("\nPrompt:", p)
        print("BASELINE:", data["baseline_text"])
        print("-----")
        print("BENT:", data["bent_text"])

def detect_and_visualize(path):
    log = load(path)

    if "per_prompt" in log and "role_counts" in log:
        visualize_masked(log)
    elif "aggregate_counts" in log:
        visualize_generation(log)
    elif "baseline_text" in log and "report" in log:
        visualize_latent_modbus(log)
    elif "per_prompt" in log and "attractor_phrases" in log:
        visualize_latent_midlayer(log)
    else:
        print("Unknown format.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_atlas.py <json-file>")
        sys.exit(1)
    detect_and_visualize(sys.argv[1])
