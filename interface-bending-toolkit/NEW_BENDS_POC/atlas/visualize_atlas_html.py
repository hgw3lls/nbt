from jinja2 import Template
import json, sys, os

TEMPLATE = Template("""
<html>
<head>
<style>
body { font-family: sans-serif; margin: 20px; }
.changed { color: red; font-weight: bold; }
td { padding: 4px 8px; border-bottom: 1px solid #ccc; }
table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
</style>
</head>
<body>
<h1>{{ title }}</h1>

{% if masked %}
<h2>Masked-LM Top-k Changes</h2>
{% for p, block in masked.items() %}
<h3>{{ p }}</h3>
<table>
<tr><th>Rank</th><th>Base</th><th>p</th><th>Bent</th><th>p</th></tr>
{% for c in block["diff"]["changes"] %}
<tr>
<td>{{ c.rank }}</td>
<td {% if c.changed %}class='changed'{% endif %}>{{ c.base_token }}</td>
<td>{{ "%.3f"|format(c.base_prob) }}</td>
<td {% if c.changed %}class='changed'{% endif %}>{{ c.bent_token }}</td>
<td>{{ "%.3f"|format(c.bent_prob) }}</td>
</tr>
{% endfor %}
</table>
{% endfor %}
{% endif %}

{% if generation %}
<h2>Generation Token Counts</h2>
<table>
<tr><th>Token</th><th>Baseline</th><th>Bent</th><th>Δ</th></tr>
{% for row in generation %}
<tr>
<td>{{ row.token }}</td>
<td>{{ row.base }}</td>
<td>{{ row.bent }}</td>
<td {% if row.delta != 0 %}class='changed'{% endif %}>{{ row.delta }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

{% if modbus %}
<h2>MODBUS Baseline vs Bent</h2>
<h3>Baseline</h3><pre>{{ modbus.base }}</pre>
<h3>Bent</h3><pre>{{ modbus.bent }}</pre>
<p><b>Mean Similarity Δ:</b> {{ modbus.delta }}</p>
{% endif %}

{% if midlayer %}
<h2>Midlayer Attractor Outputs</h2>
{% for p, bp, bt in midlayer %}
<h3>{{ p }}</h3>
<h4>Baseline</h4><pre>{{ bp }}</pre>
<h4>Bent</h4><pre>{{ bt }}</pre>
{% endfor %}
{% endif %}

</body>
</html>
""")

def load_json(path): return json.load(open(path))

def visualize_json(path, output="atlas_report.html"):
    log = load_json(path)

    data = {
        "title": f"Atlas Visualization: {log.get('bend_name', '')}",
        "masked": None,
        "generation": None,
        "modbus": None,
        "midlayer": None,
    }

    # Detect type
    if "per_prompt" in log and "role_counts" in log:
        data["masked"] = log["per_prompt"]

    elif "aggregate_counts" in log:
        rows = []
        for t in log["target_tokens"]:
            base = log["aggregate_counts"]["baseline"][t]
            bent = log["aggregate_counts"]["bent"][t]
            rows.append({"token": t, "base": base, "bent": bent, "delta": bent-base})
        data["generation"] = rows

    elif "baseline_text" in log and "report" in log:
        rep = log["report"]
        data["modbus"] = {
            "base": log["baseline_text"],
            "bent": log["bent_text"],
            "delta": rep["bent_mean_similarity"] - rep["baseline_mean_similarity"]
        }

    elif "attractor_phrases" in log:
        mids = []
        for p, block in log["per_prompt"].items():
            mids.append((p, block["baseline_text"], block["bent_text"]))
        data["midlayer"] = mids

    html = TEMPLATE.render(**data)
    open(output, "w").write(html)
    print("wrote:", output)

if __name__ == "__main__":
    visualize_json(sys.argv[1])
