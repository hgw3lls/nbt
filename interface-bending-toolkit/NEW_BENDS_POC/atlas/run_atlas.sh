
#!/usr/bin/env bash
set -e

# Optional: create base runs directory if not present
mkdir -p runs

echo "[Atlas] Running Plate I — Embedding Neighborhood Drift (Scientist / Midwife)..."
python nb_embedding_semantic_toolkit.py --config plate_I_scientist_midwife_masked.yml

echo "[Atlas] Running Plate II — Pairwise Drift (Citizen / Resident, Police / Community Care)..."
python nb_embedding_semantic_toolkit.py --config plate_II_citizen_police_pairwise_masked.yml

echo "[Atlas] Running Plate III — Semantic Inversion (Police ↔ Community Care)..."
python nb_embedding_semantic_toolkit.py --config plate_III_semantic_inversion_police.yml

echo "[Atlas] Running Plate IV — Generation-Level Public Safety Narratives..."
python nb_embedding_semantic_toolkit.py --config plate_IV_public_safety_generation.yml

echo "[Atlas] Running Plate V — Mid-Layer Latent Attractor (Relational Futures)..."
python nb_embedding_semantic_toolkit.py --config plate_V_midlayer_relational_futures.yml

echo "[Atlas] Running Plate VI — Final Hidden MOD BUS (Care Attractor vs Public Safety)..."
python nb_embedding_semantic_toolkit.py --config plate_VI_latent_modbus_public_safety.yml

echo ""
echo "[Atlas] All plates completed. Logs written under ./runs/"
