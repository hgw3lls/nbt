# Atlas of Failures — Neural Bending Experiments

This folder collects a small “Atlas of Failures”: a set of neural bending
experiments that document **where and how the model defends its world**.

Each config file defines a **plate** in the Atlas. Running the config produces a
JSON log you can read, annotate, and cite directly in the dissertation.

All experiments are driven by:

- `nb_embedding_semantic_toolkit.py` — unified bending toolkit
- config files (`.yml`) — one per plate

## 0. Setup

Install dependencies (minimal):

```bash
pip install torch transformers pyyaml
