# Patch Files

A `patch.json` file is a serialized patch graph for the Neural Bending Console runtime: nodes (modules), edges (patch cables), and globals (session metadata). It follows the console patch schema and uses a modular-synth metaphor where signal types (TEXT/CV/METRIC/EMBEDDING/IMAGE_PATH/...) flow between ports.

## Starter patch

The repository includes starter patches under `patches/`, including:

- `patches/starter_8_bends_ab.json`

## Load in the Console UI

1. Start backend:

```bash
nbt console serve --host 127.0.0.1 --port 8000
```

2. Start React UI:

```bash
cd console_ui
npm install
npm run dev
```

3. In the UI header, click **Load Patch** and choose `patches/starter_8_bends_ab.json`.

## Run headless

- Run backend only:

```bash
nbt console serve
```

- Then connect UI (or another WS client) and load/start the patch.

## Notes on disabled modules

- Diffusion nodes are **disabled by default** in the starter patch (`diffusion_baseline`, `diffusion_bent`) for safer out-of-the-box behavior.
- Some deep-hook bends are included as structural placeholders and may be bypassed until corresponding internals are fully implemented (attention/residual/norm hooks).

## Dissertation bend mapping (1–8)

1. **EmbeddingContaminationNode** — ontology as geometry; blends embedding manifolds.
2. **StratigraphySamplerNode** — corpus stratigraphy via sampling-pressure controls.
3. **InterHeadDriftNode** — governance-as-parliament proxy for attention head drift.
4. **GovernanceDissonanceNode** — contradiction/conflict injection and destabilization.
5. **ResidualDistortionNode** — residual stream temporal distortion proxy.
6. **NormPerturbationNode** — normalization perturbation / fluency discipline proxy.
7. **JusticeReweightingNode** — justice-oriented attractor reweighting.
8. **AttractorSeederNode** — persistent attractor basin seeding/stabilization.
