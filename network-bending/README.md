# Network Bending

A self-contained Python package that braids media theory with neural network bending. It exposes philosophically annotated bends across embedding space, attention, residual streams, normalization, positional encoding, multimodal alignment, and substrate-level drift. Each bend is categorized as revelatory, disruptive, or recoherent, making explicit the epistemic stakes of each technical intervention.

## Installation

```bash
pip install -e .
```

The package uses PyTorch and Hugging Face Transformers for the exemplar bends. Install GPU-enabled builds as needed for your environment.

## Usage

```python
from network_bending import apply_bend, list_bend_summaries

# Discover bends that target the residual stream
print(list_bend_summaries(domain="residual"))

# Apply a bend
bent_model = apply_bend(
    bend_name="residual_bottleneck_minimal_forms",
    model=my_model,
    bottleneck_dim=128,
    mix=0.7,
)
```

All bend registrations occur on import, so simply importing the package is enough to access `apply_bend` and the discovery helpers.
