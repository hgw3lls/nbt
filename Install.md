# Installation Guide

This guide covers local and CI-friendly installation paths for the **Neural Bending Toolkit**.

## Requirements

- Python 3.10+
- `pip` 23+
- Optional: `venv` (recommended for local development)

## Standard install (runtime only)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

Verify install:

```bash
nbt --help
```

## Editable install (development)

Use this mode when working on source code and tests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Run quality checks:

```bash
ruff check .
black --check .
pytest
```

## Optional feature extras

Install only what you need:

```bash
pip install -e .[llm]
pip install -e .[diffusion]
pip install -e .[gan]
pip install -e .[audio]
pip install -e .[analysis]
```

## CI / clean-environment install check

To mimic a fresh environment and verify packaging:

```bash
python -m pip install --upgrade pip build
python -m build
pip install dist/*.whl
nbt --help
```

## Troubleshooting

- If `nbt` is not found, make sure your virtual environment is activated.
- If dependency resolution fails, upgrade pip first (`python -m pip install --upgrade pip`).
- If optional adapters fail to import, install the matching extra (for example `.[llm]` or `.[diffusion]`).
