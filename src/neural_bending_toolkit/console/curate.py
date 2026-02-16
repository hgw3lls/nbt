"""Dissertation export helpers for console takes."""

from __future__ import annotations

import shutil
from pathlib import Path


def curate_take(run_dir: Path, slug: str, exports_root: Path = Path("dissertation/exports")) -> Path:
    target = exports_root / slug
    target.mkdir(parents=True, exist_ok=True)

    copy_candidates = [
        run_dir / "theory_memo.md",
        run_dir / "outputs",
        run_dir / "figure_specs",
        run_dir / "patch.json",
        run_dir / "metrics.jsonl",
        run_dir / "report.md",
    ]

    for src in copy_candidates:
        if not src.exists():
            continue
        dst = target / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    return target
