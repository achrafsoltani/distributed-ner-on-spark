"""
Teacher-vs-gold micro-F1 with posting-level bootstrap CIs, computed from the
shipped stand-off files alone: no posting text and no rehydration needed.

Every score in the paper matches entities as a set of (text, type) pairs per
posting, and both the stand-off gold annotations and the teachers' gold-set
predictions carry each span's string, so the text itself is not required.
Resampling mirrors bootstrap_ci.py exactly (10,000 posting-level resamples,
seed 42, Sonnet before Haiku from one generator), so the intervals equal the
teacher rows that bootstrap_ci.py prints after rehydration.

Usage:
    python -m pipeline.scripts.teacher_f1_standoff
    (writes pipeline/training/teacher_f1_standoff.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.scripts.bootstrap_ci import (
    N_BOOTSTRAP,
    SEED,
    _point_f1,
    bootstrap_f1,
    per_posting_contributions,
    teacher_predictions,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pipeline/training/teacher_f1_standoff.json"


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = {}
    # Same order and generator as bootstrap_ci.py, so the resamples are identical.
    for teacher in ("sonnet", "haiku"):
        gold = pd.read_parquet(ROOT / f"pipeline/training/{teacher}/gold-standoff.parquet")
        gold_map = dict(zip(gold["job_link"], gold["entities"]))
        contribs = per_posting_contributions(gold_map, teacher_predictions(teacher))
        tp, fp, fn = (sum(c[i] for c in contribs) for i in range(3))
        r = bootstrap_f1(contribs, N_BOOTSTRAP, rng)
        r.update(point_estimate=_point_f1(contribs), tp=tp, fp=fp, fn=fn,
                 precision=tp / (tp + fp), recall=tp / (tp + fn))
        results[f"teacher_{teacher}"] = r
        print(f"teacher_{teacher}: F1={r['point_estimate']:.4f} "
              f"CI=[{r['f1_ci_lo']:.4f}, {r['f1_ci_hi']:.4f}] "
              f"P={r['precision']:.3f} R={r['recall']:.3f} "
              f"TP/FP/FN={tp}/{fp}/{fn} n={r['n_postings']}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
