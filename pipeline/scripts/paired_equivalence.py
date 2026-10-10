"""Paired bootstrap + equivalence bounds for the matched-budget A/B (RQ1).

Comparing S2, B0 and Teacher Haiku through their *marginal* bootstrap confidence
intervals is not a valid test: overlapping CIs neither establish equivalence nor
rule out a real difference, and reading "indistinguishable" from the overlap
accepts the null from a failure to reject.

All three systems are evaluated on the *same* 515 gold postings
(pipeline/training/haiku/gold.parquet), so the correct instrument is a paired
bootstrap on the difference: resample posting indices once per iteration and
score both systems on that same resample, then take the delta. This removes the
between-system correlation that the marginal intervals throw away, and yields a
CI on the difference itself. The largest effect the data can exclude at 95 %
is then read directly off the interval bounds, which is an equivalence bound
rather than a p-value.

Systems compared, all on the identical posting set:
  B0  the spaCy students' pipeline trained on the gold set alone, 5-fold CV:
      the out-of-fold predictions written by gold_only_baseline.py (run here
      first if they are absent), so the marginal and the paired analyses
      score one and the same B0
  S2  spaCy trained on Haiku labels (pipeline/training/experiments/outputs/
      s2_spacy_haiku/model-best)
  TH  Teacher Haiku raw predictions on the gold postings

Caveat retained in the output: B0's out-of-fold predictions come from five
separately trained models, so posting-level resampling captures posting variance
but not between-fold training variance. The per-fold F1 spread is reported
alongside so the reader can see the size of the component the interval omits.

    python -m pipeline.scripts.paired_equivalence

From the public package, rebuild the text-bearing gold.parquet first (the
release ships stand-off annotations only):

    python -m pipeline.scripts.make_standoff_gold rehydrate --teacher haiku \
        --source <path to job_summary.csv from the Kaggle release>
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

from pipeline.scripts import gold_only_baseline as b0

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "pipeline/training/haiku/gold.parquet"
S2_MODEL = ROOT / "pipeline/training/experiments/outputs/s2_spacy_haiku/model-best"
S2_HUB_ID = "AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1"
HAIKU_LABELS = ROOT / "pipeline/labels/haiku"
HAIKU_PUBLIC = ROOT / "pipeline/labels-public/haiku/gold-predictions.jsonl"
OUT = ROOT / "pipeline/training/paired_equivalence.json"

SEED = 42
N_BOOTSTRAP = 10_000


# --------------------------------------------------------------------------- #
# scoring primitives (identical semantics to gold_only_baseline.py)
# --------------------------------------------------------------------------- #
entity_set = b0.entity_set


def contribs_for(gold_map: dict, pred_map: dict, links: list[str]) -> np.ndarray:
    """Per-posting (tp, fp, fn), ordered by `links` so rows align across systems."""
    rows = []
    for link in links:
        g = entity_set(gold_map.get(link))
        p = entity_set(pred_map.get(link, []))
        rows.append((len(g & p), len(p - g), len(g - p)))
    return np.array(rows, dtype=np.int64)


def f1_of(arr: np.ndarray) -> float:
    tp, fp, fn = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def f1_vector(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """F1 for each bootstrap resample; idx is (n_boot, n) posting indices."""
    s = arr[idx]
    tp = s[:, :, 0].sum(axis=1)
    fp = s[:, :, 1].sum(axis=1)
    fn = s[:, :, 2].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        rc = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        return np.where((pr + rc) > 0, 2 * pr * rc / (pr + rc), 0.0)


# --------------------------------------------------------------------------- #
# B0: the out-of-fold predictions of gold_only_baseline.py
# --------------------------------------------------------------------------- #
def b0_out_of_fold(gold_df: pd.DataFrame) -> tuple[dict, list[float], float]:
    """B0's out-of-fold predictions, per-fold F1 and aggregate F1, exactly as
    gold_only_baseline.py produced them for this gold set."""
    if not b0.PREDICTIONS.exists():
        print(f"  {b0.PREDICTIONS.relative_to(ROOT)} not found; running "
              f"gold_only_baseline first", flush=True)
        b0.main()
    rec = json.loads(b0.PREDICTIONS.read_text(encoding="utf-8"))
    summary = json.loads(b0.OUT.read_text(encoding="utf-8"))
    fingerprint = b0.gold_fingerprint(gold_df)
    if rec["gold_fingerprint"] != fingerprint or summary.get("gold_fingerprint") != fingerprint:
        raise SystemExit(
            f"B0 predictions were made on a different gold set than {GOLD}; "
            f"re-run python -m pipeline.scripts.gold_only_baseline")
    return rec["predictions"], rec["fold_f1"], summary["aggregate"]["F1"]


def teacher_haiku_preds(links: set[str]) -> dict:
    """Raw Haiku teacher predictions for the gold postings.

    Resolution order: the full teacher batch outputs, which are not part of the
    public package, else the published per-gold-posting subset in
    pipeline/labels-public/. The two were verified identical for all 515 gold
    links (entity lists compare equal record-for-record, 2026-08-16), so the
    fallback cannot change any score.
    """
    sources = sorted(HAIKU_LABELS.glob("batch_*.jsonl")) or [HAIKU_PUBLIC]
    if not sources[0].exists():
        raise FileNotFoundError(
            f"no teacher predictions found: neither {HAIKU_LABELS}/batch_*.jsonl "
            f"nor {HAIKU_PUBLIC} is present")
    preds: dict = {}
    for f in sources:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                link = rec.get("job_link")
                if link in links and link not in preds:
                    ents = rec.get("entities") or []
                    preds[link] = [e for e in ents if isinstance(e, dict)]
    return preds


# --------------------------------------------------------------------------- #
def paired(a: np.ndarray, b: np.ndarray, idx: np.ndarray, name_a: str, name_b: str) -> dict:
    """Paired bootstrap of F1(a) - F1(b) under a shared resample of postings."""
    fa, fb = f1_vector(a, idx), f1_vector(b, idx)
    d = fa - fb
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    # two-sided paired bootstrap p-value for H0: delta = 0
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {
        "comparison": f"{name_a} - {name_b}",
        "point_f1_a": round(f1_of(a), 4),
        "point_f1_b": round(f1_of(b), 4),
        "delta_point": round(f1_of(a) - f1_of(b), 4),
        "delta_mean": round(float(d.mean()), 4),
        "delta_ci_lo": round(lo, 4),
        "delta_ci_hi": round(hi, 4),
        "p_two_sided": round(float(min(p, 1.0)), 4),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "equivalence_bound": round(float(max(abs(lo), abs(hi))), 4),
    }


def main() -> None:
    gold_df = pd.read_parquet(GOLD).reset_index(drop=True)
    links = gold_df["job_link"].tolist()
    gold_map = dict(zip(gold_df["job_link"], gold_df["entities"]))
    print(f"gold postings: {len(gold_df)}", flush=True)

    print("\n=== S2 (spaCy on Haiku labels) ===", flush=True)
    if S2_MODEL.exists():
        s2 = spacy.load(str(S2_MODEL))
    else:
        # Same Hub fallback as bootstrap_ci.py: materialise the published S2
        # weights to a local snapshot and load from there.
        from huggingface_hub import snapshot_download
        print(f"  Local model not found at {S2_MODEL}; pulling from HF Hub: {S2_HUB_ID}",
              flush=True)
        s2 = spacy.load(snapshot_download(repo_id=S2_HUB_ID))
    s2_preds = b0.predict(s2, gold_df)

    print("\n=== Teacher Haiku (raw) ===", flush=True)
    th_preds = teacher_haiku_preds(set(links))
    print(f"matched teacher records: {len(th_preds)}/{len(links)}", flush=True)

    print("\n=== B0 (gold-only, 5-fold CV) ===", flush=True)
    b0_preds, per_fold_f1, b0_reported_f1 = b0_out_of_fold(gold_df)
    print(f"out-of-fold predictions: {len(b0_preds)}/{len(links)}", flush=True)

    A_S2 = contribs_for(gold_map, s2_preds, links)
    A_B0 = contribs_for(gold_map, b0_preds, links)
    A_TH = contribs_for(gold_map, th_preds, links)
    # One B0 for both analyses: the marginal F1 recomputed here from the shared
    # predictions must equal the one gold_only_baseline.py reported.
    assert round(f1_of(A_B0), 4) == b0_reported_f1, (
        f"B0 F1 {f1_of(A_B0):.4f} here vs {b0_reported_f1} in {b0.OUT.name}")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(links), size=(N_BOOTSTRAP, len(links)))

    result = {
        "n_postings": len(links),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "spacy_version": spacy.__version__,
        "b0_source": str(b0.PREDICTIONS.relative_to(ROOT)),
        "gold_fingerprint": b0.gold_fingerprint(gold_df),
        "marginal_f1": {
            "S2": round(f1_of(A_S2), 4),
            "B0": round(f1_of(A_B0), 4),
            "TeacherHaiku": round(f1_of(A_TH), 4),
        },
        "b0_per_fold_f1": per_fold_f1,
        "b0_per_fold_spread": round(max(per_fold_f1) - min(per_fold_f1), 4),
        "paired": [
            paired(A_S2, A_B0, idx, "S2", "B0"),
            paired(A_S2, A_TH, idx, "S2", "TeacherHaiku"),
            paired(A_B0, A_TH, idx, "B0", "TeacherHaiku"),
        ],
        "note": ("B0 out-of-fold predictions come from five separately trained "
                 "models; posting-level resampling captures posting variance but "
                 "not between-fold training variance. See b0_per_fold_spread."),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print("\n" + json.dumps(result["marginal_f1"], indent=2))
    for c in result["paired"]:
        print(f"\n{c['comparison']}: delta {c['delta_point']:+.4f} "
              f"95% CI [{c['delta_ci_lo']:+.4f}, {c['delta_ci_hi']:+.4f}] "
              f"p={c['p_two_sided']} excludes_zero={c['excludes_zero']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
