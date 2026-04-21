"""
Posting-level bootstrap CI for teacher and student micro-F1 on the 516-posting gold set.

Resamples postings with replacement 10,000 iterations, recomputes micro-F1 from the
resampled postings' per-posting (tp_per_type, fp_per_type, fn_per_type) tuples, and
reports the 2.5th/97.5th percentiles.

Usage:
    python -m pipeline.scripts.bootstrap_ci
    (writes results to pipeline/training/bootstrap_ci.json)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_HAIKU = ROOT / "pipeline/training/haiku/gold.parquet"
GOLD_SONNET = ROOT / "pipeline/training/sonnet/gold.parquet"
OUT = ROOT / "pipeline/training/bootstrap_ci.json"
N_BOOTSTRAP = 10_000
SEED = 42


def entity_set(entities) -> set[tuple[str, str]]:
    if entities is None:
        return set()
    if hasattr(entities, "tolist"):
        entities = entities.tolist()
    out = set()
    for e in entities:
        if isinstance(e, dict) and e.get("text") and e.get("type"):
            out.add((e["text"], e["type"]))
    return out


def per_posting_contributions(gold_map, pred_map) -> list[tuple[int, int, int]]:
    """For each gold posting, return (tp, fp, fn) against the matching prediction (or empty)."""
    rows = []
    for link, gold_ents in gold_map.items():
        gold_s = entity_set(gold_ents)
        pred_s = entity_set(pred_map.get(link, []))
        tp = len(gold_s & pred_s)
        fp = len(pred_s - gold_s)
        fn = len(gold_s - pred_s)
        rows.append((tp, fp, fn))
    return rows


def bootstrap_f1(contribs: list[tuple[int, int, int]], n_boot: int, rng) -> dict:
    arr = np.array(contribs)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    sampled = arr[idx]
    tp = sampled[:, :, 0].sum(axis=1)
    fp = sampled[:, :, 1].sum(axis=1)
    fn = sampled[:, :, 2].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {
        "f1_mean": float(f1.mean()),
        "f1_sd": float(f1.std(ddof=1)),
        "f1_ci_lo": float(np.percentile(f1, 2.5)),
        "f1_ci_hi": float(np.percentile(f1, 97.5)),
        "prec_mean": float(prec.mean()),
        "rec_mean": float(rec.mean()),
        "n_postings": n,
    }


def teacher_predictions(teacher: str) -> dict:
    """Load teacher predictions from labels/{teacher}/batch_*.jsonl."""
    preds = {}
    for batch in (ROOT / f"pipeline/labels/{teacher}").glob("batch_*.jsonl"):
        with open(batch) as f:
            for line in f:
                rec = json.loads(line)
                preds[rec["job_link"]] = rec.get("entities", [])
    return preds


def spacy_predictions(model_path: Path, gold_df: pd.DataFrame) -> dict:
    import spacy

    nlp = spacy.load(str(model_path))
    preds = {}
    for _, row in gold_df.iterrows():
        doc = nlp(row["job_summary"])
        preds[row["job_link"]] = [
            {"text": ent.text, "type": ent.label_} for ent in doc.ents
        ]
    return preds


def jobbert_predictions(model_path: Path, gold_df: pd.DataFrame) -> dict:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    tok = AutoTokenizer.from_pretrained(str(model_path))
    mdl = AutoModelForTokenClassification.from_pretrained(str(model_path))
    ner = pipeline("token-classification", model=mdl, tokenizer=tok,
                   aggregation_strategy="simple")
    preds = {}
    for _, row in gold_df.iterrows():
        text = row["job_summary"]
        # Truncate by encoding+decoding through the tokenizer so the pipeline's
        # internal call stays under the 512-token positional-embedding limit.
        enc = tok(text, truncation=True, max_length=510, add_special_tokens=False)
        safe_text = tok.decode(enc["input_ids"], skip_special_tokens=True)
        out = ner(safe_text) if safe_text.strip() else []
        preds[row["job_link"]] = [
            {"text": e["word"], "type": e["entity_group"]} for e in out
        ]
    return preds


def onnx_predictions(model_path: Path, gold_df: pd.DataFrame) -> dict:
    from transformers import AutoTokenizer, pipeline
    from optimum.onnxruntime import ORTModelForTokenClassification
    tok = AutoTokenizer.from_pretrained(str(model_path))
    mdl = ORTModelForTokenClassification.from_pretrained(
        str(model_path), file_name="model_quantized.onnx")
    ner = pipeline("token-classification", model=mdl, tokenizer=tok,
                   aggregation_strategy="simple")
    preds = {}
    for _, row in gold_df.iterrows():
        text = row["job_summary"]
        # Truncate by encoding+decoding through the tokenizer so the pipeline's
        # internal call stays under the 512-token positional-embedding limit.
        enc = tok(text, truncation=True, max_length=510, add_special_tokens=False)
        safe_text = tok.decode(enc["input_ids"], skip_special_tokens=True)
        out = ner(safe_text) if safe_text.strip() else []
        preds[row["job_link"]] = [
            {"text": e["word"], "type": e["entity_group"]} for e in out
        ]
    return preds


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = {}

    gold_s = pd.read_parquet(GOLD_SONNET)
    gold_h = pd.read_parquet(GOLD_HAIKU)

    # Teachers
    for teacher, gold_df in [("sonnet", gold_s), ("haiku", gold_h)]:
        gold_map = dict(zip(gold_df["job_link"], gold_df["entities"]))
        pred_map = teacher_predictions(teacher)
        contribs = per_posting_contributions(gold_map, pred_map)
        r = bootstrap_f1(contribs, N_BOOTSTRAP, rng)
        r["point_estimate"] = _point_f1(contribs)
        results[f"teacher_{teacher}"] = r
        print(f"teacher_{teacher}: F1={r['point_estimate']:.4f} "
              f"CI=[{r['f1_ci_lo']:.4f}, {r['f1_ci_hi']:.4f}] n={r['n_postings']}")

    # Students S1/S2 (spaCy)
    for sid, teacher, gold_df in [
        ("s1_spacy_sonnet", "sonnet", gold_s),
        ("s2_spacy_haiku", "haiku", gold_h),
    ]:
        model_path = ROOT / f"pipeline/training/experiments/outputs/{sid}/model-best/model-best"
        if not model_path.exists():
            model_path = ROOT / f"pipeline/training/experiments/outputs/{sid}/model-best"
        if not model_path.exists():
            print(f"Skipping {sid}: model not at {model_path}")
            continue
        gold_map = dict(zip(gold_df["job_link"], gold_df["entities"]))
        pred_map = spacy_predictions(model_path, gold_df)
        contribs = per_posting_contributions(gold_map, pred_map)
        r = bootstrap_f1(contribs, N_BOOTSTRAP, rng)
        r["point_estimate"] = _point_f1(contribs)
        results[sid] = r
        print(f"{sid}: F1={r['point_estimate']:.4f} "
              f"CI=[{r['f1_ci_lo']:.4f}, {r['f1_ci_hi']:.4f}] n={r['n_postings']}")

    # Students S3/S4 (dense JobBERT via transformers)
    for sid, gold_df in [
        ("s3_jobbert_sonnet", gold_s),
        ("s4_jobbert_haiku", gold_h),
    ]:
        model_path = ROOT / f"pipeline/training/experiments/outputs/{sid}/checkpoint-best/checkpoint-best"
        if not model_path.exists():
            print(f"Skipping {sid}: model not at {model_path}")
            continue
        pred_map = jobbert_predictions(model_path, gold_df)
        gold_map = dict(zip(gold_df["job_link"], gold_df["entities"]))
        contribs = per_posting_contributions(gold_map, pred_map)
        r = bootstrap_f1(contribs, N_BOOTSTRAP, rng)
        r["point_estimate"] = _point_f1(contribs)
        results[sid] = r
        print(f"{sid}: F1={r['point_estimate']:.4f} "
              f"CI=[{r['f1_ci_lo']:.4f}, {r['f1_ci_hi']:.4f}] n={r['n_postings']}")

    # Students S5/S6 (ONNX int8 via optimum)
    for sid, gold_df in [
        ("s5_jobbert_onnx_sonnet", gold_s),
        ("s6_jobbert_onnx_haiku", gold_h),
    ]:
        base = ROOT / f"pipeline/training/experiments/outputs/{sid}"
        model_path = base / "model-quantized" / "model-quantized"
        if not model_path.exists():
            model_path = base / "model-quantized"
        if not model_path.exists():
            print(f"Skipping {sid}: model not at {model_path}")
            continue
        pred_map = onnx_predictions(model_path, gold_df)
        gold_map = dict(zip(gold_df["job_link"], gold_df["entities"]))
        contribs = per_posting_contributions(gold_map, pred_map)
        r = bootstrap_f1(contribs, N_BOOTSTRAP, rng)
        r["point_estimate"] = _point_f1(contribs)
        results[sid] = r
        print(f"{sid}: F1={r['point_estimate']:.4f} "
              f"CI=[{r['f1_ci_lo']:.4f}, {r['f1_ci_hi']:.4f}] n={r['n_postings']}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


def _point_f1(contribs: list[tuple[int, int, int]]) -> float:
    tp = sum(c[0] for c in contribs)
    fp = sum(c[1] for c in contribs)
    fn = sum(c[2] for c in contribs)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


if __name__ == "__main__":
    main()
