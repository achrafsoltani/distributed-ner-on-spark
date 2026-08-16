"""
Evaluate a trained student model against dev and gold sets.

Uses the same entity-level (text, type) set matching as from_label_studio.py
to ensure teacher-vs-gold and student-vs-gold metrics are comparable.

Can be used standalone or imported by train_spacy.py / train_jobbert.py.

Usage:
    python -m pipeline.scripts.evaluate_student \
        --model-type spacy \
        --model-path outputs/s1_spacy_sonnet/model-best \
        --dev pipeline/training/sonnet/dev.parquet \
        --gold pipeline/training/sonnet/gold.parquet \
        --output outputs/s1_spacy_sonnet/eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def entity_set_from_list(entities: list[dict]) -> set[tuple[str, str]]:
    """Build a (text, type) set from entity dicts, matching from_label_studio.py."""
    return {(e["text"], e["type"]) for e in entities if e.get("text") and e.get("type")}


def compute_f1(gold_ents: set, pred_ents: set) -> tuple[Counter, Counter, Counter]:
    """Compute per-type TP/FP/FN from two entity sets."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for ent in gold_ents & pred_ents:
        tp[ent[1]] += 1
    for ent in pred_ents - gold_ents:
        fp[ent[1]] += 1
    for ent in gold_ents - pred_ents:
        fn[ent[1]] += 1
    return tp, fp, fn


def evaluate_predictions(
    gold_df: pd.DataFrame,
    predict_fn: Callable[[str], list[dict]],
    split_name: str = "gold",
) -> dict:
    """Run predict_fn on each posting and compare to gold entities."""
    per_type_tp, per_type_fp, per_type_fn = Counter(), Counter(), Counter()
    latencies_ms = []
    processed = 0

    for _, row in gold_df.iterrows():
        text = row["job_summary"]
        gold_entities = row["entities"]
        if not isinstance(gold_entities, list):
            gold_entities = list(gold_entities) if hasattr(gold_entities, '__iter__') else []

        gold_set = entity_set_from_list(gold_entities)

        t0 = time.perf_counter()
        pred_entities = predict_fn(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        pred_set = entity_set_from_list(pred_entities)
        tp, fp, fn = compute_f1(gold_set, pred_set)
        per_type_tp += tp
        per_type_fp += fp
        per_type_fn += fn
        processed += 1

    # Aggregate per-type metrics
    per_type = {}
    all_types = set(per_type_tp) | set(per_type_fp) | set(per_type_fn)
    for t in sorted(all_types):
        tp_v, fp_v, fn_v = per_type_tp[t], per_type_fp[t], per_type_fn[t]
        prec = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
        rec = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_type[t] = {
            "tp": tp_v, "fp": fp_v, "fn": fn_v,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    # Micro averages
    total_tp = sum(per_type_tp.values())
    total_fp = sum(per_type_fp.values())
    total_fn = sum(per_type_fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p50_latency = sorted(latencies_ms)[len(latencies_ms) // 2] if latencies_ms else 0.0
    p99_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)] if latencies_ms else 0.0

    return {
        "split": split_name,
        "postings_evaluated": processed,
        "micro": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
        },
        "per_type": per_type,
        "latency_ms": {
            "mean": round(avg_latency, 2),
            "p50": round(p50_latency, 2),
            "p99": round(p99_latency, 2),
        },
    }


def load_spacy_predictor(model_path: str) -> Callable[[str], list[dict]]:
    """Load a spaCy model and return a predict function."""
    import spacy
    nlp = spacy.load(model_path)

    def predict(text: str) -> list[dict]:
        doc = nlp(text[:10_000])
        return [
            {"text": ent.text, "type": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
    return predict


def load_jobbert_predictor(model_path: str) -> Callable[[str], list[dict]]:
    """Load a HuggingFace token-classification model and return a predict function."""
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        "token-classification",
        model=model_path,
        aggregation_strategy="simple",
        device=0,
    )

    def predict(text: str) -> list[dict]:
        results = pipe(text[:10_000])
        return [
            {"text": r["word"], "type": r["entity_group"], "start": r["start"], "end": r["end"]}
            for r in results
        ]
    return predict


def load_onnx_predictor(model_path: str) -> Callable[[str], list[dict]]:
    """Load an ONNX token-classification model and return a predict function."""
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer, pipeline as hf_pipeline

    model = ORTModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = hf_pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    def predict(text: str) -> list[dict]:
        results = pipe(text[:10_000])
        return [
            {"text": r["word"], "type": r["entity_group"], "start": r["start"], "end": r["end"]}
            for r in results
        ]
    return predict


LOADERS = {
    "spacy": load_spacy_predictor,
    "jobbert": load_jobbert_predictor,
    "onnx": load_onnx_predictor,
}


def print_results(results: dict, label: str = ""):
    """Pretty-print evaluation results."""
    m = results["micro"]
    print(f"\n{'=' * 60}")
    print(f"{label} — {results['split']} ({results['postings_evaluated']} postings)")
    print(f"{'=' * 60}")
    print(f"  Micro P/R/F1: {m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    lat = results["latency_ms"]
    print(f"  Latency: mean={lat['mean']:.1f}ms  p50={lat['p50']:.1f}ms  p99={lat['p99']:.1f}ms")
    print(f"  Per-type:")
    for t, s in results["per_type"].items():
        print(f"    {t:<20} P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Evaluate a student model against gold/dev sets")
    parser.add_argument("--model-type", choices=["spacy", "jobbert", "onnx"], required=True)
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--dev", type=Path, help="Dev parquet path")
    parser.add_argument("--gold", type=Path, help="Gold parquet path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    loader = LOADERS[args.model_type]
    predict_fn = loader(args.model_path)

    all_results = {}
    for split_name, path in [("dev", args.dev), ("gold", args.gold)]:
        if path and path.exists():
            logger.info(f"Evaluating on {split_name}: {path}")
            df = pd.read_parquet(path)
            results = evaluate_predictions(df, predict_fn, split_name)
            all_results[split_name] = results
            print_results(results, label=f"{args.model_type} ({args.model_path})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
