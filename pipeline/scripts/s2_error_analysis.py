"""
Error analysis on S2 (spaCy + Haiku teacher) against the 516-posting gold set.

Categorises false positives and false negatives into failure-mode buckets and
writes both a JSON summary and a paper-ready table of examples.

Usage: python -m pipeline.scripts.s2_error_analysis
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "pipeline/training/experiments/outputs/s2_spacy_haiku/model-best"
HUB_ID = "AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1"
GOLD = ROOT / "pipeline/training/haiku/gold.parquet"
OUT = ROOT / "pipeline/training/s2_error_analysis.json"


def resolve_s2_path() -> str:
    """Local first, HuggingFace Hub fallback. Mirrors the resolver in bootstrap_ci.py."""
    if MODEL.exists():
        return str(MODEL)
    print(f"Local S2 model not found at {MODEL}; pulling from HF Hub: {HUB_ID}")
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=HUB_ID)


def entity_set(entities):
    if entities is None:
        return set()
    if hasattr(entities, "tolist"):
        entities = entities.tolist()
    return {
        (e["text"], e["type"])
        for e in entities
        if isinstance(e, dict) and e.get("text") and e.get("type")
    }


def main():
    random.seed(0)
    nlp = spacy.load(resolve_s2_path())
    gold_df = pd.read_parquet(GOLD)

    fp_by_type = defaultdict(list)
    fn_by_type = defaultdict(list)
    total_fp = Counter()
    total_fn = Counter()

    for _, row in gold_df.iterrows():
        text = row["job_summary"]
        gold = entity_set(row["entities"])
        doc = nlp(text)
        pred = {(e.text, e.label_) for e in doc.ents}

        for t, typ in pred - gold:
            fp_by_type[typ].append({"text": t, "job_link": row["job_link"]})
            total_fp[typ] += 1
        for t, typ in gold - pred:
            fn_by_type[typ].append({"text": t, "job_link": row["job_link"]})
            total_fn[typ] += 1

    report = {
        "total_fp_by_type": dict(total_fp),
        "total_fn_by_type": dict(total_fn),
        "fp_top_texts": {},
        "fn_top_texts": {},
        "examples": {},
    }

    for typ, lst in fp_by_type.items():
        c = Counter(e["text"].lower() for e in lst)
        report["fp_top_texts"][typ] = c.most_common(10)
    for typ, lst in fn_by_type.items():
        c = Counter(e["text"].lower() for e in lst)
        report["fn_top_texts"][typ] = c.most_common(10)

    # Draw a small sample of examples per type for qualitative inspection
    for typ in sorted(set(list(fp_by_type) + list(fn_by_type))):
        report["examples"][typ] = {
            "fp_sample": random.sample(fp_by_type.get(typ, []),
                                       min(8, len(fp_by_type.get(typ, [])))),
            "fn_sample": random.sample(fn_by_type.get(typ, []),
                                       min(8, len(fn_by_type.get(typ, [])))),
        }

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT}")
    print("\nTotal FP/FN by type:")
    for typ in sorted(set(list(fp_by_type) + list(fn_by_type))):
        print(f"  {typ:18s} FP={total_fp.get(typ, 0):4d}  FN={total_fn.get(typ, 0):4d}")

    print("\nFP top texts per type (sanity):")
    for typ in ["SKILL", "LOCATION", "EXPERIENCE_LEVEL", "COMPANY"]:
        print(f"  {typ}: {report['fp_top_texts'].get(typ, [])[:5]}")


if __name__ == "__main__":
    main()
