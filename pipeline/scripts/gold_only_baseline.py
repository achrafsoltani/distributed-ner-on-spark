"""
Non-distillation baseline: train spaCy en_core_web_lg on the 516-posting gold set
alone, 5-fold cross-validation. Answers the reviewer question: "does the LLM
teacher add anything beyond plain supervision on a small human-annotated set?"

Produces pipeline/training/gold_only_baseline.json with aggregate + per-fold F1,
P, R, and a bootstrap CI over the concatenated fold predictions.

Usage:
    python -m pipeline.scripts.gold_only_baseline
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import DocBin, Doc, Span
from spacy.training import Example
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "pipeline/training/haiku/gold.parquet"
OUT = ROOT / "pipeline/training/gold_only_baseline.json"
N_FOLDS = 5
N_ITER = 15
DROPOUT = 0.2
SEED = 42
N_BOOTSTRAP = 10_000
BATCH_SIZE = 8


def entity_set(entities) -> set:
    if entities is None:
        return set()
    if hasattr(entities, "tolist"):
        entities = entities.tolist()
    return {(e["text"], e["type"]) for e in entities
            if isinstance(e, dict) and e.get("text") and e.get("type")}


def find_offsets(text: str, span_text: str) -> tuple[int, int] | None:
    idx = text.find(span_text)
    return (idx, idx + len(span_text)) if idx != -1 else None


def make_docs(df: pd.DataFrame, nlp) -> list[Doc]:
    docs = []
    for _, row in df.iterrows():
        text = row["job_summary"]
        doc = nlp.make_doc(text)
        ents = []
        seen = set()
        entities = row["entities"]
        if hasattr(entities, "tolist"):
            entities = entities.tolist()
        for e in entities:
            if not (isinstance(e, dict) and e.get("text") and e.get("type")):
                continue
            off = find_offsets(text, e["text"])
            if off is None:
                continue
            start, end = off
            if (start, end) in seen:
                continue
            span = doc.char_span(start, end, label=e["type"], alignment_mode="contract")
            if span is None:
                continue
            ents.append(span)
            seen.add((start, end))
        # Resolve overlaps: keep longest non-overlapping
        ents.sort(key=lambda s: (s.start, -(s.end - s.start)))
        filtered: list[Span] = []
        taken: set[int] = set()
        for s in ents:
            if any(i in taken for i in range(s.start, s.end)):
                continue
            filtered.append(s)
            taken.update(range(s.start, s.end))
        try:
            doc.set_ents(filtered)
        except Exception:
            pass
        docs.append(doc)
    return docs


def train_fold(train_df: pd.DataFrame, nlp_base_path: str, labels: set,
               n_iter: int = N_ITER, dropout: float = DROPOUT, seed: int = SEED,
               batch_size: int = BATCH_SIZE):
    nlp = spacy.load(nlp_base_path, disable=["tagger", "attribute_ruler", "lemmatizer", "parser"])
    if "ner" in nlp.pipe_names:
        nlp.remove_pipe("ner")
    ner = nlp.add_pipe("ner")
    for lab in labels:
        ner.add_label(lab)

    train_docs = make_docs(train_df, nlp)
    examples = [Example.from_dict(d, {"entities": [(s.start_char, s.end_char, s.label_) for s in d.ents]}) for d in train_docs]

    optimizer = nlp.initialize(lambda: examples)
    rng = np.random.default_rng(seed)
    for i in range(n_iter):
        rng.shuffle(examples)
        losses = {}
        # Minibatch
        for start in range(0, len(examples), batch_size):
            nlp.update(examples[start:start + batch_size], drop=dropout, sgd=optimizer, losses=losses)
        print(f"    iter {i+1:2d}/{n_iter}: loss_ner={losses.get('ner', 0):.3f}", flush=True)
    return nlp


def predict(nlp, df: pd.DataFrame) -> dict:
    preds = {}
    texts = df["job_summary"].tolist()
    links = df["job_link"].tolist()
    for link, doc in zip(links, nlp.pipe(texts, batch_size=16)):
        preds[link] = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
    return preds


def score(gold_map: dict, pred_map: dict) -> tuple[Counter, Counter, Counter, list[tuple[int, int, int]]]:
    tp, fp, fn = Counter(), Counter(), Counter()
    contribs = []
    for link, gold_ents in gold_map.items():
        g = entity_set(gold_ents)
        p = entity_set(pred_map.get(link, []))
        tp_e = g & p
        fp_e = p - g
        fn_e = g - p
        for _, typ in tp_e: tp[typ] += 1
        for _, typ in fp_e: fp[typ] += 1
        for _, typ in fn_e: fn[typ] += 1
        contribs.append((len(tp_e), len(fp_e), len(fn_e)))
    return tp, fp, fn, contribs


def f1_from_counters(tp: Counter, fp: Counter, fn: Counter) -> dict:
    Tp = sum(tp.values()); Fp = sum(fp.values()); Fn = sum(fn.values())
    P = Tp / (Tp + Fp) if (Tp + Fp) else 0.0
    R = Tp / (Tp + Fn) if (Tp + Fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    per_type = {}
    for typ in set(list(tp) + list(fp) + list(fn)):
        t = tp[typ]; f = fp[typ]; m = fn[typ]
        p = t / (t + f) if (t + f) else 0.0
        r = t / (t + m) if (t + m) else 0.0
        fval = 2 * p * r / (p + r) if (p + r) else 0.0
        per_type[typ] = {"tp": t, "fp": f, "fn": m, "P": round(p, 4), "R": round(r, 4), "F1": round(fval, 4)}
    return {"TP": Tp, "FP": Fp, "FN": Fn,
            "P": round(P, 4), "R": round(R, 4), "F1": round(F, 4),
            "per_type": per_type}


def bootstrap_f1(contribs, n_boot, rng):
    arr = np.array(contribs); n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    sampled = arr[idx]
    tp = sampled[:, :, 0].sum(axis=1)
    fp = sampled[:, :, 1].sum(axis=1)
    fn = sampled[:, :, 2].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where((tp+fp)>0, tp/(tp+fp), 0.0)
        rc = np.where((tp+fn)>0, tp/(tp+fn), 0.0)
        f1 = np.where((pr+rc)>0, 2*pr*rc/(pr+rc), 0.0)
    return {
        "f1_mean": float(f1.mean()),
        "f1_sd": float(f1.std(ddof=1)),
        "f1_ci_lo": float(np.percentile(f1, 2.5)),
        "f1_ci_hi": float(np.percentile(f1, 97.5)),
    }


def main():
    gold_df = pd.read_parquet(GOLD).reset_index(drop=True)
    # Collect label set from gold
    all_labels = set()
    for ents in gold_df["entities"]:
        if hasattr(ents, "tolist"):
            ents = ents.tolist()
        for e in ents:
            if isinstance(e, dict) and e.get("type"):
                all_labels.add(e["type"])
    print(f"Gold postings: {len(gold_df)}, labels: {sorted(all_labels)}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_contribs = []
    all_tp, all_fp, all_fn = Counter(), Counter(), Counter()
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gold_df)):
        train_df = gold_df.iloc[train_idx].reset_index(drop=True)
        test_df = gold_df.iloc[test_idx].reset_index(drop=True)
        print(f"\n[fold {fold_idx+1}/{N_FOLDS}] train={len(train_df)} test={len(test_df)}")
        nlp = train_fold(train_df, "en_core_web_lg", all_labels, seed=SEED + fold_idx)
        pred_map = predict(nlp, test_df)
        gold_map = dict(zip(test_df["job_link"], test_df["entities"]))
        tp, fp, fn, contribs = score(gold_map, pred_map)
        fold_res = f1_from_counters(tp, fp, fn)
        fold_results.append({"fold": fold_idx+1, **fold_res})
        all_contribs.extend(contribs)
        for k, v in tp.items(): all_tp[k] += v
        for k, v in fp.items(): all_fp[k] += v
        for k, v in fn.items(): all_fn[k] += v
        print(f"  F1={fold_res['F1']} P={fold_res['P']} R={fold_res['R']}")

    aggregate = f1_from_counters(all_tp, all_fp, all_fn)
    rng = np.random.default_rng(SEED)
    boot = bootstrap_f1(all_contribs, N_BOOTSTRAP, rng)
    aggregate["bootstrap"] = boot
    report = {
        "n_gold": len(gold_df),
        "n_folds": N_FOLDS,
        "n_iter": N_ITER,
        "dropout": DROPOUT,
        "base_model": "en_core_web_lg",
        "seed": SEED,
        "labels": sorted(all_labels),
        "aggregate": aggregate,
        "fold_results": fold_results,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT}")
    print(f"\n== AGGREGATE ({N_FOLDS}-fold CV on {len(gold_df)} gold postings) ==")
    print(f"F1={aggregate['F1']}  P={aggregate['P']}  R={aggregate['R']}")
    print(f"Bootstrap CI=[{boot['f1_ci_lo']:.4f}, {boot['f1_ci_hi']:.4f}]")


if __name__ == "__main__":
    main()
