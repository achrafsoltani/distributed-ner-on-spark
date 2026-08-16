"""Paired bootstrap + equivalence bounds for the matched-budget A/B (RQ1).

The published analysis compared S2, B0 and Teacher Haiku through their *marginal*
bootstrap confidence intervals and concluded "indistinguishable" from the fact
that the intervals overlap. That is not a valid test: overlapping CIs neither
establish equivalence nor rule out a real difference, and the conclusion as
drawn accepts the null from a failure to reject.

All three systems are evaluated on the *same* 515 gold postings
(pipeline/training/haiku/gold.parquet), so the correct instrument is a paired
bootstrap on the difference: resample posting indices once per iteration and
score both systems on that same resample, then take the delta. This removes the
between-system correlation that the marginal intervals throw away, and yields a
CI on the difference itself. The largest effect the data can exclude at 95 %
is then read directly off the interval bounds, which is an equivalence bound
rather than a p-value.

Systems compared, all on the identical posting set:
  B0  gold-only spaCy, 5-fold CV, out-of-fold predictions (replicates
      gold_only_baseline.py exactly: KFold(5, shuffle, seed 42), 15 iters,
      dropout 0.2, batch 8, per-fold seed 42+fold_idx)
  S2  spaCy fine-tuned on Haiku labels (pipeline/training/experiments/outputs/
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
        --source <path to the Kaggle postings CSV>
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import KFold
from spacy.tokens import Doc, Span
from spacy.training import Example

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "pipeline/training/haiku/gold.parquet"
S2_MODEL = ROOT / "pipeline/training/experiments/outputs/s2_spacy_haiku/model-best"
S2_HUB_ID = "AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1"
HAIKU_LABELS = ROOT / "pipeline/labels/haiku"
HAIKU_PUBLIC = ROOT / "pipeline/labels-public/haiku/gold-predictions.jsonl"
OUT = ROOT / "pipeline/training/paired_equivalence.json"

N_FOLDS = 5
N_ITER = 15
DROPOUT = 0.2
SEED = 42
BATCH_SIZE = 8
N_BOOTSTRAP = 10_000


# --------------------------------------------------------------------------- #
# scoring primitives (identical semantics to gold_only_baseline.py)
# --------------------------------------------------------------------------- #
def entity_set(entities) -> set:
    if entities is None:
        return set()
    if hasattr(entities, "tolist"):
        entities = entities.tolist()
    return {(e["text"], e["type"]) for e in entities
            if isinstance(e, dict) and e.get("text") and e.get("type")}


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
# B0: replicate the 5-fold CV, keeping out-of-fold predictions
# --------------------------------------------------------------------------- #
def make_docs(df: pd.DataFrame, nlp) -> list[Doc]:
    docs = []
    for _, row in df.iterrows():
        text = row["job_summary"]
        doc = nlp.make_doc(text)
        ents, seen = [], set()
        entities = row["entities"]
        if hasattr(entities, "tolist"):
            entities = entities.tolist()
        for e in entities:
            if not (isinstance(e, dict) and e.get("text") and e.get("type")):
                continue
            i = text.find(e["text"])
            if i == -1 or (i, i + len(e["text"])) in seen:
                continue
            span = doc.char_span(i, i + len(e["text"]), label=e["type"],
                                 alignment_mode="contract")
            if span is None:
                continue
            ents.append(span)
            seen.add((i, i + len(e["text"])))
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


def train_fold(train_df: pd.DataFrame, labels: set, seed: int):
    nlp = spacy.load("en_core_web_lg",
                     disable=["tagger", "attribute_ruler", "lemmatizer", "parser"])
    if "ner" in nlp.pipe_names:
        nlp.remove_pipe("ner")
    ner = nlp.add_pipe("ner")
    for lab in labels:
        ner.add_label(lab)
    docs = make_docs(train_df, nlp)
    examples = [Example.from_dict(
        d, {"entities": [(s.start_char, s.end_char, s.label_) for s in d.ents]})
        for d in docs]
    # spacy-lookups-data 1.0.5 no longer ships an `en` lexeme_norm table, which
    # the default initialize config requests. The table is irrelevant to NER
    # training, so it is disabled rather than substituted.
    nlp.config["initialize"]["lookups"] = None
    optimizer = nlp.initialize(lambda: examples)
    rng = np.random.default_rng(seed)
    for i in range(N_ITER):
        rng.shuffle(examples)
        losses = {}
        for start in range(0, len(examples), BATCH_SIZE):
            nlp.update(examples[start:start + BATCH_SIZE], drop=DROPOUT,
                       sgd=optimizer, losses=losses)
        print(f"    iter {i+1:2d}/{N_ITER}: loss_ner={losses.get('ner', 0):.1f}",
              flush=True)
    return nlp


def predict(nlp, df: pd.DataFrame) -> dict:
    out = {}
    for link, doc in zip(df["job_link"].tolist(),
                         nlp.pipe(df["job_summary"].tolist(), batch_size=16)):
        out[link] = [{"text": e.text, "type": e.label_} for e in doc.ents]
    return out


def b0_out_of_fold(gold_df: pd.DataFrame, labels: set) -> tuple[dict, list[float]]:
    preds: dict = {}
    per_fold_f1: list[float] = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (tr, te) in enumerate(kf.split(gold_df)):
        train_df = gold_df.iloc[tr].reset_index(drop=True)
        test_df = gold_df.iloc[te].reset_index(drop=True)
        print(f"\n[B0 fold {fold_idx+1}/{N_FOLDS}] train={len(train_df)} test={len(test_df)}",
              flush=True)
        nlp = train_fold(train_df, labels, seed=SEED + fold_idx)
        fold_preds = predict(nlp, test_df)
        preds.update(fold_preds)
        gm = dict(zip(test_df["job_link"], test_df["entities"]))
        arr = contribs_for(gm, fold_preds, list(test_df["job_link"]))
        per_fold_f1.append(round(f1_of(arr), 4))
        print(f"  fold F1={per_fold_f1[-1]}", flush=True)
    return preds, per_fold_f1


def teacher_haiku_preds(links: set[str]) -> dict:
    """Raw Haiku teacher predictions for the gold postings.

    Resolution order: the full private batch outputs (internal mirror), else the
    published per-gold-posting subset in pipeline/labels-public/. The two were
    verified identical for all 515 gold links (entity lists compare equal
    record-for-record, 2026-08-16), so the fallback cannot change any score.
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
    labels = set()
    for ents in gold_df["entities"]:
        if hasattr(ents, "tolist"):
            ents = ents.tolist()
        for e in ents:
            if isinstance(e, dict) and e.get("type"):
                labels.add(e["type"])
    print(f"gold postings: {len(gold_df)} | labels: {sorted(labels)}", flush=True)

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
    s2_preds = predict(s2, gold_df)

    print("\n=== Teacher Haiku (raw) ===", flush=True)
    th_preds = teacher_haiku_preds(set(links))
    print(f"matched teacher records: {len(th_preds)}/{len(links)}", flush=True)

    print("\n=== B0 (gold-only, 5-fold CV) ===", flush=True)
    b0_preds, per_fold_f1 = b0_out_of_fold(gold_df, labels)

    A_S2 = contribs_for(gold_map, s2_preds, links)
    A_B0 = contribs_for(gold_map, b0_preds, links)
    A_TH = contribs_for(gold_map, th_preds, links)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(links), size=(N_BOOTSTRAP, len(links)))

    result = {
        "n_postings": len(links),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "spacy_version": spacy.__version__,
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
