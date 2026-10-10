"""
Non-distillation baseline B0: the spaCy students' pipeline trained on the
515-posting gold set alone, 5-fold cross-validation, scored on the held-out
folds. It asks whether the LLM teacher adds anything beyond plain supervision on
a small gold set.

B0 differs from the spaCy students only in its training data and its fixed
schedule. It mirrors train_spacy.py, the script that trained S1 and S2, and keeps
its own copy of the relevant parts because that script is not part of the public
package:

  pipeline   the students' config: tok2vec sourced from en_core_web_lg and
             fine-tuned, a new NER listening to it (TransitionBasedParser.v2,
             hidden width 64, maxout 2), en_core_web_lg's static vectors, no
             lookups table, seed 42. It is initialised by spaCy's init_nlp, as
             `spacy train` does, so tok2vec keeps its pretrained weights and only
             the NER starts from scratch.
  data       the students' conversion: posting text cut to 10,000 characters,
             gold spans at their stored offsets (char_span, alignment "expand"),
             the longest span kept where spans overlap. Examples are read back
             through spacy.training.Corpus, so no entity is preset on the
             predicted docs.
  optimiser  the students' Adam settings and warm-up schedule (the values in the
             config.cfg shipped with S1 and S2), stepped once per batch as
             `spacy train` steps it.
  schedule   B0's own: 15 passes over the fold's postings, dropout 0.2, batches
             of 8 postings shuffled with seed 42 + fold index, no early stopping.

Predictions are made on the full posting text, as for every evaluated system.
Each fold runs in a fresh worker process that seeds itself and uses one BLAS
thread, so the result does not depend on how many folds run at once.

Produces pipeline/training/gold_only_baseline.json with aggregate + per-fold F1,
P, R, and a bootstrap CI over the concatenated fold predictions, and
pipeline/training/gold_only_baseline_predictions.json with the out-of-fold
predictions themselves, which paired_equivalence.py scores as B0.

Usage:
    python -m pipeline.scripts.gold_only_baseline [--workers N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training.initialize import init_nlp
from spacy.util import fix_random_seed, resolve_dot_names
from sklearn.model_selection import KFold
from thinc.api import Config
from threadpoolctl import threadpool_limits

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "pipeline/training/haiku/gold.parquet"
OUT = ROOT / "pipeline/training/gold_only_baseline.json"
PREDICTIONS = ROOT / "pipeline/training/gold_only_baseline_predictions.json"
BASE_MODEL = "en_core_web_lg"
N_FOLDS = 5
N_ITER = 15
DROPOUT = 0.2
SEED = 42
# spaCy's random state for every fold: the students' seed ([system] seed).
TRAIN_SEED = 42
N_BOOTSTRAP = 10_000
BATCH_SIZE = 8
MAX_CHARS = 10_000  # the students' training texts are cut here (train_spacy.py)

# The students' training config, as train_spacy.generate_config writes it, with
# the values of the config.cfg S1 and S2 were trained with (warm-up 50 steps).
# B0 uses its pipeline, initialisation and optimiser; the [training] loop
# settings (epochs, batcher, dropout, patience) are replaced by B0's schedule.
STUDENT_CONFIG = """
[paths]
train = null
dev = null

[system]
gpu_allocator = "pytorch"
seed = 42

[nlp]
lang = "en"
pipeline = ["tok2vec", "ner"]
batch_size = 1000
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers": "spacy.Tokenizer.v1"}

[components]

[components.tok2vec]
source = "en_core_web_lg"

[components.ner]
factory = "ner"
incorrect_spans_key = null

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = 96
upstream = "*"

[initialize]
vectors = "en_core_web_lg"

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = 42
gpu_allocator = "pytorch"
accumulate_gradient = 1
patience = 1000
max_epochs = 20
max_steps = 0
eval_frequency = 500
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 50
total_steps = 20000
initial_rate = 0.001

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null
"""


def entity_set(entities) -> set:
    if entities is None:
        return set()
    if hasattr(entities, "tolist"):
        entities = entities.tolist()
    return {(e["text"], e["type"]) for e in entities
            if isinstance(e, dict) and e.get("text") and e.get("type")}


def gold_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of the gold set (links, texts, entities, in row order).

    It ignores the parquet encoding, so a rehydrated copy of the gold set and the
    original fingerprint alike when, and only when, their content is identical.
    """
    h = hashlib.sha256()
    for link, text, entities in zip(df["job_link"], df["job_summary"], df["entities"]):
        if hasattr(entities, "tolist"):
            entities = entities.tolist()
        rec = [link, text, [[int(e["start"]), int(e["end"]), e["text"], e["type"]]
                            for e in entities]]
        h.update(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def check_offsets(df: pd.DataFrame) -> None:
    """Every gold span must sit at its stored offsets in this copy of the text."""
    for link, text, entities in zip(df["job_link"], df["job_summary"], df["entities"]):
        for e in entities:
            if text[int(e["start"]):int(e["end"])] != e["text"]:
                raise ValueError(f"{link}: offsets [{e['start']}, {e['end']}) do not hold "
                                 f"{e['text']!r}; the gold text differs from the annotated one")


def to_docbin(df: pd.DataFrame, nlp, output_path: Path | None = None) -> dict:
    """The students' conversion (train_spacy.parquet_to_docbin) of gold rows to a
    DocBin. Returns the same conversion stats."""
    db = DocBin()
    stats = {"total_docs": 0, "total_entities": 0, "aligned": 0, "dropped_overlap": 0, "dropped_misalign": 0}

    for _, row in df.iterrows():
        text = str(row["job_summary"])[:MAX_CHARS]
        doc = nlp.make_doc(text)
        entities = row["entities"]
        if not isinstance(entities, list):
            entities = list(entities) if hasattr(entities, '__iter__') else []

        spans = []
        for ent in entities:
            span = doc.char_span(
                ent["start"], ent["end"], label=ent["type"],
                alignment_mode="expand"
            )
            if span is not None:
                spans.append(span)
                stats["aligned"] += 1
            else:
                stats["dropped_misalign"] += 1

        # Filter overlapping spans: keep longest
        spans = sorted(spans, key=lambda s: s.end - s.start, reverse=True)
        filtered = []
        occupied = set()
        for span in spans:
            token_indices = set(range(span.start, span.end))
            if not token_indices & occupied:
                filtered.append(span)
                occupied |= token_indices
            else:
                stats["dropped_overlap"] += 1

        doc.ents = filtered
        db.add(doc)
        stats["total_docs"] += 1
        stats["total_entities"] += len(entities)

    if output_path is not None:
        db.to_disk(output_path)
    return stats


def params_digest(model) -> str:
    h = hashlib.md5()
    for node in model.walk():
        for name in sorted(node.param_names):
            if node.has_param(name):
                h.update(np.ascontiguousarray(node.get_param(name)).tobytes())
    return h.hexdigest()


def init_checks(nlp, base) -> dict:
    """Assert that the initialised pipeline is the students' one, before training."""
    tok2vec, ner = nlp.get_pipe("tok2vec"), nlp.get_pipe("ner")
    vectors, base_vectors = nlp.vocab.vectors, base.vocab.vectors
    checks = {
        "tok2vec_pretrained": params_digest(tok2vec.model) == params_digest(base.get_pipe("tok2vec").model),
        "ner_listens_to_tok2vec": (tok2vec.listening_components == ["ner"]
                                   and any(n.name == "tok2vec-listener" for n in ner.model.walk())),
        "static_vectors": (any(n.name == "static_vectors" for n in tok2vec.model.walk())
                           and vectors.shape == base_vectors.shape
                           and hashlib.md5(np.ascontiguousarray(vectors.data).tobytes()).digest()
                           == hashlib.md5(np.ascontiguousarray(base_vectors.data).tobytes()).digest()),
        "no_lookups": not nlp.vocab.lookups.tables,
        "ner_labels": sorted(ner.labels),
    }
    failed = [k for k, v in checks.items() if v is False]
    assert not failed, f"B0 is not the students' pipeline: {failed}"
    return checks


def train_fold(train_df: pd.DataFrame, n_iter: int = N_ITER, dropout: float = DROPOUT,
               seed: int = SEED, batch_size: int = BATCH_SIZE, train_seed: int = TRAIN_SEED):
    fix_random_seed(train_seed)
    # train_spacy.py converts with the base model's tokenizer
    base = spacy.load(BASE_MODEL, disable=["ner"])
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "train.spacy"
        conversion = to_docbin(train_df, base, corpus_path)
        # dev is required by the config but unused: B0 has no early stopping
        config = Config().from_str(STUDENT_CONFIG, overrides={
            "paths.train": str(corpus_path), "paths.dev": str(corpus_path)})
        nlp = init_nlp(config, use_gpu=-1)
        checks = init_checks(nlp, base)
        train_corpus = resolve_dot_names(nlp.config.interpolate(), ["corpora.train"])[0]
        examples = list(train_corpus(nlp))

    optimizer = nlp.create_optimizer()
    rng = np.random.default_rng(seed)
    for i in range(n_iter):
        rng.shuffle(examples)
        losses = {}
        for start in range(0, len(examples), batch_size):
            # One step as spacy train takes it (spacy/training/loop.py): update
            # every component, then apply the gradients, then step the schedule.
            nlp.update(examples[start:start + batch_size], drop=dropout, sgd=False, losses=losses)
            for name, proc in nlp.pipeline:
                if (hasattr(proc, "is_trainable") and proc.is_trainable
                        and proc.model not in (True, False, None)):
                    proc.finish_update(optimizer)
            optimizer.step_schedules()
        print(f"    [seed {seed}] iter {i+1:2d}/{n_iter}: loss_ner={losses.get('ner', 0):.3f}",
              flush=True)
    return nlp, conversion, checks


def predict(nlp, df: pd.DataFrame) -> dict:
    preds = {}
    texts = df["job_summary"].tolist()
    links = df["job_link"].tolist()
    for link, doc in zip(links, nlp.pipe(texts, batch_size=16)):
        preds[link] = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
    return preds


def run_fold(fold_idx: int, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Train B0 on one fold's training postings and predict its held-out ones."""
    with threadpool_limits(limits=1):
        nlp, conversion, checks = train_fold(train_df, seed=SEED + fold_idx)
        preds = predict(nlp, test_df)
    return {"conversion": conversion, "init_checks": checks, "predictions": preds}


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
    for typ in sorted(set(list(tp) + list(fp) + list(fn))):
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


def main(workers: int | None = None):
    workers = workers or min(N_FOLDS, os.cpu_count() or 1)
    gold_df = pd.read_parquet(GOLD).reset_index(drop=True)
    check_offsets(gold_df)
    # Collect label set from gold
    all_labels = set()
    for ents in gold_df["entities"]:
        if hasattr(ents, "tolist"):
            ents = ents.tolist()
        for e in ents:
            if isinstance(e, dict) and e.get("type"):
                all_labels.add(e["type"])
    print(f"Gold postings: {len(gold_df)}, labels: {sorted(all_labels)}")

    # The students' conversion over the whole gold set, for the record
    base = spacy.load(BASE_MODEL, disable=["ner"])
    conversion = to_docbin(gold_df, base)
    print(f"Conversion (whole gold set): {conversion}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = [(gold_df.iloc[tr].reset_index(drop=True), gold_df.iloc[te].reset_index(drop=True))
              for tr, te in kf.split(gold_df)]
    print(f"Training {N_FOLDS} folds in {workers} worker process(es)", flush=True)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_fold, k, train_df, test_df)
                   for k, (train_df, test_df) in enumerate(splits)]
        runs = [f.result() for f in futures]

    all_contribs = []
    all_tp, all_fp, all_fn = Counter(), Counter(), Counter()
    fold_results = []
    all_preds, fold_of = {}, {}

    for fold_idx, ((train_df, test_df), run) in enumerate(zip(splits, runs)):
        pred_map = run["predictions"]
        gold_map = dict(zip(test_df["job_link"], test_df["entities"]))
        tp, fp, fn, contribs = score(gold_map, pred_map)
        fold_res = f1_from_counters(tp, fp, fn)
        fold_results.append({"fold": fold_idx+1, "n_train": len(train_df), "n_test": len(test_df),
                             "conversion": run["conversion"], "init_checks": run["init_checks"],
                             **fold_res})
        all_preds.update(pred_map)
        fold_of.update({link: fold_idx+1 for link in test_df["job_link"]})
        all_contribs.extend(contribs)
        for k, v in tp.items(): all_tp[k] += v
        for k, v in fp.items(): all_fp[k] += v
        for k, v in fn.items(): all_fn[k] += v
        print(f"[fold {fold_idx+1}/{N_FOLDS}] train={len(train_df)} test={len(test_df)} "
              f"F1={fold_res['F1']} P={fold_res['P']} R={fold_res['R']}")

    aggregate = f1_from_counters(all_tp, all_fp, all_fn)
    rng = np.random.default_rng(SEED)
    boot = bootstrap_f1(all_contribs, N_BOOTSTRAP, rng)
    aggregate["bootstrap"] = boot
    fingerprint = gold_fingerprint(gold_df)
    report = {
        "n_gold": len(gold_df),
        "n_folds": N_FOLDS,
        "n_iter": N_ITER,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "base_model": BASE_MODEL,
        "base_model_version": base.meta["version"],
        "seed": SEED,
        "train_seed": TRAIN_SEED,
        "shuffle_seed": "seed + fold index",
        "pipeline": ("the spaCy students' (train_spacy.py): tok2vec sourced from en_core_web_lg "
                     "and fine-tuned, new NER listening to it (TransitionBasedParser.v2, "
                     "hidden_width 64, maxout_pieces 2, use_upper), en_core_web_lg static "
                     "vectors, no lookups"),
        "initialisation": ("spacy.training.initialize.init_nlp, as spacy train: sourced tok2vec "
                           "resumed with its pretrained weights, NER initialised"),
        "training_data": (f"the students' conversion: text cut to {MAX_CHARS:,} characters, stored "
                          "offsets via char_span(alignment_mode='expand'), longest span kept where "
                          "spans overlap; Examples from spacy.training.Corpus (unannotated "
                          "predicted docs)"),
        "optimizer": ("the students' [training.optimizer]: Adam.v1 (L2 0.01, grad_clip 1.0, no "
                      "averages), warmup_linear from 0.001 (50 warm-up steps, 20,000 total), "
                      "stepped once per batch as in spacy train"),
        "schedule": (f"{N_ITER} passes, dropout {DROPOUT}, batches of {BATCH_SIZE} postings "
                     "shuffled with seed + fold index, no early stopping, final weights"),
        "prediction": "full posting text",
        "workers": workers,
        "conversion": conversion,
        "python_version": platform.python_version(),
        "spacy_version": spacy.__version__,
        "gold_fingerprint": fingerprint,
        "labels": sorted(all_labels),
        "aggregate": aggregate,
        "fold_results": fold_results,
    }
    OUT.write_text(json.dumps(report, indent=2))
    PREDICTIONS.write_text(json.dumps({
        "gold_fingerprint": fingerprint,
        "n_postings": len(gold_df),
        "fold_f1": [f["F1"] for f in fold_results],
        "fold_of": fold_of,
        "predictions": all_preds,
    }, ensure_ascii=False))
    print(f"\nWrote {OUT}")
    print(f"Wrote {PREDICTIONS}")
    print(f"\n== AGGREGATE ({N_FOLDS}-fold CV on {len(gold_df)} gold postings) ==")
    print(f"F1={aggregate['F1']}  P={aggregate['P']}  R={aggregate['R']}")
    print(f"Bootstrap CI=[{boot['f1_ci_lo']:.4f}, {boot['f1_ci_hi']:.4f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gold-only baseline B0, 5-fold CV")
    ap.add_argument("--workers", type=int, default=None,
                    help=f"folds trained at once (default: min({N_FOLDS}, CPU count)); "
                         f"each fold runs in its own process, so the result is the same")
    main(ap.parse_args().workers)
