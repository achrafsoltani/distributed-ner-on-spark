# Distributed NER on Spark — Reproducibility Package

Companion code and gold-set data for the paper *A Scalable LLM-Distilled Named Entity Recognition Framework for Labour-Market Analytics using Apache Spark* (Soltani and Hanine, 2026).

This repository is **not** the paper. It is the minimum needed to reproduce the paper's numerical claims — including the teacher and student bootstrap CIs, the non-distillation B0 baseline, the paired-bootstrap equivalence analysis, the error-mode catalogue, the sliding-window-inference and chunked-window-training ablations, and the long-tail latency figure — against the released models, the annotated gold set, and the on-request artefacts documented at the end.

A reproducibility walk on a fresh clone (2026-04-22) verified that all four main evaluation scripts PASS or PASS-within-bootstrap-noise. In August 2026 the repository history was re-initialised when the gold set moved to stand-off form, so that no reachable commit carries the posting text; the code and models are unchanged.

## Published models

Ten student models trained and evaluated in the paper are on HuggingFace Hub. The six v1 students cover the 3-architecture × 2-teacher matrix from the main paper; the four v2 chunked variants (added 2026-04-25) extend the JobBERT students with 450-token chunked-window training to lift F1 by +0.044 and to remove the 512-token coverage cap.

| Student | Architecture | Teacher | Hub identifier |
|---|---|---|---|
| S1 | spaCy `en_core_web_lg` | Claude Sonnet 4.6 | [`AchrafSoltani/spacy-lg-jobposting-ner-sonnet-v1`](https://huggingface.co/AchrafSoltani/spacy-lg-jobposting-ner-sonnet-v1) |
| **S2** | spaCy `en_core_web_lg` | Claude Haiku 4.5 | [`AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1`](https://huggingface.co/AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1) **(production pick)** |
| S3 | `jjzha/jobbert-base-cased` | Claude Sonnet 4.6 | [`AchrafSoltani/jobbert-ner-sonnet-v1`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v1) |
| S4 | `jjzha/jobbert-base-cased` | Claude Haiku 4.5 | [`AchrafSoltani/jobbert-ner-haiku-v1`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v1) |
| S5 | JobBERT → ONNX int8 | Claude Sonnet 4.6 | [`AchrafSoltani/jobbert-ner-sonnet-v1-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v1-onnx) |
| S6 | JobBERT → ONNX int8 | Claude Haiku 4.5 | [`AchrafSoltani/jobbert-ner-haiku-v1-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v1-onnx) |
| S3' | JobBERT chunked v2 (450-token windows, 225 stride) | Claude Sonnet 4.6 | [`AchrafSoltani/jobbert-ner-sonnet-v2`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v2) |
| S4' | JobBERT chunked v2 (450-token windows, 225 stride) | Claude Haiku 4.5 | [`AchrafSoltani/jobbert-ner-haiku-v2`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v2) |
| S5' | S3' → ONNX int8 | Claude Sonnet 4.6 | [`AchrafSoltani/jobbert-ner-sonnet-v2-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v2-onnx) |
| S6' | S4' → ONNX int8 | Claude Haiku 4.5 | [`AchrafSoltani/jobbert-ner-haiku-v2-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v2-onnx) |

**v2 chunked headline numbers** (gold-set 95 % bootstrap CIs, 10 000 posting-level resamples, seed 42; reproducible via `pipeline/scripts/bootstrap_ci_chunked.py`):

- S3' Sonnet: F1 = 0.3255 [0.312, 0.339] (+0.044 over S3)
- S4' Haiku:  F1 = 0.3284 [0.313, 0.344] (+0.044 over S4)
- S5' Sonnet ONNX-int8: F1 = 0.3022 [0.289, 0.315] (~7 % F1 cost vs S3' for ~4× size reduction)
- S6' Haiku ONNX-int8:  F1 = 0.3089 [0.294, 0.324]

The v2 family closes ~17–20 % of the JobBERT-vs-spaCy F1 gap. The residual is concentrated on noun-phrase soft-boundary entity types (SKILL, EDUCATION, EXPERIENCE_LEVEL) and survives both chunked-inference and chunked-training interventions, suggesting an architecture-task-fit constraint rather than a coverage artefact. The production pick **remains S2** for full-text + sub-100 ms latency on commodity CPU; the v2 family is published for downstream use cases where chunked-window extraction is preferred.

## What's in this repository

```
pipeline/
├── annotation/
│   └── ANNOTATION-GUIDELINES.md              — rules the 516-posting gold set follows (8 entity types, R1–R8, edge cases)
├── labels-public/
│   ├── sonnet/gold-predictions.jsonl         — 516 Sonnet teacher predictions on the gold set (sufficient to reproduce the teacher-vs-gold evaluation from this package alone)
│   └── haiku/gold-predictions.jsonl          — 515 Haiku teacher predictions on the gold set
├── output-sample/
│   └── scaler-sample-10k.parquet             — 10,000-posting reservoir sample of the 1.3M Phase 8 output (sample-mode default for longtail_scatter.py; sample stats match population to within sampling noise)
├── scripts/
│   ├── bootstrap_ci.py                       — reproduces the 95% bootstrap CIs for the teacher-vs-gold and student-vs-gold tables; Hub-aware: pulls student weights on demand if not present locally
│   ├── evaluate_student.py                   — reusable per-student evaluation (single-shot inference)
│   ├── evaluate_student_sliding.py           — sliding-window inference predictor
│   ├── bootstrap_ci_sliding.py               — 4-cell bootstrap CIs for sliding-window inference on S3/S4/S5/S6
│   ├── train_jobbert_chunked.py              — chunked-window training; self-contained, ~470 lines
│   ├── bootstrap_ci_chunked.py               — bootstrap CIs for chunked-trained S3'/S4' weights; self-contained
│   ├── gold_only_baseline.py                 — reproduces the B0 non-distillation baseline (top row of the student-vs-gold table)
│   ├── paired_equivalence.py                 — paired bootstrap + equivalence bounds for S2 vs B0 vs Teacher Haiku on the shared 515 postings (the paper's paired-comparison table)
│   ├── make_standoff_gold.py                 — `make` writes the text-free stand-off gold files; `rehydrate` rebuilds gold.parquet from your own Kaggle download (offset-verified)
│   ├── s2_error_analysis.py                  — reproduces the S2 error-mode catalogue; Hub-aware
│   └── longtail_scatter.py                   — reproduces the long-tail latency scatter; uses output-sample/ by default, falls back to full Phase 8 outputs if available locally
└── training/
    ├── haiku/gold-standoff.parquet           — 515 gold postings, stand-off form: job_link + typed character-offset spans, no posting text
    ├── sonnet/gold-standoff.parquet          — 516 gold postings, stand-off form
    └── experiments/specs/*.yaml              — training hyperparameter specs for the six v1 students plus the chunked S3'/S4' v2 specs

requirements.txt                              — pinned Python dependencies (includes spacy-lookups-data + huggingface_hub for the Hub-aware loaders)
.env.example                                  — environment-variable template (HuggingFace token)
```

## Gold set: stand-off annotations

The 516-posting gold set ships as **stand-off annotations**: `job_link` plus typed entity spans with character offsets into the original posting text. The text itself remains under the original Kaggle/LinkedIn terms and is **not** re-hosted here. To run the evaluation scripts, rebuild the text-bearing `gold.parquet` locally by joining the stand-off files to your own copy of the corpus:

```bash
# One-time: download the corpus from Kaggle (asaniczka/1-3m-linkedin-jobs-and-skills-2024),
# then rehydrate each teacher's gold file. Every span is re-verified as
# text[start:end] == span text, so the result is provably the file the paper's
# numbers were computed on.
python -m pipeline.scripts.make_standoff_gold rehydrate --teacher haiku  --source /path/to/linkedin_job_postings.csv
python -m pipeline.scripts.make_standoff_gold rehydrate --teacher sonnet --source /path/to/linkedin_job_postings.csv
```

## Reproducing the paper's numbers

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU torch first
pip install -r requirements.txt
python -m spacy download en_core_web_lg
# ...then rehydrate the gold set as shown above.

# Eight student/teacher bootstrap CIs on the gold set (Hub-aware: downloads ~2 GB of model weights on first run, cached thereafter)
python -m pipeline.scripts.bootstrap_ci

# B0 non-distillation baseline (5-fold CV + bootstrap), top row of the student-vs-gold table
python -m pipeline.scripts.gold_only_baseline

# Paired bootstrap + equivalence bounds (S2 vs B0 vs Teacher Haiku on the shared 515 postings)
python -m pipeline.scripts.paired_equivalence

# S2 error-mode catalogue
python -m pipeline.scripts.s2_error_analysis

# Long-tail latency scatter — uses the shipped 10K sample by default
python -m pipeline.scripts.longtail_scatter
```

Sliding-window inference ablation — runs end-to-end from this package and the Hub-published v1 weights:

```bash
python -m pipeline.scripts.bootstrap_ci_sliding --source hub
```

Chunked-window training — script ships in this package; runnable end-to-end against the chunked checkpoints (which are an on-request artefact, see below). Documents the methodology for any reviewer who wants to re-train from teacher labels.

```bash
python -m pipeline.scripts.train_jobbert_chunked --spec pipeline/training/experiments/specs/s4_jobbert_chunked_haiku.yaml
python -m pipeline.scripts.bootstrap_ci_chunked --source local
```

## On-request artefacts

These are not in the repo; contact the authors:

- **Chunked-trained S3$'$ and S4$'$ weights** (412 MB each) — can be regenerated from `train_jobbert_chunked.py` + the curated teacher train/dev parquets (also on request)
- **Curated teacher train/dev parquets** — `pipeline/training/{sonnet,haiku}/{train,dev}.parquet`, the 80/10/10 split outputs of the Phase 5b curator
- **Phase 8 cluster-lifecycle orchestrator** + cloud-init scripts
- **Full raw Phase 8 scaler output parquet** (~647 MB across 43 files) — the `output-sample/scaler-sample-10k.parquet` is a reservoir sample of this
- **Analyst intermediate outputs**

The 1.3M-posting primary corpus is publicly available at [Kaggle — asaniczka/1-3m-linkedin-jobs-and-skills-2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024).

## Licensing

- **Code** (`pipeline/scripts/`): Apache 2.0 — see [`LICENSE-CODE`](LICENSE-CODE).
- **Research content** (annotation guidelines, gold set, gold-set teacher predictions): CC BY-NC 4.0 — see [`LICENSE-RESEARCH`](LICENSE-RESEARCH).
- **Model weights on HuggingFace Hub**: CC BY-NC 4.0 — research and non-commercial evaluation.

## Citation

```bibtex
@unpublished{soltani2026distilledner,
  author = {Achraf Soltani and Mohamed Hanine},
  title  = {A Scalable {LLM}-Distilled Named Entity Recognition Framework for Labour-Market Analytics using {Apache Spark}},
  year   = {2026},
  note   = {Manuscript, August 2026},
}
```

Paper venue will be linked here once published.

## Contact

- Achraf Soltani — [www.achrafsoltani.com](https://www.achrafsoltani.com)
- Prof. Mohamed Hanine — Université Chouaib Doukkali, El Jadida, Morocco
