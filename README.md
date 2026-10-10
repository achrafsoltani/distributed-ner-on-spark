# Distributed NER on Spark — Reproducibility Package

Companion code and gold-set data for the paper *A Scalable LLM-Distilled Named Entity Recognition Framework for Labour-Market Analytics Using Apache Spark* (Soltani and Hanine, 2026).

This repository is **not** the paper. It holds what is needed to reproduce the paper's evaluation numbers from the published models, the gold set and the public Kaggle release: the teacher and student bootstrap confidence intervals, the gold-only B0 baseline, the paired-bootstrap comparison, the S2 error-mode catalogue, the sliding-window-inference and chunked-window-training ablations, and the long-tail latency figure. The teacher-labelling pipeline, the training code of the six v1 students and the Spark deployment code are not part of it.

## Published models

Ten students are on the Hugging Face Hub. The six v1 students cover three architectures trained on each of the two teachers' labels; the four v2 students retrain the JobBERT students on 450-token windows with a 225-token stride, together with their int8 ONNX exports.

| Student | Architecture | Teacher | Gold micro-F1 [95 % CI] | Hub identifier |
|---|---|---|---|---|
| S1 | spaCy `en_core_web_lg` | Claude Sonnet 4.6 | 0.508 [0.494, 0.523] | [`AchrafSoltani/spacy-lg-jobposting-ner-sonnet-v1`](https://huggingface.co/AchrafSoltani/spacy-lg-jobposting-ner-sonnet-v1) |
| **S2** | spaCy `en_core_web_lg` | Claude Haiku 4.5 | **0.534 [0.517, 0.551]** | [`AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1`](https://huggingface.co/AchrafSoltani/spacy-lg-jobposting-ner-haiku-v1) **(recommended)** |
| S3 | `jjzha/jobbert-base-cased` | Claude Sonnet 4.6 | 0.298 [0.282, 0.314] | [`AchrafSoltani/jobbert-ner-sonnet-v1`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v1) |
| S4 | `jjzha/jobbert-base-cased` | Claude Haiku 4.5 | 0.300 [0.284, 0.316] | [`AchrafSoltani/jobbert-ner-haiku-v1`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v1) |
| S5 | S3 → ONNX int8 | Claude Sonnet 4.6 | 0.284 [0.269, 0.299] | [`AchrafSoltani/jobbert-ner-sonnet-v1-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v1-onnx) |
| S6 | S4 → ONNX int8 | Claude Haiku 4.5 | 0.283 [0.268, 0.299] | [`AchrafSoltani/jobbert-ner-haiku-v1-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v1-onnx) |
| S3′ | JobBERT, chunked-window training (v2) | Claude Sonnet 4.6 | 0.326 [0.312, 0.339] | [`AchrafSoltani/jobbert-ner-sonnet-v2`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v2) |
| S4′ | JobBERT, chunked-window training (v2) | Claude Haiku 4.5 | 0.328 [0.313, 0.344] | [`AchrafSoltani/jobbert-ner-haiku-v2`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v2) |
| S5′ | S3′ → ONNX int8 | Claude Sonnet 4.6 | 0.302 [0.289, 0.315] | [`AchrafSoltani/jobbert-ner-sonnet-v2-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-sonnet-v2-onnx) |
| S6′ | S4′ → ONNX int8 | Claude Haiku 4.5 | 0.309 [0.294, 0.324] | [`AchrafSoltani/jobbert-ner-haiku-v2-onnx`](https://huggingface.co/AchrafSoltani/jobbert-ner-haiku-v2-onnx) |

Scores are micro-F1 over per-posting sets of (text, type) pairs, with 95 % intervals from 10,000 posting-level bootstrap resamples (seed 42). Sonnet-side students are scored on all 516 gold postings and Haiku-side students on the 515 that carry Haiku teacher output. The spaCy students read the full posting; the v1 JobBERT and ONNX students read its first 510 tokens; the v2 students use the 450-token sliding-window inference they were trained for.

Chunked-window training gains +0.044 F1 over sliding-window inference on the v1 weights (+0.028 over single-shot v1 inference) but closes only about 17–20 % of the gap to spaCy. The residual concentrates on the soft-boundary types (`SKILL`, `CERT`, `EDUCATION`, `EXPERIENCE_LEVEL`), which points to the token-level training objective rather than to input coverage. Int8 quantisation costs 6–7 % of the v2 F1 for a fourfold reduction in size (431 MB → 109 MB). S2 remains the recommended model: it covers the full posting text, and in the paper's 1.3M-posting Spark run it averaged 86.4 ms per posting inside the executors.

## What's in this repository

```
pipeline/
├── annotation/
│   └── ANNOTATION-GUIDELINES.md         — the rules the 516-posting gold set follows (8 entity types, R1–R8, edge cases)
├── labels-public/
│   ├── sonnet/gold-predictions.jsonl    — Sonnet teacher predictions on the 516 gold postings
│   └── haiku/gold-predictions.jsonl     — Haiku teacher predictions on the 515 Haiku-side gold postings
├── output-sample/
│   └── scaler-sample-10k.parquet        — reservoir sample (9,965 postings) of per-posting latency, entity count and length from the 1.3M-posting Spark run
├── scripts/
│   ├── teacher_f1_standoff.py           — teacher-vs-gold F1 and CIs from the stand-off files alone (no posting text needed)
│   ├── make_standoff_gold.py            — `rehydrate` rebuilds the text-bearing gold.parquet from your copy of the Kaggle release (offset-verified); `make` is the inverse
│   ├── bootstrap_ci.py                  — bootstrap CIs for the two teachers and the six v1 students; pulls student weights from the Hub when not present locally
│   ├── evaluate_student.py              — training-time per-student evaluator (single-shot inference)
│   ├── evaluate_student_sliding.py      — sliding-window inference predictor
│   ├── bootstrap_ci_sliding.py          — bootstrap CIs for sliding-window inference on S3–S6
│   ├── train_jobbert_chunked.py         — chunked-window training of the v2 JobBERT students (self-contained)
│   ├── bootstrap_ci_chunked.py          — bootstrap CIs for the chunked-trained S3′ and S4′ (self-contained)
│   ├── gold_only_baseline.py            — the gold-only B0 baseline (5-fold cross-validation + bootstrap)
│   ├── paired_equivalence.py            — paired bootstrap for S2, B0 and Teacher Haiku on the shared 515 postings
│   ├── s2_error_analysis.py             — the S2 error-mode catalogue; pulls S2 from the Hub when not present locally
│   └── longtail_scatter.py              — the long-tail latency figure, from the shipped sample by default
└── training/
    ├── haiku/gold-standoff.parquet      — 515 gold postings in stand-off form: job_link plus typed spans (character offsets and span strings), no posting text
    ├── sonnet/gold-standoff.parquet     — 516 gold postings in stand-off form
    └── experiments/specs/*.yaml         — training specifications for the six v1 students and the chunked S3′ and S4′

.github/workflows/ci.yml                 — smoke test on every push and weekly: install, imports, compilation, and a check that the stand-off files carry no posting text
requirements.txt                         — Python dependencies (minimum versions)
.env.example                             — optional environment variables (Hugging Face token and cache location)
```

## Gold set

The gold set holds 516 postings from the dev partition, annotated by a human annotator against `pipeline/annotation/ANNOTATION-GUIDELINES.md`. 515 of the 516 postings also carry Haiku teacher output and form the Haiku-side evaluation file.

It ships as **stand-off annotations**: `job_link` plus typed entity spans, each with its character offsets and its span string. The posting text belongs to the Kaggle release and remains under its terms; it is not re-hosted here. Because every score matches entities as (text, type) sets, the teacher F1 can be computed from the stand-off files directly:

```bash
python -m pipeline.scripts.teacher_f1_standoff
# teacher_sonnet: F1=0.5171 CI=[0.5034, 0.5304] ... n=516
# teacher_haiku: F1=0.5411 CI=[0.5243, 0.5575] ... n=515
```

The student evaluations run the models on the posting text, so first rebuild the text-bearing `gold.parquet` from your own copy of the corpus. Download `job_summary.csv` (5.1 GB), the file of the Kaggle release [asaniczka/1-3m-linkedin-jobs-and-skills-2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) that holds the posting text, then run:

```bash
python -m pipeline.scripts.make_standoff_gold rehydrate --teacher haiku  --source /path/to/job_summary.csv
python -m pipeline.scripts.make_standoff_gold rehydrate --teacher sonnet --source /path/to/job_summary.csv
```

Every span is re-verified as `text[start:end] == span string`, so the rebuilt file is provably the one the paper's numbers were computed on. The command streams the CSV and takes under a minute and about 3 GB of memory per teacher. The rebuilt files contain posting text: they are git-ignored and must not be redistributed.

## Reproducing the paper's numbers

Python 3.11 or 3.12:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only torch first
pip install -r requirements.txt
python -m spacy download en_core_web_lg
# ...then rehydrate the gold set as shown above.

# Bootstrap CIs for the two teachers and the six v1 students (downloads about 2 GB of weights from the Hub on first run)
python -m pipeline.scripts.bootstrap_ci

# Gold-only B0 baseline (5-fold cross-validation + bootstrap)
python -m pipeline.scripts.gold_only_baseline

# Paired bootstrap for S2, B0 and Teacher Haiku on the shared 515 postings
python -m pipeline.scripts.paired_equivalence

# S2 error-mode catalogue
python -m pipeline.scripts.s2_error_analysis

# Long-tail latency figure from the shipped sample (writes pipeline/training/longtail_scatter.pdf)
python -m pipeline.scripts.longtail_scatter
```

Each script prints its results and writes them under `pipeline/training/`, where outputs are git-ignored. `paired_equivalence` scores the out-of-fold B0 predictions that `gold_only_baseline` saves to `pipeline/training/gold_only_baseline_predictions.json`, running `gold_only_baseline` first if that file is absent, so both report the same B0.

Sliding-window inference on the published v1 weights:

```bash
python -m pipeline.scripts.bootstrap_ci_sliding --source hub
```

Chunked-window training: the evaluation runs on the published v2 weights (about 40 minutes on CPU with four threads). Retraining additionally needs a CUDA GPU (the specifications train in fp16), the curated teacher train/dev splits (available on request, see below) and `pip install accelerate datasets evaluate seqeval`.

```bash
python -m pipeline.scripts.bootstrap_ci_chunked --source hub
python -m pipeline.scripts.train_jobbert_chunked --spec pipeline/training/experiments/specs/s4_jobbert_chunked_haiku.yaml
```

The int8 ONNX v2 students (S5′, S6′) were scored by a quantisation script that is not part of this package; they use the same chunk-and-aggregate inference as `bootstrap_ci_chunked.py`.

## Artefacts available on request

These are not in the repository; contact the authors:

- **Curated teacher train/dev splits** — `pipeline/training/{sonnet,haiku}/{train,dev}.parquet`, the 80/10/10 partitions every student was trained on
- **Full per-posting output of the 1.3M-posting Spark run** (~647 MB across 43 parquet files), of which `output-sample/scaler-sample-10k.parquet` is a sample
- **Cluster orchestration and cloud-init scripts** for that run
- **Intermediate outputs of the corpus-level analysis**

The 1.3M-posting corpus itself is public on Kaggle: [asaniczka/1-3m-linkedin-jobs-and-skills-2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024).

## Licensing

- **Code** (`pipeline/scripts/`, the training specifications, the CI workflow, `requirements.txt` and `.env.example`): Apache 2.0 — see [`LICENSE-CODE`](LICENSE-CODE).
- **Research content** (annotation guidelines, stand-off gold annotations, teacher predictions on the gold set, the output sample and this documentation): CC BY-NC 4.0 — see [`LICENSE-RESEARCH`](LICENSE-RESEARCH).
- **Model weights on the Hugging Face Hub**: CC BY-NC 4.0, for research and non-commercial evaluation; each model card carries its base model's notice.
- **Posting text**: not distributed here; it remains under the terms of the Kaggle release.

[`LICENSE`](LICENSE) maps each path to its licence.

## Citation

```bibtex
@unpublished{soltani2026distilledner,
  author = {Achraf Soltani and Mohamed Hanine},
  title  = {A Scalable {LLM}-Distilled Named Entity Recognition Framework for Labour-Market Analytics Using {Apache Spark}},
  year   = {2026},
  note   = {Manuscript},
}
```

The publication will be linked here once it appears.

## Contact

- Achraf Soltani — [www.achrafsoltani.com](https://www.achrafsoltani.com)
- Prof. Mohamed Hanine — Université Chouaib Doukkali, El Jadida, Morocco
