"""
Produce a long-tail latency scatter plot for §4.5: per-posting processing_ms
vs posting length (characters) and entity count.

Writes to paper/figures/longtail_scatter.pdf.

Two input modes, tried in order:

  (1) Public-package sample (default for fresh clones):
        pipeline/output-sample/scaler-sample-10k.parquet
      A 10,000-posting reservoir sample (seed 42) of the full 1.3M Phase 8
      output, pre-joined with character lengths so no Kaggle CSV is required.
      Sample stats match population: processing_ms mean 86.6 ms (vs canonical
      86.4), p99 220 ms (vs 218), char_len mean 3830 (vs 3996). Sufficient to
      reproduce the figure shape and the paper's Pearson correlation claim.

  (2) Full Phase 8 outputs (authors / on-request):
        pipeline/output/extraction/s2_spacy_haiku/raw/*.parquet  (~647 MB, 43 files)
        data-primary/job_summary.csv                              (~5.8 GiB)
      Listed in the paper's "Data and Code Availability" section as available
      from the authors on request. When present, the script uses them and
      reports the population numbers used in the paper.

If neither mode's inputs are present, the script fails with a helpful pointer
to the Data and Code Availability section. To reproduce only the eight Table 5
confidence intervals (which do NOT require these inputs), use bootstrap_ci.py.
"""

from __future__ import annotations

import sys
from glob import glob
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PARQUET = ROOT / "pipeline/output-sample/scaler-sample-10k.parquet"
RAW_GLOB = str(ROOT / "pipeline/output/extraction/s2_spacy_haiku/raw/*.parquet")
INPUT_CSV = str(ROOT / "data-primary/job_summary.csv")
FIG_OUT = ROOT / "paper/figures/longtail_scatter.pdf"


def _load_data():
    """Return (df, mode) where mode is 'sample' or 'full' for figure annotation."""
    raw_matches = glob(RAW_GLOB)
    csv_present = Path(INPUT_CSV).exists()
    if raw_matches and csv_present:
        print(f"[mode=full] Reading {len(raw_matches)} Phase 8 raw parquets and primary CSV via DuckDB...")
        con = duckdb.connect(config={"memory_limit": "30GB"})
        df = con.execute(f"""
          SELECT r.job_link,
                 r.processing_ms,
                 r.entity_count,
                 LENGTH(i.job_summary) AS char_len
          FROM read_parquet('{RAW_GLOB}') r
          LEFT JOIN read_csv_auto('{INPUT_CSV}', header=true, ignore_errors=true, quote='"', escape='"') i
            ON i.job_link = r.job_link
          WHERE r.processing_ms IS NOT NULL
            AND i.job_summary IS NOT NULL
        """).df()
        return df, "full"
    if SAMPLE_PARQUET.exists():
        print(f"[mode=sample] Phase 8 outputs not present; using public-package 10K sample at {SAMPLE_PARQUET.relative_to(ROOT)}.")
        import pandas as pd
        df = pd.read_parquet(SAMPLE_PARQUET)
        return df, "sample"
    print("ERROR: longtail_scatter.py requires either:", file=sys.stderr)
    print(f"  (1) {SAMPLE_PARQUET.relative_to(ROOT)}  — 10K sample shipped in the public package", file=sys.stderr)
    print(f"  (2) {RAW_GLOB}  + {INPUT_CSV}  — full Phase 8 outputs (~647 MB) + primary Kaggle CSV (~5.8 GiB)", file=sys.stderr)
    print("Both are listed in the paper's Data and Code Availability section.", file=sys.stderr)
    print("To reproduce only the eight Table 5 confidence intervals (which do NOT require these inputs), use bootstrap_ci.py.", file=sys.stderr)
    sys.exit(2)


def main():
    df, mode = _load_data()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rows: {len(df):,}; processing_ms mean={df.processing_ms.mean():.2f} p99={df.processing_ms.quantile(0.99):.2f}")

    # Downsample for plotting
    rng = np.random.default_rng(42)
    sample = df.sample(n=min(30000, len(df)), random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    if mode == "sample":
        fig.suptitle(
            f"Long-tail latency (10K-posting reservoir sample of the 1.3M Phase 8 output; full-population figure in paper §4.5)",
            fontsize=9, color="grey", y=1.02,
        )
    ax1.scatter(sample.char_len, sample.processing_ms, s=2, alpha=0.25, edgecolors="none")
    ax1.set_xlabel("Posting length (characters)")
    ax1.set_ylabel("Per-posting processing time (ms)")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_title("Latency vs posting length")
    ax1.grid(alpha=0.3, which="both")

    ax2.scatter(sample.entity_count, sample.processing_ms, s=2, alpha=0.25, edgecolors="none", color="C1")
    ax2.set_xlabel("Entities extracted (per posting)")
    ax2.set_title("Latency vs entity count")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3, which="both")

    # Mark the 14 ms smoke-test mean and 86.4 ms full-run mean on both panels
    for ax in (ax1, ax2):
        ax.axhline(14.1, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(86.4, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(ax.get_xlim()[0]*1.5, 14.1*1.05, "10K smoke mean 14.1 ms",
                fontsize=8, color="grey")
        ax.text(ax.get_xlim()[0]*1.5, 86.4*1.05, "Population mean 86.4 ms",
                fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig(FIG_OUT, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")

    # Also report correlations
    corr_len = np.corrcoef(df.char_len.fillna(0), df.processing_ms)[0, 1]
    corr_ent = np.corrcoef(df.entity_count.fillna(0), df.processing_ms)[0, 1]
    print(f"Pearson corr(char_len, ms) = {corr_len:.3f}")
    print(f"Pearson corr(entity_count, ms) = {corr_ent:.3f}")


if __name__ == "__main__":
    main()
