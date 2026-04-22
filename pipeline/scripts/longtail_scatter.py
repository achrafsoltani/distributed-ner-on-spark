"""
Produce a long-tail latency scatter plot for §4.5: per-posting processing_ms
vs posting length (characters) and entity count, over the full 1.3M corpus.

Writes to paper/figures/longtail_scatter.pdf.

Inputs required (NOT in the public reproducibility package — these are the
1.3-million-posting Phase 8 inference outputs and the primary Kaggle dataset,
which are listed in the paper's "Data and Code Availability" section as
"available from the authors on request"):

  pipeline/output/extraction/s2_spacy_haiku/raw/*.parquet   (~647 MB across 43 files)
  data-primary/job_summary.csv                              (~5.8 GiB)

If you want to reproduce only the eight Table 5 CIs, you do NOT need this
script — use bootstrap_ci.py instead. Contact the authors if you need the
Phase 8 outputs.
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
ENTITIES_PARQUET = ROOT / "pipeline/output/extraction/s2_spacy_haiku/entities/part-00000.snappy.parquet"
RAW_GLOB = str(ROOT / "pipeline/output/extraction/s2_spacy_haiku/raw/*.parquet")
INPUT_CSV = str(ROOT / "data-primary/job_summary.csv")
FIG_OUT = ROOT / "paper/figures/longtail_scatter.pdf"


def _check_inputs_or_explain() -> None:
    """Fail loudly with a helpful message if the Phase 8 inputs are not present."""
    raw_matches = glob(RAW_GLOB)
    csv_present = Path(INPUT_CSV).exists()
    if raw_matches and csv_present:
        return
    print("ERROR: longtail_scatter.py requires the Phase 8 inference outputs and the primary corpus,", file=sys.stderr)
    print("       both of which are NOT included in the public reproducibility package.", file=sys.stderr)
    print(f"       Missing: ", file=sys.stderr)
    if not raw_matches:
        print(f"         (1) {RAW_GLOB}  — Phase 8 raw inference parquet (~647 MB)", file=sys.stderr)
    if not csv_present:
        print(f"         (2) {INPUT_CSV}  — primary Kaggle dataset (~5.8 GiB)", file=sys.stderr)
    print("       These are listed in the paper's Data and Code Availability section as", file=sys.stderr)
    print("       'available from the authors on request'. To reproduce only the eight Table 5", file=sys.stderr)
    print("       confidence intervals (which do NOT require these inputs), use bootstrap_ci.py.", file=sys.stderr)
    sys.exit(2)


def main():
    _check_inputs_or_explain()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={"memory_limit": "30GB"})
    # Read per-posting timing + entity count from raw, join char length from input
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
    print(f"Rows: {len(df):,}; processing_ms mean={df.processing_ms.mean():.2f} p99={df.processing_ms.quantile(0.99):.2f}")

    # Downsample for plotting
    rng = np.random.default_rng(42)
    sample = df.sample(n=min(30000, len(df)), random_state=42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
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
