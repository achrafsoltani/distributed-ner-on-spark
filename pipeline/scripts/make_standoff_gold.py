"""Stand-off gold annotations: rehydrate them against the corpus, or write them.

The gold set ships as stand-off files: job_link plus the entity list (type,
character offsets and span string), with no posting text. The posting text
stays under the source corpus's terms, so each user rebuilds the text-bearing
gold.parquet from their own copy of the Kaggle release.

  rehydrate rebuilds pipeline/training/{teacher}/gold.parquet, which every
            text-based evaluation script reads, by joining the stand-off file
            to job_summary.csv from the Kaggle release
            asaniczka/1-3m-linkedin-jobs-and-skills-2024 (5.1 GB).
  make      the inverse: writes gold-standoff.parquet from a text-bearing
            gold.parquet (overwriting the shipped stand-off file).

    python -m pipeline.scripts.make_standoff_gold rehydrate --teacher haiku \
        --source /path/to/job_summary.csv

Offsets are verified against the text on both paths: text[start:end] must equal
the span string for every span, so a rehydrated file is provably identical in
content to the one the published numbers were computed on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "pipeline/training"


def _entities(row) -> list[dict]:
    ents = row
    if hasattr(ents, "tolist"):
        ents = ents.tolist()
    return [e for e in (ents or []) if isinstance(e, dict)]


def verify_offsets(df: pd.DataFrame) -> tuple[int, int]:
    """Return (checked, mismatched) for text[start:end] == span text."""
    checked = mismatched = 0
    for _, row in df.iterrows():
        text = row["job_summary"]
        for e in _entities(row["entities"]):
            if "start" not in e or "end" not in e:
                continue
            checked += 1
            if text[e["start"]:e["end"]] != e.get("text"):
                mismatched += 1
    return checked, mismatched


def cmd_make(args: argparse.Namespace) -> int:
    rc = 0
    for teacher in ("sonnet", "haiku"):
        src = TRAINING / teacher / "gold.parquet"
        if not src.exists():
            print(f"SKIP {teacher}: {src} absent")
            continue
        df = pd.read_parquet(src)
        checked, bad = verify_offsets(df)
        out = TRAINING / teacher / "gold-standoff.parquet"
        df[["job_link", "entities"]].to_parquet(out, index=False)
        status = "OK" if bad == 0 else f"WARNING {bad} offset mismatches"
        print(f"{teacher}: {len(df)} postings, {checked} spans offset-verified "
              f"({status}) -> {out.relative_to(ROOT)}")
        if bad:
            rc = 1
    print("\nThese stand-off files carry no posting text. The text-bearing gold.parquet\n"
          "is git-ignored; do not redistribute it.")
    return rc


def cmd_rehydrate(args: argparse.Namespace) -> int:
    standoff = TRAINING / args.teacher / "gold-standoff.parquet"
    if not standoff.exists():
        print(f"error: {standoff} not found; run `make` first or fetch it from the release")
        return 1
    source = Path(args.source)
    if not source.exists():
        print(f"error: source corpus {source} not found. Download the Kaggle "
              f"release and pass its job_summary.csv via --source.")
        return 1

    ann = pd.read_parquet(standoff)
    usecols = None
    head = pd.read_csv(source, nrows=1)
    text_col = next((c for c in ("job_summary", "job_description", "description")
                     if c in head.columns), None)
    if text_col is None:
        print(f"error: no recognised text column in {source} (saw {list(head.columns)}); "
              f"pass job_summary.csv, the file of the Kaggle release that holds the posting text")
        return 1
    usecols = ["job_link", text_col]

    # Stream the 5.1 GB CSV in chunks and keep only the gold postings, so memory
    # stays small; file order is preserved, so the first duplicate still wins.
    wanted = set(ann["job_link"])
    chunks = pd.read_csv(source, usecols=usecols, chunksize=100_000)
    src = pd.concat([c[c["job_link"].isin(wanted)] for c in chunks], ignore_index=True)
    src = src.rename(columns={text_col: "job_summary"})
    merged = ann.merge(src.drop_duplicates("job_link"), on="job_link", how="left")
    missing = int(merged["job_summary"].isna().sum())
    if missing:
        print(f"error: {missing}/{len(merged)} postings absent from the source corpus")
        return 1

    checked, bad = verify_offsets(merged)
    if bad:
        print(f"error: {bad}/{checked} spans do not match the rehydrated text; "
              f"the source corpus differs from the one used for the paper")
        return 1

    out = TRAINING / args.teacher / "gold.parquet"
    merged[["job_link", "job_summary", "entities"]].to_parquet(out, index=False)
    print(f"rehydrated {len(merged)} postings, {checked} spans offset-verified "
          f"-> {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make", help="write text-free stand-off gold files")
    r = sub.add_parser("rehydrate", help="rebuild gold.parquet from your own corpus copy")
    r.add_argument("--teacher", choices=("sonnet", "haiku"), required=True)
    r.add_argument("--source", required=True,
                   help="path to job_summary.csv from the Kaggle release")
    args = ap.parse_args()
    return cmd_make(args) if args.cmd == "make" else cmd_rehydrate(args)


if __name__ == "__main__":
    sys.exit(main())
