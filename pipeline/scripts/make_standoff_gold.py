"""Produce genuinely stand-off gold artefacts, and a rehydrator to rebuild them.

The paper states twice that the posting text is not re-hosted, but the published
`pipeline/training/{sonnet,haiku}/gold.parquet` each carry a `job_summary`
column holding the full posting body (mean ~4,000 characters over 516 + 515
postings). That is re-hosting LinkedIn text under the repository's licences,
and it contradicts both the Data-availability section and the Ethics section.

This script resolves it in the direction the paper already claims, rather than
by weakening the claim:

  make      writes gold-standoff-{sonnet,haiku}.parquet — job_link plus the
            entity list with character offsets, and no posting text. These are
            the files intended for publication.
  rehydrate rebuilds a working gold.parquet by joining the stand-off file to the
            consumer's own Kaggle download, so reproduction still works from a
            fresh clone once the source corpus is in place.

    python -m pipeline.scripts.make_standoff_gold make
    python -m pipeline.scripts.make_standoff_gold rehydrate --teacher haiku \
        --source data-primary/<kaggle-postings>.csv

Offsets are verified against the text on both paths: `text[start:end] == text`
must hold for every span, so a rehydrated file is provably identical in content
to the one used for the published numbers.
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
    print("\nThese stand-off files carry no posting text and are the ones to publish.\n"
          "Remove the text-bearing gold.parquet from the public tree before the next push.")
    return rc


def cmd_rehydrate(args: argparse.Namespace) -> int:
    standoff = TRAINING / args.teacher / "gold-standoff.parquet"
    if not standoff.exists():
        print(f"error: {standoff} not found; run `make` first or fetch it from the release")
        return 1
    source = Path(args.source)
    if not source.exists():
        print(f"error: source corpus {source} not found. Download the Kaggle "
              f"redistribution and pass its postings CSV via --source.")
        return 1

    ann = pd.read_parquet(standoff)
    usecols = None
    head = pd.read_csv(source, nrows=1)
    text_col = next((c for c in ("job_summary", "job_description", "description")
                     if c in head.columns), None)
    if text_col is None:
        print(f"error: no recognised text column in {source} (saw {list(head.columns)})")
        return 1
    usecols = ["job_link", text_col]

    src = pd.read_csv(source, usecols=usecols).rename(columns={text_col: "job_summary"})
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
    r.add_argument("--source", required=True, help="path to the Kaggle postings CSV")
    args = ap.parse_args()
    return cmd_make(args) if args.cmd == "make" else cmd_rehydrate(args)


if __name__ == "__main__":
    sys.exit(main())
