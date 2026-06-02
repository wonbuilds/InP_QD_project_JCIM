#!/usr/bin/env python3
"""Extract the InP-only subset (`inp_subset.csv`) from a DD curated-corpus CSV.

By default, if `inp_subset.csv` already exists it is reused — the public
repository ships the deposited subset, whose SHA-256 is verified separately
by `scripts/reproduce_all.py`. Pass `--force` to regenerate, and
`--source <path>` to point at a DD curated-corpus CSV to filter (for example
the 247-record DD corpus from Zenodo, DOI 10.5281/zenodo.20137306).

The filter selects rows with `corpus == "InP (L-series)"` using the Python
stdlib `csv` module, equivalent to:

    df = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[])
    inp = df[df["corpus"] == "InP (L-series)"]

so every cell is preserved as a string and blank cells stay "" (not NaN).

Validation: the extracted InP subset must be 132 rows / 52 papers / 22
distinct system_tags — an invariant across DD corpus versions (the V-series
duplicate-DOI resolution does not affect the InP subset). If that assertion
fails, the script refuses to write inp_subset.csv.

Stdlib-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST = PROJECT_ROOT / "data" / "from_dd" / "inp_subset.csv"

EXPECTED_COLUMNS = 28
EXPECTED_INP_ROWS = 132
EXPECTED_INP_PAPERS = 52
EXPECTED_INP_SYSTEM_TAGS = 22

INP_LITERAL = "InP (L-series)"
VSERIES_LITERAL = "I-III-VI (V-series)"

# JCIM target / input columns to summarize n_missing for InP-only
SUMMARY_COLUMNS = [
    "PL_peak_nm_final",
    "PL_peak_nm_core",
    "QY_percent_final",
    "FWHM_nm_final",
    "T_growth_C",
    "time_min",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract the InP-only subset (inp_subset.csv) from a DD "
                    "curated-corpus CSV.",
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="DD curated-corpus CSV to filter (e.g. the 247-record corpus from "
             "Zenodo, DOI 10.5281/zenodo.20137306). If omitted, the bundled "
             "inp_subset.csv is reused when present.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate inp_subset.csv even if it already exists.",
    )
    args = parser.parse_args(argv)

    # 1. Reuse the deposited subset when present (default path for the public repo).
    if DEST.exists() and not args.force:
        print(
            f"inp_subset.csv already exists ({DEST.relative_to(PROJECT_ROOT)}); "
            "skipping extraction (its SHA-256 is verified separately by "
            "reproduce_all.py). Use --force to regenerate."
        )
        return 0

    # 2. Resolve the source corpus.
    if args.source is None:
        print(
            "Cannot (re)generate inp_subset.csv: no --source corpus was given.\n"
            "  Options:\n"
            "    - Omit --force to reuse the bundled data/from_dd/inp_subset.csv "
            "when present, or\n"
            "    - Download the 247-record DD corpus from Zenodo (DOI "
            "10.5281/zenodo.20137306) and pass it via --source <path>.",
            file=sys.stderr,
        )
        return 1
    source = args.source
    if not source.exists():
        print(f"FAIL source not found: {source}", file=sys.stderr)
        return 1

    actual_sha = sha256_of(source)

    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if len(fieldnames) != EXPECTED_COLUMNS:
        print(
            f"FAIL column count: expected {EXPECTED_COLUMNS}, got {len(fieldnames)}",
            file=sys.stderr,
        )
        return 3

    inp = [r for r in rows if r["corpus"] == INP_LITERAL]
    vseries = [r for r in rows if r["corpus"] == VSERIES_LITERAL]
    other = [r for r in rows if r["corpus"] not in (INP_LITERAL, VSERIES_LITERAL)]
    if other:
        unknown = sorted({r["corpus"] for r in other})
        print(f"FAIL unknown corpus literal(s): {unknown}", file=sys.stderr)
        return 5

    inp_papers = sorted({r["paper_id"] for r in inp})
    inp_system_tags = sorted({r["system_tag"] for r in inp})
    vseries_papers = sorted({r["paper_id"] for r in vseries})
    vseries_system_tags = sorted({r["system_tag"] for r in vseries})

    # InP-subset invariants — the load-bearing correctness checks. These hold
    # across DD corpus versions (the V-series duplicate-DOI resolution does not
    # affect the InP subset), so they are enforced regardless of source corpus.
    checks = [
        ("InP rows", len(inp), EXPECTED_INP_ROWS),
        ("InP distinct papers", len(inp_papers), EXPECTED_INP_PAPERS),
        ("InP distinct system_tags", len(inp_system_tags), EXPECTED_INP_SYSTEM_TAGS),
    ]
    failed = [(name, got, want) for name, got, want in checks if got != want]
    if failed:
        for name, got, want in failed:
            print(f"FAIL {name}: expected {want}, got {got}", file=sys.stderr)
        return 6

    with DEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inp)

    print("=== DD InP subset extraction — PASS ===")
    print(f"  Source              : {source}")
    print(f"  Source SHA-256      : {actual_sha}")
    print(f"  Total rows          : {len(rows)}")
    print(f"  Columns             : {len(fieldnames)}")
    print(f"  InP rows            : {len(inp)}  (expected {EXPECTED_INP_ROWS})")
    print(
        f"  InP papers          : {len(inp_papers)}  (expected {EXPECTED_INP_PAPERS})"
    )
    print(
        f"  InP system_tags     : {len(inp_system_tags)}  (expected {EXPECTED_INP_SYSTEM_TAGS})"
    )
    print(f"  V-series rows       : {len(vseries)}  (informational; varies by corpus version)")
    print(f"  V-series papers     : {len(vseries_papers)}")
    print(f"  V-series system_tags: {len(vseries_system_tags)}")
    print()
    print(f"  Wrote               : {DEST.relative_to(PROJECT_ROOT)}")
    print(f"  Output SHA-256      : {sha256_of(DEST)}")
    print()

    print("--- InP subset: data_quality_tier distribution ---")
    for tier, n in Counter(r["data_quality_tier"] for r in inp).most_common():
        print(f"  {tier:36s} {n:>4d}")
    print()

    print("--- InP subset: route distribution ---")
    for route, n in Counter(r["route"] for r in inp).most_common():
        print(f"  {route:36s} {n:>4d}")
    print()

    print("--- InP subset: measurement_state distribution ---")
    for state, n in Counter(r["measurement_state"] for r in inp).most_common():
        print(f"  {state:36s} {n:>4d}")
    print()

    print("--- InP subset: shell_present_flag distribution ---")
    for flag, n in Counter(r["shell_present_flag"] for r in inp).most_common():
        print(f"  shell_present_flag={flag:6s}            {n:>4d}")
    print()

    print("--- InP subset: caution_count distribution ---")
    for cc, n in sorted(Counter(r["caution_count"] for r in inp).items()):
        print(f"  caution_count={cc:6s}                 {n:>4d}")
    print()

    print("--- InP subset: n_missing for JCIM target/input columns ---")
    for col in SUMMARY_COLUMNS:
        n_missing = sum(1 for r in inp if r[col] == "")
        n_present = len(inp) - n_missing
        print(f"  {col:24s}  present={n_present:>4d}  missing={n_missing:>4d}")
    print()

    print("--- InP subset: distinct system_tags ---")
    for tag in inp_system_tags:
        cnt = sum(1 for r in inp if r["system_tag"] == tag)
        print(f"  ({cnt:>3d})  {tag}")
    print()

    print("--- InP subset: distinct papers ---")
    print("  " + ", ".join(inp_papers))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
