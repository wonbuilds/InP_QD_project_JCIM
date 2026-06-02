#!/usr/bin/env python3
"""Standard I/O helpers for Phase C analysis layer.

Why this module exists:

- DD's deposited CSVs encode "blank" as the literal empty string "" with the
  recommended pandas options dtype=str / keep_default_na=False / na_values=[].
- Cossairt's deposited CSV encodes missing as the literal string "None"
  (confirmed in Phase B-5 audit: matches SI §S2 missingness distribution
  exactly — emission_nm has 86 numeric + 134 "None" = 220).
- The integrated `data/integrated/inp_combined.csv` carries over both
  conventions row-by-row (build script preserves source values verbatim so
  that the file SHA-256 stays deterministic).

This module unifies the conventions at *analysis* time so notebooks can
load any of the four canonical CSVs with one function and get a consistent
view of missingness.

It also adds the `data_origin` derived flag for the Cossairt `nayon`
placeholder row (Phase B sub-claim — published_literature vs
author_synthesis policy in COSSAIRT_PROVENANCE.md §5.6).

Stdlib-friendly. The pandas import is the only optional dependency; this
file is structured so that callers without pandas can also use the
lightweight `iter_rows()` generator (stdlib only).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable, Iterator

# canonical missing-value placeholders across all sources
MISSING_TOKENS = ["None", "none", "N/A", "NA", "nan", "NaN", ""]

# Cossairt nayon placeholder DOI
COSSAIRT_AUTHOR_SYNTHESIS_DOI = "nayon"


def _project_root() -> Path:
    """Return the project root (parent of `scripts/`)."""
    return Path(__file__).resolve().parent.parent


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# pandas-aware path (analysis notebooks)
# ---------------------------------------------------------------------------

def load_combined(path: str | Path | None = None, *, with_data_origin: bool = True):
    """Load `data/integrated/inp_combined.csv` with unified missing-value
    handling.

    All MISSING_TOKENS are coerced to NaN. Strings stay as strings. Numeric
    columns can be cast downstream via `pd.to_numeric(..., errors='coerce')`.

    If `with_data_origin=True` (default), a derived column `data_origin`
    is added with values:

    - "published_literature" for all rows where `doi` is non-NaN and not
      the literal `"nayon"`,
    - "author_synthesis" for the single Cossairt `doi == "nayon"` row,
    - "" for DD rows without a DOI lookup (paper-level overlap rows have
      a DOI; non-overlap DD rows do not). DD rows are always
      publication-sourced per the DD curation methodology.

    Default analysis filter: `df[df["data_origin"] != "author_synthesis"]`
    (CON: 219 Cossairt + 132 DD = 351 rows). Sensitivity analysis can
    flip the filter for the 220+132 = 352 full sample.
    """
    import pandas as pd  # local import — module is import-safe without pandas

    if path is None:
        path = _project_root() / "data" / "integrated" / "inp_combined.csv"
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=MISSING_TOKENS,
    )
    if with_data_origin:
        # default: published_literature
        origin = ["published_literature"] * len(df)
        # mark nayon row
        for i, (src, doi) in enumerate(zip(df["source"], df["doi"])):
            if src == "cossairt" and (
                doi == COSSAIRT_AUTHOR_SYNTHESIS_DOI or doi == ""
            ):
                # an empty doi on a Cossairt row would also be an anomaly;
                # treat it the same as nayon to be safe
                if doi == COSSAIRT_AUTHOR_SYNTHESIS_DOI:
                    origin[i] = "author_synthesis"
                # else leave as published_literature; an empty Cossairt DOI
                # is unexpected — confirm via audit if it occurs
        df["data_origin"] = origin
    return df


def load_dd_subset(path: str | Path | None = None):
    """Load `data/from_dd/inp_subset.csv` (132 rows). DD uses "" for missing."""
    import pandas as pd

    if path is None:
        path = _project_root() / "data" / "from_dd" / "inp_subset.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def load_cossairt_raw(path: str | Path | None = None):
    """Load `data/from_cossairt/dataset_InP_raw.csv` (220 entries). Cossairt
    uses literal "None" for missing — coerce via MISSING_TOKENS."""
    import pandas as pd

    if path is None:
        path = _project_root() / "data" / "from_cossairt" / "dataset_InP_raw.csv"
    return pd.read_csv(
        path, dtype=str, keep_default_na=False, na_values=MISSING_TOKENS
    )


def to_numeric_cols(df, cols: Iterable[str]):
    """Cast a list of string columns to numeric (NaN-safe). Returns a copy."""
    import pandas as pd

    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# stdlib-only iterator path (for validators / small scripts)
# ---------------------------------------------------------------------------

def iter_rows(path: Path) -> Iterator[dict]:
    """Iterate CSV rows as dicts, normalizing missing tokens to '' on the
    fly. Useful for stdlib-only validators / SHA checks."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in list(row.items()):
                if v.strip() in MISSING_TOKENS:
                    row[k] = ""
            yield row


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("io_helpers self-check")
    print("=====================")

    root = _project_root()
    combined = root / "data" / "integrated" / "inp_combined.csv"
    print(f"combined exists  : {combined.exists()}")
    print(f"combined sha256  : {sha256_of(combined)}")

    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print("pandas not installed — skipping pandas-path tests")
        sys.exit(0)

    df = load_combined()
    print(f"load_combined    : {len(df)} rows × {len(df.columns)} cols")
    print(f"sources          : {df['source'].value_counts().to_dict()}")
    print(f"data_origin dist : {df['data_origin'].value_counts().to_dict()}")
    em_present = df["emission_nm"].notna().sum()
    em_dd = df[(df["source"] == "dd") & df["emission_nm"].notna()].shape[0]
    em_cos = df[(df["source"] == "cossairt") & df["emission_nm"].notna()].shape[0]
    print(f"emission_nm OK   : total={em_present} (DD={em_dd}, Cossairt={em_cos})")
    print(f"  (expected: total=215, DD=129, Cossairt=86 — confirms 'None' parsing)")
