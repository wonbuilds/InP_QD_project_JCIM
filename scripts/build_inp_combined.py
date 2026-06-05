#!/usr/bin/env python3
"""Build integrated InP dataset by union-stacking DD InP subset (132 rows)
and Cossairt raw dataset (220 entries) on a common-schema set of columns.

Output: `data/integrated/inp_combined.csv` (long/stacked format).

Stacking strategy (the project analysis plan §4.3, §4.4):
- Each output row carries a `source` field (`dd` or `cossairt`) so downstream
  analyses can stratify trivially.
- Comparable cross-source columns are normalized to a single shared name
  (emission_nm, T_growth_C, time_min). Source-specific columns are kept as
  separate fields and filled with "" for the non-applicable source.
- `overlap_flag = 1` is set on rows whose paper appears in the
  DD ∩ Cossairt overlap (10 papers, established by the user's paper-level
  matching in `cossairt_reference_master_paper_level.csv`). The overlap is
  encoded by DD `paper_id` for the DD side and Cossairt `doi` for the
  Cossairt side, with a paper-id ↔ doi lookup built from
  `cossairt_ref_values_dd_matched.csv`.
- Values stay as strings (no numeric coercion) so blanks remain "" — same
  philosophy as the DD `dtype=str / keep_default_na=False` extraction.

stdlib only (DD validators-style). Deterministic — re-running produces an
identical file (SHA-256 stable).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DD_SUBSET = PROJECT_ROOT / "data" / "from_dd" / "inp_subset.csv"
COSSAIRT_RAW = PROJECT_ROOT / "data" / "from_cossairt" / "dataset_InP_raw.csv"
COSSAIRT_PAPER_LEVEL = (
    PROJECT_ROOT / "data" / "from_cossairt" / "cossairt_reference_master_paper_level.csv"
)
COSSAIRT_DD_MATCHED = (
    PROJECT_ROOT / "data" / "from_cossairt" / "cossairt_ref_values_dd_matched.csv"
)
SCHEMA_OUT = PROJECT_ROOT / "data" / "integrated" / "common_schema.json"
CSV_OUT = PROJECT_ROOT / "data" / "integrated" / "inp_combined.csv"

OUTPUT_COLUMNS = [
    "source",                # "dd" or "cossairt"
    "row_id",                # recipe_id (DD) or cos_raw_row_<n> (Cossairt)
    "paper_id_dd",           # L### (DD only)
    "doi",                   # DOI (Cossairt always; DD only for the 10 overlap papers, via matching)
    "overlap_flag",          # "1" if paper in DD ∩ Cossairt overlap else "0"
    # comparable cross-source numeric fields
    "emission_nm",           # DD: PL_peak_nm_final; Cossairt: emission_nm
    "T_growth_C",            # DD: T_growth_C; Cossairt: temp_c
    "time_min",              # DD: time_min; Cossairt: time_min
    # DD-only fields
    "FWHM_nm",               # DD: FWHM_nm_final
    "QY_percent",            # DD: QY_percent_final
    "PL_peak_nm_core",       # DD: PL_peak_nm_core (3 rows non-blank)
    "route",                 # DD: route
    "system_tag",            # DD: system_tag
    "shell_present_flag",    # DD: shell_present_flag (constant 1 in InP subset)
    "data_quality_tier",     # DD: data_quality_tier (constant literature_structured in InP subset)
    "measurement_state",     # DD: measurement_state (constant final_purified in InP subset)
    "caution_count",         # DD: caution_count
    # Cossairt-only fields
    "abs_nm",                # Cossairt: abs_nm
    "diameter_nm",           # Cossairt: diameter_nm
    "in_source",             # Cossairt: in_source
    "in_amount_mmol",        # Cossairt: in_amount_mmol
    "p_source",              # Cossairt: p_source
    "p_amount_mmol",         # Cossairt: p_amount_mmol
    "ligand_source",         # Cossairt: ligand_source
    "ligand_amount_mmol",    # Cossairt: ligand_amount_mmol
    "first_sol",             # Cossairt: first_sol
    "second_sol",            # Cossairt: second_sol
    "acid",                  # Cossairt: acid
    "acid_amount_mmol",      # Cossairt: acid_amount_mmol
    "total_volume_ml",       # Cossairt: total_volume_ml
    "cossairt_user",         # Cossairt: user (Hao / Florence / Nayon ...)
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def empty_row() -> dict:
    return {c: "" for c in OUTPUT_COLUMNS}


def build_paper_id_to_doi(matched_rows: list[dict]) -> dict[str, str]:
    """L-series paper_id -> DOI (lowercased) from user's match work (10 papers)."""
    mapping = {}
    for r in matched_rows:
        pid = r["paper_id"].strip()
        doi = r["doi"].strip().lower()
        if pid and doi:
            mapping[pid] = doi
    return mapping


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the integrated InP dataset (inp_combined.csv) by "
                    "union-stacking the DD InP subset and the Cossairt raw dataset.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild inp_combined.csv even if it already exists.",
    )
    args = parser.parse_args(argv)

    # 1. Reuse the deposited output when present (default path for the public
    #    repo: the shipped inp_combined.csv is verified separately by
    #    reproduce_all.py). Mirrors scripts/extract_inp_subset.py.
    if CSV_OUT.exists() and not args.force:
        print(
            f"inp_combined.csv already exists ({CSV_OUT.relative_to(PROJECT_ROOT)}); "
            "skipping rebuild (its SHA-256 is verified separately by "
            "reproduce_all.py). Use --force to regenerate."
        )
        return 0

    # 2. Friendly error if a build input is missing and there is no output to reuse.
    missing = [p for p in (DD_SUBSET, COSSAIRT_RAW, COSSAIRT_DD_MATCHED, COSSAIRT_PAPER_LEVEL)
               if not p.exists()]
    if missing:
        names = ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing)
        print(
            f"Cannot (re)build inp_combined.csv: missing input(s): {names}.\n"
            "  Reuse the shipped inp_combined.csv (omit --force), or supply the "
            "missing input(s) before regenerating.",
            file=sys.stderr,
        )
        return 1

    # ----- read sources (preserve strings, blanks stay "") -----
    with DD_SUBSET.open(newline="", encoding="utf-8") as f:
        dd_rows = list(csv.DictReader(f))
    with COSSAIRT_RAW.open(newline="", encoding="utf-8") as f:
        cos_rows = list(csv.DictReader(f))
    with COSSAIRT_DD_MATCHED.open(newline="", encoding="utf-8") as f:
        cos_matched = list(csv.DictReader(f))
    with COSSAIRT_PAPER_LEVEL.open(newline="", encoding="utf-8") as f:
        cos_paper = list(csv.DictReader(f))

    if len(dd_rows) != 132:
        print(f"FAIL dd_rows count: expected 132, got {len(dd_rows)}", file=sys.stderr)
        return 2
    if len(cos_rows) != 220:
        print(f"FAIL cos_rows count: expected 220, got {len(cos_rows)}", file=sys.stderr)
        return 3

    # ----- build paper-id ↔ DOI map for DD overlap rows -----
    paper_to_doi = build_paper_id_to_doi(cos_matched)
    # also derive the set of overlap DOIs (lowercased) from the user's paper-level CSV
    overlap_dois_from_master: set[str] = set()
    overlap_paper_ids: set[str] = set()
    for r in cos_paper:
        if r["matched_to_L_series"].strip().lower() == "yes":
            overlap_paper_ids.add(r["matched_paper_id"].strip())
            d = r["doi"].strip().lower()
            if d and d != "nayon":
                overlap_dois_from_master.add(d)

    # ----- emit DD rows -----
    out_rows: list[dict] = []
    for r in dd_rows:
        pid = r["paper_id"].strip()
        o = empty_row()
        o["source"] = "dd"
        o["row_id"] = r["recipe_id"].strip()
        o["paper_id_dd"] = pid
        o["doi"] = paper_to_doi.get(pid, "")
        o["overlap_flag"] = "1" if pid in overlap_paper_ids else "0"
        o["emission_nm"] = r["PL_peak_nm_final"].strip()
        o["T_growth_C"] = r["T_growth_C"].strip()
        o["time_min"] = r["time_min"].strip()
        o["FWHM_nm"] = r["FWHM_nm_final"].strip()
        o["QY_percent"] = r["QY_percent_final"].strip()
        o["PL_peak_nm_core"] = r["PL_peak_nm_core"].strip()
        o["route"] = r["route"].strip()
        o["system_tag"] = r["system_tag"].strip()
        o["shell_present_flag"] = r["shell_present_flag"].strip()
        o["data_quality_tier"] = r["data_quality_tier"].strip()
        o["measurement_state"] = r["measurement_state"].strip()
        o["caution_count"] = r["caution_count"].strip()
        out_rows.append(o)

    # ----- emit Cossairt rows -----
    for idx, r in enumerate(cos_rows, start=1):
        doi_l = r["doi"].strip().lower()
        o = empty_row()
        o["source"] = "cossairt"
        o["row_id"] = f"cos_raw_row_{idx:03d}"
        o["paper_id_dd"] = ""
        o["doi"] = doi_l
        o["overlap_flag"] = "1" if doi_l in overlap_dois_from_master else "0"
        o["emission_nm"] = r["emission_nm"].strip()
        o["T_growth_C"] = r["temp_c"].strip()
        o["time_min"] = r["time_min"].strip()
        o["abs_nm"] = r["abs_nm"].strip()
        o["diameter_nm"] = r["diameter_nm"].strip()
        o["in_source"] = r["in_source"].strip()
        o["in_amount_mmol"] = r["in_amount_mmol"].strip()
        o["p_source"] = r["p_source"].strip()
        o["p_amount_mmol"] = r["p_amount_mmol"].strip()
        o["ligand_source"] = r["ligand_source"].strip()
        o["ligand_amount_mmol"] = r["ligand_amount_mmol"].strip()
        o["first_sol"] = r["first_sol"].strip()
        o["second_sol"] = r["second_sol"].strip()
        o["acid"] = r["acid"].strip()
        o["acid_amount_mmol"] = r["acid_amount_mmol"].strip()
        o["total_volume_ml"] = r["total_volume_ml"].strip()
        o["cossairt_user"] = r["user"].strip()
        out_rows.append(o)

    # ----- sort: DD first (by recipe_id), then Cossairt (by row_id) -----
    out_rows.sort(key=lambda r: (r["source"] != "dd", r["row_id"]))

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    # ----- summary -----
    dd_emitted = sum(1 for r in out_rows if r["source"] == "dd")
    cos_emitted = sum(1 for r in out_rows if r["source"] == "cossairt")
    dd_overlap = sum(1 for r in out_rows if r["source"] == "dd" and r["overlap_flag"] == "1")
    cos_overlap = sum(
        1 for r in out_rows if r["source"] == "cossairt" and r["overlap_flag"] == "1"
    )
    dd_with_doi = sum(1 for r in out_rows if r["source"] == "dd" and r["doi"])
    cos_with_doi = sum(1 for r in out_rows if r["source"] == "cossairt" and r["doi"])

    print("=== inp_combined.csv build — PASS ===")
    print(f"  DD rows emitted     : {dd_emitted}  (expected 132)")
    print(f"  Cossairt rows       : {cos_emitted}  (expected 220)")
    print(f"  Total rows          : {len(out_rows)}  (expected 352)")
    print(f"  DD overlap_flag=1   : {dd_overlap}  (expected 10)")
    print(f"  Cossairt overlap=1  : {cos_overlap}  (>=10 — counted by DOI literal in 220 entries)")
    print(f"  DD rows with doi    : {dd_with_doi}  (expected 10, only overlap papers)")
    print(f"  Cossairt with doi   : {cos_with_doi}")
    print(f"  Output              : {CSV_OUT.relative_to(PROJECT_ROOT)}")
    print(f"  Output SHA-256      : {sha256_of(CSV_OUT)}")
    print()

    # emission_nm presence stratified
    em_dd = sum(1 for r in out_rows if r["source"] == "dd" and r["emission_nm"])
    em_cos = sum(1 for r in out_rows if r["source"] == "cossairt" and r["emission_nm"])
    t_dd = sum(1 for r in out_rows if r["source"] == "dd" and r["T_growth_C"])
    t_cos = sum(1 for r in out_rows if r["source"] == "cossairt" and r["T_growth_C"])
    tm_dd = sum(1 for r in out_rows if r["source"] == "dd" and r["time_min"])
    tm_cos = sum(1 for r in out_rows if r["source"] == "cossairt" and r["time_min"])

    print("--- comparable-column presence (non-blank) ---")
    print(f"  emission_nm  DD: {em_dd:>3d}/132   Cossairt: {em_cos:>3d}/220")
    print(f"  T_growth_C   DD: {t_dd:>3d}/132   Cossairt: {t_cos:>3d}/220")
    print(f"  time_min     DD: {tm_dd:>3d}/132   Cossairt: {tm_cos:>3d}/220")
    print()

    # write schema metadata
    schema = {
        "version": "1.0",
        "created": "2026-05-21",
        "description": (
            "Common schema for the integrated InP dataset. Stacks DD InP subset "
            "(132 rows from data/from_dd/inp_subset.csv) and Cossairt raw "
            "dataset (220 entries from data/from_cossairt/dataset_InP_raw.csv) "
            "on cross-comparable columns. Source-specific columns are preserved "
            "as separate fields filled with empty string for non-applicable rows."
        ),
        "row_counts": {
            "dd_rows": dd_emitted,
            "cossairt_rows": cos_emitted,
            "total": len(out_rows),
            "dd_overlap_rows": dd_overlap,
            "cossairt_overlap_rows": cos_overlap,
        },
        "filter_rules": {
            "dd_filter": "df[df[\"corpus\"] == \"InP (L-series)\"] (authoritative InP-subset filter)",
            "cossairt_filter": "all 220 entries from dataset_InP_raw.csv (GitHub Cossairt-Lab/Indium-Phosphide master, SHA-256 verified MIT)",
        },
        "overlap_definition": (
            "10 papers in DD ∩ Cossairt, derived from the user's prior "
            "paper-level matching (matched_to_L_series == 'yes' in "
            "cossairt_reference_master_paper_level.csv)."
        ),
        "overlap_paper_ids_dd": sorted(overlap_paper_ids),
        "comparable_columns": {
            "emission_nm": {
                "dd_source": "PL_peak_nm_final (nm)",
                "cossairt_source": "emission_nm (nm)",
                "unit": "nm",
                "notes": "primary cross-validation target (Part A)",
            },
            "T_growth_C": {
                "dd_source": "T_growth_C (°C)",
                "cossairt_source": "temp_c (°C)",
                "unit": "°C",
                "notes": "primary input for cross-source feature comparison",
            },
            "time_min": {
                "dd_source": "time_min (min)",
                "cossairt_source": "time_min (min)",
                "unit": "min",
                "notes": "growth-time definition consistency to verify in Phase C.1",
            },
        },
        "dd_only_columns": [
            "FWHM_nm (FWHM_nm_final, nm)",
            "QY_percent (QY_percent_final, %)",
            "PL_peak_nm_core (nm, 3 rows non-blank, all L037)",
            "route (string vocabulary)",
            "system_tag (string, 22 distinct in InP subset)",
            "shell_present_flag (constant 1 in InP subset)",
            "data_quality_tier (constant literature_structured in InP subset)",
            "measurement_state (constant final_purified in InP subset)",
            "caution_count (DD audit-load metric, do not use as ML target)",
        ],
        "cossairt_only_columns": [
            "abs_nm (absorption peak, nm)",
            "diameter_nm (QD diameter, nm; DD does not report size)",
            "in_source / in_amount_mmol",
            "p_source / p_amount_mmol",
            "ligand_source / ligand_amount_mmol",
            "first_sol / second_sol",
            "acid / acid_amount_mmol",
            "total_volume_ml",
            "cossairt_user (Hao / Florence / Nayon / ...)",
        ],
        "unit_normalization_rules": {
            "T_growth_C": "kept as °C (no conversion); Cossairt temp_c and DD T_growth_C are both °C per audit",
            "time_min": "kept as min (no conversion); both sources report minutes",
            "emission_nm": "kept as nm (no conversion); both sources report PL/emission peak in nm",
            "QY_percent": "DD reports QY in % (0-100); Cossairt does not provide QY",
            "FWHM_nm": "DD reports FWHM in nm; Cossairt does not provide FWHM",
        },
        "missing_value_convention": "empty string \"\" for blank; never NaN/None/NULL in CSV",
        "sources": {
            "dd_subset_path": "data/from_dd/inp_subset.csv",
            "dd_subset_sha256": "9c1d56d9897dbef05acadad01b4b7b4c83e7bebcf34d59a5c8a839e5bab324a5",
            "cossairt_raw_path": "data/from_cossairt/dataset_InP_raw.csv",
            "cossairt_raw_sha256": "33539b8dbd60d434a6d9c27d52b9e51e84c83416405c47ecb6059675dbeedc09",
            "cossairt_paper_level_path": "data/from_cossairt/cossairt_reference_master_paper_level.csv",
            "cossairt_dd_matched_path": "data/from_cossairt/cossairt_ref_values_dd_matched.csv",
        },
        "build_script": "scripts/build_inp_combined.py",
        "output_columns": OUTPUT_COLUMNS,
    }

    with SCHEMA_OUT.open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"  Schema              : {SCHEMA_OUT.relative_to(PROJECT_ROOT)}")
    print(f"  Schema SHA-256      : {sha256_of(SCHEMA_OUT)}")
    print()

    # overlap detail
    print("--- DD overlap rows by paper_id_dd ---")
    overlap_dd_rows = [r for r in out_rows if r["source"] == "dd" and r["overlap_flag"] == "1"]
    paper_count = Counter(r["paper_id_dd"] for r in overlap_dd_rows)
    for pid, n in sorted(paper_count.items()):
        doi = paper_to_doi.get(pid, "(no DOI lookup)")
        print(f"  {pid}  rows={n}  doi={doi}")
    print()

    print("--- Cossairt overlap rows by DOI ---")
    overlap_cos_rows = [r for r in out_rows if r["source"] == "cossairt" and r["overlap_flag"] == "1"]
    doi_count = Counter(r["doi"] for r in overlap_cos_rows)
    for d, n in sorted(doi_count.items()):
        print(f"  doi={d}  rows={n}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
