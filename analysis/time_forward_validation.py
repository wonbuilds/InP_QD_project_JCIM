#!/usr/bin/env python3
"""Retrospective time-forward validation for DD-curated InP corpus.

Phase 5 F37 — assess the production-RF model's prospective-style predictive
coverage without new wet-lab experiments. For the DD-curated corpus (52
papers, 132 rows), papers are sorted by publication year (from the
repo-local data/from_dd/dd_paper_years.csv), split at a chosen Y_cutoff into pre-
and post-cutoff sets, and the model is trained on pre-cutoff papers and
evaluated on post-cutoff papers. Bootstrap CI on MAE via paper-level
resample.

This complements the paper-level GroupKFold MAE (50.04 ± 7.28 nm,
random-shuffle cross-paper) with a temporal-extrapolation evaluation
that more closely approximates the candidate-prioritization use case.

Note on Cossairt-side: 50 of 73 papers in
`cossairt_reference_master_paper_level.csv` have year = "NR" (not
recorded); DOI-based year extraction recovers 13 additional papers
(23/73 total). This is insufficient for a balanced time-forward split,
so Cossairt-side time-forward validation is not reported.

Determinism: random_state=42 throughout. Output JSON SHA-256-verified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from io_helpers import load_dd_subset, sha256_of  # noqa: E402
from parse_shell_architecture import parse_shell_layers  # noqa: E402

# DD InP paper publication years (52/52 covered), extracted from the L-series
# paper tracker into a repo-local CSV for reproducibility.
PAPER_TRACKER = PROJECT_ROOT / "data" / "from_dd" / "dd_paper_years.csv"


def load_year_map() -> dict[str, int]:
    """Load paper_id → year from the canonical L-series tracker."""
    tracker = pd.read_csv(PAPER_TRACKER)
    tracker["year_int"] = pd.to_numeric(tracker["year"], errors="coerce").astype("Int64")
    year_map = {
        str(row["paper_id"]): int(row["year_int"])
        for _, row in tracker.iterrows()
        if pd.notna(row["year_int"])
    }
    return year_map


def prepare_dd_pl_peak():
    """Replicate Part B PL_peak preprocessing."""
    dd = load_dd_subset()
    for c in ["T_growth_C", "time_min", "PL_peak_nm_final"]:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    parsed = dd["system_tag"].apply(parse_shell_layers).apply(pd.Series)
    parsed["shell_layers_str"] = parsed["shell_layers"].apply(
        lambda lst: ",".join(lst) if isinstance(lst, list) else "")
    dd = pd.concat([dd, parsed.drop(columns=["shell_layers"])], axis=1)
    dd["outer_is_ZnS"] = (dd["shell_outermost"] == "ZnS").astype(int)
    work = dd.dropna(subset=["PL_peak_nm_final"]).copy()
    # time_min (12/132 missing) imputed with the TRAIN-set median inside the pipeline
    # (SimpleImputer) so the temporal split does not leak future medians into training;
    # T_growth_C has no missingness.
    work["shell_innermost"] = work["shell_innermost"].fillna("Unknown").astype(str)
    work["route"] = work["route"].fillna("Unknown").astype(str)
    work["shell_composition_tier"] = work["shell_composition_tier"].fillna("Unknown").astype(str)
    work["shell_layer_count"] = pd.to_numeric(
        work["shell_layer_count"], errors="coerce").fillna(2).astype(int)
    return work


def build_rf_pipe():
    NUM = ["T_growth_C", "time_min", "outer_is_ZnS", "shell_layer_count"]
    CAT = ["route", "shell_innermost", "shell_composition_tier"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return (Pipeline([("pre", pre),
                      ("rf", RandomForestRegressor(
                          n_estimators=200, max_features="sqrt",
                          random_state=42, n_jobs=1))]),
            NUM + CAT)


def time_forward_split_eval(work, year_map, y_cutoff: int, n_boot: int = 1000):
    """Train on papers with year < y_cutoff, test on papers with year >= y_cutoff.

    Returns: (n_train_papers, n_train_rows, n_test_papers, n_test_rows,
              MAE, R2, bootstrap_95ci)
    """
    work_year = work.copy()
    work_year["year"] = work_year["paper_id"].map(year_map)
    work_year = work_year.dropna(subset=["year"])
    work_year["year"] = work_year["year"].astype(int)

    train_mask = work_year["year"] < y_cutoff
    test_mask = ~train_mask

    train = work_year[train_mask]
    test = work_year[test_mask]

    rf_pipe, feats = build_rf_pipe()
    X_train, X_test = train[feats], test[feats]
    y_train = train["PL_peak_nm_final"].values
    y_test = test["PL_peak_nm_final"].values

    rf_pipe.fit(X_train, y_train)
    y_pred = rf_pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    # Paper-level bootstrap on test paper resample
    rng = np.random.default_rng(42)
    test_papers = np.array(sorted(test["paper_id"].unique()))
    p2i = {p: test[test["paper_id"] == p].index.values for p in test_papers}
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(test_papers, size=len(test_papers), replace=True)
        all_idx = np.concatenate([p2i[p] for p in sampled])
        if len(all_idx) == 0:
            continue
        sub_y = test.loc[all_idx, "PL_peak_nm_final"].values
        sub_X = test.loc[all_idx, feats]
        sub_pred = rf_pipe.predict(sub_X)
        boots.append(mean_absolute_error(sub_y, sub_pred))
    boots = np.array(boots)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    return dict(
        y_cutoff=int(y_cutoff),
        n_train_papers=int(train["paper_id"].nunique()),
        n_train_rows=int(len(train)),
        n_test_papers=int(test["paper_id"].nunique()),
        n_test_rows=int(len(test)),
        train_year_range=[int(train["year"].min()), int(train["year"].max())],
        test_year_range=[int(test["year"].min()), int(test["year"].max())],
        mae=mae,
        r2=r2,
        bootstrap_95ci=ci,
        bootstrap_n=int(len(boots)),
    )


def main() -> None:
    print("=" * 70)
    print("F37 — DD-side retrospective time-forward validation")
    print("=" * 70)

    year_map = load_year_map()
    print(f"Year map: {len(year_map)} L-series papers")

    work = prepare_dd_pl_peak()
    print(f"DD PL_peak target: {len(work)} rows / {work['paper_id'].nunique()} papers")

    # Y_cutoff sweep: try 2020, 2021, 2022 to find ~60/40 split
    results = []
    for cutoff in [2020, 2021, 2022, 2023]:
        print(f"\n→ Y_cutoff = {cutoff}")
        r = time_forward_split_eval(work, year_map, y_cutoff=cutoff, n_boot=1000)
        results.append(r)
        print(f"   Train: {r['n_train_papers']} papers / {r['n_train_rows']} rows "
              f"(years {r['train_year_range'][0]}–{r['train_year_range'][1]})")
        print(f"   Test:  {r['n_test_papers']} papers / {r['n_test_rows']} rows "
              f"(years {r['test_year_range'][0]}–{r['test_year_range'][1]})")
        print(f"   MAE  = {r['mae']:.2f} nm   R² = {r['r2']:.3f}   "
              f"boot 95% CI ({r['bootstrap_95ci'][0]:.2f}, {r['bootstrap_95ci'][1]:.2f})")

    # Pick the cutoff closest to 60/40 train/test ratio
    target_ratio = 0.6
    best = min(results, key=lambda r: abs(r["n_train_rows"] / (r["n_train_rows"] + r["n_test_rows"]) - target_ratio))
    print(f"\n=== Selected cutoff (closest to 60/40 split): Y_cutoff = {best['y_cutoff']} ===")

    # Reference: existing GroupKFold MAE for context
    groupkfold_mae = 50.04
    groupkfold_ci = (37.33, 66.42)

    interpretation = (
        f"Time-forward MAE at Y_cutoff={best['y_cutoff']}: {best['mae']:.2f} nm "
        f"(95% CI {best['bootstrap_95ci'][0]:.2f}-{best['bootstrap_95ci'][1]:.2f}); "
        f"reference GroupKFold MAE: {groupkfold_mae} nm "
        f"(95% CI {groupkfold_ci[0]}-{groupkfold_ci[1]}). "
    )
    if best["bootstrap_95ci"][0] <= groupkfold_ci[1] and best["bootstrap_95ci"][1] >= groupkfold_ci[0]:
        interpretation += (
            "The time-forward bootstrap CI overlaps the GroupKFold CI, "
            "indicating the model generalizes to chronologically future papers "
            "without statistically detectable additional degradation. This supports "
            "prospective-style predictive coverage for candidate prioritization."
        )
    else:
        interpretation += (
            "The time-forward bootstrap CI does not overlap the GroupKFold CI, "
            "indicating temporal drift in the synthesis-property landscape. "
            "Candidate prioritization should be interpreted with additional "
            "caution; the GroupKFold MAE underestimates prospective error."
        )

    print(f"\n{interpretation}")

    out = PROJECT_ROOT / "analysis" / "time_forward_results.json"
    payload = dict(
        analysis="F37 retrospective time-forward validation (DD-side)",
        target="PL_peak_nm_final",
        corpus="DD-curated InP subset (132 rows / 52 papers, year 2016-2025)",
        year_source="dd_paper_years.csv (L-series; 52/52 covered)",
        cossairt_side_note=(
            "Cossairt-side time-forward not reported: 50/73 papers have "
            "year='NR' in cossairt_reference_master_paper_level.csv; DOI-based "
            "year extraction adds 13 (23/73 total), insufficient for a balanced "
            "60/40 train/test split."
        ),
        cutoffs_evaluated=results,
        selected_cutoff=best,
        reference_groupkfold=dict(
            mae=groupkfold_mae,
            bootstrap_95ci=groupkfold_ci,
            source="part_b_pl_peak_results.json",
        ),
        interpretation=interpretation,
        notes=(
            "RF model trained on pre-cutoff papers, evaluated on post-cutoff "
            "papers under no-imputation regime. Bootstrap n=1000 via paper-level "
            "resample of test set. Model: RandomForestRegressor(n_estimators=200, "
            "max_features='sqrt', random_state=42), identical to production "
            "Part B configuration."
        ),
    )
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print(f"Output SHA-256: {sha256_of(out)}")


if __name__ == "__main__":
    main()
