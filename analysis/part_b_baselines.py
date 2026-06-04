#!/usr/bin/env python3
"""Part B baseline comparison for PL_peak — hypothesis-prioritization framing.

Phase 3 F18 — quantitative baseline hierarchy for the PL_peak_nm_final
target under paper-level GroupKFold (5-fold), bootstrap n=1000 paper-level
resample. Supports the main-text framing (Discussion §4.4) that the Part B
model functions as a hypothesis-prioritization tool whose value lies in its
improvement over coarse interpretable baselines, not as a high-accuracy
quantitative predictor.

Baselines (in order of complexity)
----------------------------------
1. Shuffled-target negative control — full RF on shuffled labels (target
   leakage prevention sanity).
2. Global mean predictor — constant prediction at training mean.
3. Paper-grouped dummy — predicts paper-mean for the test paper if present in
   training history, else global mean (tests whether paper-level structure
   alone carries signal).
4. T_growth_C only — single-feature linear regression.
5. shell_innermost only — categorical OneHotEncoded linear regression.
6. T + log(time) — 2-feature linear regression (Part A.4 baseline-equivalent).
7. Full RF model — production random forest (Part B configuration).

Determinism: random_state=42 throughout. Output JSON SHA-256-verified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from io_helpers import load_dd_subset, sha256_of  # noqa: E402
from parse_shell_architecture import parse_shell_layers  # noqa: E402


def load_pl_peak_dataset():
    """Replicate the Part B PL_peak preprocessing exactly."""
    dd = load_dd_subset()
    for c in ["T_growth_C", "time_min", "PL_peak_nm_final"]:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    parsed = dd["system_tag"].apply(parse_shell_layers).apply(pd.Series)
    parsed["shell_layers_str"] = parsed["shell_layers"].apply(
        lambda lst: ",".join(lst) if isinstance(lst, list) else "")
    dd = pd.concat([dd, parsed.drop(columns=["shell_layers"])], axis=1)
    dd["outer_is_ZnS"] = (dd["shell_outermost"] == "ZnS").astype(int)
    work = dd.dropna(subset=["PL_peak_nm_final"]).copy()
    # T_growth_C has no missingness; time_min (12/132 missing) is imputed PER FOLD
    # inside each model pipeline (SimpleImputer) to avoid held-out-fold leakage.
    work["log10_time_min"] = np.log10(work["time_min"].clip(lower=1.0))
    work["shell_innermost"] = work["shell_innermost"].fillna("Unknown").astype(str)
    work["route"] = work["route"].fillna("Unknown").astype(str)
    work["shell_composition_tier"] = work["shell_composition_tier"].fillna("Unknown").astype(str)
    work["shell_layer_count"] = pd.to_numeric(
        work["shell_layer_count"], errors="coerce").fillna(2).astype(int)
    return work


def cv_eval(model, X, y, groups, *, n_boot: int = 1000) -> dict:
    """5-fold paper-level GroupKFold + paper-level bootstrap on MAE."""
    gkf = GroupKFold(n_splits=5)
    mae_fold = []
    r2_fold = []
    for tr, te in gkf.split(X, y, groups=groups):
        m = _clone(model)
        m.fit(X.iloc[tr], y[tr])
        pred = m.predict(X.iloc[te])
        mae_fold.append(mean_absolute_error(y[te], pred))
        r2_fold.append(r2_score(y[te], pred))
    mae_fold = np.array(mae_fold)
    r2_fold = np.array(r2_fold)

    # Paper-level bootstrap on MAE
    rng = np.random.default_rng(42)
    papers = np.array(sorted(set(groups)))
    p2i = {p: np.where(groups == p)[0] for p in papers}
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(papers, size=len(papers), replace=True)
        train_i = np.concatenate([p2i[p] for p in sampled])
        held = np.setdiff1d(np.arange(len(y)), np.unique(train_i))
        if len(held) == 0:
            continue
        m = _clone(model)
        m.fit(X.iloc[train_i], y[train_i])
        pred = m.predict(X.iloc[held])
        boots.append(mean_absolute_error(y[held], pred))
    boots = np.array(boots)
    return dict(
        mae_mean=float(mae_fold.mean()),
        mae_std=float(mae_fold.std()),
        mae_per_fold=[float(v) for v in mae_fold],
        r2_mean=float(r2_fold.mean()),
        r2_std=float(r2_fold.std()),
        r2_per_fold=[float(v) for v in r2_fold],
        bootstrap_n=int(len(boots)),
        bootstrap_95ci=(float(np.percentile(boots, 2.5)),
                        float(np.percentile(boots, 97.5))),
    )


def _clone(model):
    """Shallow clone via sklearn.base.clone — preserves hyperparameters."""
    from sklearn.base import clone
    return clone(model)


def make_full_rf():
    """Replicate Part B production RF (caution_count OFF)."""
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


def main() -> None:
    print("=" * 70)
    print("F18 — Part B baselines for PL_peak_nm_final")
    print("=" * 70)
    work = load_pl_peak_dataset()
    y = work["PL_peak_nm_final"].values
    groups = work["paper_id"].values
    print(f"PL_peak target: n={len(y)} rows, papers={len(set(groups))}")

    results = {}

    # 1. Shuffled-target negative control
    print("\n→ Baseline 1: Shuffled-target negative control (full RF)")
    rf_model, rf_feats = make_full_rf()
    X_rf = work[rf_feats]
    rng = np.random.default_rng(42)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    res = cv_eval(rf_model, X_rf, y_shuffled, groups, n_boot=1000)
    results["1_shuffled_negative_control"] = dict(
        label="Shuffled-target negative control (full RF)",
        features=rf_feats, **res)
    print(f"   MAE = {res['mae_mean']:.2f} ± {res['mae_std']:.2f}   R² = {res['r2_mean']:.3f}")

    # 2. Global mean predictor
    print("\n→ Baseline 2: Global mean predictor (constant)")
    dummy_mean = DummyRegressor(strategy="mean")
    X_dummy = work[["T_growth_C"]]  # single col placeholder
    res = cv_eval(dummy_mean, X_dummy, y, groups, n_boot=1000)
    results["2_global_mean"] = dict(
        label="Global mean predictor (constant = training mean)",
        features=[], **res)
    print(f"   MAE = {res['mae_mean']:.2f} ± {res['mae_std']:.2f}   R² = {res['r2_mean']:.3f}")

    # 3. Paper-grouped dummy: predict paper-mean if paper in training, else global mean
    # Custom implementation
    print("\n→ Baseline 3: Paper-grouped dummy")
    gkf = GroupKFold(n_splits=5)
    pg_mae, pg_r2 = [], []
    for tr, te in gkf.split(work[["T_growth_C"]], y, groups=groups):
        train_papers = set(groups[tr])
        global_mean = y[tr].mean()
        paper_means = {p: y[tr][groups[tr] == p].mean() for p in train_papers}
        pred = np.array([paper_means.get(p, global_mean) for p in groups[te]])
        pg_mae.append(mean_absolute_error(y[te], pred))
        pg_r2.append(r2_score(y[te], pred))
    pg_mae = np.array(pg_mae)
    pg_r2 = np.array(pg_r2)
    # Bootstrap CI for paper-grouped dummy
    rng = np.random.default_rng(42)
    papers = np.array(sorted(set(groups)))
    p2i = {p: np.where(groups == p)[0] for p in papers}
    boots = []
    for _ in range(1000):
        sampled = rng.choice(papers, size=len(papers), replace=True)
        train_i = np.concatenate([p2i[p] for p in sampled])
        held = np.setdiff1d(np.arange(len(y)), np.unique(train_i))
        if len(held) == 0:
            continue
        train_papers = set(groups[train_i])
        global_mean = y[train_i].mean()
        paper_means = {p: y[train_i][groups[train_i] == p].mean() for p in train_papers}
        pred = np.array([paper_means.get(p, global_mean) for p in groups[held]])
        boots.append(mean_absolute_error(y[held], pred))
    boots = np.array(boots)
    results["3_paper_grouped_dummy"] = dict(
        label="Paper-grouped dummy (paper-mean if seen, else global mean)",
        features=["paper_id"],
        mae_mean=float(pg_mae.mean()),
        mae_std=float(pg_mae.std()),
        mae_per_fold=[float(v) for v in pg_mae],
        r2_mean=float(pg_r2.mean()),
        r2_std=float(pg_r2.std()),
        r2_per_fold=[float(v) for v in pg_r2],
        bootstrap_n=int(len(boots)),
        bootstrap_95ci=(float(np.percentile(boots, 2.5)),
                        float(np.percentile(boots, 97.5))),
    )
    print(f"   MAE = {pg_mae.mean():.2f} ± {pg_mae.std():.2f}   R² = {pg_r2.mean():.3f}")

    # 4. T_growth_C only (linear)
    print("\n→ Baseline 4: T_growth_C only (linear)")
    lin1 = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("lr", LinearRegression())])
    X1 = work[["T_growth_C"]]
    res = cv_eval(lin1, X1, y, groups, n_boot=1000)
    results["4_T_only_linear"] = dict(
        label="T_growth_C only (single-feature linear regression)",
        features=["T_growth_C"], **res)
    print(f"   MAE = {res['mae_mean']:.2f} ± {res['mae_std']:.2f}   R² = {res['r2_mean']:.3f}")

    # 5. shell_innermost only (categorical OHE + linear)
    print("\n→ Baseline 5: shell_innermost only (categorical linear)")
    pre_cat = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["shell_innermost"]),
    ])
    lin_cat = Pipeline([("pre", pre_cat), ("lr", LinearRegression())])
    X_cat = work[["shell_innermost"]]
    res = cv_eval(lin_cat, X_cat, y, groups, n_boot=1000)
    results["5_shell_innermost_only_linear"] = dict(
        label="shell_innermost only (categorical linear regression)",
        features=["shell_innermost"], **res)
    print(f"   MAE = {res['mae_mean']:.2f} ± {res['mae_std']:.2f}   R² = {res['r2_mean']:.3f}")

    # 6. T + log(time) (2-feature linear, Part A.4 baseline-equivalent)
    print("\n→ Baseline 6: T + log(time) (2-feature linear)")
    lin2 = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("lr", LinearRegression())])
    X2 = work[["T_growth_C", "log10_time_min"]]
    res = cv_eval(lin2, X2, y, groups, n_boot=1000)
    results["6_T_plus_logtime_linear"] = dict(
        label="T_growth_C + log10(time_min) (2-feature linear regression)",
        features=["T_growth_C", "log10_time_min"], **res)
    print(f"   MAE = {res['mae_mean']:.2f} ± {res['mae_std']:.2f}   R² = {res['r2_mean']:.3f}")

    # 7. Full RF (production)
    print("\n→ Baseline 7: Full RF (Part B production)")
    res = cv_eval(rf_model, X_rf, y, groups, n_boot=1000)
    results["7_full_rf_production"] = dict(
        label="Full random forest (Part B production model)",
        features=rf_feats, **res)
    print(f"   MAE = {res['mae_mean']:.2f} ± {res['mae_std']:.2f}   R² = {res['r2_mean']:.3f}")

    # Compute key gaps
    rf_mae = results["7_full_rf_production"]["mae_mean"]
    t_only_mae = results["4_T_only_linear"]["mae_mean"]
    t_log_t_mae = results["6_T_plus_logtime_linear"]["mae_mean"]
    shuffled_mae = results["1_shuffled_negative_control"]["mae_mean"]
    summary = dict(
        improvement_over_T_only=float(t_only_mae - rf_mae),
        improvement_over_T_plus_logtime=float(t_log_t_mae - rf_mae),
        full_vs_shuffled_gap=float(shuffled_mae - rf_mae),
        interpretation=(
            "Full RF improves on T_growth_C-only by improvement_over_T_only nm "
            "and on T + log(time) by improvement_over_T_plus_logtime nm. "
            "The shuffled-target negative control demonstrates the RF's signal "
            "is not numerical artifact (a substantial full_vs_shuffled_gap "
            "indicates meaningful target dependence)."
        ),
    )

    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k} = {v:+.2f} nm")
        else:
            print(f"  {k}: {v}")

    out = PROJECT_ROOT / "analysis" / "part_b_baselines_results.json"
    payload = dict(
        analysis="F18 Part B baselines hierarchy for PL_peak_nm_final",
        target="PL_peak_nm_final",
        n_rows=int(len(y)),
        n_papers=int(len(set(groups))),
        cv="paper-level GroupKFold (5-fold)",
        bootstrap_n=1000,
        bootstrap_strategy="paper-level resample",
        baselines=results,
        summary=summary,
    )
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print(f"Output SHA-256: {sha256_of(out)}")


if __name__ == "__main__":
    main()
