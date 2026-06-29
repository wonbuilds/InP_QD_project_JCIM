#!/usr/bin/env python3
"""Cossairt-side emission ablation: 2 x 2 (imputation x split) regime grid.

Phase 3 F17 — quantitative decomposition of the Cossairt-published 11.46 nm
(main text §3.1.3 / §A.3.1) vs audit-grade 27.28 nm gap into two
contributions: imputation effect (regime 1 vs 2) and leakage effect (regime 1
vs 3). The fourth regime (no imputation + paper-level GroupKFold) corresponds
to the audit-grade evaluation adopted in the main text.

Regimes
-------
1. Per-fold KNN-imputed features + random KFold (imputation-sensitivity regime; KNNImputer fit INSIDE each CV fold on train features only; n_rows = 85)
2. No imputation (zero-fill) + random KFold (n_rows = 85)
3. Per-fold KNN-imputed features + paper-level GroupKFold by doi (n_rows = 85)
4. No imputation (zero-fill) + paper-level GroupKFold by doi (audit-grade strict; n_rows = 85)

All four regimes share the SAME target-present 85-row sample; the only axes
are (imputer: per-fold KNN vs zero-fill) x (split: random KFold vs GroupKFold).
The target (emission_nm) is never imputed nor placed in the feature matrix.

Model: ExtraTreesRegressor with Cossairt's official hyperparameters
(n_estimators=3, max_features=13, random_state=51). Mirrors part_a3
reproduction.

Bootstrap CI: paper-level resample (1000 iterations) for paper-level regimes;
row-level resample (1000 iterations) for random-split regimes.

Outputs
-------
- analysis/cossairt_ablation_results.json — full results dict including
  per-regime MAE, R², per-fold values, bootstrap CI, decomposition arithmetic.

Determinism: random_state=42 throughout where applicable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from io_helpers import load_cossairt_raw, sha256_of  # noqa: E402

COS_HP = dict(n_estimators=3, max_features=13, random_state=51)

COS_NUMERIC = [
    "in_amount_mmol", "p_amount_mmol", "first_sol_amount_ml",
    "second_sol_amount_ml", "acid_amount_mmol", "ligand_amount_mmol",
    "other_1_amount_mmol", "other_2_amount_mmol", "total_volume_ml",
    "temp_c", "time_min",
]
COS_CATEGORICAL = [
    "in_source", "p_source", "first_sol", "second_sol",
    "acid", "ligand_source", "other_1", "other_2",
]


def prepare_features(cos_df: pd.DataFrame, *, impute: str) -> tuple:
    """Prepare X, y, groups for a given imputation strategy.

    impute='knn': feature NaNs are LEFT INTACT (the per-fold KNNImputer in the
                  CV pipeline imputes them, fit on train features only); the
                  target is never imputed nor placed in X. Target-present rows
                  only (n=85). Categorical 'None' fill.
    impute='none': fillna(0.0) on numeric features (Cossairt's actual fallback);
                  target-present rows only (n=85). Categorical 'None' fill.
    """
    # Defensive: the unpublished "nayon" author-synthesis row was removed from
    # the public deposit (GROUP A2); this filter is kept as a safety net.
    pub = cos_df[cos_df["doi"].astype(str) != "nayon"].copy()

    # Categorical fill is identical in both regimes
    for c in COS_CATEGORICAL:
        pub[c] = pub[c].fillna("None").astype(str)

    if impute == "knn":
        # NO global imputation. Feature NaNs are kept and imputed PER FOLD inside
        # the CV pipeline (KNNImputer on numeric features only — see build_pipeline).
        # The target is never imputed nor included in the feature matrix. Use the
        # SAME target-present 85 rows as the 'none' regime.
        for c in COS_NUMERIC:
            pub[c] = pd.to_numeric(pub[c], errors="coerce")  # keep NaN (no fill)
        mask = pd.to_numeric(pub["emission_nm"], errors="coerce").notna()
        sub = pub[mask].copy()
        X = sub[COS_NUMERIC + COS_CATEGORICAL]
        y = pd.to_numeric(sub["emission_nm"], errors="coerce").values
        groups = sub["doi"].values
        return X, y, groups

    if impute == "none":
        # Cossairt's actual fallback for missing numeric: fillna(0.0)
        for c in COS_NUMERIC:
            pub[c] = pd.to_numeric(pub[c], errors="coerce").fillna(0.0)
        # Target subset: drop rows with missing emission_nm
        mask = pd.to_numeric(pub["emission_nm"], errors="coerce").notna()
        sub = pub[mask].copy()
        X = sub[COS_NUMERIC + COS_CATEGORICAL]
        y = pd.to_numeric(sub["emission_nm"], errors="coerce").values
        groups = sub["doi"].values
        return X, y, groups

    raise ValueError(f"unknown impute={impute!r}")


def build_pipeline(impute: str = "none") -> Pipeline:
    if impute == "knn":
        # Per-fold KNN imputation of NUMERIC features only (fit on each train fold).
        num_tf = Pipeline([
            ("knn", KNNImputer(n_neighbors=5, weights="uniform")),
            ("sc", StandardScaler()),
        ])
    else:
        num_tf = StandardScaler()
    pre = ColumnTransformer([
        ("num", num_tf, COS_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COS_CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("et", ExtraTreesRegressor(**COS_HP))])


def evaluate_regime(X, y, groups, *, split: str, impute: str = "none", n_boot: int = 1000) -> dict:
    """Run 5-fold CV under the requested split strategy + bootstrap CI on MAE."""
    pipe = build_pipeline(impute)
    if split == "random":
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_args = dict(cv=cv)
    elif split == "groupkfold_doi":
        cv = GroupKFold(n_splits=5)
        cv_args = dict(cv=cv, groups=groups)
    else:
        raise ValueError(f"unknown split={split!r}")

    mae_arr = -cross_val_score(pipe, X, y, scoring="neg_mean_absolute_error",
                               n_jobs=1, **cv_args)
    r2_arr = cross_val_score(pipe, X, y, scoring="r2", n_jobs=1, **cv_args)

    # Bootstrap CI on MAE
    rng = np.random.default_rng(42)
    n = len(y)
    if split == "groupkfold_doi":
        # Paper-level resample
        unique_papers = np.array(sorted(set(groups)))
        p2i = {p: np.where(groups == p)[0] for p in unique_papers}
        boots = []
        for _ in range(n_boot):
            sampled = rng.choice(unique_papers, size=len(unique_papers), replace=True)
            train_i = np.concatenate([p2i[p] for p in sampled])
            held = np.setdiff1d(np.arange(n), np.unique(train_i))
            if len(held) == 0:
                continue
            pipe.fit(X.iloc[train_i], y[train_i])
            pred = pipe.predict(X.iloc[held])
            boots.append(mean_absolute_error(y[held], pred))
        boots = np.array(boots)
    else:
        # Row-level resample
        # Fit once on a random 85/15 hold-out, then bootstrap-resample the test predictions
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.15, random_state=42, shuffle=True)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, len(y_te), len(y_te))
            boots[i] = mean_absolute_error(y_te[idx], y_pred[idx])

    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    return dict(
        n_rows=int(n),
        n_papers=int(len(set(groups))),
        mae_mean=float(mae_arr.mean()),
        mae_std=float(mae_arr.std()),
        mae_per_fold=[float(v) for v in mae_arr],
        r2_mean=float(r2_arr.mean()),
        r2_std=float(r2_arr.std()),
        r2_per_fold=[float(v) for v in r2_arr],
        bootstrap_n=int(len(boots)),
        bootstrap_95ci=(ci_lo, ci_hi),
    )


def main() -> None:
    print("=" * 70)
    print("F17 — Cossairt-side emission ablation (2x2 regime grid)")
    print("=" * 70)

    cos_raw = load_cossairt_raw()
    print(f"Cossairt raw: {len(cos_raw)} rows")

    # Prepare X, y, groups for each imputation regime
    X_knn, y_knn, g_knn = prepare_features(cos_raw, impute="knn")
    X_none, y_none, g_none = prepare_features(cos_raw, impute="none")
    print(f"  KNN-imputed: X {X_knn.shape}, y {len(y_knn)}, papers {len(set(g_knn))}")
    print(f"  No-imputation: X {X_none.shape}, y {len(y_none)}, papers {len(set(g_none))}")

    regimes = {}
    for regime_id, label, X, y, g, split, impute in [
        ("R1", "Per-fold KNN features + random KFold", X_knn, y_knn, g_knn, "random", "knn"),
        ("R2", "No imputation (zero-fill) + random KFold", X_none, y_none, g_none, "random", "none"),
        ("R3", "Per-fold KNN features + paper-level GroupKFold (doi)", X_knn, y_knn, g_knn, "groupkfold_doi", "knn"),
        ("R4", "No imputation (zero-fill) + paper-level GroupKFold (doi)", X_none, y_none, g_none, "groupkfold_doi", "none"),
    ]:
        print(f"\n→ Regime {regime_id}: {label}")
        result = evaluate_regime(X, y, g, split=split, impute=impute, n_boot=1000)
        result["label"] = label
        result["split"] = split
        result["imputation"] = impute
        regimes[regime_id] = result
        print(f"   n_rows={result['n_rows']} (papers={result['n_papers']})")
        print(f"   MAE = {result['mae_mean']:.2f} ± {result['mae_std']:.2f} nm   "
              f"R² = {result['r2_mean']:.3f}   "
              f"boot 95% CI ({result['bootstrap_95ci'][0]:.2f}, "
              f"{result['bootstrap_95ci'][1]:.2f}) nm")

    # Decomposition arithmetic
    gap_total = regimes["R4"]["mae_mean"] - regimes["R1"]["mae_mean"]
    imputation_only = regimes["R2"]["mae_mean"] - regimes["R1"]["mae_mean"]
    leakage_only = regimes["R3"]["mae_mean"] - regimes["R1"]["mae_mean"]
    combined_simple_sum = imputation_only + leakage_only
    interaction_residual = gap_total - combined_simple_sum

    decomposition = dict(
        gap_R4_minus_R1=float(gap_total),
        imputation_effect_R2_minus_R1=float(imputation_only),
        leakage_effect_R3_minus_R1=float(leakage_only),
        simple_sum=float(combined_simple_sum),
        interaction_residual=float(interaction_residual),
        interpretation=(
            "gap_total = imputation_effect + leakage_effect + interaction. "
            "If interaction_residual is small (within bootstrap noise), the "
            "two effects are approximately additive and the main-text "
            "claim 'gap is largely explained by stricter evaluation' holds."
        ),
    )

    print("\n=== Decomposition ===")
    for k, v in decomposition.items():
        if isinstance(v, float):
            print(f"  {k} = {v:.2f} nm")
        else:
            print(f"  {k}: {v}")

    out = PROJECT_ROOT / "analysis" / "cossairt_ablation_results.json"
    payload = dict(
        analysis="F17 Cossairt-side emission 2x2 ablation",
        target="emission_nm",
        model=("ExtraTreesRegressor(n_estimators=3, max_features=13, random_state=51) "
               "with Cossairt's official hyperparameters"),
        regimes=regimes,
        decomposition=decomposition,
        notes=("Regimes R1-R4 cover the Cartesian product of imputation "
               "(KNNImputer-based vs none) and split (random KFold vs paper-level GroupKFold). "
               "R1 uses our own KNNImputer-based imputation-sensitivity setting (NOT a "
               "reproduction of Cossairt's sequential model-based imputation procedure); "
               "R4 is the audit-grade strict regime adopted in the main text. R2 and R3 "
               "isolate the imputation and leakage effects respectively. Bootstrap n=1000."),
    )
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print(f"Output SHA-256: {sha256_of(out)}")


if __name__ == "__main__":
    main()
