#!/usr/bin/env python3
"""Leave-one-paper-out SHAP stability analysis for Part B PL_peak.

Phase 3 F19 — robustness check on the shell_innermost = ZnSe SHAP attribution
(main-text §3.2.1). For each of the 52 DD-side papers, refit the production
gradient-boosting model with that paper's rows held out, compute SHAP mean
magnitudes, and track top-5 feature rank stability across the 52 folds.

If shell_innermost = ZnSe remains top-3 in ≥ 90 % of folds, the attribution
is robust to single-paper removal and unlikely to be driven by a single
dominant publication.

Determinism: random_state=42 throughout. Output JSON SHA-256-verified.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
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
    # time_min (12/132 missing) imputed PER FOLD inside the GBM pipeline (SimpleImputer);
    # T_growth_C has no missingness. No global fill (avoids LOPO held-out leakage).
    work["shell_innermost"] = work["shell_innermost"].fillna("Unknown").astype(str)
    work["route"] = work["route"].fillna("Unknown").astype(str)
    work["shell_composition_tier"] = work["shell_composition_tier"].fillna("Unknown").astype(str)
    work["shell_layer_count"] = pd.to_numeric(
        work["shell_layer_count"], errors="coerce").fillna(2).astype(int)
    return work


def build_gbm_pipe():
    NUM = ["T_growth_C", "time_min", "outer_is_ZnS", "shell_layer_count"]
    CAT = ["route", "shell_innermost", "shell_composition_tier"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("gbm", GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=42)),
    ])
    return pipe, NUM + CAT


def fold_shap(work, holdout_paper):
    """Refit on all papers except `holdout_paper`; return top-10 SHAP mean magnitudes."""
    train_mask = work["paper_id"] != holdout_paper
    sub = work[train_mask]
    pipe, feats = build_gbm_pipe()
    X = sub[feats]
    y = sub["PL_peak_nm_final"].values
    pipe.fit(X, y)
    Xt = pipe.named_steps["pre"].transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)
    fnames = pipe.named_steps["pre"].get_feature_names_out()
    expl = shap.TreeExplainer(pipe.named_steps["gbm"])
    sv = expl.shap_values(Xt)
    mean_abs = pd.DataFrame(sv, columns=fnames).abs().mean().sort_values(ascending=False)
    return mean_abs


def main() -> None:
    print("=" * 70)
    print("F19 — Leave-one-paper-out SHAP stability for PL_peak_nm_final")
    print("=" * 70)
    work = load_pl_peak_dataset()
    papers = sorted(work["paper_id"].unique())
    print(f"PL_peak target: n={len(work)} rows, papers={len(papers)}")

    # Baseline full-data SHAP for reference (no leave-out)
    baseline = fold_shap(work, holdout_paper="__none__")
    baseline_top5 = baseline.head(5)
    print("\nBaseline (full-data) top-5 SHAP:")
    for i, (name, val) in enumerate(baseline_top5.items(), 1):
        print(f"  {i}. {name}  {val:.4f}")

    # LOPO: 52 folds
    fold_ranks = defaultdict(list)   # feature -> list of ranks (0-indexed)
    fold_magnitudes = defaultdict(list)
    fold_top1 = Counter()
    fold_top3 = Counter()
    fold_top5 = Counter()
    fold_records = []
    for i, paper in enumerate(papers, 1):
        print(f"  Fold {i}/{len(papers)}: hold out {paper}", end=" ", flush=True)
        sv = fold_shap(work, paper)
        rank_map = {feat: int(rank) for rank, feat in enumerate(sv.index)}
        for feat, val in sv.items():
            fold_ranks[feat].append(rank_map[feat])
            fold_magnitudes[feat].append(float(val))
        top1 = sv.index[0]
        top3 = set(sv.index[:3])
        top5 = set(sv.index[:5])
        fold_top1[top1] += 1
        for f in top3:
            fold_top3[f] += 1
        for f in top5:
            fold_top5[f] += 1
        fold_records.append(dict(
            holdout_paper=paper,
            top5=[(str(k), float(v)) for k, v in sv.head(5).items()],
        ))
        print(f"top-1={top1}")

    # Per-feature aggregate
    rank_summary = []
    n_folds = len(papers)
    for feat, ranks in fold_ranks.items():
        ranks_arr = np.array(ranks)
        rank_summary.append(dict(
            feature=feat,
            rank_median=float(np.median(ranks_arr)) + 1.0,  # 1-indexed for display
            rank_iqr_low=float(np.percentile(ranks_arr, 25)) + 1.0,
            rank_iqr_high=float(np.percentile(ranks_arr, 75)) + 1.0,
            rank_min=int(ranks_arr.min()) + 1,
            rank_max=int(ranks_arr.max()) + 1,
            top1_pct=float(fold_top1[feat] / n_folds * 100),
            top3_pct=float(fold_top3[feat] / n_folds * 100),
            top5_pct=float(fold_top5[feat] / n_folds * 100),
            magnitude_mean=float(np.mean(fold_magnitudes[feat])),
            magnitude_std=float(np.std(fold_magnitudes[feat])),
        ))
    rank_summary.sort(key=lambda r: r["rank_median"])

    print("\n=== Top-10 features by median LOPO rank ===")
    print(f"{'rank':>4}  {'feature':40s}  {'rank_med':>8}  {'top-1%':>7}  {'top-3%':>7}  {'top-5%':>7}")
    for i, r in enumerate(rank_summary[:10], 1):
        print(f"  {i:2d}.  {r['feature'][:40]:40s}  {r['rank_median']:>8.1f}  "
              f"{r['top1_pct']:>6.1f}%  {r['top3_pct']:>6.1f}%  {r['top5_pct']:>6.1f}%")

    # Specific ZnSe stability check
    znse_features = [r for r in rank_summary if r["feature"] == "cat__shell_innermost_ZnSe"]
    print("\n=== shell_innermost = ZnSe stability ===")
    for r in znse_features:
        print(f"  {r['feature']}: rank median={r['rank_median']:.1f}, "
              f"top-1 in {r['top1_pct']:.1f}% folds, top-3 in {r['top3_pct']:.1f}% folds, "
              f"top-5 in {r['top5_pct']:.1f}% folds")

    out = PROJECT_ROOT / "analysis" / "part_b_shap_stability_results.json"
    payload = dict(
        analysis="F19 LOPO SHAP stability for PL_peak_nm_final",
        target="PL_peak_nm_final",
        n_papers=int(n_folds),
        n_rows_full=int(len(work)),
        model="GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)",
        baseline_full_data_top10=baseline.head(10).to_dict(),
        rank_summary=rank_summary,
        per_fold_top5=fold_records,
        notes=(
            "Each fold refits the production GBM with one paper's rows held "
            "out and recomputes SHAP TreeExplainer mean-magnitude on the "
            "refit training set. Top-3/top-5 percentages reflect the "
            "fraction of folds where each feature appears in the top-3/top-5 "
            "by SHAP magnitude. A feature that remains top-3 in >= 90 % of "
            "folds is interpreted as robust to single-paper removal."
        ),
    )
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print(f"Output SHA-256: {sha256_of(out)}")


if __name__ == "__main__":
    main()
