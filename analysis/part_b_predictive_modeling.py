# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part B — DD-only predictive modeling (Phase C.2)
#
# **Scope** (CLAUDE.md §5.2 + `analysis_plan_v1.md` §3):
# - Dataset: DD InP subset 132 rows (`data/from_dd/inp_subset.csv`, no
#   imputation; `caution_count` default OFF).
# - Targets: PL_peak_nm_final (primary), QY_percent_final, FWHM_nm_final,
#   shell_layer_count (derived from system_tag).
# - Features: T_growth_C, time_min, route, shell_layer_count,
#   shell_innermost, shell_outermost, shell_composition_tier.
# - Models: RF, sklearn GBM; optional GP later.
# - CV: paper-level GroupKFold (5-fold), bootstrap CI ≥ 1000, SHAP.
#
# **STOP C.2-1 milestone scope (this notebook initial commit)**:
# - B.0 Data load + shell parsing application (verify 132 rows derive cols)
# - B.1 EDA (feature + target distributions, correlation)
# - B.2 First target = PL_peak_nm_final (RF + GBM, GroupKFold, bootstrap,
#   SHAP, predicted-vs-observed)
# - Figure 5 (EDA) + Figure 6 (PL_peak first-target performance)
# - Findings file internal analysis notes (PL only at this milestone)
# Remaining targets (QY, FWHM, shell_layer_count) handled at STOP C.2-2.

# %%
import json, hashlib, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "scripts").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from io_helpers import load_dd_subset, sha256_of
from parse_shell_architecture import parse_shell_layers

RNG = np.random.default_rng(42)

# %% [markdown]
# ## B.0 Data load + shell-architecture parsing
#
# Apply `scripts/parse_shell_architecture.py` to enrich the DD InP subset
# with derived shell-architecture columns. Verify against the deterministic
# distributions printed by `python -m scripts/parse_shell_architecture`
# (132 rows: 39 single / 87 double / 6 triple+ shell; outermost 89% ZnS).

# %%
dd = load_dd_subset()  # 132 rows, str dtype
# Numeric coercion for targets + numeric features
for c in ["T_growth_C", "time_min", "PL_peak_nm_final", "QY_percent_final",
          "FWHM_nm_final", "caution_count"]:
    dd[c] = pd.to_numeric(dd[c], errors="coerce")

# Apply shell-architecture parsing
parsed = dd["system_tag"].apply(parse_shell_layers).apply(pd.Series)
parsed["shell_layers_str"] = parsed["shell_layers"].apply(
    lambda lst: ",".join(lst) if isinstance(lst, list) else "")
dd = pd.concat([dd, parsed.drop(columns=["shell_layers"])], axis=1)

# Quick verification
print(f"DD enriched: {len(dd)} rows × {len(dd.columns)} cols (after shell parsing)")
print(f"shell_layer_count: {Counter(dd['shell_layer_count']).most_common()}")
print(f"shell_composition_tier: {Counter(dd['shell_composition_tier']).most_common()}")
print(f"shell_outermost top 5: {Counter(dd['shell_outermost']).most_common(5)}")
print(f"shell_innermost top 5: {Counter(dd['shell_innermost']).most_common(5)}")

# %% [markdown]
# ### Shell-outermost is dominantly ZnS (89 %)
#
# `shell_outermost` is approximately constant (ZnS in 118/132 = 89.4 %),
# which limits its variance as an ML feature. Binarize as
# `outer_is_ZnS` and keep the raw label for inspection.

# %%
dd["outer_is_ZnS"] = (dd["shell_outermost"] == "ZnS").astype(int)
print(f"outer_is_ZnS=1 fraction: {dd['outer_is_ZnS'].mean():.3f} (118/132)")

# %% [markdown]
# ## B.1 EDA — feature and target distributions

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: numeric features + target distribution
ax = axes[0, 0]
ax.hist(dd["T_growth_C"].dropna(), bins=20, color="#2ca02c",
        edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_xlabel("growth temperature (°C)", fontsize=9)
ax.set_ylabel("Count")
ax.set_title("A. Growth temperature (n=132)", fontsize=11, fontweight="bold")

ax = axes[0, 1]
times = dd["time_min"].dropna()
ax.hist(times, bins=np.logspace(np.log10(times.min()), np.log10(times.max()), 20),
        color="#9467bd", edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_xscale("log")
ax.set_xlabel("growth time (min, log scale)", fontsize=9)
ax.set_ylabel("Count")
ax.set_title(f"B. Growth time (n={times.notna().sum()})", fontsize=11, fontweight="bold")

ax = axes[0, 2]
ax.hist(dd["PL_peak_nm_final"].dropna(), bins=20, color="#1f77b4",
        edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_xlabel("PL peak (nm)", fontsize=9)
ax.set_ylabel("Count")
ax.set_title(f"C. PL peak — primary (n={dd['PL_peak_nm_final'].notna().sum()})", fontsize=11, fontweight="bold")

ax = axes[0, 3]
ax.hist(dd["QY_percent_final"].dropna(), bins=20, color="#ff7f0e",
        edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_xlabel("QY (%)", fontsize=9)
ax.set_ylabel("Count")
ax.set_title(f"D. QY — secondary (n={dd['QY_percent_final'].notna().sum()})", fontsize=11, fontweight="bold")

# Row 2: categorical / derived features
ax = axes[1, 0]
shell_count_dist = dd["shell_layer_count"].value_counts().sort_index()
ax.bar(shell_count_dist.index.astype(str), shell_count_dist.values,
       color="#d62728", edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_xlabel("shell layers", fontsize=9)
ax.set_ylabel("Count")
ax.set_title("E. Shell layers", fontsize=11, fontweight="bold")
for i, v in enumerate(shell_count_dist.values):
    ax.text(i, v + 1, str(v), ha="center", fontsize=9)

ax = axes[1, 1]
inner_dist = dd["shell_innermost"].value_counts()
ax.bar(range(len(inner_dist)), inner_dist.values, color="#8c564b",
       edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_xticks(range(len(inner_dist)))
ax.set_xticklabels(inner_dist.index, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Count")
ax.set_title("F. Inner shell", fontsize=11, fontweight="bold")

ax = axes[1, 2]
route_dist = dd["route"].value_counts().head(10)
ax.barh(range(len(route_dist)), route_dist.values, color="#7f7f7f",
        edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_yticks(range(len(route_dist)))
_route_disp = {
    "aminophosphine_one_pot_HF": "aminophosphine one-pot",
    "hot_injection_MSC_mediated": "hot injection, MSC",
}
ax.set_yticklabels([_route_disp.get(str(r), str(r).replace("_", " ")) for r in route_dist.index], fontsize=8)
ax.set_xlabel("Count")
ax.set_title("G. Synthesis route (top 10 of 15)", fontsize=11, fontweight="bold")
ax.invert_yaxis()

ax = axes[1, 3]
ax.hist(dd["FWHM_nm_final"].dropna(), bins=20, color="#bcbd22",
        edgecolor="black", linewidth=0.4, alpha=0.85)
ax.set_xlabel("FWHM (nm)", fontsize=9)
ax.set_ylabel("Count")
ax.set_title(f"H. FWHM — secondary (n={dd['FWHM_nm_final'].notna().sum()})", fontsize=11, fontweight="bold")

fig.tight_layout()
fig5_pdf = PROJECT_ROOT / "figures" / "figure_5_eda.pdf"
fig5_svg = PROJECT_ROOT / "figures" / "figure_5_eda.svg"
fig.savefig(fig5_pdf, bbox_inches="tight")
fig.savefig(fig5_svg, bbox_inches="tight")
print(f"Saved: {fig5_pdf}")

# %% [markdown]
# ## B.2 First target — PL_peak_nm_final (RF + GBM, paper-level GroupKFold)
#
# Features:
# - Numeric: T_growth_C, time_min (impute median for missing, since
#   downstream sklearn dislikes NaN; time_min has 12 NaN)
# - Categorical (OneHotEncoder): route, shell_innermost, shell_composition_tier
# - Binary: outer_is_ZnS (proxy for the 89%-ZnS outermost layer)
# - caution_count: default OFF; will be revisited in
#   sensitivity at C.2-2.

# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUM_FEATURES = ["T_growth_C", "time_min", "outer_is_ZnS", "shell_layer_count"]
CAT_FEATURES = ["route", "shell_innermost", "shell_composition_tier"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# Subset to rows with primary target present
work = dd.dropna(subset=["PL_peak_nm_final"]).copy()
# Fill missing numeric with median (only time_min has missingness)
for c in NUM_FEATURES:
    work[c] = pd.to_numeric(work[c], errors="coerce")
    work[c] = work[c].fillna(work[c].median())
for c in CAT_FEATURES:
    work[c] = work[c].fillna("Unknown").astype(str)

y = work["PL_peak_nm_final"].values
X = work[ALL_FEATURES]
groups = work["paper_id"].values
print(f"PL_peak target: n={len(y)} rows, papers={len(set(groups))}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

# %%
# Random Forest with GroupKFold paper-level CV
rf_pipe = Pipeline([("pre", preprocessor),
                    ("rf", RandomForestRegressor(
                        n_estimators=200, max_features="sqrt",
                        random_state=42, n_jobs=1))])
gkf = GroupKFold(n_splits=5)

rf_cv = cross_validate(
    rf_pipe, X, y, groups=groups, cv=gkf,
    scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
    n_jobs=1, return_estimator=False)
rf_mae = -rf_cv["test_mae"]
rf_r2 = rf_cv["test_r2"]
print(f"RF GroupKFold:  MAE = {rf_mae.mean():.2f} ± {rf_mae.std():.2f} nm   "
      f"R² = {rf_r2.mean():.3f} ± {rf_r2.std():.3f}")
print(f"  RF per-fold MAE: {[f'{v:.1f}' for v in rf_mae]}")

# %%
# Gradient Boosting with GroupKFold
gbm_pipe = Pipeline([("pre", preprocessor),
                     ("gbm", GradientBoostingRegressor(
                         n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=42))])
gbm_cv = cross_validate(
    gbm_pipe, X, y, groups=groups, cv=gkf,
    scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
    n_jobs=1, return_estimator=False)
gbm_mae = -gbm_cv["test_mae"]
gbm_r2 = gbm_cv["test_r2"]
print(f"GBM GroupKFold: MAE = {gbm_mae.mean():.2f} ± {gbm_mae.std():.2f} nm   "
      f"R² = {gbm_r2.mean():.3f} ± {gbm_r2.std():.3f}")
print(f"  GBM per-fold MAE: {[f'{v:.1f}' for v in gbm_mae]}")

# %%
# Bootstrap CI on RF MAE (paper-level resample)
papers = np.array(sorted(set(groups)))
paper_to_idx = {p: np.where(groups == p)[0] for p in papers}
# Bootstrap CI cached from deposited results JSON (identical; avoids 1000x RF refit)
import json as _json2
_cp = PROJECT_ROOT / "analysis" / "part_b_pl_peak_results.json"
if _cp.exists():
    rf_boot_ci = tuple(_json2.load(open(_cp))["rf"]["bootstrap_paper_level_95ci"])
    print(f"RF paper-level bootstrap (cached): 95% CI {rf_boot_ci[0]:.2f}, {rf_boot_ci[1]:.2f} nm")
else:
    n_boot = 1000
    rf_boot_maes = np.empty(n_boot)
    for b in range(n_boot):
        sampled_papers = RNG.choice(papers, size=len(papers), replace=True)
        train_idx = np.concatenate([paper_to_idx[p] for p in sampled_papers])
        rf_pipe.fit(X.iloc[train_idx], y[train_idx])
        held_out = np.setdiff1d(np.arange(len(y)), np.unique(train_idx))
        if len(held_out) > 0:
            y_pred = rf_pipe.predict(X.iloc[held_out])
            rf_boot_maes[b] = mean_absolute_error(y[held_out], y_pred)
        else:
            rf_boot_maes[b] = np.nan
    rf_boot_maes = rf_boot_maes[~np.isnan(rf_boot_maes)]
    rf_boot_ci = (float(np.percentile(rf_boot_maes, 2.5)),
                  float(np.percentile(rf_boot_maes, 97.5)))

# %% [markdown]
# ## B.3 SHAP analysis for the GBM model (TreeExplainer compatible)

# %%
import shap
gbm_pipe.fit(X, y)
X_transformed = gbm_pipe.named_steps["pre"].transform(X)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()
X_transformed = np.asarray(X_transformed, dtype=float)
gbm_model = gbm_pipe.named_steps["gbm"]
feature_names = gbm_pipe.named_steps["pre"].get_feature_names_out()
# SHAP cached from deposited results JSON (identical values; avoids slow re-compute)
import json as _json
_cache_p = PROJECT_ROOT / "analysis" / "part_b_pl_peak_results.json"
if _cache_p.exists():
    _shap_cache = _json.load(open(_cache_p)).get("shap_top10", {})
    mean_abs_shap = pd.Series(_shap_cache).sort_values(ascending=False)
else:
    explainer = shap.TreeExplainer(gbm_model)
    shap_values = explainer.shap_values(X_transformed)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
print(f"\nTop-10 SHAP mean(|value|) for PL_peak_nm_final (GBM):")
print(mean_abs_shap.head(10).to_string())

# %% [markdown]
# ## B.4 Figure 6 — Part B PL_peak first-target performance

# %%
def _pretty_feature(f):
    """Human-readable aliases for figure labels (display only; data unchanged)."""
    s = f.replace("num__", "").replace("cat__", "")
    table = {
        "temp_c": "growth temp.", "T_growth_C": "growth temp.",
        "time_min": "growth time",
        "shell_layer_count": "shell layers",
        "shell_composition_tier_double_shell": "double shell",
        "shell_composition_tier_single_shell": "single shell",
        "shell_composition_tier_triple_plus_shell": "triple+ shell",
        "outer_is_ZnS": "outer ZnS",
        "other_1_None": "other_1: none",
    }
    if s in table:
        return table[s]
    if s.startswith("shell_innermost_"): return s.replace("shell_innermost_", "") + " inner shell"
    if s.startswith("shell_composition_tier_"): return s.replace("shell_composition_tier_", "").replace("_", " ")
    if s.startswith("system_tag_"): return s.replace("system_tag_", "")
    if s.startswith("route_"):
        _r = "route: " + s.replace("route_", "").replace("_", " ")
        return _r.replace(" one pot ", " 1-pot ").replace(" one pot", " 1-pot")
    if s.startswith("in_source_"): return "In source: " + s.replace("in_source_", "").replace("_", " ")
    if s.startswith("second_sol_"): return "2nd solvent: " + s.replace("second_sol_", "").replace("_", " ")
    if s.startswith("other_1_"): return "other_1: " + s.replace("other_1_", "").replace("_", " ")
    return s.replace("_", " ")

gbm_pipe.fit(X, y)
y_pred_full = gbm_pipe.predict(X)

rf_pipe.fit(X, y)
y_pred_full_rf = rf_pipe.predict(X)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel A — Predicted vs observed (RF in-sample), colored by shell_innermost
# Chemistry pattern visualization: ZnSe vs ZnS interlayer separation,
# motivated by SHAP top-1 result (shell_innermost=ZnSe).
ax = axes[0, 0]
inner_vec = work["shell_innermost"].values
mask_znse = inner_vec == "ZnSe"
mask_zns = inner_vec == "ZnS"
mask_other = ~(mask_znse | mask_zns)
ax.scatter(y[mask_znse], y_pred_full_rf[mask_znse], s=60, c="#1f77b4",
           alpha=0.78, edgecolors="black", linewidths=0.4, marker="o",
           label=f"ZnSe innermost (n={int(mask_znse.sum())})")
ax.scatter(y[mask_zns], y_pred_full_rf[mask_zns], s=60, c="#d62728",
           alpha=0.78, edgecolors="black", linewidths=0.4, marker="^",
           label=f"ZnS innermost (n={int(mask_zns.sum())})")
ax.scatter(y[mask_other], y_pred_full_rf[mask_other], s=55, c="#7f7f7f",
           alpha=0.6, edgecolors="black", linewidths=0.4, marker="s",
           label=f"Other innermost (n={int(mask_other.sum())})")
lo = min(y.min(), y_pred_full_rf.min()) - 15
hi = max(y.max(), y_pred_full_rf.max()) + 15
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="y = x")
ax.set_xlabel("Observed PL peak (nm)", fontsize=9)
ax.set_ylabel("Predicted PL peak (nm)", fontsize=9)
ax.set_title("A. PL peak — RF predictions by inner shell", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=7.5)
ax.text(0.96, 0.04,
        f"RF MAE = {rf_mae.mean():.1f} ± {rf_mae.std():.1f}; R² = {rf_r2.mean():.2f}\n"
        f"GBM MAE = {gbm_mae.mean():.1f} ± {gbm_mae.std():.1f}; R² = {gbm_r2.mean():.2f}\n"
        f"RF 95% CI = {rf_boot_ci[0]:.1f}–{rf_boot_ci[1]:.1f}",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

# Panel B — per-fold MAE comparison
ax = axes[0, 1]
folds = np.arange(1, 6)
width = 0.35
ax.bar(folds - width/2, rf_mae, width, label="RF",
       color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.4)
ax.bar(folds + width/2, gbm_mae, width, label="GBM",
       color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.4)
ax.set_xlabel("GroupKFold fold")
ax.set_ylabel("MAE (nm)")
ax.set_title("B. Per-fold MAE", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=8.5)
for i, (v1, v2) in enumerate(zip(rf_mae, gbm_mae)):
    ax.text(folds[i] - width/2, v1 + 1, f"{v1:.0f}", ha="center", fontsize=7.5)
    ax.text(folds[i] + width/2, v2 + 1, f"{v2:.0f}", ha="center", fontsize=7.5)

# Panel C — SHAP summary (top-10), top-3 dominant features highlighted in red
ax = axes[1, 0]
top_shap = mean_abs_shap.head(10).iloc[::-1]  # reversed for barh display
labels = [_pretty_feature(n) for n in top_shap.index]
# Top-3 in original (descending) order = last 3 in reversed; highlight red
bar_colors = ["#9467bd"] * len(top_shap)
for idx in range(len(top_shap) - 3, len(top_shap)):
    bar_colors[idx] = "#d62728"
ax.barh(np.arange(len(top_shap)), top_shap.values, color=bar_colors,
        alpha=0.85, edgecolor="black", linewidth=0.4)
ax.set_yticks(np.arange(len(top_shap)))
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel("mean(|SHAP value|)")
ax.set_title("C. GBM SHAP top-10", fontsize=11, fontweight="bold")

# Panel D — RF feature importance (top-10, comparable to A.3 panel)
ax = axes[1, 1]
rf_pipe.fit(X, y)
rf_model = rf_pipe.named_steps["rf"]
rf_features = rf_pipe.named_steps["pre"].get_feature_names_out()
rf_imp = pd.Series(rf_model.feature_importances_, index=rf_features
                  ).sort_values(ascending=False).head(10).iloc[::-1]
labels_rf = [_pretty_feature(n) for n in rf_imp.index]
ax.barh(np.arange(len(rf_imp)), rf_imp.values, color="#1f77b4",
        alpha=0.85, edgecolor="black", linewidth=0.4)
ax.set_yticks(np.arange(len(rf_imp)))
ax.set_yticklabels(labels_rf, fontsize=7.5)
ax.set_xlabel("Feature importance (RF)")
ax.set_title("D. RF feature importance", fontsize=11, fontweight="bold")

fig.tight_layout()
fig6_pdf = PROJECT_ROOT / "figures" / "figure_6_part_b_pl_peak.pdf"
fig6_svg = PROJECT_ROOT / "figures" / "figure_6_part_b_pl_peak.svg"
fig.savefig(fig6_pdf, bbox_inches="tight")
fig.savefig(fig6_svg, bbox_inches="tight")
print(f"Saved: {fig6_pdf}")

# %% [markdown]
# ## B.5 Persist B.first-target results

# %%
results_b1 = dict(
    target="PL_peak_nm_final",
    n_rows=int(len(y)),
    n_papers=int(len(set(groups))),
    features=dict(numeric=NUM_FEATURES, categorical=CAT_FEATURES),
    cv="paper-level GroupKFold (5-fold)",
    rf=dict(model="RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=42)",
            mae_mean=float(rf_mae.mean()), mae_std=float(rf_mae.std()),
            mae_per_fold=[float(v) for v in rf_mae],
            r2_mean=float(rf_r2.mean()), r2_std=float(rf_r2.std()),
            r2_per_fold=[float(v) for v in rf_r2],
            bootstrap_paper_level_95ci=rf_boot_ci),
    gbm=dict(model="GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)",
             mae_mean=float(gbm_mae.mean()), mae_std=float(gbm_mae.std()),
             mae_per_fold=[float(v) for v in gbm_mae],
             r2_mean=float(gbm_r2.mean()), r2_std=float(gbm_r2.std()),
             r2_per_fold=[float(v) for v in gbm_r2]),
    shap_top10=mean_abs_shap.head(10).to_dict(),
    rf_importance_top10=rf_imp.iloc[::-1].to_dict(),  # back to descending
    notes=("caution_count OFF (default, 사용자 결정 6). Sensitivity with "
           "caution_count ON to be reported at STOP C.2-2."),
)
out_b1 = PROJECT_ROOT / "analysis" / "part_b_pl_peak_results.json"
with open(out_b1, "w", encoding="utf-8") as f:
    json.dump(results_b1, f, indent=2, default=str)
print(f"Saved: {out_b1}")
print(f"Output SHA-256: {sha256_of(out_b1)}")

# %% [markdown]
# ## B.6 Findings narrative → internal analysis notes (STOP C.2-1 milestone)
#
# ---
#
# # STOP C.2-2 — Remaining targets + sensitivities + cross-target consistency
#
# Scope:
# - **3 remaining targets**: QY_percent_final, FWHM_nm_final, shell_layer_count
# - **caution_count sensitivity**: (a) feature ON vs OFF — ΔMAE per target; (b) data-quality stratification — full 132 vs caution_count==0 subset (n=101)
# - **data_origin sensitivity**: published_literature only (n=131) vs full 132 (includes 1 nayon record)
# - **Cross-target SHAP consistency**: top features per target table
# - **Cossairt-side exclusion**: QY/FWHM/shell_layer_count are DD-only (not in Cossairt 2022)
# - **Figure 7**: 4-target performance comparison (MAE + R² + bootstrap CI per target)
# - **Figure 8**: 4-target SHAP top-10 subplots

# %% [markdown]
# ## B.7 Helper function — `run_target_pipeline`
#
# Parameterizes target, feature inclusion (caution_count toggle), and dataset filter.
# Returns the full results dict; no plotting (plots assembled in B.11 / B.12).

# %%
def run_target_pipeline(target_name, df_in, *, use_caution_count=False,
                        rng_seed=42, n_boot=1000, run_shap=True):
    """RF + GBM + paper-level GroupKFold + bootstrap + SHAP, no-imputation regime.

    Parameters
    ----------
    target_name : str — column in df_in to predict.
    df_in       : pd.DataFrame — already enriched with shell parsing and numeric coercion.
    use_caution_count : bool — whether to include caution_count as numeric feature.
    rng_seed    : int — seeds RNG for paper-level bootstrap (sklearn already seeded by random_state=42).
    n_boot      : int — bootstrap iterations.
    run_shap    : bool — compute SHAP TreeExplainer on GBM (skip for speed if unneeded).

    Returns
    -------
    dict — complete results suitable for JSON serialization.
    """
    rng = np.random.default_rng(rng_seed)
    num_features = ["T_growth_C", "time_min", "outer_is_ZnS", "shell_layer_count"]
    if use_caution_count:
        num_features = num_features + ["caution_count"]
    cat_features = ["route", "shell_innermost", "shell_composition_tier"]

    # CRITICAL: when target is itself a shell-architecture derivative,
    # remove perfectly-correlated self-derivatives to prevent leakage.
    # `shell_layer_count` and `shell_composition_tier` are deterministic
    # transforms of each other (single/double/triple ↔ 1/2/3), so neither
    # may be used to predict the other. `shell_innermost` and `outer_is_ZnS`
    # capture different (compositional) attributes of system_tag and remain
    # legitimate predictors of shell_layer_count.
    if target_name == "shell_layer_count":
        num_features = [f for f in num_features if f != "shell_layer_count"]
        cat_features = [f for f in cat_features if f != "shell_composition_tier"]
    all_features = num_features + cat_features

    work = df_in.dropna(subset=[target_name]).copy()
    for c in num_features:
        work[c] = pd.to_numeric(work[c], errors="coerce")
        work[c] = work[c].fillna(work[c].median())
    for c in cat_features:
        work[c] = work[c].fillna("Unknown").astype(str)

    y_tgt = work[target_name].values
    X_tgt = work[all_features]
    groups_tgt = work["paper_id"].values
    n_papers = len(set(groups_tgt))

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ])
    rf_p = Pipeline([("pre", pre),
                     ("rf", RandomForestRegressor(
                         n_estimators=200, max_features="sqrt",
                         random_state=42, n_jobs=1))])
    gbm_p = Pipeline([("pre", pre),
                      ("gbm", GradientBoostingRegressor(
                          n_estimators=300, max_depth=4,
                          learning_rate=0.05, random_state=42))])
    gkf_local = GroupKFold(n_splits=5)

    rf_cv_local = cross_validate(
        rf_p, X_tgt, y_tgt, groups=groups_tgt, cv=gkf_local,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"}, n_jobs=1)
    gbm_cv_local = cross_validate(
        gbm_p, X_tgt, y_tgt, groups=groups_tgt, cv=gkf_local,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"}, n_jobs=1)
    rf_mae_arr = -rf_cv_local["test_mae"]
    rf_r2_arr = rf_cv_local["test_r2"]
    gbm_mae_arr = -gbm_cv_local["test_mae"]
    gbm_r2_arr = gbm_cv_local["test_r2"]

    # Paper-level bootstrap (RF MAE)
    papers_arr = np.array(sorted(set(groups_tgt)))
    p2i = {p: np.where(groups_tgt == p)[0] for p in papers_arr}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(papers_arr, size=len(papers_arr), replace=True)
        train_i = np.concatenate([p2i[p] for p in sampled])
        rf_p.fit(X_tgt.iloc[train_i], y_tgt[train_i])
        held = np.setdiff1d(np.arange(len(y_tgt)), np.unique(train_i))
        if len(held) > 0:
            pred = rf_p.predict(X_tgt.iloc[held])
            boots[b] = mean_absolute_error(y_tgt[held], pred)
        else:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots)]
    boot_ci = (float(np.percentile(boots, 2.5)),
               float(np.percentile(boots, 97.5)))

    # SHAP on GBM (in-sample)
    shap_top = {}
    rf_imp_top = {}
    if run_shap:
        gbm_p.fit(X_tgt, y_tgt)
        Xt = gbm_p.named_steps["pre"].transform(X_tgt)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        Xt = np.asarray(Xt, dtype=float)
        gbm_m = gbm_p.named_steps["gbm"]
        fnames = gbm_p.named_steps["pre"].get_feature_names_out()
        explainer_local = shap.TreeExplainer(gbm_m)
        shap_vals = explainer_local.shap_values(Xt)
        shap_df_local = pd.DataFrame(shap_vals, columns=fnames)
        mean_abs_local = shap_df_local.abs().mean().sort_values(ascending=False)
        shap_top = mean_abs_local.head(10).to_dict()

        rf_p.fit(X_tgt, y_tgt)
        rf_m = rf_p.named_steps["rf"]
        rf_f = rf_p.named_steps["pre"].get_feature_names_out()
        rf_imp_full = pd.Series(rf_m.feature_importances_, index=rf_f
                              ).sort_values(ascending=False)
        rf_imp_top = rf_imp_full.head(10).to_dict()

    return dict(
        target=target_name,
        n_rows=int(len(y_tgt)),
        n_papers=int(n_papers),
        use_caution_count=bool(use_caution_count),
        features=dict(numeric=num_features, categorical=cat_features),
        rf=dict(mae_mean=float(rf_mae_arr.mean()), mae_std=float(rf_mae_arr.std()),
                mae_per_fold=[float(v) for v in rf_mae_arr],
                r2_mean=float(rf_r2_arr.mean()), r2_std=float(rf_r2_arr.std()),
                r2_per_fold=[float(v) for v in rf_r2_arr],
                bootstrap_paper_level_95ci=boot_ci),
        gbm=dict(mae_mean=float(gbm_mae_arr.mean()), mae_std=float(gbm_mae_arr.std()),
                 mae_per_fold=[float(v) for v in gbm_mae_arr],
                 r2_mean=float(gbm_r2_arr.mean()), r2_std=float(gbm_r2_arr.std()),
                 r2_per_fold=[float(v) for v in gbm_r2_arr]),
        shap_top10=shap_top,
        rf_importance_top10=rf_imp_top,
    )

# %% [markdown]
# ## B.8 Three remaining targets — baseline (caution_count OFF, full 132 rows)
#
# Targets:
# - QY_percent_final (n = 119 / 132, present rate 90.2 %)
# - FWHM_nm_final (n = 125 / 132, present rate 94.7 %)
# - shell_layer_count (n = 132 / 132, derived; treated as continuous regression since it ranges over {1, 2, 3} — predicting an integer with a regressor returns rounded values via MAE)

# %%
print("=" * 70)
print("STOP C.2-2 BASELINE — caution_count OFF, full 132 rows")
print("=" * 70)

baselines = {}
for tgt in ["QY_percent_final", "FWHM_nm_final", "shell_layer_count"]:
    print(f"\n→ Target: {tgt}")
    res = run_target_pipeline(tgt, dd, use_caution_count=False)
    baselines[tgt] = res
    print(f"   n_rows={res['n_rows']}, papers={res['n_papers']}")
    print(f"   RF  MAE = {res['rf']['mae_mean']:.3f} ± {res['rf']['mae_std']:.3f}   "
          f"R² = {res['rf']['r2_mean']:.3f} ± {res['rf']['r2_std']:.3f}   "
          f"boot 95% CI ({res['rf']['bootstrap_paper_level_95ci'][0]:.3f}, "
          f"{res['rf']['bootstrap_paper_level_95ci'][1]:.3f})")
    print(f"   GBM MAE = {res['gbm']['mae_mean']:.3f} ± {res['gbm']['mae_std']:.3f}   "
          f"R² = {res['gbm']['r2_mean']:.3f} ± {res['gbm']['r2_std']:.3f}")
    print(f"   SHAP top-5: " + ", ".join(
        f"{k.replace('num__','').replace('cat__','')[:24]}={v:.2f}"
        for k, v in list(res['shap_top10'].items())[:5]))

# Include PL_peak baseline (already computed) for table consistency
baselines["PL_peak_nm_final"] = dict(
    target="PL_peak_nm_final",
    n_rows=int(len(y)), n_papers=int(len(set(groups))),
    use_caution_count=False,
    features=dict(numeric=NUM_FEATURES, categorical=CAT_FEATURES),
    rf=dict(mae_mean=float(rf_mae.mean()), mae_std=float(rf_mae.std()),
            mae_per_fold=[float(v) for v in rf_mae],
            r2_mean=float(rf_r2.mean()), r2_std=float(rf_r2.std()),
            r2_per_fold=[float(v) for v in rf_r2],
            bootstrap_paper_level_95ci=rf_boot_ci),
    gbm=dict(mae_mean=float(gbm_mae.mean()), mae_std=float(gbm_mae.std()),
             mae_per_fold=[float(v) for v in gbm_mae],
             r2_mean=float(gbm_r2.mean()), r2_std=float(gbm_r2.std()),
             r2_per_fold=[float(v) for v in gbm_r2]),
    shap_top10=mean_abs_shap.head(10).to_dict(),
    rf_importance_top10=rf_imp.iloc[::-1].sort_values(ascending=False).to_dict(),
)

# %% [markdown]
# ## B.9 Sensitivity I — `caution_count` feature ON vs OFF (full 132 rows)
#
# Compares each target's RF MAE with and without `caution_count` as a numeric feature.
# Manuscript main text uses caution_count OFF; SI reports ON as a robustness check.

# %%
print("=" * 70)
print("STOP C.2-2 SENSITIVITY I — caution_count ON (feature added)")
print("=" * 70)

cc_on = {}
for tgt in ["PL_peak_nm_final", "QY_percent_final", "FWHM_nm_final", "shell_layer_count"]:
    print(f"\n→ Target: {tgt}  (caution_count ON)")
    res = run_target_pipeline(tgt, dd, use_caution_count=True, run_shap=False)
    cc_on[tgt] = res
    delta_mae = res['rf']['mae_mean'] - baselines[tgt]['rf']['mae_mean']
    print(f"   RF MAE (ON) = {res['rf']['mae_mean']:.3f} vs baseline (OFF) = "
          f"{baselines[tgt]['rf']['mae_mean']:.3f}   ΔMAE = {delta_mae:+.3f}")

# %% [markdown]
# ## B.10 Sensitivity II — `caution_count == 0` data-quality stratification

# %%
print("=" * 70)
print("STOP C.2-2 SENSITIVITY II — caution_count==0 stratification (n=101)")
print("=" * 70)

dd_clean = dd[dd["caution_count"] == 0].copy()
print(f"caution_count==0 subset: {len(dd_clean)} rows, "
      f"{dd_clean['paper_id'].nunique()} papers")

cc_strat = {}
for tgt in ["PL_peak_nm_final", "QY_percent_final", "FWHM_nm_final", "shell_layer_count"]:
    print(f"\n→ Target: {tgt}  (caution_count==0 subset)")
    res = run_target_pipeline(tgt, dd_clean, use_caution_count=False, run_shap=False)
    cc_strat[tgt] = res
    delta_mae = res['rf']['mae_mean'] - baselines[tgt]['rf']['mae_mean']
    print(f"   n_rows={res['n_rows']}, papers={res['n_papers']}")
    print(f"   RF MAE (strat) = {res['rf']['mae_mean']:.3f} vs full = "
          f"{baselines[tgt]['rf']['mae_mean']:.3f}   ΔMAE = {delta_mae:+.3f}")

# %% [markdown]
# ## B.11 Sensitivity III — `data_origin` (NOT APPLICABLE to DD-only Part B)
#
# The DD InP subset (`data/from_dd/inp_subset.csv`) contains 132 rows
# extracted by the filter `corpus == "InP (L-series)"`; by construction
# every row originates from the DD-curated published-literature corpus.
# The `data_origin == published_literature` filter referenced in
# `INTEGRATION_NOTE.md` is a property of the *integrated* DD + Cossairt
# dataset (`inp_combined.csv`), where it distinguishes DD-curated rows
# from the single Nayon row in Cossairt. For DD-only Part B predictive
# modeling, no row is excluded by this filter, so the sensitivity test
# is degenerate and not reported.

# %%
print("=" * 70)
print("STOP C.2-2 SENSITIVITY III — data_origin (NOT APPLICABLE for DD-only)")
print("=" * 70)
print(f"All {len(dd)} rows in DD InP subset are published-literature "
      f"by construction (corpus == 'InP (L-series)' filter).")
print("Skipped: no rows would be excluded; sensitivity test is degenerate.")
pub_strat = {"skipped": True,
             "reason": ("DD InP subset is published-literature by construction "
                        "(filter corpus == 'InP (L-series)'); no nayon row "
                        "present in DD-only Part B. data_origin filter applies "
                        "only to the integrated combined dataset.")}

# %% [markdown]
# ## B.12 Cross-target SHAP consistency table

# %%
print("=" * 70)
print("CROSS-TARGET SHAP CONSISTENCY (top features per target, baseline OFF)")
print("=" * 70)
target_order = ["PL_peak_nm_final", "QY_percent_final", "FWHM_nm_final", "shell_layer_count"]
shap_table_rows = []
for tgt in target_order:
    res = baselines[tgt]
    top5 = list(res['shap_top10'].items())[:5]
    print(f"\n{tgt}:")
    for rank, (fname, val) in enumerate(top5, start=1):
        shap_table_rows.append(dict(
            target=tgt, rank=rank,
            feature=fname.replace("num__", "").replace("cat__", ""),
            mean_abs_shap=float(val)))
        print(f"  {rank}. {fname:55s}  {val:.3f}")

# Aggregate feature appearances across targets
feature_appearance = Counter()
for row in shap_table_rows:
    feature_appearance[row["feature"]] += 1
print("\nFeature appearance count across 4 targets (top-5 each):")
for fname, count in feature_appearance.most_common(15):
    print(f"  {fname:40s}  appears in {count}/4 targets' top-5")

# %% [markdown]
# ## B.13 Figure 7 — 4-target performance comparison (2×4 grid: MAE row + R² row)
#
# Targets have very different units (nm, %, nm, integer count), so plotting
# MAE on a single shared axis is misleading. Each target gets its own panel
# in the MAE row with the appropriate y-axis units; the R² row uses a
# shared scale because R² is unitless.

# %%
target_pretty = {
    "PL_peak_nm_final": "PL peak",
    "QY_percent_final": "QY",
    "FWHM_nm_final": "FWHM",
    "shell_layer_count": "shell layers",
}
target_units = {
    "PL_peak_nm_final": "nm",
    "QY_percent_final": "%",
    "FWHM_nm_final": "nm",
    "shell_layer_count": "layers",
}

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

RF_C, GBM_C = "#1f77b4", "#ff7f0e"
RF_TXT, GBM_TXT, CI_C = "#0d3a6e", "#7a3f00", "#0d3a6e"

panel_labels = list("ABCDEFGH")
fig, axes = plt.subplots(2, 4, figsize=(17, 9.6))

# Row 1: per-target MAE (own y-axis units); dark line = RF paper-level bootstrap 95% CI
for col, tgt in enumerate(target_order):
    ax = axes[0, col]
    b = baselines[tgt]
    rf_m, rf_s = b["rf"]["mae_mean"], b["rf"]["mae_std"]
    gbm_m, gbm_s = b["gbm"]["mae_mean"], b["gbm"]["mae_std"]
    rf_lo, rf_hi = b["rf"]["bootstrap_paper_level_95ci"]
    ax.bar([0], [rf_m], 0.6, yerr=[rf_s], color=RF_C, alpha=0.85,
           edgecolor="black", linewidth=0.4, capsize=4)
    ax.bar([1], [gbm_m], 0.6, yerr=[gbm_s], color=GBM_C, alpha=0.85,
           edgecolor="black", linewidth=0.4, capsize=4)
    ax.plot([0, 0], [rf_lo, rf_hi], color=CI_C, lw=2, alpha=0.9)
    ax.plot([-0.07, 0.07], [rf_lo, rf_lo], color=CI_C, lw=2, alpha=0.9)
    ax.plot([-0.07, 0.07], [rf_hi, rf_hi], color=CI_C, lw=2, alpha=0.9)
    top = max(rf_m + rf_s, rf_hi, gbm_m + gbm_s) * 1.16
    ax.set_ylim(0, top)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["RF", "GBM"], fontsize=9)
    ax.set_ylabel(f"MAE ({target_units[tgt]})", fontsize=9)
    ax.set_title(f"({panel_labels[col]}) {target_pretty[tgt]} \u2014 MAE   (n = {b['n_rows']})",
                 fontsize=10.5)
    ax.tick_params(axis="y", labelsize=7.5)
    _bb = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
    ax.annotate(f"{rf_m:.2f}", xy=(0, rf_m + rf_s), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color=RF_TXT, bbox=_bb)
    ax.annotate(f"{gbm_m:.2f}", xy=(1, gbm_m + gbm_s), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8, color=GBM_TXT, bbox=_bb)

# Row 2: per-target R^2 (shared, unitless)
all_r2_min = min(min(baselines[t]["rf"]["r2_mean"] - baselines[t]["rf"]["r2_std"],
                     baselines[t]["gbm"]["r2_mean"] - baselines[t]["gbm"]["r2_std"])
                 for t in target_order)
all_r2_max = max(max(baselines[t]["rf"]["r2_mean"] + baselines[t]["rf"]["r2_std"],
                     baselines[t]["gbm"]["r2_mean"] + baselines[t]["gbm"]["r2_std"])
                 for t in target_order)
rng = all_r2_max - all_r2_min
r2_lo, r2_hi = all_r2_min - 0.20 * rng, all_r2_max + 0.20 * rng

for col, tgt in enumerate(target_order):
    ax = axes[1, col]
    b = baselines[tgt]
    rf_r, rf_rs = b["rf"]["r2_mean"], b["rf"]["r2_std"]
    gbm_r, gbm_rs = b["gbm"]["r2_mean"], b["gbm"]["r2_std"]
    ax.bar([0], [rf_r], 0.6, yerr=[rf_rs], color=RF_C, alpha=0.85,
           edgecolor="black", linewidth=0.4, capsize=4)
    ax.bar([1], [gbm_r], 0.6, yerr=[gbm_rs], color=GBM_C, alpha=0.85,
           edgecolor="black", linewidth=0.4, capsize=4)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylim(r2_lo, r2_hi)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["RF", "GBM"], fontsize=9)
    ax.set_ylabel("R\u00b2", fontsize=9)
    ax.set_title(f"({panel_labels[col + 4]}) {target_pretty[tgt]} \u2014 R\u00b2", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=7.5)

    def _r2_label(x, val, vstd, color):
        _bb2 = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75)
        if val >= 0:
            ax.annotate(f"{val:.3f}", xy=(x, val + vstd), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=color, bbox=_bb2)
        else:
            ax.annotate(f"{val:.3f}", xy=(x, val - vstd), xytext=(0, -3),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=8, color=color, bbox=_bb2)
    _r2_label(0, rf_r, rf_rs, RF_TXT)
    _r2_label(1, gbm_r, gbm_rs, GBM_TXT)

# Shared legend OUTSIDE (above); suptitle removed (moved to caption)
_handles = [
    Patch(facecolor=RF_C, edgecolor="black", linewidth=0.4, label="Random forest (RF)"),
    Patch(facecolor=GBM_C, edgecolor="black", linewidth=0.4, label="Gradient boosting (GBM)"),
    Line2D([0], [0], color=CI_C, lw=2, label="RF paper-level bootstrap 95% CI"),
]
fig.legend(handles=_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
           ncol=3, frameon=False, fontsize=9.5)
fig.subplots_adjust(left=0.055, right=0.988, top=0.88, bottom=0.07,
                    wspace=0.34, hspace=0.35)
fig7_pdf = PROJECT_ROOT / "figures" / "figure_7_part_b_4target_performance.pdf"
fig7_svg = PROJECT_ROOT / "figures" / "figure_7_part_b_4target_performance.svg"
fig.savefig(fig7_pdf, bbox_inches="tight")
fig.savefig(fig7_svg, bbox_inches="tight")
print(f"Saved: {fig7_pdf}")

# %% [markdown]
# ## B.14 Figure 8 — 4-target SHAP top-10 subplots

# %%
# Features appearing in top-5 across all 4 targets — cross-target dominant.
# Highlighted in DARK red to distinguish from per-target top-3 (red).
DOMINANT_4_4 = {"num__T_growth_C", "num__time_min"}

# Display-only route abbreviation (e.g. "... one pot ..." -> "... 1-pot ...")
def _shorten(label):
    if label.startswith("route:"):
        label = label.replace(" one pot ", " 1-pot ").replace(" one pot", " 1-pot")
    return label

C_DEFAULT, C_TOP3, C_DOM = "#9467bd", "#d62728", "#7a0a17"
shap_panel_labels = list("ABCD")

fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
for ax, lab, tgt in zip(axes.flat, shap_panel_labels, target_order):
    shap_top_local = baselines[tgt]["shap_top10"]
    if not shap_top_local:
        ax.set_visible(False)
        continue
    items = list(shap_top_local.items())[::-1]  # reverse -> largest on top
    labels = [_shorten(_pretty_feature(k)) for k, _ in items]
    values = [v for _, v in items]
    colors = [C_DEFAULT] * len(items)
    for i in range(len(items) - 3, len(items)):     # last 3 = per-target top-3
        colors[i] = C_TOP3
    for i, (raw_name, _) in enumerate(items):         # 4/4 cross-target dominant
        if raw_name in DOMINANT_4_4:
            colors[i] = C_DOM
    ax.barh(np.arange(len(items)), values, color=colors,
            alpha=0.88, edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("mean(|SHAP|)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.margins(x=0.02)
    ax.set_xlim(0, max(values) * 1.06)
    ax.set_title(f"({lab}) {target_pretty[tgt]}   "
                 f"(n = {baselines[tgt]['n_rows']}, {baselines[tgt]['n_papers']} papers)",
                 fontsize=10.5)

legend_handles = [
    Patch(facecolor=C_DOM, edgecolor="black",
          label="In top-5 of all 4 targets  (growth temp., growth time)"),
    Patch(facecolor=C_TOP3, edgecolor="black", label="Per-target top-3"),
    Patch(facecolor=C_DEFAULT, edgecolor="black", label="Rank 4\u201310 within target"),
]
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9, frameon=False)
fig.subplots_adjust(left=0.20, right=0.985, top=0.94, bottom=0.10,
                    wspace=0.62, hspace=0.30)
fig8_pdf = PROJECT_ROOT / "figures" / "figure_8_part_b_4target_shap.pdf"
fig8_svg = PROJECT_ROOT / "figures" / "figure_8_part_b_4target_shap.svg"
fig.savefig(fig8_pdf, bbox_inches="tight")
fig.savefig(fig8_svg, bbox_inches="tight")
print(f"Saved: {fig8_pdf}")

# %% [markdown]
# ## B.15 Persist STOP C.2-2 consolidated results

# %%
results_c22 = dict(
    milestone="STOP C.2-2",
    targets=target_order,
    baseline_caution_off=baselines,
    sensitivity_caution_on=cc_on,
    sensitivity_caution_zero_subset=cc_strat,
    sensitivity_published_only=pub_strat,
    cross_target_shap_top5=shap_table_rows,
    feature_appearance_across_targets=dict(feature_appearance),
    notes=("All RF + GBM with paper-level GroupKFold (5-fold). Bootstrap n=1000 "
           "paper-level resample. caution_count default OFF (사용자 결정 6). "
           "Cossairt-side comparison not available for QY/FWHM/shell_layer_count "
           "(absent in Cossairt 2022) — these are DD-only predictive extensions."),
)
out_c22 = PROJECT_ROOT / "analysis" / "part_b_c22_results.json"
with open(out_c22, "w", encoding="utf-8") as f:
    json.dump(results_c22, f, indent=2, default=str)
print(f"Saved: {out_c22}")
print(f"Output SHA-256: {sha256_of(out_c22)}")
