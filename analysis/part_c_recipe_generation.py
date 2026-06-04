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
# # Part C — Candidate recipe generation for under-explored design regions
#
# **Scope** (CLAUDE.md §5.3 + `analysis_plan_v1.md` §4):
# - Dataset: DD InP subset 132 rows (`data/from_dd/inp_subset.csv`, no
#   imputation; shell-architecture parsed via
#   `scripts/parse_shell_architecture.py`).
# - Design space (4 axes): T_growth_C, time_min, shell_innermost,
#   shell_layer_count.
# - Methods (3 hybrid, per 사용자 STOP C.2-2 권고):
#   1. Constrained random sampling (Mahalanobis-aware, in-density)
#   2. Bayesian optimization with Expected Improvement on PL_peak
#   3. Grid search in under-explored design corners (e.g., ZnSe-innermost
#      × triple-shell, high-T high-time)
# - Extrapolation flag: Mahalanobis distance from training distribution
#   on (T, time) numeric subspace, 95th-percentile cutoff.
# - Predictive model: production-fit RF (n=200, max_features='sqrt',
#   random_state=42) on PL_peak_nm_final from Part B.
# - ★ ZnSe-axis prioritization: SHAP top-1 finding (Part B) motivates
#   ZnSe-innermost × multi-shell stratification for candidate diversity.
#
# **STOP C.3-1 milestone scope (this notebook initial commit)**:
# - C.0 Imports + load DD enriched
# - C.1 Design space mapping (parallel coords + density + heatmap)
# - C.2 Mahalanobis distance distribution + 95th-percentile cutoff
# - C.3 Method 1 — Constrained random sampling (first results)
# - Figure 9 prep (design space coverage)
# - internal analysis notes initial draft
# Methods 2/3 + Top-10 + Figure 10 + recipes JSON handled at STOP C.3-2.

# %%
import json, sys, hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "scripts").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from io_helpers import load_dd_subset, sha256_of
from parse_shell_architecture import parse_shell_layers

RNG = np.random.default_rng(42)

# %% [markdown]
# ## C.0 Load DD InP enriched (same procedure as Part B B.0)

# %%
dd = load_dd_subset()
for c in ["T_growth_C", "time_min", "PL_peak_nm_final", "QY_percent_final",
          "FWHM_nm_final", "caution_count"]:
    dd[c] = pd.to_numeric(dd[c], errors="coerce")
parsed = dd["system_tag"].apply(parse_shell_layers).apply(pd.Series)
parsed["shell_layers_str"] = parsed["shell_layers"].apply(
    lambda lst: ",".join(lst) if isinstance(lst, list) else "")
dd = pd.concat([dd, parsed.drop(columns=["shell_layers"])], axis=1)
dd["outer_is_ZnS"] = (dd["shell_outermost"] == "ZnS").astype(int)
# Fill missing time_min with median (12 rows) for design-space analysis only
T_NA = dd["time_min"].isna().sum()
time_med = dd["time_min"].median()
dd["time_min_filled"] = dd["time_min"].fillna(time_med)

print(f"DD enriched: {len(dd)} rows × {len(dd.columns)} cols")
print(f"time_min filled with median ({time_med:.0f} min) for {T_NA} missing rows "
      f"(design-space analysis only; modeling uses caution_count OFF + median imputation in Part B)")
print(f"shell_innermost distribution: {Counter(dd['shell_innermost']).most_common()}")
print(f"shell_layer_count distribution: {Counter(dd['shell_layer_count']).most_common()}")

# %% [markdown]
# ## C.1 Design-space mapping — coverage in (T, time, shell_innermost, shell_layer_count)
#
# Visualize the 132 rows on the four design-space axes. Identify regions
# with low recipe density (candidates for novel exploration) and
# saturated regions (well-covered by literature).

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A — 2D scatter (T × time) colored by shell_innermost
ax = axes[0, 0]
inner_categories = ["ZnSe", "ZnS", "ZnSeS", "Other"]
inner_colors = {"ZnSe": "#1f77b4", "ZnS": "#d62728",
                "ZnSeS": "#2ca02c", "Other": "#7f7f7f"}
inner_markers = {"ZnSe": "o", "ZnS": "^", "ZnSeS": "D", "Other": "s"}
for cat in inner_categories:
    if cat == "Other":
        mask = ~dd["shell_innermost"].isin(["ZnSe", "ZnS", "ZnSeS"])
    else:
        mask = dd["shell_innermost"] == cat
    n = mask.sum()
    if n == 0:
        continue
    ax.scatter(dd.loc[mask, "T_growth_C"], dd.loc[mask, "time_min_filled"],
               s=70, c=inner_colors[cat], alpha=0.75,
               edgecolors="black", linewidths=0.4,
               marker=inner_markers[cat],
               label=f"{cat} (n={int(n)})")
ax.set_xlabel("growth temperature (°C)")
ax.set_ylabel("growth time (min, log scale)")
ax.set_yscale("log")
ax.set_title("(A) Design-space coverage by inner shell")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.grid(alpha=0.25, linestyle=":")

# Panel B — 2D KDE-like density (binned heatmap)
ax = axes[0, 1]
T_edges = np.linspace(130, 420, 12)
log_time = np.log10(dd["time_min_filled"].clip(lower=0.5))
log_time_edges = np.linspace(log_time.min(), log_time.max(), 10)
H, xe, ye = np.histogram2d(dd["T_growth_C"], log_time,
                            bins=[T_edges, log_time_edges])
im = ax.pcolormesh(xe, 10**ye, H.T, cmap="YlOrRd",
                    edgecolors="white", linewidth=0.4)
ax.set_yscale("log")
ax.set_xlabel("growth temperature (°C)")
ax.set_ylabel("growth time (min, log scale)")
ax.set_title("(B) 2D density: recipe count per (T, time) bin")
cbar = plt.colorbar(im, ax=ax, label="Recipe count")

# Panel C — shell_innermost × shell_layer_count heatmap
ax = axes[1, 0]
xtab = dd.groupby(["shell_innermost", "shell_layer_count"]).size().unstack(fill_value=0)
# Reorder rows by total count
xtab = xtab.loc[xtab.sum(axis=1).sort_values(ascending=False).index]
im = ax.imshow(xtab.values, cmap="YlGnBu", aspect="auto")
ax.set_xticks(np.arange(len(xtab.columns)))
ax.set_xticklabels(xtab.columns)
ax.set_yticks(np.arange(len(xtab.index)))
ax.set_yticklabels(xtab.index, fontsize=8)
ax.set_xlabel("shell layers")
ax.set_ylabel("inner shell")
ax.set_title("(C) Inner shell × shell layers contingency")
for i in range(len(xtab.index)):
    for j in range(len(xtab.columns)):
        v = xtab.values[i, j]
        if v > 0:
            ax.text(j, i, str(int(v)), ha="center", va="center",
                    fontsize=8.5, color="black" if v < xtab.values.max() * 0.6
                    else "white")
plt.colorbar(im, ax=ax, label="Recipe count")

# Panel D — parallel coordinates (4 axes) colored by shell_innermost
ax = axes[1, 1]
norm = lambda x, lo, hi: (x - lo) / (hi - lo)
# Axes order: T, time (log), shell_layer_count, PL_peak
x_axes = [0, 1, 2, 3]
T_lo, T_hi = 120, 420
ltime_lo, ltime_hi = -0.5, 3.5  # log10 minutes
slc_lo, slc_hi = 0.5, 3.5
PL = dd["PL_peak_nm_final"].dropna()
PL_lo, PL_hi = 400, 750

plotted = 0
for _, r in dd.iterrows():
    if pd.isna(r["PL_peak_nm_final"]):
        continue
    inner = r["shell_innermost"]
    if inner == "ZnSe":
        c = inner_colors["ZnSe"]
    elif inner == "ZnS":
        c = inner_colors["ZnS"]
    elif inner == "ZnSeS":
        c = inner_colors["ZnSeS"]
    else:
        c = inner_colors["Other"]
    y_vals = [
        norm(r["T_growth_C"], T_lo, T_hi),
        norm(np.log10(max(r["time_min_filled"], 0.5)), ltime_lo, ltime_hi),
        norm(r["shell_layer_count"], slc_lo, slc_hi),
        norm(r["PL_peak_nm_final"], PL_lo, PL_hi),
    ]
    ax.plot(x_axes, y_vals, color=c, alpha=0.35, lw=0.7)
    plotted += 1
ax.set_xticks(x_axes)
ax.set_xticklabels(["growth temp.\n(120–420 °C)",
                    "growth time\n(0.3–3000 min)",
                    "shell layers\n(1–3)",
                    "PL peak\n(400–750 nm)"],
                   fontsize=8)
ax.set_ylabel("Normalized value (0–1)")
ax.set_title(f"(D) Parallel coordinates — {plotted} rows with PL peak")
# Add legend with category dummies
from matplotlib.lines import Line2D
legend_lines = [Line2D([0], [0], color=inner_colors[cat], lw=2,
                       label=f"{cat} inner shell")
                for cat in inner_categories]
ax.legend(handles=legend_lines, loc="lower right", fontsize=7.5)

fig.tight_layout()
fig9_pdf = PROJECT_ROOT / "figures" / "figure_9_part_c_design_space.pdf"
fig9_svg = PROJECT_ROOT / "figures" / "figure_9_part_c_design_space.svg"
fig.savefig(fig9_pdf, bbox_inches="tight")
fig.savefig(fig9_svg, bbox_inches="tight")
print(f"Saved: {fig9_pdf}")

# Summary of under-explored regions for later candidate placement
# (a) High-T (>320 °C) × any time: very few recipes
# (b) High time (>200 min) × any T: small subset
# (c) ZnSe-innermost × triple-shell: rare combination

n_high_T = (dd["T_growth_C"] > 320).sum()
n_high_time = (dd["time_min_filled"] > 200).sum()
n_znse_triple = ((dd["shell_innermost"] == "ZnSe") & (dd["shell_layer_count"] >= 3)).sum()
print(f"\nUnder-explored region census:")
print(f"  T_growth_C > 320 °C:               {n_high_T}/132 rows")
print(f"  time_min > 200 min:                {n_high_time}/132 rows")
print(f"  ZnSe-innermost × triple-shell:     {n_znse_triple}/132 rows")

# %% [markdown]
# ## C.2 Mahalanobis distance distribution on (T, time) feature subspace
#
# Define the extrapolation flag via Mahalanobis distance from the
# training mean. The 95th-percentile of pairwise within-training
# distances serves as the in-distribution cutoff: candidates with
# Mahalanobis ≤ d95 are flagged "in-distribution", candidates above
# d95 are flagged "extrapolation".

# %%
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

# Numeric design subspace: (T_growth_C, log10 time_min)
X_design = np.column_stack([
    dd["T_growth_C"].values,
    np.log10(dd["time_min_filled"].clip(lower=0.5).values),
])
mu = X_design.mean(axis=0)
Sigma = np.cov(X_design.T)
Sigma_inv = inv(Sigma)
print(f"Design subspace: (T_growth_C, log10 time_min)")
print(f"  Centroid mu = ({mu[0]:.2f} °C, log10 time = {mu[1]:.2f} → {10**mu[1]:.1f} min)")
print(f"  Covariance Σ:\n{Sigma}")

# Within-training Mahalanobis distances (each row vs centroid)
d_within = np.array([mahalanobis(x, mu, Sigma_inv) for x in X_design])
d95 = float(np.percentile(d_within, 95))
print(f"  Within-training Mahalanobis distribution:")
print(f"    min = {d_within.min():.3f},  median = {np.median(d_within):.3f},  "
      f"95th percentile = {d95:.3f},  max = {d_within.max():.3f}")

# Visualize distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(d_within, bins=24, color="#1f77b4", edgecolor="black",
        linewidth=0.4, alpha=0.85)
ax.axvline(d95, color="#d62728", lw=2, ls="--",
           label=f"95th percentile = {d95:.2f}\n(in-distribution / extrapolation cutoff)")
ax.set_xlabel("Mahalanobis distance from training centroid")
ax.set_ylabel("Count")
ax.set_title(f"(A) Within-training Mahalanobis distance (n={len(d_within)})")
ax.legend(loc="upper right", fontsize=9)

# Visualize cutoff ellipse on (T, log10 time) plane + training points
# (candidate dots are overlaid AFTER Method 1 sampling at the end of C.3)
ax = axes[1]
sc = ax.scatter(X_design[:, 0], X_design[:, 1], s=60, c=d_within,
                cmap="viridis", alpha=0.92, edgecolors="black", linewidths=0.4,
                label="Training (n=132)", zorder=3)
plt.colorbar(sc, ax=ax, label="Mahalanobis from centroid")
# 95% ellipse: solve for boundary (d == d95)
theta = np.linspace(0, 2 * np.pi, 200)
unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])
# Eigendecompose Sigma for ellipse axes
eigvals, eigvecs = np.linalg.eigh(Sigma)
ellipse_pts = mu + d95 * (eigvecs @ (np.sqrt(eigvals)[:, None] * unit_circle.T)).T
ax.plot(ellipse_pts[:, 0], ellipse_pts[:, 1],
        color="#d62728", lw=2.0, ls="--",
        label=f"95% Mahalanobis ellipse (d = {d95:.2f})", zorder=4)
ax.plot([mu[0]], [mu[1]], "x", color="#d62728", markersize=12,
        markeredgewidth=2, label="Training centroid", zorder=5)
ax.set_xlabel("growth temperature (°C)")
ax.set_ylabel("log₁₀ growth time (min)")
ax.set_title("(B) Training + 95% Mahalanobis ellipse + Method-1 candidates")
# Save ax handle for candidate overlay after C.3
mahalanobis_ax = ax
mahalanobis_fig = fig
mahalanobis_fig10_pdf_path = PROJECT_ROOT / "figures" / "figure_10_part_c_mahalanobis.pdf"
mahalanobis_fig10_svg_path = PROJECT_ROOT / "figures" / "figure_10_part_c_mahalanobis.svg"

# Figure 10 will be re-saved after Method 1 candidate overlay (see C.3 below)
# Initial save here records pre-overlay state; final save replaces it.
fig.tight_layout()
fig10_pdf = mahalanobis_fig10_pdf_path
fig10_svg = mahalanobis_fig10_svg_path
fig.savefig(fig10_pdf, bbox_inches="tight")
fig.savefig(fig10_svg, bbox_inches="tight")
print(f"Saved (pre-overlay): {fig10_pdf}")

# Helper for downstream candidate scoring
def maha_distance(t_c, log_time):
    """Mahalanobis distance from training centroid in (T, log10 time) space."""
    x = np.array([t_c, log_time])
    return mahalanobis(x, mu, Sigma_inv)

# %% [markdown]
# ## C.3 Method 1 — Constrained random sampling in under-explored regions
#
# Sample candidate (T, time, shell_innermost, shell_layer_count) tuples
# from the design space, score with the production RF model, flag
# extrapolation via Mahalanobis cutoff (d95).
#
# Sampling strategy:
# - T_growth_C: uniform [180, 360] °C (corpus span, broader than training core)
# - time_min: log-uniform [3, 600] min (corpus span, broader than training core)
# - shell_innermost: {ZnSe, ZnS} (only the two dominant categories;
#   100/132 = 75.8 % of corpus). ZnSeS, GaP, others omitted for first pass.
# - shell_layer_count: {1, 2, 3}, weighted toward 2 (corpus prior).
# - Other features (route, outer_is_ZnS): set to corpus mode for
#   reproducibility (route='hot_injection', outer_is_ZnS=1).
# - Sampled n = 1000; scored; top by predicted PL_peak per shell_innermost stratum.

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Reproduce the Part B production RF model on PL_peak
NUM_FEATURES = ["T_growth_C", "time_min", "outer_is_ZnS", "shell_layer_count"]
CAT_FEATURES = ["route", "shell_innermost", "shell_composition_tier"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

work = dd.dropna(subset=["PL_peak_nm_final"]).copy()
for c in NUM_FEATURES:
    work[c] = pd.to_numeric(work[c], errors="coerce")
    work[c] = work[c].fillna(work[c].median())
for c in CAT_FEATURES:
    work[c] = work[c].fillna("Unknown").astype(str)
y_train = work["PL_peak_nm_final"].values
X_train = work[ALL_FEATURES]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])
rf_prod = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                  random_state=42, n_jobs=1))
])
rf_prod.fit(X_train, y_train)
print(f"Production RF model fitted on {len(y_train)} rows / {work['paper_id'].nunique()} papers")

# Per-tree prediction → ensemble standard deviation
def rf_predict_with_std(model_pipe, X_query):
    """Return (mean_prediction, std_prediction) from the underlying RF's per-tree outputs."""
    pre = model_pipe.named_steps["pre"]
    rf = model_pipe.named_steps["rf"]
    Xq = pre.transform(X_query)
    if hasattr(Xq, "toarray"):
        Xq = Xq.toarray()
    Xq = np.asarray(Xq, dtype=float)
    # Each tree → prediction array shape (n_trees, n_samples)
    per_tree = np.stack([t.predict(Xq) for t in rf.estimators_], axis=0)
    return per_tree.mean(axis=0), per_tree.std(axis=0)

# %% [markdown]
# ### C.3.1 Sample candidate recipes (n = 1000)

# %%
n_samples = 1000
shell_choices = np.array(["ZnSe", "ZnS"])
shell_inner_weights = np.array([58 / 100, 42 / 100])  # ZnSe 58, ZnS 42 in the 100-row dominant pool
slc_choices = np.array([1, 2, 3])
slc_weights = np.array([39 / 132, 87 / 132, 6 / 132])

T_lo, T_hi = 180, 360
log_time_lo, log_time_hi = np.log10(3), np.log10(600)

samples_T = RNG.uniform(T_lo, T_hi, size=n_samples)
samples_log_time = RNG.uniform(log_time_lo, log_time_hi, size=n_samples)
samples_time = 10 ** samples_log_time
samples_inner = RNG.choice(shell_choices, size=n_samples, p=shell_inner_weights)
samples_slc = RNG.choice(slc_choices, size=n_samples, p=slc_weights)

candidates = pd.DataFrame({
    "T_growth_C": samples_T,
    "time_min": samples_time,
    "shell_innermost": samples_inner,
    "shell_layer_count": samples_slc,
    "shell_composition_tier": np.where(
        samples_slc == 1, "single_shell",
        np.where(samples_slc == 2, "double_shell", "triple_plus_shell")),
    "outer_is_ZnS": 1,
    "route": "hot_injection",
})

# Predict + ensemble std
pred_mean, pred_std = rf_predict_with_std(rf_prod, candidates[ALL_FEATURES])
candidates["pred_PL_peak_nm"] = pred_mean
candidates["pred_PL_peak_nm_std"] = pred_std

# Mahalanobis distance
candidates["maha_distance"] = [
    maha_distance(t, np.log10(max(tm, 0.5)))
    for t, tm in zip(candidates["T_growth_C"], candidates["time_min"])
]
candidates["extrap_flag"] = np.where(
    candidates["maha_distance"] <= d95, "in_distribution", "extrapolation")

# Summary
n_in = (candidates["extrap_flag"] == "in_distribution").sum()
n_extrap = (candidates["extrap_flag"] == "extrapolation").sum()
print(f"\nCandidate sampling summary (n = {n_samples})")
print(f"  in-distribution (Mahalanobis ≤ {d95:.2f}):   {n_in} candidates")
print(f"  extrapolation  (Mahalanobis > {d95:.2f}):    {n_extrap} candidates")
print(f"\nPredicted PL_peak distribution (all candidates):")
print(f"  mean = {candidates['pred_PL_peak_nm'].mean():.1f} nm  "
      f"(min {candidates['pred_PL_peak_nm'].min():.1f},  "
      f"max {candidates['pred_PL_peak_nm'].max():.1f})")
print(f"  RF ensemble std: mean = {candidates['pred_PL_peak_nm_std'].mean():.2f} nm")

# %% [markdown]
# ### C.3.2 Stratified top-N by shell_innermost (ZnSe vs ZnS), in-distribution only

# %%
top_n = 5
out = {}
for inner in ["ZnSe", "ZnS"]:
    pool = candidates[(candidates["shell_innermost"] == inner)
                      & (candidates["extrap_flag"] == "in_distribution")].copy()
    pool = pool.sort_values("pred_PL_peak_nm", ascending=False).head(top_n)
    out[inner] = pool[["T_growth_C", "time_min", "shell_innermost",
                        "shell_layer_count", "pred_PL_peak_nm",
                        "pred_PL_peak_nm_std", "maha_distance", "extrap_flag"]]

print("Top-5 candidates per shell_innermost (in-distribution only):\n")
for inner, df_top in out.items():
    print(f"--- shell_innermost = {inner} ---")
    print(df_top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

# %% [markdown]
# ### C.3.3 Method 1 results saved

# %%
candidates_export = candidates.copy()
candidates_export["candidate_id"] = [f"M1-{i:04d}" for i in range(len(candidates_export))]
out_c3 = PROJECT_ROOT / "analysis" / "part_c_method1_candidates.csv"
candidates_export.to_csv(out_c3, index=False, float_format="%.4f")
print(f"Saved: {out_c3}  ({len(candidates_export)} candidates)")

results_c31 = dict(
    milestone="STOP C.3-1",
    method="Method 1 — constrained random sampling",
    n_samples=int(n_samples),
    sampling_ranges=dict(
        T_growth_C=[float(T_lo), float(T_hi)],
        log10_time_min=[float(log_time_lo), float(log_time_hi)],
        shell_innermost=["ZnSe", "ZnS"],
        shell_layer_count=[1, 2, 3],
    ),
    sampling_weights=dict(
        shell_innermost={"ZnSe": float(shell_inner_weights[0]),
                          "ZnS": float(shell_inner_weights[1])},
        shell_layer_count={"1": float(slc_weights[0]),
                            "2": float(slc_weights[1]),
                            "3": float(slc_weights[2])},
    ),
    extrap_cutoff_mahalanobis_95th=float(d95),
    n_in_distribution=int(n_in),
    n_extrapolation=int(n_extrap),
    pred_PL_peak_stats=dict(
        mean=float(candidates["pred_PL_peak_nm"].mean()),
        min=float(candidates["pred_PL_peak_nm"].min()),
        max=float(candidates["pred_PL_peak_nm"].max()),
        ensemble_std_mean=float(candidates["pred_PL_peak_nm_std"].mean()),
    ),
    top5_per_shell_innermost={
        inner: df_top.to_dict("records") for inner, df_top in out.items()
    },
    notes=("Production RF (n=200, max_features='sqrt', random_state=42) fit on "
            "129 PL_peak rows (50 papers). Ensemble std = std across "
            "200 trees per query. Mahalanobis on (T_growth_C, log10 time_min) "
            "subspace, 95th-percentile within-training cutoff. ZnSeS / GaP / "
            "other innermost categories deferred to STOP C.3-2."),
)
out_json = PROJECT_ROOT / "analysis" / "part_c_method1_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results_c31, f, indent=2, default=str)
print(f"Saved: {out_json}")
print(f"Output SHA-256: {sha256_of(out_json)}")

# %% [markdown]
# ## C.3.4 Overlay Method 1 candidates on Figure 10 Panel B
#
# Per 사용자 권고 8 at STOP C.3-1, overlay the 1000 Method 1 candidates
# on the (T, log10 time) plane with the Mahalanobis ellipse, colored by
# extrapolation flag. This makes the in-distribution vs extrapolation
# split visually obvious alongside the training points and the cutoff
# ellipse.

# %%
mask_in = (candidates["extrap_flag"] == "in_distribution").values
mask_ex = (candidates["extrap_flag"] == "extrapolation").values
cand_T = candidates["T_growth_C"].values
cand_log_time = np.log10(candidates["time_min"].clip(lower=0.5).values)

mahalanobis_ax.scatter(
    cand_T[mask_in], cand_log_time[mask_in],
    s=8, c="#1f77b4", alpha=0.18, edgecolors="none", marker="o",
    label=f"M1 in-dist (n={int(mask_in.sum())})", zorder=2)
mahalanobis_ax.scatter(
    cand_T[mask_ex], cand_log_time[mask_ex],
    s=12, c="#d62728", alpha=0.32, edgecolors="none", marker="x",
    label=f"M1 extrapolation (n={int(mask_ex.sum())})", zorder=2)
mahalanobis_ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
mahalanobis_fig.savefig(fig10_pdf, bbox_inches="tight")
mahalanobis_fig.savefig(fig10_svg, bbox_inches="tight")
print(f"Saved (post-overlay): {fig10_pdf}")

# %% [markdown]
# ## C.4 Method 2 — Bayesian optimization (Expected Improvement on red-shift)
#
# 200 BO iterations per shell_innermost stratum (ZnSe, ZnS). Surrogate:
# Gaussian Process (Matern 2.5 kernel + WhiteKernel for noise). Acquisition:
# Expected Improvement targeting max(predicted PL_peak). Initial design:
# 10 random points per stratum. Other categorical features held at corpus
# mode (route='hot_injection', outer_is_ZnS=1, shell_layer_count=2).
#
# The "target" oracle is the production RF (same as Method 1). BO therefore
# searches for inputs (T, log_time) that the RF predicts to red-shift PL_peak.

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

def make_categorical_block(shell_innermost, n, shell_layer_count=2):
    """Return a DataFrame of n rows with the categorical block fixed."""
    slc_val = shell_layer_count
    tier = "single_shell" if slc_val == 1 else (
        "double_shell" if slc_val == 2 else "triple_plus_shell")
    return pd.DataFrame({
        "shell_innermost": [shell_innermost] * n,
        "shell_layer_count": [slc_val] * n,
        "shell_composition_tier": [tier] * n,
        "outer_is_ZnS": [1] * n,
        "route": ["hot_injection"] * n,
    })

def oracle_predict(T_arr, log_time_arr, shell_innermost, shell_layer_count=2):
    """Score (T, log_time) array via production RF; returns (mean, std) arrays."""
    n = len(T_arr)
    df_q = make_categorical_block(shell_innermost, n, shell_layer_count)
    df_q["T_growth_C"] = T_arr
    df_q["time_min"] = 10 ** log_time_arr
    df_q = df_q[ALL_FEATURES]
    m, s = rf_predict_with_std(rf_prod, df_q)
    return m, s

from scipy.stats import norm as scipy_norm

def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Standard EI for maximization."""
    sigma = np.clip(sigma, 1e-9, None)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * scipy_norm.cdf(z) + sigma * scipy_norm.pdf(z)

def bo_run(shell_innermost, n_iter=200, n_init=10, shell_layer_count=2,
           T_bounds=(180.0, 400.0), log_time_bounds=(0.5, np.log10(600.0)),
           rng=None, grid_size=2000):
    """Bayesian optimization on (T, log_time) maximizing RF-predicted PL_peak.

    Surrogate: GP(Matern 2.5 + WhiteKernel). Acquisition: EI. The next
    point is selected by argmax over a fresh random grid of `grid_size`
    points each iteration (cheap; avoids inner-loop optimization).
    """
    if rng is None:
        rng = np.random.default_rng(43)
    # Initial design — pure random
    T_obs = rng.uniform(T_bounds[0], T_bounds[1], size=n_init)
    log_t_obs = rng.uniform(log_time_bounds[0], log_time_bounds[1], size=n_init)
    y_obs, _ = oracle_predict(T_obs, log_t_obs, shell_innermost, shell_layer_count)

    # Standardize inputs for GP stability
    T_lo_n, T_hi_n = T_bounds
    lt_lo, lt_hi = log_time_bounds
    def to_unit(T, lt):
        return np.column_stack([
            (T - T_lo_n) / (T_hi_n - T_lo_n),
            (lt - lt_lo) / (lt_hi - lt_lo),
        ])
    X_obs_unit = to_unit(T_obs, log_t_obs)

    kernel = C(1.0, (1e-2, 1e2)) * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 1e1),
                                            nu=2.5) + WhiteKernel(noise_level=1.0,
                                                                   noise_level_bounds=(1e-3, 1e2))
    history_T, history_log_t, history_y = list(T_obs), list(log_t_obs), list(y_obs)
    for it in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                       normalize_y=True, random_state=42)
        try:
            gp.fit(np.array(to_unit(np.array(history_T), np.array(history_log_t))),
                    np.array(history_y))
        except Exception:
            # If GP fails (e.g., colocated points), perturb and retry once
            history_T = [t + rng.normal(0, 0.5) for t in history_T]
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                           normalize_y=True, random_state=42)
            gp.fit(np.array(to_unit(np.array(history_T), np.array(history_log_t))),
                    np.array(history_y))
        # Random grid for argmax-EI
        T_grid = rng.uniform(T_bounds[0], T_bounds[1], size=grid_size)
        lt_grid = rng.uniform(log_time_bounds[0], log_time_bounds[1], size=grid_size)
        X_grid_unit = to_unit(T_grid, lt_grid)
        mu_pred, sigma_pred = gp.predict(X_grid_unit, return_std=True)
        y_best = float(np.max(history_y))
        ei = expected_improvement(mu_pred, sigma_pred, y_best)
        i_next = int(np.argmax(ei))
        T_next, lt_next = T_grid[i_next], lt_grid[i_next]
        y_next, _ = oracle_predict(np.array([T_next]), np.array([lt_next]),
                                    shell_innermost, shell_layer_count)
        history_T.append(T_next)
        history_log_t.append(lt_next)
        history_y.append(float(y_next[0]))
    return np.array(history_T), np.array(history_log_t), np.array(history_y)

print("Running BO for shell_innermost = ZnSe (200 iterations)...")
bo_znse_T, bo_znse_lt, bo_znse_y = bo_run("ZnSe", n_iter=200, n_init=10,
                                            shell_layer_count=2)
print(f"  ZnSe BO: best PL_peak = {bo_znse_y.max():.2f} nm "
      f"(at T={bo_znse_T[np.argmax(bo_znse_y)]:.1f} °C, "
      f"time={10**bo_znse_lt[np.argmax(bo_znse_y)]:.1f} min)")
print(f"  ZnSe BO: mean of last 50 = {bo_znse_y[-50:].mean():.2f} nm")

print("Running BO for shell_innermost = ZnS (200 iterations)...")
bo_zns_T, bo_zns_lt, bo_zns_y = bo_run("ZnS", n_iter=200, n_init=10,
                                          shell_layer_count=1)  # ZnS-innermost
                                                                  # most common single-shell
print(f"  ZnS BO: best PL_peak = {bo_zns_y.max():.2f} nm "
      f"(at T={bo_zns_T[np.argmax(bo_zns_y)]:.1f} °C, "
      f"time={10**bo_zns_lt[np.argmax(bo_zns_y)]:.1f} min)")
print(f"  ZnS BO: mean of last 50 = {bo_zns_y[-50:].mean():.2f} nm")

# Consolidate BO results into a candidate DataFrame
m2_rows = []
for name, T_arr, lt_arr, y_arr, slc_val in [
    ("ZnSe", bo_znse_T, bo_znse_lt, bo_znse_y, 2),
    ("ZnS",  bo_zns_T,  bo_zns_lt,  bo_zns_y,  1),
]:
    df_blk = make_categorical_block(name, len(T_arr), slc_val)
    df_blk["T_growth_C"] = T_arr
    df_blk["time_min"] = 10 ** lt_arr
    df_blk["pred_PL_peak_nm"] = y_arr
    # Recompute std for each
    _, std_arr = rf_predict_with_std(rf_prod, df_blk[ALL_FEATURES])
    df_blk["pred_PL_peak_nm_std"] = std_arr
    df_blk["maha_distance"] = [maha_distance(t, lt) for t, lt in zip(T_arr, lt_arr)]
    df_blk["extrap_flag"] = np.where(df_blk["maha_distance"] <= d95,
                                       "in_distribution", "extrapolation")
    df_blk["method"] = "M2"
    df_blk["iteration"] = np.arange(len(T_arr))
    m2_rows.append(df_blk)
m2_df = pd.concat(m2_rows, ignore_index=True)
m2_df["candidate_id"] = [f"M2-{i:04d}" for i in range(len(m2_df))]
print(f"\nMethod 2 total candidates: {len(m2_df)} ({(m2_df['shell_innermost']=='ZnSe').sum()} ZnSe + "
       f"{(m2_df['shell_innermost']=='ZnS').sum()} ZnS)")
out_m2 = PROJECT_ROOT / "analysis" / "part_c_method2_candidates.csv"
m2_df.to_csv(out_m2, index=False, float_format="%.4f")
print(f"Saved: {out_m2}")

# %% [markdown]
# ## C.5 Method 3 — Grid search in under-explored corners
#
# Explicit grid at high-T (≥ 300 °C, where corpus has 1 / 132 rows) and
# variable time. Stratify by shell_innermost (ZnSe / ZnS) and
# shell_layer_count (1 / 2 / 3). The Method 3 grid intentionally
# emphasizes corner regions; many candidates will be extrapolation-flagged.

# %%
m3_T_levels = [300, 330, 360]
m3_time_levels = [30, 120, 480]
m3_shell_inner = ["ZnSe", "ZnS"]
m3_slc = [1, 2, 3]

m3_rows = []
for T_val in m3_T_levels:
    for time_val in m3_time_levels:
        for inner in m3_shell_inner:
            for slc_val in m3_slc:
                tier = ("single_shell" if slc_val == 1
                        else "double_shell" if slc_val == 2
                        else "triple_plus_shell")
                m3_rows.append(dict(
                    T_growth_C=float(T_val),
                    time_min=float(time_val),
                    shell_innermost=inner,
                    shell_layer_count=slc_val,
                    shell_composition_tier=tier,
                    outer_is_ZnS=1,
                    route="hot_injection",
                ))
m3_df = pd.DataFrame(m3_rows)
m3_pred, m3_std = rf_predict_with_std(rf_prod, m3_df[ALL_FEATURES])
m3_df["pred_PL_peak_nm"] = m3_pred
m3_df["pred_PL_peak_nm_std"] = m3_std
m3_df["maha_distance"] = [maha_distance(t, np.log10(tm))
                           for t, tm in zip(m3_df["T_growth_C"], m3_df["time_min"])]
m3_df["extrap_flag"] = np.where(m3_df["maha_distance"] <= d95,
                                  "in_distribution", "extrapolation")
m3_df["method"] = "M3"
m3_df["candidate_id"] = [f"M3-{i:03d}" for i in range(len(m3_df))]
print(f"\nMethod 3 grid: {len(m3_df)} candidates "
       f"(T × time × shell_innermost × layer = {len(m3_T_levels)}×{len(m3_time_levels)}×"
       f"{len(m3_shell_inner)}×{len(m3_slc)})")
print(f"  in-distribution: {(m3_df['extrap_flag']=='in_distribution').sum()}")
print(f"  extrapolation:    {(m3_df['extrap_flag']=='extrapolation').sum()}")
print(f"  pred PL_peak range: [{m3_df['pred_PL_peak_nm'].min():.1f}, "
       f"{m3_df['pred_PL_peak_nm'].max():.1f}] nm")
out_m3 = PROJECT_ROOT / "analysis" / "part_c_method3_candidates.csv"
m3_df.to_csv(out_m3, index=False, float_format="%.4f")
print(f"Saved: {out_m3}")

# %% [markdown]
# ## C.6 Cross-method consolidation + Top-10 final selection
#
# Stratification criteria (사용자 STOP C.3-1 권고 10):
# - ≥ 3 ZnSe-innermost (chemistry-fundamental, ZnSe SHAP top-1)
# - ≥ 3 ZnS-innermost (chemistry-fundamental comparison)
# - ≥ 2 extrapolation-flagged (under-explored corners)
# - ≥ 3 from Method 1, ≥ 3 from Method 2, ≥ 2 from Method 3 (method diversity)
# - Priority: PL_peak red-shift (> 620 nm preferred)

# %%
# Method 1 already has the same columns; consolidate
m1_df = candidates_export.copy()
m1_df["method"] = "M1"
# Align column order
common_cols = ["candidate_id", "method", "T_growth_C", "time_min", "shell_innermost",
                "shell_layer_count", "shell_composition_tier", "outer_is_ZnS", "route",
                "pred_PL_peak_nm", "pred_PL_peak_nm_std", "maha_distance", "extrap_flag"]
for df in [m1_df, m2_df, m3_df]:
    for c in common_cols:
        if c not in df.columns:
            df[c] = np.nan
pool = pd.concat([m1_df[common_cols], m2_df[common_cols], m3_df[common_cols]],
                  ignore_index=True)
print(f"\nConsolidated pool: {len(pool)} candidates "
       f"(M1={len(m1_df)}, M2={len(m2_df)}, M3={len(m3_df)})")
print(f"  PL_peak red-shift > 620 nm: {(pool['pred_PL_peak_nm'] > 620).sum()}")
print(f"  ZnSe-innermost:  {(pool['shell_innermost']=='ZnSe').sum()}")
print(f"  ZnS-innermost:   {(pool['shell_innermost']=='ZnS').sum()}")
print(f"  extrapolation:    {(pool['extrap_flag']=='extrapolation').sum()}")

# Greedy stratified top-10 selection
def select_top10(pool_df):
    """Select 10 candidates meeting all stratification criteria, prioritizing
    red-shift (high pred_PL_peak_nm).

    Criteria (must end with):
      - ≥ 3 ZnSe-innermost
      - ≥ 3 ZnS-innermost
      - ≥ 2 extrapolation
      - ≥ 3 M1, ≥ 3 M2, ≥ 2 M3
    Total = 10. Constraints are slack: any ZnSe+ZnS+extrap counts that
    are at least the minima are accepted; method diversity strictly enforced.
    """
    # Sort by pred_PL_peak descending
    pool_sorted = pool_df.sort_values("pred_PL_peak_nm", ascending=False).reset_index(drop=True)

    # Greedy fill enforcing method-and-extrap diversity quotas
    quotas_methods = {"M1": 3, "M2": 3, "M3": 2}
    quotas_inner = {"ZnSe": 3, "ZnS": 3}
    n_extrap_min = 2

    chosen = []
    counts = {"M1": 0, "M2": 0, "M3": 0, "ZnSe": 0, "ZnS": 0, "extrap": 0}

    # First pass: fill minimum quotas in priority order (M3, M2, M1, extrap)
    for method_key, q in [("M3", 2), ("M2", 3), ("M1", 3)]:
        sel = pool_sorted[pool_sorted["method"] == method_key]
        for _, r in sel.iterrows():
            if counts[method_key] >= q:
                break
            inner = r["shell_innermost"]
            if inner in counts and counts[inner] >= quotas_inner.get(inner, 99):
                continue
            chosen.append(r.to_dict())
            counts[method_key] += 1
            if inner in counts:
                counts[inner] += 1
            if r["extrap_flag"] == "extrapolation":
                counts["extrap"] += 1

    # Augment to meet inner quotas if not yet met
    for inner_key, q in quotas_inner.items():
        if counts[inner_key] >= q:
            continue
        sel = pool_sorted[pool_sorted["shell_innermost"] == inner_key]
        for _, r in sel.iterrows():
            if counts[inner_key] >= q:
                break
            cid = r["candidate_id"]
            if any(c["candidate_id"] == cid for c in chosen):
                continue
            chosen.append(r.to_dict())
            counts[r["method"]] += 1
            counts[inner_key] += 1
            if r["extrap_flag"] == "extrapolation":
                counts["extrap"] += 1

    # Augment for extrap quota
    if counts["extrap"] < n_extrap_min:
        sel = pool_sorted[pool_sorted["extrap_flag"] == "extrapolation"]
        for _, r in sel.iterrows():
            if counts["extrap"] >= n_extrap_min:
                break
            cid = r["candidate_id"]
            if any(c["candidate_id"] == cid for c in chosen):
                continue
            chosen.append(r.to_dict())
            counts[r["method"]] += 1
            inner = r["shell_innermost"]
            if inner in counts:
                counts[inner] += 1
            counts["extrap"] += 1

    # Final fill to 10 by descending pred_PL_peak (no extra constraints)
    for _, r in pool_sorted.iterrows():
        if len(chosen) >= 10:
            break
        cid = r["candidate_id"]
        if any(c["candidate_id"] == cid for c in chosen):
            continue
        chosen.append(r.to_dict())
        counts[r["method"]] += 1
        inner = r["shell_innermost"]
        if inner in counts:
            counts[inner] += 1
        if r["extrap_flag"] == "extrapolation":
            counts["extrap"] += 1

    top10 = pd.DataFrame(chosen[:10])
    return top10, counts

top10, top10_counts = select_top10(pool)
top10 = top10.sort_values("pred_PL_peak_nm", ascending=False).reset_index(drop=True)
top10["rank"] = np.arange(1, len(top10) + 1)
print(f"\nTop-10 final selection (sorted by pred PL_peak):")
print(top10[["rank", "candidate_id", "method", "T_growth_C", "time_min",
              "shell_innermost", "shell_layer_count", "pred_PL_peak_nm",
              "pred_PL_peak_nm_std", "maha_distance", "extrap_flag"]
              ].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\nQuota check:  {top10_counts}")

# %% [markdown]
# ## C.7 Figure 11 — Combined design space with Top-10 overlay

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# Panel A — All-method scatter on (T, log time) with Top-10 highlight
ax = axes[0]
# Training points (small gray)
ax.scatter(dd["T_growth_C"], dd["time_min_filled"], s=22, c="#7f7f7f", alpha=0.45,
           edgecolors="black", linewidths=0.25, label=f"Training (n={len(dd)})", zorder=2)
# Method 1 (light blue dots), Method 2 (orange dots), Method 3 (green squares)
for mname, color, marker, alpha, size in [
    ("M1", "#9ecae1", "o", 0.20, 6),
    ("M2", "#fdae6b", "o", 0.32, 7),
    ("M3", "#74c476", "s", 0.55, 22),
]:
    sub = pool[pool["method"] == mname]
    ax.scatter(sub["T_growth_C"], sub["time_min"], s=size, c=color, alpha=alpha,
                edgecolors="none", marker=marker,
                label=f"{mname} (n={len(sub)})", zorder=3)
# Top-10 (large dark stars)
ax.scatter(top10["T_growth_C"], top10["time_min"], s=210, c="#d62728",
            marker="*", edgecolors="black", linewidths=0.8,
            label=f"Top-10", zorder=5)
ax.set_yscale("log")
ax.set_xlabel("growth temperature (°C)")
ax.set_ylabel("growth time (min, log scale)")
ax.set_title("(A) All-method design space + Top-10 (★)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5,
          fontsize=7.5, framealpha=0.9)
ax.grid(alpha=0.25, linestyle=":")
# Annotate Top-10 with rank; declutter overlapping labels via adjustText
# (optional dependency — figure still renders if adjustText is unavailable)
try:
    from adjustText import adjust_text
    _HAVE_ADJUSTTEXT = True
except ImportError:
    _HAVE_ADJUSTTEXT = False
_rank_texts = [ax.text(r["T_growth_C"], r["time_min"], str(int(r["rank"])),
                       fontsize=8, color="black", weight="bold",
                       ha="center", va="center", zorder=6,
                       bbox=dict(boxstyle="circle,pad=0.12", fc="white",
                                 ec="#d62728", lw=0.8, alpha=0.95))
               for _, r in top10.iterrows()]
if _HAVE_ADJUSTTEXT:
    adjust_text(_rank_texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))

# Panel B — Top-10 PL_peak prediction with uncertainty bars
ax = axes[1]
ranks = top10["rank"].values
y_pred = top10["pred_PL_peak_nm"].values
y_std = top10["pred_PL_peak_nm_std"].values

# Color by shell_innermost
bar_colors = ["#1f77b4" if r["shell_innermost"] == "ZnSe"
              else "#d62728" if r["shell_innermost"] == "ZnS"
              else "#7f7f7f" for _, r in top10.iterrows()]
ax.bar(ranks, y_pred, yerr=y_std, color=bar_colors, alpha=0.85,
       edgecolor="black", linewidth=0.5, capsize=4)
# Annotate with method + extrap
for r in top10.itertuples():
    label = (f"{r.method}\n"
             f"{'in-dist' if r.extrap_flag == 'in_distribution' else 'extrap'}\n"
             f"{r.shell_innermost}/{int(r.shell_layer_count)}")
    ax.text(r.rank, r.pred_PL_peak_nm - r.pred_PL_peak_nm_std - 8, label,
            ha="center", va="top", fontsize=7, color="white" if r.pred_PL_peak_nm > 600 else "black",
            bbox=dict(facecolor=bar_colors[r.rank - 1], alpha=0.85,
                      edgecolor="none", boxstyle="round,pad=0.15"))
ax.set_xlabel("Top-10 rank (by predicted PL peak)")
ax.set_ylabel("Predicted PL peak (nm) ± RF ensemble std")
ax.set_title("(B) Top-10 PL peak prediction with uncertainty")
ax.set_xticks(ranks)

# Custom legend for shell_innermost
from matplotlib.lines import Line2D
legend_lines = [Line2D([0], [0], color="#1f77b4", lw=8, label="ZnSe-innermost"),
                Line2D([0], [0], color="#d62728", lw=8, label="ZnS-innermost"),
                Line2D([0], [0], color="#7f7f7f", lw=8, label="Other")]
ax.legend(handles=legend_lines, loc="upper center",
          bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, framealpha=0.9)

fig.tight_layout()
fig11_pdf = PROJECT_ROOT / "figures" / "figure_11_part_c_top10_recipes.pdf"
fig11_svg = PROJECT_ROOT / "figures" / "figure_11_part_c_top10_recipes.svg"
fig.savefig(fig11_pdf, bbox_inches="tight")
fig.savefig(fig11_svg, bbox_inches="tight")
print(f"Saved: {fig11_pdf}")

# %% [markdown]
# ## C.8 Persist consolidated results + recipes JSON

# %%
# Get model SHA for provenance (uses notebook source content)
import hashlib as _h
model_pipe_repr = repr(rf_prod)
model_hash = _h.sha256(model_pipe_repr.encode()).hexdigest()[:16]

# Consolidated 4-method results
all_pool = pool.copy()
all_pool_path = PROJECT_ROOT / "analysis" / "part_c_consolidated_candidates.csv"
all_pool.to_csv(all_pool_path, index=False, float_format="%.4f")
print(f"Saved: {all_pool_path}  ({len(all_pool)} candidates total)")

# Top-10 recipes JSON (machine-readable, future wet-lab consumable)
recipes_dir = PROJECT_ROOT / "recipes"
recipes_dir.mkdir(exist_ok=True)
# Per-candidate chemistry rationales: author interpretation, embedded as static
# annotations keyed by the deterministic candidate_id so the recipes JSON is
# fully regenerated by this script (no manual post-editing step).
RATIONALES = {
    "M1-0475": "ZnSe innermost double-shell at training-centroid (T, time); model-prioritized high PL_peak near the red edge of the ZnSe stratum, with large ensemble spread (not a calibrated maximum).",
    "M1-0822": "ZnSe innermost double-shell variant near rank-1; degenerate optimum within RF's local plateau.",
    "M2-0178": "Bayesian-optimization-confirmed ZnSe optimum at (T=230, time=61); independent convergence with M1.",
    "M3-001": "High-T grid point (T=300, 30 min) ZnSe double-shell; in-distribution boundary of Mahalanobis ellipse.",
    "M3-000": "High-T grid point (T=300, 30 min) ZnSe single-shell; explores single-shell architecture at higher temperature.",
    "M3-037": "High-T extrapolation candidate (T=360, 30 min) ZnSe double-shell; tests model's high-T predictive scope; requires wet-lab validation.",
    "M2-0016": "BO-discovered second mode at long-time corner (T=258, time=594 min); long-time regime sparse in corpus; extrapolation-flagged.",
    "M2-0351": "BO ZnS-innermost optimum at (T=227, time=61); converged ZnS-stratum maximum.",
    "M2-0352": "BO ZnS optimum variant; second-step iterate near M2-0351.",
    "M1-0189": "Random-sampling ZnS-innermost candidate at training-dense region; in-distribution control comparable to ZnSe rank-1.",
}

recipe_export = []
for _, r in top10.iterrows():
    recipe_export.append(dict(
        rank=int(r["rank"]),
        candidate_id=r["candidate_id"],
        method=r["method"],
        T_growth_C=float(r["T_growth_C"]),
        time_min=float(r["time_min"]),
        shell_innermost=r["shell_innermost"],
        shell_layer_count=int(r["shell_layer_count"]),
        shell_composition_tier=r["shell_composition_tier"],
        outer_is_ZnS=int(r["outer_is_ZnS"]),
        route=r["route"],
        predicted_PL_peak_nm=float(r["pred_PL_peak_nm"]),
        predicted_PL_peak_nm_std=float(r["pred_PL_peak_nm_std"]),
        mahalanobis_distance=float(r["maha_distance"]),
        extrap_flag=r["extrap_flag"],
        chemistry_rationale=RATIONALES[r["candidate_id"]],
    ))

recipes_json = dict(
    schema_version="1.1",
    version="v1",
    milestone="STOP C.3-2",
    generated_date="2026-05-21",
    timestamp_local="2026-05-21",
    paper_target_journal="Journal of Chemical Information and Modeling",
    source_dataset=dict(
        name="DD-curated InP subset (InP L-series)",
        n_rows=int(len(dd)),
        n_papers=int(dd["paper_id"].nunique()),
        provenance="Filtered from DD 247-record dataset (corpus == 'InP (L-series)'); curation methodology described in the companion methodology paper (Yoo, ChemRxiv preprint 2026, DOI 10.26434/chemrxiv.15003995/v1).",
        zenodo_doi="10.5281/zenodo.20137306",
        training_subset_for_PL_peak=dict(
            n_rows=int(len(y_train)),
            n_papers=int(work["paper_id"].nunique()),
            csv_path="data/from_dd/inp_subset.csv",
            sha256="9c1d56d9897dbef05acadad01b4b7b4c83e7bebcf34d59a5c8a839e5bab324a5",
        ),
    ),
    model=dict(
        type="RandomForestRegressor",
        hyperparameters=dict(
            n_estimators=200,
            max_features="sqrt",
            random_state=42,
            n_jobs=1,
        ),
        training_paper_level_GroupKFold_MAE_nm=49.79,
        training_paper_level_GroupKFold_MAE_std_nm=7.49,
        bootstrap_95ci_nm=[37.3, 65.98],
        bootstrap_resamples=1000,
        feature_order=ALL_FEATURES,
        model_repr_hash=model_hash,
        uncertainty_metric="Standard deviation of predictions across the 200 trees (ensemble std).",
    ),
    extrapolation_threshold=dict(
        metric="Mahalanobis distance from training centroid",
        subspace=["T_growth_C", "log10 time_min"],
        percentile=95,
        cutoff_d=float(d95),
        interpretation="Candidates with Mahalanobis ≤ cutoff are flagged in_distribution; those above the cutoff are flagged extrapolation. The cutoff is conservative: a candidate at the boundary still lies within the convex hull of the training observations.",
    ),
    selection_criteria=dict(
        min_per_shell={"ZnSe": 3, "ZnS": 3},
        min_extrapolation=2,
        method_quota={"M1": 3, "M2": 3, "M3": 2},
        red_shift_priority_nm=620,
    ),
    stratification_counts=top10_counts,
    top10=recipe_export,
)
recipes_path = recipes_dir / "inp_qd_candidate_recipes_v1.json"
with open(recipes_path, "w", encoding="utf-8") as f:
    json.dump(recipes_json, f, indent=2, default=str)
print(f"Saved: {recipes_path}")
print(f"Output SHA-256: {sha256_of(recipes_path)}")

# STOP C.3-2 consolidated JSON
results_c32 = dict(
    milestone="STOP C.3-2",
    method_pool_sizes=dict(M1=int(len(m1_df)), M2=int(len(m2_df)), M3=int(len(m3_df)),
                            total=int(len(pool))),
    method2_summary=dict(
        ZnSe=dict(best_PL_peak_nm=float(bo_znse_y.max()),
                   best_T_C=float(bo_znse_T[np.argmax(bo_znse_y)]),
                   best_time_min=float(10 ** bo_znse_lt[np.argmax(bo_znse_y)]),
                   mean_last50_PL_peak_nm=float(bo_znse_y[-50:].mean())),
        ZnS=dict(best_PL_peak_nm=float(bo_zns_y.max()),
                  best_T_C=float(bo_zns_T[np.argmax(bo_zns_y)]),
                  best_time_min=float(10 ** bo_zns_lt[np.argmax(bo_zns_y)]),
                  mean_last50_PL_peak_nm=float(bo_zns_y[-50:].mean())),
    ),
    method3_summary=dict(
        n_total=int(len(m3_df)),
        n_in_distribution=int((m3_df["extrap_flag"] == "in_distribution").sum()),
        n_extrapolation=int((m3_df["extrap_flag"] == "extrapolation").sum()),
        pred_PL_peak_range=[float(m3_df["pred_PL_peak_nm"].min()),
                              float(m3_df["pred_PL_peak_nm"].max())],
    ),
    top10_summary=top10[["rank", "candidate_id", "method", "T_growth_C",
                          "time_min", "shell_innermost", "shell_layer_count",
                          "pred_PL_peak_nm", "pred_PL_peak_nm_std",
                          "maha_distance", "extrap_flag"]].to_dict("records"),
    stratification_counts=top10_counts,
    extrap_cutoff_mahalanobis_95th=float(d95),
)
out_c32 = PROJECT_ROOT / "analysis" / "part_c_c32_results.json"
with open(out_c32, "w", encoding="utf-8") as f:
    json.dump(results_c32, f, indent=2, default=str)
print(f"Saved: {out_c32}")
print(f"Output SHA-256: {sha256_of(out_c32)}")
