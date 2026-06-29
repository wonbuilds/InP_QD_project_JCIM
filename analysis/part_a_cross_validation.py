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
# # Part A — Cross-validation (JCIM application paper)
#
# **Stages** (the project analysis plan §5.1 v2.1, the analysis design plan §2):
#
# - A.1 Paper-level paired comparison (9 overlap papers, this notebook).
# - A.2 Distribution-level comparison (independent samples — separate notebook section).
# - A.3 Reproduction of Cossairt PCA/RF/GB on full Cossairt 220 (separate section).
# - A.4 Cross-prediction.
#
# **Analysis design notes outcome (2026-05-21)**:
# Option B adopted — Part A.1 paired comparison uses `emission_nm` (primary)
# + `T_growth_C` (secondary, with paper-median aggregation and caveat for L002 / L009).
# `time_min` is excluded from A.1 paired comparison and analyzed only in A.2
# (distribution-level).
#
# **Bootstrap policy**: paper-level resample with
# replacement, n = 1000 iterations. Row-level bootstrap is *not* used because
# within-paper rows are dependent (recipe variants from the same paper).

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from adjustText import adjust_text
except ImportError:  # graceful no-op fallback; figures still render (labels just not de-cluttered)
    def adjust_text(*_a, **_k):
        return None

PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / "scripts").exists() is False:
    PROJECT_ROOT = PROJECT_ROOT.parent  # if launched from analysis/
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from io_helpers import load_combined, sha256_of

RNG = np.random.default_rng(42)

# %% [markdown]
# ## Provenance — SHA-256 of inputs (reproducibility cell)

# %%
inp_combined = PROJECT_ROOT / "data" / "integrated" / "inp_combined.csv"
print("inp_combined.csv SHA-256:", sha256_of(inp_combined))
print("Expected (INTEGRATION_NOTE.md): b3113d969c5f9bde5ae7b10b632578f34602d9f7679b19a72690a6bf7056bc6a")

# %%
df = load_combined(inp_combined)
for col in ["emission_nm", "T_growth_C", "time_min", "FWHM_nm",
            "QY_percent", "PL_peak_nm_core", "abs_nm", "diameter_nm"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Loaded inp_combined: {len(df)} rows × {len(df.columns)} cols")
print(f"source counts: {df['source'].value_counts().to_dict()}")
print(f"data_origin: {df['data_origin'].value_counts().to_dict()}")

# %% [markdown]
# ## A.1.0 Overlap subset prep
#
# Exclude L031 paper (Cossairt-only, no DD InP corpus match).

# %%
L031_DOI = "10.1021/acsnano.8b06692"
ovl = df[(df["overlap_flag"] == "1") & (df["doi"].fillna("") != L031_DOI)].copy()
dd_ovl = ovl[ovl["source"] == "dd"].copy()
cos_ovl = ovl[ovl["source"] == "cossairt"].copy()

print(f"Overlap rows after L031 exclude: {len(ovl)} (DD {len(dd_ovl)} + Cossairt {len(cos_ovl)})")
print(f"Expected: 52 (DD 29 + Cossairt 23)")

# paper_id (L*) <-> doi map from DD overlap rows
paper_to_doi = (dd_ovl.dropna(subset=["doi"])
                      .groupby("paper_id_dd")["doi"].first().to_dict())
print(f"\n9 overlap papers (paper_id ↔ doi):")
for pid, doi in sorted(paper_to_doi.items()):
    print(f"  {pid:5s} ↔ {doi}")

# %% [markdown]
# ## A.1.1 Paper-median aggregation
#
# Per overlap paper, compute the median of each numeric column for the
# DD subset and the Cossairt subset. `"None"` values were already coerced to
# NaN by `load_combined` (via `na_values=["None", ...]`).

# %%
def paper_median(rows: pd.DataFrame, cols):
    return {c: rows[c].dropna().median() if rows[c].notna().any() else np.nan
            for c in cols}

pairs = []
for pid in sorted(paper_to_doi):
    doi = paper_to_doi[pid]
    dd_p = dd_ovl[dd_ovl["paper_id_dd"] == pid]
    cos_p = cos_ovl[cos_ovl["doi"] == doi]
    cols = ["emission_nm", "T_growth_C", "time_min"]
    dd_med = paper_median(dd_p, cols)
    cos_med = paper_median(cos_p, cols)
    pairs.append(dict(
        paper=pid, doi=doi,
        dd_n=len(dd_p), cos_n=len(cos_p),
        dd_em=dd_med["emission_nm"], cos_em=cos_med["emission_nm"],
        dd_T=dd_med["T_growth_C"], cos_T=cos_med["T_growth_C"],
        dd_t=dd_med["time_min"], cos_t=cos_med["time_min"],
    ))
pair_df = pd.DataFrame(pairs)
pair_df["em_diff"] = pair_df["dd_em"] - pair_df["cos_em"]
pair_df["T_diff"] = pair_df["dd_T"] - pair_df["cos_T"]
print(pair_df[["paper", "dd_n", "cos_n",
               "dd_em", "cos_em", "em_diff",
               "dd_T", "cos_T", "T_diff"]].to_string(index=False))

# %% [markdown]
# ### A.1.1 sub-finding
#
# - **Emission paper-median paired**: 3 papers (L009, L023, L038) have both-source
#   numeric emission. Other 6 overlap papers have Cossairt emission_nm = NaN
#   (i.e. paper present in Cossairt corpus but emission not reported in those
#   rows). This makes the emission paired analysis N = 3 — descriptive only.
# - **T paper-median paired**: 9 papers, 7 perfectly match, 2 mismatch
#   (L002 +40 °C from DD STAGE-FIX, L009 -90 °C from sample misalignment
#   confirmed in analysis design notes).

# %% [markdown]
# ## A.1.2 Bland–Altman + scatter (paper-level)
#
# Bias = mean(DD − Cossairt). Limits of Agreement (LoA) = bias ± 1.96 × SD
# (Bland–Altman 1986). Reported with the caveat that N = 3 for emission and
# N = 9 for T_growth.

# %%
def bland_altman_stats(diffs: np.ndarray):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return dict(n=len(diffs), bias=float(diffs.mean()) if len(diffs) else np.nan,
                    sd=np.nan, loa_low=np.nan, loa_high=np.nan)
    bias = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    return dict(n=len(diffs), bias=bias, sd=sd,
                loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd)

em_paired = pair_df.dropna(subset=["dd_em", "cos_em"]).copy()
T_paired = pair_df.dropna(subset=["dd_T", "cos_T"]).copy()

em_ba = bland_altman_stats(em_paired["em_diff"].values)
T_ba = bland_altman_stats(T_paired["T_diff"].values)
print("Paper-level Bland–Altman (emission_nm):", em_ba)
print("Paper-level Bland–Altman (T_growth_C):", T_ba)

# %% [markdown]
# ## A.1.3 Paper-level bootstrap CI (n = 1000)
#
# Resample papers with replacement (paper as the unit of independence). Returns
# 95 % percentile CI for the mean difference.

# %%
def paper_bootstrap_ci(diffs: np.ndarray, n_iter=1000, ci=95, rng=RNG):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2:
        return dict(n=n, mean=float(diffs.mean()) if n else np.nan,
                    ci_low=np.nan, ci_high=np.nan, n_iter=n_iter)
    means = np.empty(n_iter)
    idx_pool = np.arange(n)
    for i in range(n_iter):
        idx = rng.choice(idx_pool, size=n, replace=True)
        means[i] = diffs[idx].mean()
    lo = float(np.percentile(means, (100 - ci) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci) / 2))
    return dict(n=n, mean=float(diffs.mean()),
                ci_low=lo, ci_high=hi, n_iter=n_iter)

em_boot = paper_bootstrap_ci(em_paired["em_diff"].values)
T_boot = paper_bootstrap_ci(T_paired["T_diff"].values)
print(f"emission_nm paper-level bootstrap (n=1000): {em_boot}")
print(f"T_growth_C paper-level bootstrap (n=1000): {T_boot}")

# %% [markdown]
# ## A.1.4 Sensitivity — sample-level paired (L023, L038 emission-sort)
#
# Sample-level paired comparison sorts both DD and Cossairt entries within a
# paper by emission_nm and pairs index-wise. Restricted to papers where both
# sources have emission numeric *and* the sample alignment is interpretable
# (L023: 5↔5; L038: 1 row each with non-NaN emission). L009 is excluded
# because sample mismatch was confirmed in analysis design notes (DD
# 150/blank vs Cossairt 240/60). L009 paper-level pair stays in the primary
# A.1.2/A.1.3 analysis as a transparently-flagged outlier.

# %%
def emission_sorted_pairs(dd_rows, cos_rows):
    dd_s = dd_rows.dropna(subset=["emission_nm"]).sort_values("emission_nm")
    cos_s = cos_rows.dropna(subset=["emission_nm"]).sort_values("emission_nm")
    n = min(len(dd_s), len(cos_s))
    if n == 0:
        return []
    dd_em = dd_s["emission_nm"].values[:n]
    cos_em = cos_s["emission_nm"].values[:n]
    return list(zip(dd_em, cos_em))

sample_pairs = []
for pid in ["L023", "L038"]:
    doi = paper_to_doi[pid]
    dd_p = dd_ovl[dd_ovl["paper_id_dd"] == pid]
    cos_p = cos_ovl[cos_ovl["doi"] == doi]
    for dd_em, cos_em in emission_sorted_pairs(dd_p, cos_p):
        sample_pairs.append(dict(paper=pid, dd_em=dd_em, cos_em=cos_em))
sample_pair_df = pd.DataFrame(sample_pairs)
sample_pair_df["em_diff"] = sample_pair_df["dd_em"] - sample_pair_df["cos_em"]
print("Sample-level paired (L023 + L038, emission-sorted):")
print(sample_pair_df.to_string(index=False))

sample_ba = bland_altman_stats(sample_pair_df["em_diff"].values)
sample_boot = paper_bootstrap_ci(sample_pair_df["em_diff"].values)  # paper-aggregated bootstrap if extended; here row-level fine for sensitivity
print(f"\nSample-level emission Bland–Altman: {sample_ba}")
print(f"Sample-level emission bootstrap (row-level n=1000, sensitivity only): {sample_boot}")

# %% [markdown]
# ## A.1.5 Figure 1 — Bland–Altman + scatter (paper-level + sample-aligned sensitivity)
#
# Hero finding (user-confirmed 2026-05-21): **L023 is the most stringent
# sample-aligned subset (5 InP recipes, single publication, both sources
# reporting emission_nm) and yields bit-identical agreement (Δ = 0 nm
# across all 5).** This is the strongest single-paper evidence available
# within the overlap subset and is highlighted in red across all panels.
# L038 is the only sample-level outlier and is shown as orange ✕ with the
# sample-misalignment caveat. L002 (DD STAGE-FIX) and L009 (sample mismatch)
# are annotated as outliers on the paper-level panels.

# %%
COLOR_BASE   = "#1f77b4"   # default blue — paper-median, non-L023
COLOR_L023   = "#d62728"   # red — hero L023
COLOR_L038   = "#ff7f0e"   # orange — L038 sample-misaligned caveat
COLOR_T_BASE = "#2ca02c"   # green — T_growth paper-median (non-outlier)
COLOR_OUTL_T = "#9467bd"   # purple — T_growth outliers (L002, L009)
OUTLIER_PAPERS_T = {"L002", "L009"}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# ------------------------- Panel A: emission scatter -------------------------
ax = axes[0, 0]
# paper-median, non-L023
non_l023 = em_paired[em_paired["paper"] != "L023"]
l023_pm = em_paired[em_paired["paper"] == "L023"]
ax.scatter(non_l023["cos_em"], non_l023["dd_em"], s=50, c=COLOR_BASE,
           edgecolors="black", linewidths=0.6, label="other paper medians", zorder=3)
# L023 paper-median (hero, large red)
ax.scatter(l023_pm["cos_em"], l023_pm["dd_em"], s=90, c=COLOR_L023,
           marker="o", edgecolors="black", linewidths=1.2,
           label="L023 median", zorder=5)
# Sample-aligned (L023 individual recipes)
l023_samples = sample_pair_df[sample_pair_df["paper"] == "L023"]
l038_samples = sample_pair_df[sample_pair_df["paper"] == "L038"]
ax.scatter(l023_samples["cos_em"], l023_samples["dd_em"], s=36, c=COLOR_L023,
           marker="o", alpha=0.55, edgecolors="black", linewidths=0.4,
           label="L023 sample-aligned rows", zorder=4)
# L038 caveat
ax.scatter(l038_samples["cos_em"], l038_samples["dd_em"], s=52, c=COLOR_L038,
           marker="x", linewidths=2.2,
           label="L038 unmatched", zorder=4)
# y = x reference
allx = pd.concat([em_paired["cos_em"], sample_pair_df["cos_em"]]).dropna()
ally = pd.concat([em_paired["dd_em"], sample_pair_df["dd_em"]]).dropna()
lo = min(allx.min(), ally.min()) - 25
hi = max(allx.max(), ally.max()) + 25
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="y = x", zorder=1)
ax.set_xlabel("reference emission (nm)", fontsize=9)
ax.set_ylabel("curated emission (nm)", fontsize=9)
ax.set_title("A. Emission: paired", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.30), fontsize=6.6, framealpha=0.95, markerscale=0.4, labelspacing=0.85, handletextpad=0.5, borderpad=0.6, ncol=2)
ax.text(0.03, 0.03,
        f"N={em_ba['n']}, bias={em_ba['bias']:.1f} nm; 95% CI=({em_boot['ci_low']:.1f}, {em_boot['ci_high']:.1f})",
        transform=ax.transAxes, fontsize=6.8, va="bottom", ha="left",
        bbox=dict(facecolor="white", alpha=0.95, edgecolor="gray"))
_texts_A = [ax.text(r["cos_em"], r["dd_em"], r["paper"], fontsize=7.5) for _, r in em_paired.iterrows() if r["paper"] in {"L023","L038","L009","L002","L006"}]
adjust_text(_texts_A, ax=ax, x=list(em_paired["cos_em"]), y=list(em_paired["dd_em"]), arrowprops=dict(arrowstyle="-", color="gray", lw=0.6), expand=(2.0,2.4), force_text=(0.6,0.9), force_static=(0.4,0.7), min_arrow_len=5)

# ----------------------- Panel B: emission Bland-Altman ----------------------
ax = axes[0, 1]
em_paired["em_mean"] = (em_paired["dd_em"] + em_paired["cos_em"]) / 2
# non-L023
ax.scatter((non_l023["dd_em"] + non_l023["cos_em"]) / 2,
           non_l023["em_diff"], s=50, c=COLOR_BASE,
           edgecolors="black", linewidths=0.6, label="other paper medians", zorder=3)
# L023 paper-median (hero)
ax.scatter(l023_pm["em_mean"] if "em_mean" in l023_pm else
           (l023_pm["dd_em"] + l023_pm["cos_em"]) / 2,
           l023_pm["em_diff"], s=90, c=COLOR_L023,
           marker="o", edgecolors="black", linewidths=1.2,
           label="L023 median", zorder=5)
ax.axhline(em_ba["bias"], color="red", lw=1.2, label=f"bias = {em_ba['bias']:.1f} nm")
ax.axhline(em_ba["loa_low"], color="red", lw=0.6, ls="--",
           label=f"LoA = ({em_ba['loa_low']:.1f}, {em_ba['loa_high']:.1f})")
ax.axhline(em_ba["loa_high"], color="red", lw=0.6, ls="--")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Mean emission (nm)", fontsize=9)
ax.set_ylabel("curated − reference (nm)", fontsize=9)
ax.set_title("B. Emission Bland–Altman", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.30), fontsize=6.6, framealpha=0.95, markerscale=0.4, labelspacing=0.85, handletextpad=0.5, borderpad=0.6, ncol=2)
_em_mean_all = (em_paired["dd_em"] + em_paired["cos_em"]) / 2
_texts_B = [ax.text(m, d, p, fontsize=7.5) for m, d, p in zip(_em_mean_all, em_paired["em_diff"], em_paired["paper"]) if p in {"L023","L038","L009","L002","L006"}]
adjust_text(_texts_B, ax=ax, x=list(_em_mean_all), y=list(em_paired["em_diff"]), arrowprops=dict(arrowstyle="-", color="gray", lw=0.6), expand=(2.0,2.4), force_text=(0.6,0.9), force_static=(0.4,0.7), min_arrow_len=5)

# ------------------------ Panel C: T_growth scatter --------------------------
ax = axes[1, 0]
T_in = T_paired[~T_paired["paper"].isin(OUTLIER_PAPERS_T)]
T_out = T_paired[T_paired["paper"].isin(OUTLIER_PAPERS_T)]
T_l023 = T_paired[T_paired["paper"] == "L023"]
ax.scatter(T_in["cos_T"], T_in["dd_T"], s=50, c=COLOR_T_BASE,
           edgecolors="black", linewidths=0.6, label="paper medians", zorder=3)
ax.scatter(T_out["cos_T"], T_out["dd_T"], s=66, c=COLOR_OUTL_T,
           marker="D", edgecolors="black", linewidths=0.8,
           label="L002 / L009 outlier", zorder=4)
ax.scatter(T_l023["cos_T"], T_l023["dd_T"], s=82, c=COLOR_L023,
           marker="o", edgecolors="black", linewidths=1.2,
           label="L023 median", zorder=5)
lo = T_paired["cos_T"].min() - 30
hi = T_paired["cos_T"].max() + 30
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="y = x", zorder=1)
ax.set_xlabel("reference growth temperature (°C)", fontsize=9)
ax.set_ylabel("curated growth temperature (°C)", fontsize=9)
ax.set_title("C. Growth temperature: paired", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.30), fontsize=6.6, framealpha=0.95, markerscale=0.4, labelspacing=0.85, handletextpad=0.5, borderpad=0.6, ncol=2)
ax.text(0.97, 0.03,
        f"N={T_ba['n']}, bias={T_ba['bias']:.1f} °C; 95% CI=({T_boot['ci_low']:.1f}, {T_boot['ci_high']:.1f})",
        transform=ax.transAxes, fontsize=6.8, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.95, edgecolor="gray"))
# L009/L038 labels placed manually (pixel offset only; data points unchanged); rest via adjustText
_manual_C = {"L009": (12, -2), "L038": (2, 11), "L006": (8, 8)}
_texts_C = [ax.text(r["cos_T"], r["dd_T"], r["paper"], fontsize=7.5)
            for _, r in T_paired.iterrows() if r["paper"] not in _manual_C and r["paper"] in {"L023","L038","L009","L002","L006"}]
_pts_C = T_paired[~T_paired["paper"].isin(_manual_C)]
adjust_text(_texts_C, ax=ax, x=list(_pts_C["cos_T"]), y=list(_pts_C["dd_T"]),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
            expand=(2.2, 2.8), force_text=(0.8, 1.1), force_static=(0.5, 0.8), min_arrow_len=5)
for _, r in T_paired[T_paired["paper"].isin(_manual_C)].iterrows():
    dx, dy = _manual_C[r["paper"]]
    ax.annotate(r["paper"], (r["cos_T"], r["dd_T"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5, ha="left", va="center")

# ------------------- Panel D: T_growth Bland-Altman --------------------------
ax = axes[1, 1]
T_paired["T_mean"] = (T_paired["dd_T"] + T_paired["cos_T"]) / 2
T_in_ba = T_paired[~T_paired["paper"].isin(OUTLIER_PAPERS_T)]
T_out_ba = T_paired[T_paired["paper"].isin(OUTLIER_PAPERS_T)]
T_l023_ba = T_paired[T_paired["paper"] == "L023"]
ax.scatter(T_in_ba["T_mean"], T_in_ba["T_diff"], s=50, c=COLOR_T_BASE,
           edgecolors="black", linewidths=0.6, label="paper medians", zorder=3)
ax.scatter(T_out_ba["T_mean"], T_out_ba["T_diff"], s=66, c=COLOR_OUTL_T,
           marker="D", edgecolors="black", linewidths=0.8,
           label="L002 / L009 outlier", zorder=4)
ax.scatter(T_l023_ba["T_mean"], T_l023_ba["T_diff"], s=82, c=COLOR_L023,
           marker="o", edgecolors="black", linewidths=1.2,
           label="L023 median", zorder=5)
ax.axhline(T_ba["bias"], color="red", lw=1.2, label=f"bias = {T_ba['bias']:.1f} °C")
ax.axhline(T_ba["loa_low"], color="red", lw=0.6, ls="--",
           label=f"LoA = ({T_ba['loa_low']:.1f}, {T_ba['loa_high']:.1f})")
ax.axhline(T_ba["loa_high"], color="red", lw=0.6, ls="--")
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_xlabel("Mean growth temperature (°C)", fontsize=9)
ax.set_ylabel("curated − reference (°C)", fontsize=9)
ax.set_title("D. Growth temperature Bland–Altman", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.30), fontsize=6.6, framealpha=0.95, markerscale=0.4, labelspacing=0.85, handletextpad=0.5, borderpad=0.6, ncol=2)
_texts_D = [ax.text(r["T_mean"], r["T_diff"], r["paper"], fontsize=7.5) for _, r in T_paired.iterrows() if r["paper"] in {"L023","L038","L009","L002","L006"}]
adjust_text(_texts_D, ax=ax, x=list(T_paired["T_mean"]), y=list(T_paired["T_diff"]), arrowprops=dict(arrowstyle="-", color="gray", lw=0.6), expand=(2.2,2.8), force_text=(0.8,1.1), force_static=(0.5,0.8), min_arrow_len=5)

for _ax in axes.flat:
    _ax.margins(0.12)
fig.tight_layout()

fig_path_pdf = PROJECT_ROOT / "figures" / "figure_1_paired_comparison.pdf"
fig_path_svg = PROJECT_ROOT / "figures" / "figure_1_paired_comparison.svg"
fig_path_pdf.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path_pdf, bbox_inches="tight")
fig.savefig(fig_path_svg, bbox_inches="tight")
print(f"Saved: {fig_path_pdf}")
print(f"Saved: {fig_path_svg}")

# %% [markdown]
# ## A.1.6 Result tables (machine-readable)

# %%
results = dict(
    paper_pairs=[{k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                  for k, v in p.items()} for p in pair_df.to_dict("records")],
    emission_paper_level=dict(
        N=em_ba["n"], bias_nm=em_ba["bias"], sd_nm=em_ba["sd"],
        loa_low_nm=em_ba["loa_low"], loa_high_nm=em_ba["loa_high"],
        bootstrap_ci=em_boot,
    ),
    T_growth_paper_level=dict(
        N=T_ba["n"], bias_C=T_ba["bias"], sd_C=T_ba["sd"],
        loa_low_C=T_ba["loa_low"], loa_high_C=T_ba["loa_high"],
        bootstrap_ci=T_boot,
    ),
    emission_sample_level_sensitivity=dict(
        N=sample_ba["n"], papers=["L023", "L038"], excluded=["L009 sample mismatch"],
        bias_nm=sample_ba["bias"], sd_nm=sample_ba["sd"],
        loa_low_nm=sample_ba["loa_low"], loa_high_nm=sample_ba["loa_high"],
        bootstrap_ci_sensitivity=sample_boot,
        pairs=sample_pair_df.to_dict("records"),
    ),
    metadata=dict(
        random_seed=42,
        bootstrap_iterations=1000,
        L031_excluded=True,
        L031_doi=L031_DOI,
        L002_note="DD STAGE-FIX (T 260→300, t 40→10) — caveat",
        L009_note="Sample mismatch (DD 150/blank vs Cos 240/60) — paper-level outlier",
    ),
)
out_json = PROJECT_ROOT / "analysis" / "part_a1_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved: {out_json}")
print(f"Output SHA-256: {sha256_of(out_json)}")

# %% [markdown]
# ## A.1.7 Findings narrative — see analysis design notes
#
# The narrative interpretation, limitations, caveats, and decisions for the
# manuscript are recorded in a separate markdown file alongside this
# notebook. Refer to that file for the human-readable conclusion.

# %% [markdown]
# # Part A.2 — Distribution-level comparison (independent samples)
#
# **Question (the analysis design plan §2, A.2)**: At the *marginal-distribution*
# level — independent of paper-level pairing — do the DD-curated InP subset
# (132 records) and the Cossairt 2022 published-literature subset (219
# records, `nayon` excluded) sample the same underlying distribution of
# growth temperature, emission wavelength, and growth time?
#
# **Tests**: Kolmogorov–Smirnov two-sample (`scipy.stats.ks_2samp`),
# Wasserstein-1 distance (`scipy.stats.wasserstein_distance`), Mann–Whitney U
# (`scipy.stats.mannwhitneyu`). KS detects differences in distributional
# shape, Wasserstein-1 quantifies effect size (mean-of-absolute-CDF-difference,
# unit = column unit), Mann–Whitney detects location shift (median test).
#
# **Multiple-testing correction**: Benjamini–Hochberg (BH, FDR) across
# 3 columns × 2 inferential tests (KS + Mann–Whitney) = 6 tests. Wasserstein
# is a distance, not a test, so not corrected.
#
# **time_min handled here** per Part A.1 Option B decision: A.1 excluded
# time_min from paired comparison because of sample-selection differences;
# A.2 includes time_min because the marginal distribution comparison is
# informative *exactly because* Cossairt's multiple-condition extraction is
# preserved in the long-tail.

# %%
from scipy import stats
try:
    from scipy.stats import false_discovery_control
except ImportError:
    false_discovery_control = None  # noqa: F811

# DD InP — all 132 rows
dd_all = df[df["source"] == "dd"].copy()
# Cossairt published-only — 219 rows (data_origin filter excludes nayon)
cos_pub = df[(df["source"] == "cossairt") & (df["data_origin"] == "published_literature")].copy()
print(f"DD InP subset:           {len(dd_all)} rows")
print(f"Cossairt published-only: {len(cos_pub)} rows  (raw deposit is 219; the nayon author-synthesis row was removed from the public deposit in A2)")

# %%
A2_TARGETS = ["emission_nm", "T_growth_C", "time_min"]

a2_records = []
for col in A2_TARGETS:
    dd_vals = dd_all[col].dropna().values.astype(float)
    cos_vals = cos_pub[col].dropna().values.astype(float)
    ks_stat, ks_p = stats.ks_2samp(dd_vals, cos_vals)
    w_dist = float(stats.wasserstein_distance(dd_vals, cos_vals))
    mw_stat, mw_p = stats.mannwhitneyu(dd_vals, cos_vals, alternative="two-sided")
    a2_records.append(dict(
        column=col,
        n_dd=int(len(dd_vals)), n_cos=int(len(cos_vals)),
        dd_median=float(np.median(dd_vals)),
        cos_median=float(np.median(cos_vals)),
        dd_mean=float(dd_vals.mean()),
        cos_mean=float(cos_vals.mean()),
        ks_stat=float(ks_stat), ks_p=float(ks_p),
        wasserstein=w_dist,
        mw_stat=float(mw_stat), mw_p=float(mw_p),
    ))

a2_df = pd.DataFrame(a2_records)
print(a2_df.to_string(index=False))

# %% [markdown]
# ## A.2.2 Multiple-testing correction (BH across 6 tests)

# %%
all_pvals = []
for r in a2_records:
    all_pvals.extend([r["ks_p"], r["mw_p"]])

if false_discovery_control is not None:
    adjusted = false_discovery_control(np.array(all_pvals), method="bh")
else:
    # Manual BH fallback (scipy < 1.12 path)
    p = np.asarray(all_pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    # enforce monotonicity from the top
    p_sorted = p[order]
    bh_sorted = (p_sorted * n) / np.arange(1, n + 1)
    bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(bh_sorted, 1.0)

for i, r in enumerate(a2_records):
    r["ks_p_bh"] = float(adjusted[2 * i])
    r["mw_p_bh"] = float(adjusted[2 * i + 1])

a2_df = pd.DataFrame(a2_records)
print("\nWith BH-corrected p-values:")
print(a2_df[["column", "n_dd", "n_cos", "ks_stat", "ks_p", "ks_p_bh",
             "wasserstein", "mw_p", "mw_p_bh"]].to_string(index=False))

# %% [markdown]
# ## A.2.3 Interpretation framework
#
# - **KS p (BH) < 0.05**: reject null that DD and Cossairt sample from the
#   same distribution for that column.
# - **Wasserstein-1**: effect-size unit = column unit. Small relative to
#   the column's natural scale → distributional convergence.
# - **time_min long-tail expectation**: if
#   Cossairt time_min distribution is heavier-tailed than DD (which is
#   expected because Cossairt extracts multiple conditions per paper
#   while DD selects one representative), the manuscript framing is
#   "discovery, not disagreement":
#   > "The DD-curated dataset preserves one representative synthesis
#   > condition per paper, while Cossairt extracted multiple conditions
#   > when reported. This methodological difference accounts for the
#   > heavier long-tail in the Cossairt distribution of growth time,
#   > without indicating disagreement in the underlying chemistry."

# %% [markdown]
# ## A.2.4 Figure 2 — overlaid histograms + violins (3 columns × 2 rows)

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

PRETTY = {
    "emission_nm": ("Emission (nm)", "Emission"),
    "T_growth_C": ("Growth temperature (°C)", "Growth temperature"),
    "time_min": ("Growth time (min, log₁₀ scale)", "Growth time"),
}

for i, col in enumerate(A2_TARGETS):
    dd_vals = dd_all[col].dropna().values.astype(float)
    cos_vals = cos_pub[col].dropna().values.astype(float)
    rec = a2_records[i]

    # Top row: overlaid histograms
    ax = axes[0, i]
    if col == "time_min":
        # log-scale x-axis for time_min (Cossairt long-tail)
        bins = np.logspace(np.log10(min(dd_vals.min(), cos_vals.min())),
                           np.log10(max(dd_vals.max(), cos_vals.max())), 25)
        ax.hist(dd_vals, bins=bins, alpha=0.55, color="#1f77b4",
                label=f"curated (n={len(dd_vals)})", density=True)
        ax.hist(cos_vals, bins=bins, alpha=0.55, color="#ff7f0e",
                label=f"reference (n={len(cos_vals)})", density=True)
        ax.set_xscale("log")
    else:
        bins = 20
        ax.hist(dd_vals, bins=bins, alpha=0.55, color="#1f77b4",
                label=f"curated (n={len(dd_vals)})", density=True)
        ax.hist(cos_vals, bins=bins, alpha=0.55, color="#ff7f0e",
                label=f"reference (n={len(cos_vals)})", density=True)
    ax.set_xlabel(PRETTY[col][0])
    ax.set_ylabel("Density")
    ax.set_title(f"{chr(ord('A')+i)}. {PRETTY[col][1]} (density)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    sig_marker = " *" if rec["ks_p_bh"] < 0.05 else ""
    _w_unit = {"emission_nm": "nm", "T_growth_C": "°C", "time_min": "min"}.get(col, "")
    ax.text(0.02, 0.98,
            f"KS p (BH) = {rec['ks_p_bh']:.3f}{sig_marker}\n"
            f"W₁ = {rec['wasserstein']:.2f} {_w_unit}\n"
            f"MW p (BH) = {rec['mw_p_bh']:.3f}",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    # Bottom row: violin
    ax = axes[1, i]
    parts = ax.violinplot([dd_vals, cos_vals], showmeans=True,
                          showmedians=True, widths=0.8)
    for j, body in enumerate(parts["bodies"]):
        body.set_facecolor("#1f77b4" if j == 0 else "#ff7f0e")
        body.set_alpha(0.55)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"curated\n(n={len(dd_vals)})", f"reference\n(n={len(cos_vals)})"])
    ax.set_ylabel(PRETTY[col][0])
    ax.set_title(f"{chr(ord('D')+i)}. {PRETTY[col][1]} (violin)", fontsize=11, fontweight="bold")
    if col == "time_min":
        ax.set_yscale("log")
    # Show medians as numerical annotations
    ax.axhline(rec["dd_median"], xmin=0.05, xmax=0.45, color="#1f77b4",
               lw=0.6, ls="--")
    ax.axhline(rec["cos_median"], xmin=0.55, xmax=0.95, color="#ff7f0e",
               lw=0.6, ls="--")
    ax.text(0.02, 0.02,
            f"curated median = {rec['dd_median']:.1f}\nreference median = {rec['cos_median']:.1f}",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

fig.tight_layout()

fig2_pdf = PROJECT_ROOT / "figures" / "figure_2_distribution_comparison.pdf"
fig2_svg = PROJECT_ROOT / "figures" / "figure_2_distribution_comparison.svg"
fig.savefig(fig2_pdf, bbox_inches="tight")
fig.savefig(fig2_svg, bbox_inches="tight")
print(f"Saved: {fig2_pdf}")
print(f"Saved: {fig2_svg}")

# %% [markdown]
# ## A.2.5 Persist A.2 results as JSON

# %%
results_a2 = dict(
    targets=A2_TARGETS,
    n_dd=len(dd_all),
    n_cos=len(cos_pub),
    cossairt_filter="data_origin == 'published_literature' (nayon excluded)",
    bh_correction="Benjamini–Hochberg across 6 tests (3 columns × 2: KS + MW)",
    per_column=a2_records,
)
out2 = PROJECT_ROOT / "analysis" / "part_a2_results.json"
with open(out2, "w", encoding="utf-8") as f:
    json.dump(results_a2, f, indent=2, default=str)
print(f"Saved: {out2}")
print(f"Output SHA-256: {sha256_of(out2)}")

# %% [markdown]
# ## A.2.6 Findings narrative → analysis design notes

# %% [markdown]
# # Part A.3 — Reproduction of Cossairt's predictive pipeline
#
# **Question**: Does Cossairt et al.'s reported emission-MAE (11.46 nm in the
# main-text abstract, on the *imputed* full dataset) reproduce in our hands
# on the *non-imputed* published-only subset (219 rows / 85-86 with emission)?
# And does the same pipeline applied to DD produce a comparable feature
# importance ranking?
#
# **A.3.0 audit — Cossairt-Lab/Indium-Phosphide GitHub repo** (verified
# against the deposited reference copy by SHA-256). The relevant
# emission-training notebook is `notebook_85/hao/3.1. model hao emission.ipynb`.
# Key hyperparameters extracted verbatim from that notebook:
#
#   - **Best emission model (single-output)**: `ExtraTreesRegressor`
#     with `n_estimators=3, max_features=13, random_state=51`.
#   - **Train/test split**: `test_size=0.15, random_state=45, shuffle=True`.
#   - **Alternate DecisionTree**: `max_depth=5, max_features=9, random_state=60`.
#   - Cossairt's grid-search optimized on a single train/test split MAE
#     (no proper CV); we will reproduce the official hyperparameters **and**
#     add k-fold CV + bootstrap CI to characterize variance honestly.
#   - Cossairt's training data is *augmented + scaled*
#     (`dataset_scaled_emission.csv`). We use the raw 219-row published-only
#     subset for honest comparison (no augmentation; missing numerics use
#     Cossairt's 0.0-fill fallback below, not statistical imputation).
#
# Cossairt requirements.txt pin: `scikit_learn==0.24.1` (2021). We run on
# `scikit-learn==1.8.0` — same algorithm but minor numerical differences
# possible; the hyperparameters are stable across versions.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (GroupKFold, KFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

from io_helpers import load_cossairt_raw

COS_HP_EMISSION = dict(n_estimators=3, max_features=13, random_state=51)
COS_SPLIT = dict(test_size=0.15, random_state=45, shuffle=True)

# Reload Cossairt raw to access the full Cossairt-side feature set
# (inp_combined.csv carries only a subset of Cossairt columns).
cos_raw = load_cossairt_raw()  # 220 rows, "None" -> NaN
print(f"Cossairt raw: {len(cos_raw)} rows, {len(cos_raw.columns)} cols")

# Non-0.0-filled snapshot of the A.4 columns (published-only), captured
# BEFORE the COS_NUMERIC 0.0-fill below, for the no-imputation A.4
# cross-prediction so the Cossairt side is symmetric with the DD side.
cos_raw_a4 = cos_raw[cos_raw["doi"].astype(str) != "nayon"][
    ["doi", "temp_c", "time_min", "emission_nm"]].copy()

# Numeric + categorical Cossairt features (mirror Cossairt SI Table S1)
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

# Coerce types
for c in COS_NUMERIC:
    cos_raw[c] = pd.to_numeric(cos_raw[c], errors="coerce").fillna(0.0)
for c in COS_CATEGORICAL:
    cos_raw[c] = cos_raw[c].fillna("None").astype(str)

# Published-only + emission-present
cos_em = cos_raw[(cos_raw["doi"].astype(str) != "nayon") &
                 cos_raw["emission_nm"].notna()].copy()
y_cos = pd.to_numeric(cos_em["emission_nm"], errors="coerce").values
X_cos = cos_em[COS_NUMERIC + COS_CATEGORICAL]
print(f"Cossairt published emission training set: {X_cos.shape} (target n={len(y_cos)})")

# %% [markdown]
# ## A.3.1 Cossairt ExtraTrees reproduction — official hyperparameters

# %%
cos_preprocessor = ColumnTransformer([
    ("num", StandardScaler(), COS_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), COS_CATEGORICAL),
])
cos_pipe = Pipeline([
    ("pre", cos_preprocessor),
    ("et", ExtraTreesRegressor(**COS_HP_EMISSION)),
])

X_train, X_test, y_train, y_test = train_test_split(X_cos, y_cos, **COS_SPLIT)
cos_pipe.fit(X_train, y_train)
y_pred = cos_pipe.predict(X_test)
cos_split_mae = float(mean_absolute_error(y_test, y_pred))
cos_split_r2 = float(r2_score(y_test, y_pred))
print(f"Cossairt 85/15 split (official hyperparameters):")
print(f"  test MAE = {cos_split_mae:.2f} nm  (Cossairt reported best 11.46 nm on imputed dataset)")
print(f"  test R²  = {cos_split_r2:.3f}")

# Bootstrap CI on test MAE (row-level since hold-out test set has no paper grouping)
rng = np.random.default_rng(42)
n_iter = 1000
test_mae_boot = np.empty(n_iter)
for i in range(n_iter):
    idx = rng.integers(0, len(y_test), len(y_test))
    test_mae_boot[i] = mean_absolute_error(y_test[idx], y_pred[idx])
cos_mae_ci = (float(np.percentile(test_mae_boot, 2.5)),
              float(np.percentile(test_mae_boot, 97.5)))
print(f"  bootstrap 95% CI on test MAE: ({cos_mae_ci[0]:.2f}, {cos_mae_ci[1]:.2f}) nm")

# k-fold CV (more robust)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cos_cv = -cross_val_score(cos_pipe, X_cos, y_cos, cv=kf,
                          scoring="neg_mean_absolute_error", n_jobs=1)
print(f"  5-fold CV MAE: {cos_cv.mean():.2f} ± {cos_cv.std():.2f} nm")
print(f"  5-fold CV MAE per fold: {[f'{v:.2f}' for v in cos_cv]}")

# %% [markdown]
# ## A.3.2 DD same pipeline (paper-level GroupKFold to prevent leakage)

# %%
dd_em = dd_all.dropna(subset=["emission_nm"]).copy()
DD_NUMERIC = ["T_growth_C", "time_min"]
DD_CATEGORICAL = ["route", "system_tag"]
for c in DD_NUMERIC:
    dd_em[c] = pd.to_numeric(dd_em[c], errors="coerce").fillna(0.0)
for c in DD_CATEGORICAL:
    dd_em[c] = dd_em[c].fillna("Unknown").astype(str)

y_dd = pd.to_numeric(dd_em["emission_nm"], errors="coerce").values
X_dd = dd_em[DD_NUMERIC + DD_CATEGORICAL]
groups_dd = dd_em["paper_id_dd"].values
print(f"DD emission training set: {X_dd.shape} (target n={len(y_dd)}, papers={len(set(groups_dd))})")

dd_preprocessor = ColumnTransformer([
    ("num", StandardScaler(), DD_NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), DD_CATEGORICAL),
])
dd_pipe = Pipeline([
    ("pre", dd_preprocessor),
    ("et", ExtraTreesRegressor(**COS_HP_EMISSION)),
])

# Paper-level CV (prevents data leakage across rows of same paper)
gkf = GroupKFold(n_splits=5)
dd_cv = -cross_val_score(dd_pipe, X_dd, y_dd, groups=groups_dd, cv=gkf,
                         scoring="neg_mean_absolute_error", n_jobs=1)
print(f"DD paper-level GroupKFold MAE: {dd_cv.mean():.2f} ± {dd_cv.std():.2f} nm")
print(f"DD paper-level MAE per fold: {[f'{v:.2f}' for v in dd_cv]}")

# Sanity: also compute non-grouped k-fold for direct comparison to Cossairt's setting
dd_kfold_cv = -cross_val_score(dd_pipe, X_dd, y_dd, cv=KFold(5, shuffle=True, random_state=42),
                               scoring="neg_mean_absolute_error", n_jobs=1)
print(f"DD non-grouped 5-fold (leakage-prone, for direct vs Cossairt only): "
      f"{dd_kfold_cv.mean():.2f} ± {dd_kfold_cv.std():.2f} nm")

# %% [markdown]
# ## A.3.3 Feature importance — Cossairt vs DD (Spearman ρ on shared features)

# %%
# Fit both pipelines on full data for importance extraction
cos_pipe.fit(X_cos, y_cos)
dd_pipe.fit(X_dd, y_dd)

cos_feat_names = cos_pipe.named_steps["pre"].get_feature_names_out()
dd_feat_names = dd_pipe.named_steps["pre"].get_feature_names_out()
cos_imp = pd.DataFrame({"feature": cos_feat_names,
                        "importance": cos_pipe.named_steps["et"].feature_importances_})
cos_imp = cos_imp.sort_values("importance", ascending=False).reset_index(drop=True)
dd_imp = pd.DataFrame({"feature": dd_feat_names,
                       "importance": dd_pipe.named_steps["et"].feature_importances_})
dd_imp = dd_imp.sort_values("importance", ascending=False).reset_index(drop=True)

print(f"Cossairt feature importance — top 10:")
print(cos_imp.head(10).to_string(index=False))
print(f"\nDD feature importance — top 10:")
print(dd_imp.head(10).to_string(index=False))

# Spearman ρ on the two shared numeric features (temp_c <-> T_growth_C, time_min <-> time_min)
# Build a small table of (name, cos_rank, dd_rank)
shared_pairs = [("num__temp_c", "num__T_growth_C"), ("num__time_min", "num__time_min")]
shared_data = []
for cos_name, dd_name in shared_pairs:
    cos_rank = cos_imp.index[cos_imp["feature"] == cos_name].tolist()
    dd_rank = dd_imp.index[dd_imp["feature"] == dd_name].tolist()
    cos_imp_val = float(cos_imp.loc[cos_imp["feature"] == cos_name, "importance"].iloc[0]) if cos_rank else np.nan
    dd_imp_val = float(dd_imp.loc[dd_imp["feature"] == dd_name, "importance"].iloc[0]) if dd_rank else np.nan
    shared_data.append(dict(
        cossairt_feature=cos_name, dd_feature=dd_name,
        cossairt_rank=(cos_rank[0] + 1) if cos_rank else None,
        dd_rank=(dd_rank[0] + 1) if dd_rank else None,
        cossairt_importance=cos_imp_val, dd_importance=dd_imp_val,
    ))
shared_df = pd.DataFrame(shared_data)
print(f"\nShared-feature rank comparison (2 features):")
print(shared_df.to_string(index=False))

# Spearman ρ across all features by name normalization (drop num__/cat__ prefixes)
def norm(name):
    name = name.replace("num__", "").replace("cat__", "")
    # NOTE: only the num__/cat__ family prefix is removed above; the per-column
    # OneHotEncoder value suffix (e.g. "in_source_indium chloride") is NOT
    # stripped, so only bare numeric features (e.g. time_min) can intersect.
    return name

cos_norm = cos_imp.assign(feat=cos_imp["feature"].map(norm))
dd_norm = dd_imp.assign(feat=dd_imp["feature"].map(norm))

# Spearman ρ on intersected feature names
joined = pd.merge(cos_norm[["feat", "importance"]].rename(columns={"importance": "cos_imp"}),
                  dd_norm[["feat", "importance"]].rename(columns={"importance": "dd_imp"}),
                  on="feat", how="inner")
print(f"\nIntersected features: {len(joined)} ({joined['feat'].tolist()})")
if len(joined) >= 3:
    rho, rho_p = stats.spearmanr(joined["cos_imp"], joined["dd_imp"])
    print(f"Spearman ρ on intersected features: ρ = {rho:.3f}, p = {rho_p:.3f}")
else:
    rho, rho_p = float("nan"), float("nan")
    print("Too few intersected features for Spearman ρ.")

# %% [markdown]
# ## A.3.4 PCA — Cossairt vs DD (separate spaces, side-by-side)

# %%
X_cos_pre = cos_pipe.named_steps["pre"].transform(X_cos)
X_dd_pre = dd_pipe.named_steps["pre"].transform(X_dd)

# Sparse output from OneHotEncoder — convert to dense
if hasattr(X_cos_pre, "toarray"):
    X_cos_pre = X_cos_pre.toarray()
if hasattr(X_dd_pre, "toarray"):
    X_dd_pre = X_dd_pre.toarray()

pca_cos = PCA(n_components=2, random_state=42).fit(X_cos_pre)
pca_dd = PCA(n_components=2, random_state=42).fit(X_dd_pre)
pcs_cos = pca_cos.transform(X_cos_pre)
pcs_dd = pca_dd.transform(X_dd_pre)
print(f"Cossairt PCA explained variance: PC1 {pca_cos.explained_variance_ratio_[0]:.3f}, PC2 {pca_cos.explained_variance_ratio_[1]:.3f}")
print(f"DD PCA explained variance: PC1 {pca_dd.explained_variance_ratio_[0]:.3f}, PC2 {pca_dd.explained_variance_ratio_[1]:.3f}")

# %% [markdown]
# ## A.3.5 Figure 3 — Cossairt vs DD predictive comparison

# %%
# Predictions for full dataset (for scatter)
cos_pred_full = cos_pipe.predict(X_cos)
dd_pred_full = dd_pipe.predict(X_dd)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))


def _pretty_feature(f):
    """Map raw column names to human-readable aliases for figure labels."""
    s = f.replace("num__", "").replace("cat__", "")
    table = {
        "temp_c": "growth temp.", "T_growth_C": "growth temp.",
        "time_min": "growth time",
        "other_1_None": "other_1: none", "other_1_amount_mmol": "other_1 amount",
        "other_1_zinc chloride": "other_1: zinc chloride",
        "other_1_zinc oleate": "other_1: zinc oleate",
        "other_1_zinc undecylenate": "other_1: zinc undecylenate",
        "outer_is_ZnS": "outer ZnS", "shell_layer_count": "shell layers",
    }
    if s in table:
        return table[s]
    # system_tag_X/Y/Z -> X/Y/Z ; route_x -> route: x ; in_source_x -> In source: x ; second_sol_x -> 2nd solvent: x
    if s.startswith("system_tag_"): return s.replace("system_tag_", "")
    if s.startswith("route_"): return "route: " + s.replace("route_", "").replace("_", " ")
    if s.startswith("in_source_"): return "In source: " + s.replace("in_source_", "").replace("_", " ")
    if s.startswith("second_sol_"): return "2nd solvent: " + s.replace("second_sol_", "").replace("_", " ")
    if s.startswith("shell_innermost_"): return s.replace("shell_innermost_", "") + " inner shell"
    if s.startswith("other_1_"): return "other_1: " + s.replace("other_1_", "").replace("_", " ")
    s2 = s.replace("_", " ")
    if s2.startswith("other 1 "): return "other_1: " + s2[len("other 1 "):]
    return s2

SHARED_HIGHLIGHT = "#d62728"  # red for shared T / time features
SHARED_NAMES = {"num__temp_c", "num__T_growth_C", "num__time_min"}

# Panel A — Cossairt observed vs predicted, with reported MAE reference line
ax = axes[0, 0]
ax.scatter(y_cos, cos_pred_full, s=50, c="#ff7f0e", edgecolors="black",
           linewidths=0.4, alpha=0.75, label="reference entries")
lo = min(y_cos.min(), cos_pred_full.min()) - 15
hi = max(y_cos.max(), cos_pred_full.max()) + 15
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="y = x", zorder=2)
ax.set_xlabel("Observed emission (nm)", fontsize=9)
ax.set_ylabel("Predicted emission (nm)", fontsize=9)
ax.set_title("A. reference reproduction", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

# F40: inset bar chart — 11.46 / 27.28 / 41.75 nm progression decomposition
# (audit-grade upper bound R4=41.75 made visible in main text)
ax_inset = ax.inset_axes([0.56, 0.07, 0.42, 0.34])  # enlarged for readability
regime_labels = ["Published", "No imp.", "Strict"]
regime_mae = [11.46, 27.28, 41.75]
regime_err = [0.0, 6.35, 7.88]
regime_colors = ["#bbbbbb", "#d62728", "#7a0a17"]
bars = ax_inset.bar(range(3), regime_mae, yerr=regime_err,
                     color=regime_colors, edgecolor="black",
                     linewidth=0.4, alpha=0.9, capsize=2)
ax_inset.set_xticks(range(3))
ax_inset.set_xticklabels(regime_labels, fontsize=7)
ax_inset.set_ylabel("MAE (nm)", fontsize=7)
ax_inset.tick_params(axis="y", labelsize=6)
for i, (v, e) in enumerate(zip(regime_mae, regime_err)):
    ax_inset.text(i, v + e + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
ax_inset.set_ylim(0, max(regime_mae) + max(regime_err) + 9)
ax_inset.set_facecolor((1, 1, 1, 0.92))
for spine in ax_inset.spines.values():
    spine.set_linewidth(0.4)

# Panel B — DD observed vs predicted (paper-level CV emphasis)
ax = axes[0, 1]
ax.scatter(y_dd, dd_pred_full, s=50, c="#1f77b4", edgecolors="black",
           linewidths=0.4, alpha=0.75, label="curated entries")
lo = min(y_dd.min(), dd_pred_full.min()) - 15
hi = max(y_dd.max(), dd_pred_full.max()) + 15
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="y = x", zorder=2)
ax.set_xlabel("Observed emission (nm)", fontsize=9)
ax.set_ylabel("Predicted emission (nm)", fontsize=9)
ax.set_title("B. curated reproduction", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

# Panel C — Cossairt feature importance with shared features highlighted
ax = axes[1, 0]
top_cos = cos_imp.head(10).iloc[::-1].reset_index(drop=True)
colors_cos = [SHARED_HIGHLIGHT if f in SHARED_NAMES else "#ff7f0e"
              for f in top_cos["feature"]]
ax.barh(np.arange(len(top_cos)), top_cos["importance"], color=colors_cos, alpha=0.85,
        edgecolor="black", linewidth=0.4)
ax.set_yticks(np.arange(len(top_cos)))
labels = []
for f in top_cos["feature"]:
    label = _pretty_feature(f)
    if f in SHARED_NAMES:
        label = label + " ★"
    labels.append(label)
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel("Feature importance", fontsize=9)
ax.set_title("C. reference feature importance", fontsize=11, fontweight="bold")
ax.text(0.97, 0.04,
        "★ growth temp. rank 2\n★ growth time rank 3",
        transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor=SHARED_HIGHLIGHT))

# Panel D — DD feature importance with shared features highlighted
ax = axes[1, 1]
top_dd = dd_imp.head(10).iloc[::-1].reset_index(drop=True)
colors_dd = [SHARED_HIGHLIGHT if f in SHARED_NAMES else "#1f77b4"
             for f in top_dd["feature"]]
ax.barh(np.arange(len(top_dd)), top_dd["importance"], color=colors_dd, alpha=0.85,
        edgecolor="black", linewidth=0.4)
ax.set_yticks(np.arange(len(top_dd)))
labels = []
for f in top_dd["feature"]:
    label = _pretty_feature(f)
    if f in SHARED_NAMES:
        label = label + " ★"
    labels.append(label)
ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlabel("Feature importance", fontsize=9)
ax.set_title("D. curated feature importance", fontsize=11, fontweight="bold")
ax.text(0.97, 0.04,
        "★ growth temp. rank 1\n★ growth time rank 3",
        transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor=SHARED_HIGHLIGHT))

fig.tight_layout()

fig3_pdf = PROJECT_ROOT / "figures" / "figure_3_cossairt_reproduction.pdf"
fig3_svg = PROJECT_ROOT / "figures" / "figure_3_cossairt_reproduction.svg"
fig.savefig(fig3_pdf, bbox_inches="tight")
fig.savefig(fig3_svg, bbox_inches="tight")
print(f"Saved: {fig3_pdf}")
print(f"Saved: {fig3_svg}")

# %% [markdown]
# ## A.3.6 Persist A.3 results

# %%
results_a3 = dict(
    cossairt_hyperparameters_source=(
        "notebook_85/hao/3.1. model hao emission.ipynb (verified SHA-256 "
        "match with GitHub Cossairt-Lab/Indium-Phosphide master)"
    ),
    hyperparameters=dict(model="ExtraTreesRegressor", **COS_HP_EMISSION,
                         split=COS_SPLIT),
    cossairt=dict(
        n=int(len(y_cos)),
        split_mae_nm=cos_split_mae,
        split_r2=cos_split_r2,
        split_mae_bootstrap_95ci=cos_mae_ci,
        kfold5_mae_mean=float(cos_cv.mean()),
        kfold5_mae_std=float(cos_cv.std()),
        kfold5_mae_per_fold=[float(v) for v in cos_cv],
        cossairt_reported_best_imputed_nm=11.46,
    ),
    dd=dict(
        n=int(len(y_dd)),
        n_papers=int(len(set(groups_dd))),
        groupkfold5_mae_mean=float(dd_cv.mean()),
        groupkfold5_mae_std=float(dd_cv.std()),
        groupkfold5_mae_per_fold=[float(v) for v in dd_cv],
        nongrouped_kfold5_mae_mean=float(dd_kfold_cv.mean()),
        nongrouped_kfold5_mae_std=float(dd_kfold_cv.std()),
    ),
    shared_feature_comparison=shared_df.to_dict("records"),
    spearman_rho=(
        dict(computed=True, rho=float(rho), p=float(rho_p),
             n_features_intersected=int(len(joined)))
        if (len(joined) >= 3 and not np.isnan(rho)) else
        dict(computed=False,
             reason=("only %d shared feature name(s) after dropping num__/cat__ "
                     "prefixes (OneHotEncoder value suffixes are not canonicalized, "
                     "so e.g. temp_c vs T_growth_C do not match); Spearman rho is "
                     "undefined for n<3 and is therefore NOT computed — "
                     "missing-by-design, not a zero/absent correlation result"
                     ) % int(len(joined)),
             n_features_intersected=int(len(joined)))
    ),
    pca=dict(
        cossairt_explained_variance=[float(v) for v in pca_cos.explained_variance_ratio_],
        dd_explained_variance=[float(v) for v in pca_dd.explained_variance_ratio_],
    ),
    top10_cossairt=cos_imp.head(10).to_dict("records"),
    top10_dd=dd_imp.head(10).to_dict("records"),
)
out3 = PROJECT_ROOT / "analysis" / "part_a3_results.json"
with open(out3, "w", encoding="utf-8") as f:
    json.dump(results_a3, f, indent=2, default=str)
print(f"Saved: {out3}")
print(f"Output SHA-256: {sha256_of(out3)}")

# %% [markdown]
# ## A.3.7 Findings narrative → analysis design notes

# %% [markdown]
# # Part A.4 — Cross-prediction (shared-features subset, no-imputation)
#
# **Goal**: complete the 4-stage cross-validation protocol by training a
# model on one corpus and predicting the other's outcomes, using only the
# *shared minimal feature set* (growth temperature, growth time). This is
# the most conservative cross-prediction we can run without inferring
# precursor / route mappings between the two corpora. The protocol
# maintains the no-imputation audit-grade regime: training uses only
# explicitly-reported records on each side.
#
# **Direction 1 (Forward, Cossairt → DD)**: train an ExtraTreesRegressor
# on Cossairt 85-emission published-only records with features (temp_c,
# time_min), predict DD 129 emission rows using the same two features
# (T_growth_C, time_min).
#
# **Direction 2 (Backward, DD → Cossairt)**: train an ExtraTreesRegressor
# on DD 129-emission records with the two shared features, predict the
# Cossairt 85 emission rows.
#
# **Asymmetry interpretation**:
# - If both directions transfer with similar MAE, the two corpora encode
#   the T/time → emission mapping consistently.
# - If one direction is substantially better, that asymmetry quantifies
#   the *information advantage* of one corpus's covered design space over
#   the other.
# - DD has more papers / fewer per-paper recipes; Cossairt has fewer
#   papers / more entries per paper. Either could be more informative
#   for cross-prediction depending on coverage.

# %%
SHARED_FEATURES_A4 = ["T_temp", "time_min"]

# Cossairt subset (published + emission-present), built from the non-0.0-
# filled raw snapshot cos_raw_a4 (NOT cos_em, which inherits the COS_NUMERIC
# 0.0-fill at L726 used for the A.3 reproduction). Coerce then drop genuinely
# missing rows (no-imputation, symmetric with the DD side).
cos_a4 = cos_raw_a4[["doi", "temp_c", "time_min", "emission_nm"]].copy()
cos_a4 = cos_a4.rename(columns={"temp_c": "T_temp"})
for c in ["T_temp", "time_min", "emission_nm"]:
    cos_a4[c] = pd.to_numeric(cos_a4[c], errors="coerce")
cos_a4 = cos_a4.dropna(subset=["T_temp", "time_min", "emission_nm"])
X_cos_a4 = cos_a4[SHARED_FEATURES_A4].values.astype(float)
y_cos_a4 = cos_a4["emission_nm"].values.astype(float)
groups_cos_a4 = cos_a4["doi"].values  # paper-level (doi) grouping — symmetric with the DD side
print(f"Cossairt A.4 subset: n={len(y_cos_a4)} rows (features={SHARED_FEATURES_A4})")

# DD subset (emission-present) with unified column name.
# Build from the raw DD frame (dd_all), NOT dd_em: dd_em fills missing
# T_growth_C / time_min with 0.0 (A.3 pipeline), which would leak spurious
# 0-minute / 0-degree rows into A.4. Here we coerce, then drop genuinely
# missing rows (no-imputation audit-grade regime).
dd_a4 = dd_all[["T_growth_C", "time_min", "emission_nm", "paper_id_dd"]].copy()
dd_a4 = dd_a4.rename(columns={"T_growth_C": "T_temp"})
for c in ["T_temp", "time_min", "emission_nm"]:
    dd_a4[c] = pd.to_numeric(dd_a4[c], errors="coerce")
dd_a4 = dd_a4.dropna(subset=["T_temp", "time_min", "emission_nm"])
X_dd_a4 = dd_a4[SHARED_FEATURES_A4].values.astype(float)
y_dd_a4 = dd_a4["emission_nm"].values.astype(float)
groups_dd_a4 = dd_a4["paper_id_dd"].values
print(f"DD A.4 subset: n={len(y_dd_a4)} rows (features={SHARED_FEATURES_A4})")

# %% [markdown]
# ## A.4.1 Forward — train Cossairt, predict DD

# %%
cos_model_a4 = ExtraTreesRegressor(**COS_HP_EMISSION)
cos_model_a4.fit(X_cos_a4, y_cos_a4)
y_dd_pred_from_cos = cos_model_a4.predict(X_dd_a4)
fwd_mae = float(mean_absolute_error(y_dd_a4, y_dd_pred_from_cos))
fwd_r2 = float(r2_score(y_dd_a4, y_dd_pred_from_cos))
print(f"Forward (Cossairt→DD):  MAE = {fwd_mae:.2f} nm   R² = {fwd_r2:.3f}")

# Row-level fixed-prediction bootstrap (descriptive only; NOT paper-level):
# predictions are computed once, then row indices are resampled with replacement.
# It characterizes row-level prediction-error spread and ignores within-paper
# dependence — it is not a paper-level generalization CI.
rng_a4 = np.random.default_rng(42)
fwd_mae_boot = np.empty(1000)
for i in range(1000):
    idx = rng_a4.integers(0, len(y_dd_a4), len(y_dd_a4))
    fwd_mae_boot[i] = mean_absolute_error(y_dd_a4[idx], y_dd_pred_from_cos[idx])
fwd_ci = (float(np.percentile(fwd_mae_boot, 2.5)),
          float(np.percentile(fwd_mae_boot, 97.5)))
print(f"  Bootstrap 95% CI: ({fwd_ci[0]:.2f}, {fwd_ci[1]:.2f}) nm")

# %% [markdown]
# ## A.4.2 Backward — train DD, predict Cossairt

# %%
dd_model_a4 = ExtraTreesRegressor(**COS_HP_EMISSION)
dd_model_a4.fit(X_dd_a4, y_dd_a4)
y_cos_pred_from_dd = dd_model_a4.predict(X_cos_a4)
bwd_mae = float(mean_absolute_error(y_cos_a4, y_cos_pred_from_dd))
bwd_r2 = float(r2_score(y_cos_a4, y_cos_pred_from_dd))
print(f"Backward (DD→Cossairt): MAE = {bwd_mae:.2f} nm   R² = {bwd_r2:.3f}")

bwd_mae_boot = np.empty(1000)
for i in range(1000):
    idx = rng_a4.integers(0, len(y_cos_a4), len(y_cos_a4))
    bwd_mae_boot[i] = mean_absolute_error(y_cos_a4[idx], y_cos_pred_from_dd[idx])
bwd_ci = (float(np.percentile(bwd_mae_boot, 2.5)),
          float(np.percentile(bwd_mae_boot, 97.5)))
print(f"  Bootstrap 95% CI: ({bwd_ci[0]:.2f}, {bwd_ci[1]:.2f}) nm")

# %% [markdown]
# ## A.4.3 Reference — within-source CV with shared-features-only
#
# To interpret the cross-prediction MAEs honestly we need the
# *within-source* MAE under the same minimal feature set. This isolates
# the feature-set cost from any cross-corpus transfer gap.

# %%
cos_within_cv = -cross_val_score(
    Pipeline([("et", ExtraTreesRegressor(**COS_HP_EMISSION))]),
    X_cos_a4, y_cos_a4, groups=groups_cos_a4, cv=GroupKFold(5),
    scoring="neg_mean_absolute_error", n_jobs=1)
cos_within_r2 = cross_val_score(
    Pipeline([("et", ExtraTreesRegressor(**COS_HP_EMISSION))]),
    X_cos_a4, y_cos_a4, groups=groups_cos_a4, cv=GroupKFold(5),
    scoring="r2", n_jobs=1)
print(f"Cossairt within-source 5-fold CV MAE (shared features only): "
      f"{cos_within_cv.mean():.2f} ± {cos_within_cv.std():.2f} nm")
print(f"Cossairt within-source 5-fold CV R² (shared features only): "
      f"{cos_within_r2.mean():.3f} ± {cos_within_r2.std():.3f}")

dd_within_cv = -cross_val_score(
    Pipeline([("et", ExtraTreesRegressor(**COS_HP_EMISSION))]),
    X_dd_a4, y_dd_a4, groups=groups_dd_a4, cv=GroupKFold(5),
    scoring="neg_mean_absolute_error", n_jobs=1)
dd_within_r2 = cross_val_score(
    Pipeline([("et", ExtraTreesRegressor(**COS_HP_EMISSION))]),
    X_dd_a4, y_dd_a4, groups=groups_dd_a4, cv=GroupKFold(5),
    scoring="r2", n_jobs=1)
print(f"DD within-source paper-level GroupKFold MAE (shared features only): "
      f"{dd_within_cv.mean():.2f} ± {dd_within_cv.std():.2f} nm")
print(f"DD within-source paper-level GroupKFold R² (shared features only): "
      f"{dd_within_r2.mean():.3f} ± {dd_within_r2.std():.3f}")

# %% [markdown]
# ## A.4.4 Asymmetry diagnostic
#
# - Forward gap  = Forward MAE  − DD within-source MAE
# - Backward gap = Backward MAE − Cossairt within-source MAE
#
# Each gap reports the *additional* error incurred by training on the
# other corpus rather than within-source under the same feature set.

# %%
fwd_gap = fwd_mae - dd_within_cv.mean()
bwd_gap = bwd_mae - cos_within_cv.mean()
print(f"Forward gap  (Cos→DD MAE − DD within): {fwd_gap:+.2f} nm")
print(f"Backward gap (DD→Cos MAE − Cos within): {bwd_gap:+.2f} nm")
asym = fwd_gap - bwd_gap
print(f"Asymmetry (forward − backward gap):     {asym:+.2f} nm")

# %% [markdown]
# ## A.4.5 Figure 4 — forward + backward scatter

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# Panel A — Forward (Cossairt → DD)
ax = axes[0]
ax.scatter(y_dd_a4, y_dd_pred_from_cos, s=55, c="#ff7f0e", alpha=0.7,
           edgecolors="black", linewidths=0.4,
           label="curated test entries")
lo = min(y_dd_a4.min(), y_dd_pred_from_cos.min()) - 20
hi = max(y_dd_a4.max(), y_dd_pred_from_cos.max()) + 20
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="y = x")
ax.set_xlabel("Observed emission (nm)", fontsize=9)
ax.set_ylabel("Predicted emission (nm)", fontsize=9)
ax.set_title("A. Forward: reference → curated", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.text(0.97, 0.04,
        f"MAE = {fwd_mae:.1f} nm\nforward gap = {fwd_gap:+.1f} nm",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))

# Panel B — Backward (DD → Cossairt)
ax = axes[1]
ax.scatter(y_cos_a4, y_cos_pred_from_dd, s=55, c="#1f77b4", alpha=0.7,
           edgecolors="black", linewidths=0.4,
           label="reference test entries")
lo = min(y_cos_a4.min(), y_cos_pred_from_dd.min()) - 20
hi = max(y_cos_a4.max(), y_cos_pred_from_dd.max()) + 20
ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="y = x")
ax.set_xlabel("Observed emission (nm)", fontsize=9)
ax.set_ylabel("Predicted emission (nm)", fontsize=9)
ax.set_title("B. Backward: curated → reference", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.text(0.97, 0.04,
        f"MAE = {bwd_mae:.1f} nm\nbackward gap = {bwd_gap:+.1f} nm",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))

# Panel C — Bar chart summary (MAEs + gaps)
ax = axes[2]
bars_labels = ["curated\nwithin-source", "reference\u2192curated\n(forward)",
               "reference\nwithin-source", "curated\u2192reference\n(backward)"]
bars_vals = [dd_within_cv.mean(), fwd_mae, cos_within_cv.mean(), bwd_mae]
bars_errs = [dd_within_cv.std(), 0, cos_within_cv.std(), 0]
bars_colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]
positions = [0, 1, 3, 4]
bars = ax.bar(positions, bars_vals, yerr=bars_errs, capsize=4,
              color=bars_colors, alpha=0.85, edgecolor="black", linewidth=0.6)
for pos, val in zip(positions, bars_vals):
    ax.text(pos, val + 2, f"{val:.1f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")
ax.text(0.97, 0.97,
        f"forward gap = {fwd_gap:+.1f} nm\nbackward gap = {bwd_gap:+.1f} nm",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))
ax.set_xticks(positions)
ax.set_xticklabels(bars_labels, fontsize=7.5)
ax.set_ylabel("MAE (nm)", fontsize=9)
ax.set_title("C. Cross-source transfer vs baselines", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(bars_vals) * 1.40)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#1f77b4", alpha=0.85, label="curated target"),
    Patch(facecolor="#ff7f0e", alpha=0.85, label="reference target"),
], loc="upper left", fontsize=7.5)

fig.tight_layout()

fig4_pdf = PROJECT_ROOT / "figures" / "figure_4_cross_prediction.pdf"
fig4_svg = PROJECT_ROOT / "figures" / "figure_4_cross_prediction.svg"
fig.savefig(fig4_pdf, bbox_inches="tight")
fig.savefig(fig4_svg, bbox_inches="tight")
print(f"Saved: {fig4_pdf}")
print(f"Saved: {fig4_svg}")

# %% [markdown]
# ## A.4.6 Persist A.4 results

# %%
# --- Overlap-excluded sensitivity ---------------------------------------
# Drop the 9 DOI-overlap publications (A.1 overlap set) from BOTH corpora,
# train + test, then recompute forward / backward / within-source under the
# same minimal shared-feature, no-imputation regime. Tests whether the
# cross-prediction transfer is inflated by shared papers.
ovl_pids = set(paper_to_doi.keys())            # 9 DD overlap paper_ids
ovl_dois = set(paper_to_doi.values())          # their Cossairt DOIs

dd_a4_ne = dd_a4[~dd_a4["paper_id_dd"].isin(ovl_pids)].copy()

# Rebuild the Cossairt A.4 frame carrying doi (identical coercion/dropna to
# the main cos_a4) so overlap rows can be excluded by DOI.
cos_a4_doi = cos_raw_a4[["temp_c", "time_min", "emission_nm", "doi"]].copy()
cos_a4_doi = cos_a4_doi.rename(columns={"temp_c": "T_temp"})
for c in ["T_temp", "time_min", "emission_nm"]:
    cos_a4_doi[c] = pd.to_numeric(cos_a4_doi[c], errors="coerce")
cos_a4_doi = cos_a4_doi.dropna(subset=["T_temp", "time_min", "emission_nm"])
cos_a4_ne = cos_a4_doi[~cos_a4_doi["doi"].astype(str).isin(ovl_dois)].copy()

X_dd_ne = dd_a4_ne[SHARED_FEATURES_A4].values.astype(float)
y_dd_ne = dd_a4_ne["emission_nm"].values.astype(float)
groups_dd_ne = dd_a4_ne["paper_id_dd"].values
X_cos_ne = cos_a4_ne[SHARED_FEATURES_A4].values.astype(float)
y_cos_ne = cos_a4_ne["emission_nm"].values.astype(float)
groups_cos_ne = cos_a4_ne["doi"].values
print(f"Overlap-excluded subsets: DD n={len(y_dd_ne)} "
      f"({len(set(groups_dd_ne))} papers), Cossairt n={len(y_cos_ne)}")

_fwd_ne = ExtraTreesRegressor(**COS_HP_EMISSION).fit(X_cos_ne, y_cos_ne)
fwd_mae_ne = float(mean_absolute_error(y_dd_ne, _fwd_ne.predict(X_dd_ne)))
_bwd_ne = ExtraTreesRegressor(**COS_HP_EMISSION).fit(X_dd_ne, y_dd_ne)
bwd_mae_ne = float(mean_absolute_error(y_cos_ne, _bwd_ne.predict(X_cos_ne)))

_n_splits_ne = max(2, min(5, len(set(groups_dd_ne))))
dd_within_ne = -cross_val_score(
    Pipeline([("et", ExtraTreesRegressor(**COS_HP_EMISSION))]),
    X_dd_ne, y_dd_ne, groups=groups_dd_ne, cv=GroupKFold(_n_splits_ne),
    scoring="neg_mean_absolute_error", n_jobs=1)
cos_within_ne = -cross_val_score(
    Pipeline([("et", ExtraTreesRegressor(**COS_HP_EMISSION))]),
    X_cos_ne, y_cos_ne, groups=groups_cos_ne,
    cv=GroupKFold(max(2, min(5, len(set(groups_cos_ne))))),
    scoring="neg_mean_absolute_error", n_jobs=1)
fwd_gap_ne = float(fwd_mae_ne - dd_within_ne.mean())
bwd_gap_ne = float(bwd_mae_ne - cos_within_ne.mean())
print(f"[overlap-excluded] forward  MAE={fwd_mae_ne:.2f}  DD within={dd_within_ne.mean():.2f}  "
      f"gap={fwd_gap_ne:+.2f} nm")
print(f"[overlap-excluded] backward MAE={bwd_mae_ne:.2f}  Cos within={cos_within_ne.mean():.2f}  "
      f"gap={bwd_gap_ne:+.2f} nm")

# %%
results_a4 = dict(
    shared_features=SHARED_FEATURES_A4,
    cossairt=dict(n=int(len(y_cos_a4)), n_papers=int(len(set(groups_cos_a4))),
                  within_groupkfold5_mae_mean=float(cos_within_cv.mean()),
                  within_groupkfold5_mae_std=float(cos_within_cv.std()),
                  within_groupkfold5_mae_per_fold=[float(v) for v in cos_within_cv],
                  within_groupkfold5_r2_mean=float(cos_within_r2.mean()),
                  within_groupkfold5_r2_std=float(cos_within_r2.std()),
                  within_groupkfold5_r2_per_fold=[float(v) for v in cos_within_r2]),
    dd=dict(n=int(len(y_dd_a4)), n_papers=int(len(set(groups_dd_a4))),
            within_groupkfold5_mae_mean=float(dd_within_cv.mean()),
            within_groupkfold5_mae_std=float(dd_within_cv.std()),
            within_groupkfold5_mae_per_fold=[float(v) for v in dd_within_cv],
            within_groupkfold5_r2_mean=float(dd_within_r2.mean()),
            within_groupkfold5_r2_std=float(dd_within_r2.std()),
            within_groupkfold5_r2_per_fold=[float(v) for v in dd_within_r2]),
    forward=dict(direction="Cossairt -> DD",
                 mae_nm=fwd_mae, r2=fwd_r2,
                 bootstrap_95ci=fwd_ci,
                 bootstrap_method="row-level fixed-prediction bootstrap (descriptive; not paper-level)",
                 gap_vs_within_source=float(fwd_gap)),
    backward=dict(direction="DD -> Cossairt",
                  mae_nm=bwd_mae, r2=bwd_r2,
                  bootstrap_95ci=bwd_ci,
                  bootstrap_method="row-level fixed-prediction bootstrap (descriptive; not paper-level)",
                  gap_vs_within_source=float(bwd_gap)),
    asymmetry_fwd_minus_bwd_gap=float(asym),
    overlap_excluded_sensitivity=dict(
        description="9 DOI-overlap publications removed from both corpora (train + test)",
        dd_n=int(len(y_dd_ne)), dd_n_papers=int(len(set(groups_dd_ne))),
        cossairt_n=int(len(y_cos_ne)),
        forward_mae_nm=fwd_mae_ne,
        dd_within_mae_nm=float(dd_within_ne.mean()),
        forward_gap_nm=fwd_gap_ne,
        backward_mae_nm=bwd_mae_ne,
        cossairt_within_mae_nm=float(cos_within_ne.mean()),
        backward_gap_nm=bwd_gap_ne,
        groupkfold_n_splits=int(_n_splits_ne),
    ),
    hyperparameters=dict(model="ExtraTreesRegressor", **COS_HP_EMISSION),
    regime="no-imputation audit-grade; shared minimal features (T + time only)",
)
out4 = PROJECT_ROOT / "analysis" / "part_a4_results.json"
with open(out4, "w", encoding="utf-8") as f:
    json.dump(results_a4, f, indent=2, default=str)
print(f"Saved: {out4}")
print(f"Output SHA-256: {sha256_of(out4)}")

# %% [markdown]
# ## A.4.7 Findings narrative → analysis design notes
