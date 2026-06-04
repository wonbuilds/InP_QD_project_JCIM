#!/usr/bin/env python3
"""Systematic feature pair audit on the DD-curated InP subset.

Phase 5 F39 — verify that no functional dependencies beyond the previously
disclosed `shell_layer_count` ↔ `shell_composition_tier` deterministic
transform leak into the modeling feature set. The audit reports:

- Categorical × categorical pairs: Cramér's V (0 = independent, 1 = deterministic)
- Numerical × numerical pairs: Pearson correlation
- Numerical × categorical pairs: eta-squared (η²) from one-way ANOVA

Specific feature pairs checked for redundancy:
- outer_is_ZnS × shell_innermost
- route × shell_innermost
- caution_count × shell_innermost
- T_growth_C × shell_innermost (continuous vs categorical)
- time_min × shell_innermost (continuous vs categorical)
- shell_layer_count × shell_innermost
- data_quality_tier × shell_innermost

Determinism: deterministic statistics; no random seed required.
Output JSON SHA-256-verified.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from io_helpers import load_dd_subset, sha256_of  # noqa: E402
from parse_shell_architecture import parse_shell_layers  # noqa: E402


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """Cramér's V with bias correction.

    Returns (V, contingency_table_min_dim).
    """
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion, correction=False)[0]
    n = confusion.values.sum()
    if n == 0:
        return 0.0, 0
    r, k = confusion.shape
    min_dim = min(r - 1, k - 1)
    if min_dim == 0:
        return 0.0, 0
    v = float(np.sqrt(chi2 / (n * min_dim)))
    return v, int(min_dim)


def eta_squared(numeric: pd.Series, categorical: pd.Series) -> float:
    """One-way ANOVA eta-squared: SS_between / SS_total.

    Range [0, 1]; 0 = no group-level mean difference, 1 = all variance
    explained by group membership.
    """
    df = pd.DataFrame({"y": numeric, "g": categorical}).dropna()
    groups = [grp["y"].values for _, grp in df.groupby("g")]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return float("nan")
    # ss_within
    grand_mean = df["y"].mean()
    ss_total = ((df["y"] - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)


def main() -> None:
    print("=" * 70)
    print("F39 — Feature pair audit (DD InP subset, n = 132)")
    print("=" * 70)

    dd = load_dd_subset()
    for c in ["T_growth_C", "time_min", "PL_peak_nm_final", "caution_count"]:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    parsed = dd["system_tag"].apply(parse_shell_layers).apply(pd.Series)
    parsed = parsed.drop(columns=["shell_layers"])
    dd = pd.concat([dd, parsed], axis=1)
    dd["outer_is_ZnS"] = (dd["shell_outermost"] == "ZnS").astype(int)
    dd["shell_layer_count"] = pd.to_numeric(dd["shell_layer_count"], errors="coerce").fillna(0).astype(int)

    # The features used in Part B modeling (caution_count OFF baseline)
    NUMERIC = ["T_growth_C", "time_min", "outer_is_ZnS", "shell_layer_count", "caution_count"]
    CATEGORICAL = ["route", "shell_innermost", "shell_composition_tier", "shell_outermost", "data_quality_tier"]

    for c in NUMERIC:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    for c in CATEGORICAL:
        dd[c] = dd[c].fillna("Unknown").astype(str)

    # ---- Categorical × Categorical: Cramér's V ----
    print("\n=== Cramér's V (categorical × categorical) ===")
    cramers_results = []
    for c1, c2 in combinations(CATEGORICAL, 2):
        v, mdim = cramers_v(dd[c1], dd[c2])
        cramers_results.append(dict(feat1=c1, feat2=c2, V=v, min_dim=mdim))
        print(f"  {c1} × {c2}: V = {v:.3f} (min_dim={mdim})")

    # ---- Numerical × Numerical: Pearson r ----
    print("\n=== Pearson r (numerical × numerical) ===")
    pearson_results = []
    for n1, n2 in combinations(NUMERIC, 2):
        sub = dd[[n1, n2]].dropna()
        if len(sub) < 3:
            r = float("nan")
        else:
            r = float(sub[n1].corr(sub[n2], method="pearson"))
        pearson_results.append(dict(feat1=n1, feat2=n2, r=r, n=int(len(sub))))
        print(f"  {n1} × {n2}: r = {r:+.3f} (n={len(sub)})")

    # ---- Numerical × Categorical: eta² ----
    print("\n=== eta² (numerical × categorical) ===")
    eta_results = []
    for n in NUMERIC:
        for c in CATEGORICAL:
            eta = eta_squared(dd[n], dd[c])
            eta_results.append(dict(numeric=n, categorical=c, eta_sq=eta))
            print(f"  {n} × {c}: η² = {eta:.3f}" if not np.isnan(eta) else f"  {n} × {c}: η² = N/A")

    # Identify suspicious pairs (potential deterministic transforms)
    print("\n=== Suspicious pairs (functional-dependence candidates) ===")
    suspicious = []
    for r in cramers_results:
        if r["V"] >= 0.9 and r["min_dim"] >= 1:
            suspicious.append(dict(
                pair=f"{r['feat1']} × {r['feat2']}",
                metric="Cramér's V",
                value=r["V"],
                threshold=0.9,
                interpretation="Near-deterministic categorical-categorical dependence.",
            ))
            print(f"  ★ {r['feat1']} × {r['feat2']}: Cramér's V = {r['V']:.3f}")
    for r in pearson_results:
        if abs(r["r"]) >= 0.9:
            suspicious.append(dict(
                pair=f"{r['feat1']} × {r['feat2']}",
                metric="Pearson r",
                value=r["r"],
                threshold=0.9,
                interpretation="Near-linear numerical-numerical dependence.",
            ))
            print(f"  ★ {r['feat1']} × {r['feat2']}: |r| = {abs(r['r']):.3f}")
    for r in eta_results:
        if not np.isnan(r["eta_sq"]) and r["eta_sq"] >= 0.9:
            suspicious.append(dict(
                pair=f"{r['numeric']} × {r['categorical']}",
                metric="eta²",
                value=r["eta_sq"],
                threshold=0.9,
                interpretation="Categorical groups capture > 90% of numeric variance.",
            ))
            print(f"  ★ {r['numeric']} × {r['categorical']}: η² = {r['eta_sq']:.3f}")
    if not suspicious:
        print("  (none beyond the pre-disclosed shell_layer_count ↔ shell_composition_tier pair)")

    out = PROJECT_ROOT / "analysis" / "feature_pair_audit_results.json"
    payload = dict(
        analysis="F39 systematic feature pair audit on DD InP subset",
        n_rows=int(len(dd)),
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        cramers_v=cramers_results,
        pearson_r=pearson_results,
        eta_squared=eta_results,
        suspicious_pairs=suspicious,
        suspicion_threshold=0.9,
        notes=(
            "Cramér's V uses no bias correction (chi2_contingency correction=False) "
            "to match the standard textbook formula. Pearson r computed with "
            "pairwise-complete observations. eta² = SS_between / SS_total from "
            "one-way ANOVA; reflects the fraction of numeric variance explained "
            "by categorical-group membership. The suspicion threshold of 0.9 is "
            "chosen to flag near-deterministic dependencies; values 0.7-0.9 "
            "indicate strong but not exclusive associations and are not flagged "
            "as suspicious (sub-deterministic associations are expected in "
            "chemistry: e.g., route × shell_innermost reflects real synthetic "
            "co-occurrence patterns, not a transform leak)."
        ),
    )
    def _nan_to_none(o):
        # RFC 8259: emit JSON null instead of the non-standard bare NaN token.
        if isinstance(o, (float, np.floating)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, dict):
            return {k: _nan_to_none(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_nan_to_none(x) for x in o]
        return o
    with out.open("w", encoding="utf-8") as f:
        json.dump(_nan_to_none(payload), f, indent=2, allow_nan=False, default=str)
    print(f"\nSaved: {out}")
    print(f"Output SHA-256: {sha256_of(out)}")


if __name__ == "__main__":
    main()
