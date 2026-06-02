#!/usr/bin/env python3
"""Parse `system_tag` strings into structured shell-architecture features.

Background (from `data/from_dd/constant_columns.md` §3): the DD InP subset
has `shell_present_flag = 1` for all 132 records — every recipe is a
core/shell QD. The architectural variability lives in the `system_tag`
slash-separated string. This module decomposes that string into:

    core                 — first slash-separated segment
    shell_layers         — list of subsequent segments
    shell_layer_count    — len(shell_layers)
    shell_innermost      — shell_layers[0] or "" if no shell
    shell_outermost      — shell_layers[-1] or "" if no shell
    shell_composition_tier — coarse category for ML stratification

stdlib-only. Used by `analysis/part_b_predictive_modeling.py`.

Tier definitions (informative, chosen for ML use):
    single_shell       : exactly 1 shell layer
    double_shell       : exactly 2 shell layers
    triple_plus_shell  : 3 or more shell layers
    no_shell           : 0 shell layers (defensive; not expected in InP subset)
"""

from __future__ import annotations

from typing import Dict


def parse_shell_layers(system_tag: str) -> Dict[str, object]:
    """Decompose a slash-separated `system_tag` like 'InP/ZnSe/ZnS' into
    structured shell-architecture fields.

    Examples:
        >>> parse_shell_layers('InP/ZnSe/ZnS')
        {'core': 'InP', 'shell_layers': ['ZnSe', 'ZnS'],
         'shell_layer_count': 2, 'shell_innermost': 'ZnSe',
         'shell_outermost': 'ZnS',
         'shell_composition_tier': 'double_shell'}
        >>> parse_shell_layers('InP/ZnS')
        {'core': 'InP', 'shell_layers': ['ZnS'],
         'shell_layer_count': 1, 'shell_innermost': 'ZnS',
         'shell_outermost': 'ZnS',
         'shell_composition_tier': 'single_shell'}
        >>> parse_shell_layers('InZnP/InGaP/GaP/ZnSeS')
        {'core': 'InZnP', 'shell_layers': ['InGaP', 'GaP', 'ZnSeS'],
         'shell_layer_count': 3, 'shell_innermost': 'InGaP',
         'shell_outermost': 'ZnSeS',
         'shell_composition_tier': 'triple_plus_shell'}
    """
    if not isinstance(system_tag, str) or not system_tag.strip():
        return _empty_result()
    parts = [seg.strip() for seg in system_tag.split("/") if seg.strip()]
    if not parts:
        return _empty_result()
    core = parts[0]
    shell_layers = parts[1:]
    n = len(shell_layers)
    if n == 0:
        tier = "no_shell"
    elif n == 1:
        tier = "single_shell"
    elif n == 2:
        tier = "double_shell"
    else:
        tier = "triple_plus_shell"
    return dict(
        core=core,
        shell_layers=shell_layers,
        shell_layer_count=n,
        shell_innermost=shell_layers[0] if shell_layers else "",
        shell_outermost=shell_layers[-1] if shell_layers else "",
        shell_composition_tier=tier,
    )


def _empty_result() -> Dict[str, object]:
    return dict(core="", shell_layers=[], shell_layer_count=0,
                shell_innermost="", shell_outermost="",
                shell_composition_tier="no_shell")


if __name__ == "__main__":
    import csv
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    inp_csv = project_root / "data" / "from_dd" / "inp_subset.csv"
    with inp_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"DD InP subset: {len(rows)} rows")
    from collections import Counter
    parsed = [parse_shell_layers(r["system_tag"]) for r in rows]
    n_by_count = Counter(p["shell_layer_count"] for p in parsed)
    n_by_tier = Counter(p["shell_composition_tier"] for p in parsed)
    n_by_innermost = Counter(p["shell_innermost"] for p in parsed)
    n_by_outermost = Counter(p["shell_outermost"] for p in parsed)
    print("\nshell_layer_count distribution:")
    for k, v in sorted(n_by_count.items()):
        print(f"  layers = {k}:  {v}")
    print("\nshell_composition_tier distribution:")
    for k, v in n_by_tier.most_common():
        print(f"  {k:24s}  {v}")
    print("\nshell_innermost distribution (top 10):")
    for k, v in n_by_innermost.most_common(10):
        print(f"  {k:24s}  {v}")
    print("\nshell_outermost distribution (top 10):")
    for k, v in n_by_outermost.most_common(10):
        print(f"  {k:24s}  {v}")
