# InP QD Literature-Mining Project (JCIM)

Analysis code, processed data, and figures for the JCIM submission
**"Leakage-Aware External Assessment of an LLM-Curated InP Quantum-Dot Synthesis
Corpus"** (Yoo).

This repository is the **reproducible analysis package** for that manuscript: it
cross-validates an LLM-curated InP quantum-dot synthesis corpus (132 records / 52
papers) against the manually curated reference of Nguyen et al. (*Chem. Mater.*
2022, 34, 6296), fits leakage-aware (paper-level GroupKFold) interpretable models
of emission/QY/FWHM/shell architecture, and prioritizes candidate synthesis
conditions. The manuscript text itself is **not** included here.

> **On "candidate conditions / recipes":** the prioritized synthesis conditions
> (`recipes/inp_qd_candidate_recipes_v1.json`; the filename is retained for
> stability) are **model-prioritized candidate hypotheses**, not experimentally
> validated syntheses — computational suggestions for under-explored design
> regions that require wet-lab confirmation.

## Repository layout

| Path | Role |
| --- | --- |
| `data/from_dd/` | LLM-curated InP subset (`inp_subset.csv`, 132 rows) + provenance docs (`constant_columns.md`, `dd_paper_years.csv`). |
| `data/from_cossairt/` | Cossairt-group reference dataset (Nguyen et al. 2022) CSVs, redeposited with SHA-256 verification. |
| `data/integrated/` | Combined dataset (`inp_combined.csv`) + `common_schema.json`. |
| `analysis/` | Analysis scripts (`part_a/b/c_*.py`, baselines, SHAP stability, Cossairt ablation, time-forward, feature-pair audit) + their `*_results.json` and the paired `.ipynb` notebooks. |
| `figures/` | Generated figures (`figure_1`–`figure_11`, PDF + SVG). Outside the checksum manifest. |
| `recipes/` | `inp_qd_candidate_recipes_v1.json` — Top-10 model-prioritized candidate conditions (hypotheses) with provenance. |
| `scripts/` | Pipeline helpers: `reproduce_all.py`, `extract_inp_subset.py`, `build_inp_combined.py`, `io_helpers.py`, `parse_shell_architecture.py`, and the SHA-256 manifest `expected_checksums.json`. |

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (a) Verify deposited outputs against the SHA-256 manifest (fast, no recompute):
python scripts/reproduce_all.py --verify-only          # expect all [ok]

# (b) Full clean rebuild, then verify (re-runs the whole pipeline):
python scripts/reproduce_all.py --strict
```

- **`--verify-only`** only hashes the deposited artifacts and compares them to
  `scripts/expected_checksums.json`. Seconds to run; it does **not** re-run any
  analysis. Note: `--verify-only` checks **file integrity, not regenerability** —
  it confirms the committed files match the manifest, not that the code rebuilds
  them bit-for-bit (use `--strict` for that).
- **`--strict`** re-runs the entire pipeline and then verifies. The Part B SHAP
  recomputation and the paper-level bootstraps dominate runtime — expect roughly
  **5–15 minutes** on a typical laptop (single-threaded, `random_state = 42`).
- All randomness is seeded (`random_state = 42`); feature imputation is performed
  per-fold inside the CV pipelines (no leakage).

### ⚠️ Version sensitivity (pinned environment required)

Exact byte-for-byte reproduction (and the SHA-256 manifest) is guaranteed
**only under the pinned environment** in `requirements.txt`, notably
`numpy==2.4.6`, `scipy==1.17.1`, `scikit-learn==1.8.0`, `shap==0.51.0`,
`pandas==3.0.3`, `matplotlib==3.10.9`. Different versions can change float
serialization, p-values, SHAP magnitudes at trailing digits, and figure hashes —
the numeric conclusions are stable, but `--strict` checksum equality is not
guaranteed off-pin. Use `--verify-only` against the deposited artifacts for a
pinned-independent integrity check of what is committed; use `--strict` for a
clean rebuild in the pinned environment. **`figures/` are regenerated from the
data and are not checksum-manifest-verified** (PDF/SVG serialization is not
guaranteed bit-stable); they are intentionally outside the manifest, and the
numbers behind every figure are checksum-verified via the `*_results.json`
artifacts.

## What each script produces

| Script | Output(s) |
| --- | --- |
| `scripts/extract_inp_subset.py` | `data/from_dd/inp_subset.csv` (reuses the deposited file unless `--force`). |
| `scripts/build_inp_combined.py` | `data/integrated/inp_combined.csv`. |
| `analysis/part_a_cross_validation.py` | `analysis/part_a{1,2,3,4}_results.json`; `figures/figure_1`–`figure_4`. |
| `analysis/part_b_predictive_modeling.py` | `analysis/part_b_pl_peak_results.json`, `analysis/part_b_c22_results.json`; `figures/figure_5`–`figure_8`. |
| `analysis/part_b_baselines.py` | `analysis/part_b_baselines_results.json`. |
| `analysis/part_b_shap_stability.py` | `analysis/part_b_shap_stability_results.json`. |
| `analysis/part_c_recipe_generation.py` | `recipes/inp_qd_candidate_recipes_v1.json`, `analysis/part_c_{method1,c32}_results.json`, candidate CSVs; `figures/figure_9`–`figure_11`. |
| `analysis/cossairt_ablation.py` | `analysis/cossairt_ablation_results.json`. |
| `analysis/time_forward_validation.py` | `analysis/time_forward_results.json`. |
| `analysis/feature_pair_audit.py` | `analysis/feature_pair_audit_results.json`. |

## Related resources

- **Dataset (Zenodo):** DOI [10.5281/zenodo.20137306](https://doi.org/10.5281/zenodo.20137306) — LLM-curated method-validation corpus.
- **External reference:** Nguyen et al., *Chemistry of Materials* **2022**, 34, 6296 (Cossairt group).
- **Companion methodology preprint (ChemRxiv):** Yoo, *"Beyond Numerical Hallucination: Context-Alignment Failures in LLM-Assisted Quantum Dot Synthesis Literature Curation,"* DOI [10.26434/chemrxiv.15003995/v1](https://doi.org/10.26434/chemrxiv.15003995/v1).

## License

MIT License — see [LICENSE](LICENSE). Copyright (c) 2026 Seungwon Yoo.
