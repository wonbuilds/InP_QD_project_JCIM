# Provenance and Licensing of Third-Party (Cossairt) Data

This repository redistributes, in modified form, experimental indium phosphide (InP)
quantum-dot synthesis data originally published by the Cossairt Laboratory.

## Source
- **Upstream repository:** Cossairt-Lab/Indium-Phosphide —
  https://github.com/Cossairt-Lab/Indium-Phosphide
- **License:** MIT License (Copyright (c) 2021 Cossairt-Lab). The upstream license is
  reproduced verbatim in `data/from_cossairt/LICENSE-upstream.txt`.
- **Associated publication:** Nguyen, H. A.; Dou, F. Y.; Park, N.; Wu, S.; Sarsito, H.;
  Diakubama, B.; Larson, H.; Nishiwaki, E.; Homer, M.; Cash, M.; Cossairt, B. M.
  *Chem. Mater.* 2022. (Use the exact DOI/volume/pages from this manuscript's Nguyen 2022
  reference entry.)

## Files derived from this source (under `data/from_cossairt/`)
- `dataset_InP_raw.csv`
- `hao_dataset.csv`
- `flo_dataset.csv`
- `cossairt_reference_master_paper_level.csv`
- `cossairt_ref_values_dd_matched.csv`

## Modifications relative to the upstream originals
The deposited files are **privacy-cleaned derivatives** and are **not byte-for-byte
identical** to the upstream originals. The following modifications were applied:

1. **Removed the `user` column** from `dataset_InP_raw.csv` and `hao_dataset.csv`. This
   column recorded the individual laboratory member associated with each synthesis record.
   It is not used as a feature in any analysis here; it was removed to avoid record-level
   attribution to named individuals.
2. **Removed one author-synthesis entry that has no literature DOI** (the row whose `doi`
   field is the sentinel value `nayon`) from `dataset_InP_raw.csv` and `hao_dataset.csv`.
   The corresponding paper-level entry in `cossairt_reference_master_paper_level.csv`
   (`cossairt_ref_id = COS-073`) is retained as a documented placeholder, but its
   unpublished synthesis parameters and emission value have been redacted.

These modifications **do not affect any reported result**: every analysis in this
repository already excludes the `nayon` entry (the code filters `doi != "nayon"`), and the
`user` column is never used as a model input. All reported numbers reproduce identically
from the deposited (cleaned) data, as verified by `reproduce_all.py --strict`.

The **unmodified upstream originals** remain available at the source repository URL above.

## Licensing scope of this repository
- Original **code and analysis** authored here are licensed under the MIT License
  (Copyright (c) 2026 Seungwon Yoo); see `LICENSE`.
- The **third-party data** under `data/from_cossairt/` is Copyright (c) 2021 Cossairt-Lab,
  redistributed under its original MIT License (see `data/from_cossairt/LICENSE-upstream.txt`).
  It is not the author's to relicense; the repository-level `LICENSE` does not extend to it.
