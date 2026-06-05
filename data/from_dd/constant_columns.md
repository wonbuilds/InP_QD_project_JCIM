# Constant columns in `inp_subset.csv`

_Auto-generated from `data/from_dd/inp_subset.csv` (132 rows); regenerable by selecting columns with `nunique() == 1`. Documentation only — no analysis value depends on this file._

Of the 28 columns in the InP subset, **4** are constant across all 132/132 records:

| Column | Constant value | Kind |
| --- | --- | --- |
| `corpus` | `InP (L-series)` (on 132/132) | constant by construction (selection key: `corpus == "InP (L-series)"`) |
| `data_quality_tier` | `literature_structured` (on 132/132) | incidental constant |
| `measurement_state` | `final_purified` (on 132/132) | incidental constant |
| `shell_present_flag` | `1` (on 132/132) | incidental constant |

None of these constant columns is a model feature. The Part B / Part C feature set is `T_growth_C`, `time_min`, `outer_is_ZnS`, `shell_layer_count`, `route`, `shell_innermost`, `shell_composition_tier`; the columns above are provenance/quality flags carried for transparency and are excluded from all modeling.

