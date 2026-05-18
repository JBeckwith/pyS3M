# SR_Functions Notebook Corrections — 2026-05-17

`fit_SM_data` and `fit_imaging_data` are now thin wrappers around `_fit_files`.
The wrappers remain in place so notebooks still work, but if the wrappers are
ever removed, every call site below must be updated.

---

## Mapping to `_fit_files`

| Old call | New call | Flags to add |
|---|---|---|
| `SupRes_F.fit_SM_data(...)` | `SupRes_F._fit_files(...)` | `accumulate_frame_numbers=False, combined_output=False` |
| `SupRes_F.fit_imaging_data(...)` | `SupRes_F._fit_files(...)` | `accumulate_frame_numbers=True, combined_output=True` |

All other arguments (`image_folder`, `smoothing_function`, calibration maps,
`pfa`, `ROI_size`, `peak_wavelength`, etc.) pass through unchanged.

---

## Notebook call sites

### `notebooks/hela_tubulin/All_Analysis_OneBook.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 11 | `SupRes_F.fit_SM_data(new_folder, ...)` | → `_fit_files(..., accumulate_frame_numbers=False, combined_output=False)` |
| 12 | `SupRes_F.fit_imaging_data(new_folder, ...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 13 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 14 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 15 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 17 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 20 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 22 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 24 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/hela_tubulin/HeLa_Tubulin_Analysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 5 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 6 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 7 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 8 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/saureus/SAureus_Analysis_RawAnalysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 5 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/asyn_aggregates/asyn_ThX_NR_Analysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 6 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/asyn_aggregates/asyn_ThX_NR_Analysis_Updated.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 10 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/asyn_aggregates/asyn_NR_Analysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 23 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/asyn_aggregates/20260211_asyn_NR_RawAnalysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 5 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/superres_dna_origami/DNA_Origami_Analysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 7 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/superres_dna_origami/DNA_Nanoruler_Analysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 6 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 7 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 8 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |
| 9 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/superres_dna_paint_cells/DNA_PAINT_Cells_Analysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 5 | `SupRes_F.fit_imaging_data(...)` | → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)` |

### `notebooks/dye_discrimination/Raw_Data_Analysis.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 5 | `SupRes_F.fit_SM_data(...)` | → `_fit_files(..., accumulate_frame_numbers=False, combined_output=False)` |

### `notebooks/dye_discrimination/DyeOnSlide_Analysis_btc.ipynb`

| Cell | Current call | Action |
|---|---|---|
| 18 | `SRes_F.fit_SM_data(image_folder, ...)` | → `_fit_files(..., accumulate_frame_numbers=False, combined_output=False)` |
| Note | Uses `SRes_F` not `SupRes_F` as the variable name | — |

### `notebooks/tracking/20260331_Dan_Track_Analysis.ipynb` and `notebooks/tracking/Example_Track_Analysis.ipynb`

Markdown cells only mention `fit_SM_data` in prose (no executable calls). Update
the text descriptions when the wrappers are removed.

---

## `src/Constants.py`

Line 247 has an example in a docstring/comment:
```python
sr.fit_SM_data(..., config=cfg)
```
Update to `sr._fit_files(..., config=cfg, accumulate_frame_numbers=False, combined_output=False)` if wrappers are removed.

---

## Summary

- **24 executable call sites** across 12 notebooks need updating.
- All `fit_imaging_data` calls → `_fit_files(..., accumulate_frame_numbers=True, combined_output=True)`.
- All `fit_SM_data` calls → `_fit_files(..., accumulate_frame_numbers=False, combined_output=False)`.
- The wrappers make this a non-urgent, batch update.
