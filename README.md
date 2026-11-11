# Material Characterization

Design  property modeling of extruded recycled-polymer blends.

## Inputs (design variables)

| Input | Type | Values |
|---|---|---|
| Polymer composition PP % / PAL % | mixture (sums to 100) | PAL 0 / 25 / 50 % |
| Extrusion temperature | continuous | 180 / 200 / 220 / 240 °C |
| Shear rate *(rheology test condition)* | continuous | 0.1 – 100 s⁻¹ |

## Outputs (responses to predict)

| Output | Units | Column |
|---|---|---|
| UTS (strength) | MPa | `max_tensile_stress_MPa` |
| Young's modulus (stiffness) | MPa | `young_modulus_MPa` |
| Toughness | MJ/m³ | `toughness_MJ_m3` |
| Viscosity / flow curve | Pa·s | `viscosity_Pa_s` (or `powerlaw_K`, `powerlaw_n`) |

**Design of experiments**
- Temperature: 180–240 °C.
- Composition: PAL 10–50 % 
- Recycle levels: 1 / 5 / 10× (virgin 0× baseline separate).
- Polymers: PP+PAL, HDPE+PAL, TPU+PAL (maybe nylon).

## Data deliveries

| Received | Content | Notebook | Outputs |
|---|---|---|---|
| 2026-06-29 | Preliminary: tensile (PP/PAL, xlsx) + rheology (recycled HDPE) | `notebooks/data_processing_v2.ipynb` | `processed_data/tensile/`, `processed_data/rheology/` |
| 2026-09-11 (tested 09-03) | **DoE batch 1, tensile**: Instron raw CSVs, 30 specimens, 9 of 12 formulations (TPU f05/f07/f11 missing, f08 single specimen; recycle level not encoded) | `notebooks/data_processing_v3.ipynb` | `processed_data/tensile_doe/`, `processed_data/figures/doe_tensile_*.png` |

Utilities live in `utils/` (`instron.py` parses the Bluehill CSV exports and compares the delivery with the DoE).

`notebooks/doe_augmentation.ipynb` (2026-09-17) explores adding the recycle level as a design factor without
dropping the 12 requested formulations, which stay fixed at 1x. Option A re-uses them at 5x/10x (exchange search,
`doe.augment_plan`); option B adds new formulations as an augmented Latin hypercube over PAL x temperature x recycle
5-10x, optimised by column permutations that keep the Latin structure (`doe.augment_lhs`). Proposal only — the
handed-off plan is unchanged until agreed with the lab.

`data_processing_v3.ipynb` and `doe_augmentation.ipynb` write nothing to `processed_data/` (or `doe/`) on their own: each cell that can save a figure or
table starts with a switch and a file name (`SAVE_FIG` / `FIG_NAME`, `SAVE_CSV` / `CSV_NAME`, `SAVE_XLSX` / `XLSX_NAME`,
`SAVE_CURVES` / `CURVES_DIR`) — set the switch to `True` to write that file under the chosen name.
