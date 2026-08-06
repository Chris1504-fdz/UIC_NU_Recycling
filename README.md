# Multi-Task Material Characterization

Design → property modeling of extruded recycled-polymer blends.
Goal: predict mechanical + rheological responses from processing inputs.

## Inputs (design variables)

| Input | Type | Values |
|---|---|---|
| Polymer composition PP % / PAL % | mixture (sums to 100) | PAL 0 / 25 / 50 % |
| Extrusion temperature | continuous | 180 / 200 / 220 / 240 °C |
| Recycle level *(rheology)* | categorical | 1× / 5× |
| Shear rate *(rheology test condition)* | continuous | 0.1 – 100 s⁻¹ |

## Outputs (responses to predict)

| Output | Units | Column |
|---|---|---|
| UTS (strength) | MPa | `max_tensile_stress_MPa` |
| Young's modulus (stiffness) | MPa | `young_modulus_MPa` |
| Toughness | MJ/m³ | `toughness_MJ_m3` |
| Viscosity / flow curve | Pa·s | `viscosity_Pa_s` (or `powerlaw_K`, `powerlaw_n`) |

## Data & code

- Raw: `raw_data/data_06_29_26/`
- Processed tables: `processed_data/tensile/tensile_summary.csv`, `processed_data/rheology/rheology_summary.csv`
- Proposed experiments: `doe/doe_formulations.csv` (LHS) + `doe/doe_measurement_plan.csv` (× recycle series)
- Logic: `utils/` (`config`, `io`, `tensile`, `rheology`, `doe`, `plots`)
- Clean driver: `notebooks/data_processing_v2.ipynb` (utils-based) — regenerate with
  `jupyter nbconvert --to notebook --execute --inplace notebooks/data_processing_v2.ipynb`
- Original monolithic: `notebooks/data_processing.ipynb`


**Design of experiments**
- Temperature: 180–240 °C (4 levels).
- Composition: PAL 10–50 % 
- Recycle levels: 0 / 1 / 5 / 10× (no intermediate cycles).
- Polymers (≥3 for LVGP): PP+PAL, HDPE+PAL, TPU+PAL (maybe nylon).

