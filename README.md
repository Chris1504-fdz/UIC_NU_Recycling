# Multi-Task Material Characterization

Design → property modeling of extruded recycled-polymer blends.
Goal: predict mechanical + rheological responses from processing inputs.

## Inputs (design variables)

| Input | Type | Values |
|---|---|---|
| Polymer composition — PP % / PAL % | mixture (sums to 100) | PAL 0 / 25 / 50 % |
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

## Data

- Raw: `raw_data/data_06_29_26/`
- Processed tables: `processed_data/tensile/tensile_summary.csv`, `processed_data/rheology/rheology_summary.csv`
- Regenerate: `jupyter nbconvert --to notebook --execute --inplace data_processing.ipynb`

## Next steps (from 08-06 meeting)

**Modeling**
- Check UTS ↔ toughness correlation before multi-objective opt (objectives should be ~uncorrelated).
- Primary: heteroscedastic-noise GP (category/input-dependent noise).
- Explore non-GP models with monotonic inductive bias for viscosity-vs-recycling degradation.
- Option: multi-source model (literature + experiment) via LVGP, source as categorical.
- Draft the design of experiments; hand to experiments for runs.

**Design of experiments**
- Temperature: 180–240 °C (4 levels).
- Composition: PAL 10–50 % (e.g. 10 / 30 / 50); drop pure 100/0 (pure polymer from literature).
- Recycle levels: 0 / 1 / 5 / 10× (no intermediate cycles).
- Polymers (≥3 for LVGP): PP+PAL, HDPE, TPU+PAL (nylon on hold).
- Rheology per polymer × composition × temperature.

**Incoming data (experiments)**
- 10× recycled + extra replicates.
- Redo basic shear-rate sweep (high-shear drop = plate-slip artifact).
- Add cleaner frequency + amplitude sweeps (storage / loss / complex viscosity).

**Direction**
- Multi-objective optimization + recyclability degradation model (max reuse cycles); later DSC/TGA.

## Notes

- Tensile = PP/PAL blends; rheology = recycled HDPE (different base material) — not yet shared; plan is to connect them.
- Rheology region of interest: shear rate **0.1–10 s⁻¹** (extrusion range); high-shear drop is an artifact, ignore.
