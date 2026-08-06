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

