# Multi-Task Material Characterization — Data & Processing

This project organizes and processes raw experimental data for two material-characterization
studies that were delivered together on **2026-06-29** (`raw_data/data_06_29_26/`). The raw
delivery was a single Google-Drive zip of unprocessed instrument exports; this repo turns those
messy workbooks into clean, analysis-ready tables and overview plots.

> **Status / framing note:** The two studies below currently live side-by-side as **parallel
> sub-studies**. Whether they are meant to connect into one story (e.g. processability *and*
> mechanical performance of recycled blends) is still an open question for the team to decide —
> this README keeps them documented independently for now. See [Open questions](#open-questions).

---

## TL;DR — the two studies

| # | Study | Material | Measures | What you get | Raw workbook |
|---|-------|----------|----------|--------------|--------------|
| 1 | **Tensile** | PP / PAL blends (extruded) | Stress vs. strain | Young's modulus, ultimate tensile strength (UTS) | `Trial Data Analysis.xlsx` |
| 2 | **Rheology** | Recycled HDPE | Viscosity vs. shear rate | Flow curves, power-law fit | `recycled HDPE-PAL-viscosity.xlsx` |

- **Tensile:** 38 specimens across **12 configurations** (blend ratio × extrusion temperature), ~3 replicates each.
- **Rheology:** 9 flow-curve runs across **2 recycle levels** (`5x` = 5×-recycled, `1x` = shredded 3D-print stock).

---

## Quick start

Everything is reproducible from the raw zip:

```bash
# from the project root
jupyter notebook data_processing.ipynb     # run all cells, top to bottom
```

The notebook unzips the raw delivery, parses every sheet, writes tidy files into
`processed_data/`, and saves overview figures. Nothing is hand-edited — re-running regenerates
all outputs.

**Requirements:** Python with `pandas`, `numpy`, `matplotlib`, `openpyxl`, `jupyter`.

---

## Directory map

```
multi_task/
├── README.md                      ← you are here
├── data_processing.ipynb          ← the pipeline: raw → tidy → plots
├── raw_data/
│   └── data_06_29_26/
│       └── drive-download-*.zip   ← original delivery (source of truth, untouched)
└── processed_data/                ← all generated outputs
    ├── tensile/
    │   ├── curves/                  Config<N>_<rep>.csv   — one stress-strain curve per specimen (38 files)
    │   ├── tensile_summary.csv      per-specimen metrics + blend metadata
    │   └── tensile_summary_by_config.csv   replicate-averaged metrics per configuration
    ├── rheology/
    │   ├── curves/                  <sample>.csv          — one viscosity flow-curve per run (9 files)
    │   └── rheology_summary.csv     per-run metadata + power-law fit
    └── figures/                     overview PNGs (stress-strain, properties-vs-temp, flow curves, recycle comparison)
```

> The raw `~$*.xlsx` files inside the zip are Excel lock/temp files and are intentionally skipped.

---

## Concepts primer (plain language)

New to this data? Here's what the key quantities mean.

**Tensile test** — a machine pulls a specimen apart at steady speed and records force vs.
stretch. Converted to engineering units:

- **Stress (MPa)** = force ÷ cross-sectional area — "how hard the material is being pulled per unit area."
- **Strain (mm/mm)** = stretch ÷ original length — "how much it has elongated, as a fraction."
- **Stress–strain curve** — plot stress (y) vs. strain (x). Its shape tells you the material's mechanical behavior.
- **Young's modulus (MPa)** — the *slope* of the initial straight part of the curve. **Stiffness**: higher = harder to stretch elastically.
- **Ultimate / max tensile strength, UTS (MPa)** — the *highest* stress reached before failure. **Strength**: how much load it survives.
- **Strain at break** — how far it stretched before the test ended (a rough ductility proxy).

**Rheology / viscosity** — measures how a *molten* polymer flows when sheared:

- **Shear rate (1/s)** — how fast the material is being deformed (the rheometer sweeps this from low to high).
- **Shear stress (Pa)** — the resistance the material pushes back with.
- **Viscosity (Pa·s)** = shear stress ÷ shear rate — "thickness" of the melt. Higher = stiffer flow.
- **Shear-thinning** — viscosity *drops* as shear rate rises (normal for polymer melts). On a log-log plot it slopes downward.
- **Power-law fit** `viscosity = K · (shear rate)^(n−1)`:
  - **n** = flow-behavior index. `n = 1` is water-like (Newtonian); `n < 1` means shear-thinning — the **lower n, the more shear-thinning**.
  - **K** = consistency — overall viscosity level.

---

## Study 1 — Tensile (stress vs. strain)

### Experimental design

Each **configuration** is a unique combination of blend composition and extrusion temperature.
The mapping comes from the `Configuration_reference` sheet in the raw workbook:

| Config | PP % | PAL % | Extrusion temp (°C) |
|-------:|-----:|------:|--------------------:|
| 9.5 | 50 | 50 | 220 |
| 10  | 50 | 50 | 180 |
| 11  | 50 | 50 | 240 |
| 12  | 50 | 50 | 200 |
| 13  | 75 | 25 | 180 |
| 14  | 75 | 25 | 200 |
| 15  | 75 | 25 | 220 |
| 16  | 100 | 0 | 200 |
| 17  | 100 | 0 | 240 |
| 18  | 100 | 0 | 220 |
| 20  | 100 | 0 | 180 |
| 21  | 75 | 25 | 240 |

Specimen sheets are named `Config<N>_<rep>` (e.g. `Config10_2` = configuration 10, replicate 2).

### What the processing does

- Reads each specimen sheet, keeps the 5 measured columns, drops the messy header/unit rows.
- Extracts the **Young's Modulus** and **Max Tensile Stress** values (which the raw file stashed in stray side cells).
- Joins each specimen back to its blend composition and extrusion temperature.

### Output files & columns

`processed_data/tensile/curves/Config<N>_<rep>.csv` — one tidy curve per specimen:

| Column | Units | Meaning |
|--------|-------|---------|
| `time_s` | s | Elapsed test time |
| `extension_mm` | mm | Crosshead displacement |
| `load_N` | N | Measured force |
| `strain` | mm/mm | Tensile strain |
| `stress_MPa` | MPa | Tensile stress |

`processed_data/tensile/tensile_summary.csv` — one row per specimen:

| Column | Meaning |
|--------|---------|
| `config`, `replicate` | Specimen identity |
| `pp_pct`, `pal_pct`, `extr_temp_C` | Blend composition & extrusion temperature |
| `young_modulus_MPa` | Stiffness (from raw file) |
| `max_tensile_stress_MPa` | UTS (from raw file) |
| `strain_at_break` | Final strain in the record |
| `n_points`, `sheet` | Provenance |

`processed_data/tensile/tensile_summary_by_config.csv` — replicate-averaged means + std for each configuration.

---

## Study 2 — Rheology (viscosity vs. shear rate)

### The column-decoding story

The raw rheometer export (`recycled HDPE-PAL-viscosity.xlsx`) had **no header row** — just
unlabeled numeric columns, one sheet per run. We identified the physical columns by using the
defining relationship **viscosity = shear stress ÷ shear rate** and confirming it numerically
(e.g. `281.3 Pa ÷ 0.1 1/s = 2813 Pa·s`):

| Raw column index | Quantity | Units |
|-----------------:|----------|-------|
| 6 | Shear stress | Pa |
| 7 | Shear rate | 1/s |
| 8 | Viscosity | Pa·s |

Sheet-name prefixes encode the **recycle level**: `5x*` = 5×-recycled HDPE, `1x*` = 1× (shredded 3D-print stock).

### Output files & columns

`processed_data/rheology/curves/<sample>.csv` — one flow curve per run:

| Column | Units | Meaning |
|--------|-------|---------|
| `point` | – | Measurement index |
| `shear_stress_Pa` | Pa | Shear stress |
| `shear_rate_1_s` | 1/s | Shear rate |
| `viscosity_Pa_s` | Pa·s | Viscosity |

`processed_data/rheology/rheology_summary.csv` — one row per run, including the
`powerlaw_K` / `powerlaw_n` fit (over the well-behaved shear-rate region ≤ 10 1/s),
zero-shear-ish viscosity, and shear-rate range.

---

## Key observations (preliminary)

- **Stiffer with more PP:** Young's modulus rises with PP content (50/50 blends ≈ 700–1000 MPa, pure-PP configs ≈ 1500–2000 MPa).
- **Recycling increases shear-thinning:** the 5× material's flow-behavior index `n` (~0.6–0.8) sits below the 1× material's (~0.83–0.87) — repeated recycling makes the melt more shear-thinning.

---

## Known issues / caveats

- **Config16 is noisy** — 5 replicates with large scatter (modulus std ≈ 940 MPa). Replicate
  `Config16_2` (modulus ≈ 128 MPa, UTS ≈ 7.6 MPa vs. ~2300 / ~25 for siblings) looks like a
  **bad or slipped pull** and is a candidate to exclude.
- **High-shear rheology noise** — above ~10 1/s the flow curves get jumpy (likely instrument
  artifacts at high shear); the power-law fit deliberately uses the ≤ 10 1/s region only.
- **Different base materials** — Study 1 is PP/PAL, Study 2 is recycled HDPE. They are *not* the
  same specimens; don't merge them without a clear rationale.

---

## Open questions

For team / advisor discussion:

1. **Do the two studies connect** into one narrative, or stay independent? (Drives how we report.)
2. **Should `Config16_2` (and any other outliers) be dropped** before averaging?
3. **Is the ≤ 10 1/s cutoff** the right region for the rheology power-law fit, or do you want the full range?
4. **What downstream use** is intended — comparison tables, statistical modeling, ML surrogate? (Affects whether we need a single merged feature table.)
