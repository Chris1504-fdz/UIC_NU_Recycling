"""Parse the tensile workbook and derive per-specimen / per-configuration metrics."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TENSILE_COLS


def _clean(x):
    """Strip zero-width spaces / stray % and whitespace from a reference cell."""
    if pd.isna(x):
        return x
    return str(x).replace("​", "").replace("%", "").strip()


def load_config_reference(xlsx: Path) -> pd.DataFrame:
    """Return the Configuration_reference lookup: config -> PP%, PAL%, extrusion temp."""
    raw = pd.read_excel(xlsx, sheet_name="Configuration_reference", header=None)
    rows = []
    for _, r in raw.iterrows():
        cfg = _clean(r[0])
        if cfg in (None, "", "nan") or "Configuration" in str(r[0]):
            continue
        rows.append({"config": cfg, "pp_pct": float(_clean(r[1])),
                     "pal_pct": float(_clean(r[2])), "extr_temp_C": float(_clean(r[3]))})
    return pd.DataFrame(rows).sort_values("extr_temp_C").reset_index(drop=True)


def parse_specimen(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Return ``(tidy_curve, young_modulus_MPa, max_tensile_stress_MPa)`` for one Config* sheet.

    The two scalars are stashed in stray side cells (label cell, value in the row below).
    """
    ym = mts = np.nan
    arr = df.astype(str).values
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if "Modulus" in v:
                ym = pd.to_numeric(df.iat[i + 1, j], errors="coerce")
            elif "Max Tensile" in v:
                mts = pd.to_numeric(df.iat[i + 1, j], errors="coerce")

    curve = df.iloc[2:, :5].copy()          # row0 = names, row1 = units
    curve.columns = TENSILE_COLS
    curve = (curve.apply(pd.to_numeric, errors="coerce")
                  .dropna(how="all")
                  .dropna(subset=["strain", "stress_MPa"])
                  .reset_index(drop=True))
    return curve, float(ym), float(mts)


def toughness(curve: pd.DataFrame) -> float:
    """Area under the stress-strain curve (modulus of toughness), MJ/m^3."""
    if len(curve) < 2:
        return np.nan
    return float(np.trapz(curve["stress_MPa"].values, curve["strain"].values))


def load_tensile(xlsx: Path, curves_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Parse every Config* sheet, write a tidy curve CSV per specimen, and return
    ``(curves, raw_summary)`` (raw_summary has no composition metadata yet).
    """
    xl = pd.ExcelFile(xlsx)
    sheets = [s for s in xl.sheet_names
              if s.startswith("Config") and s != "Configuration_reference"]
    curves, rows = {}, []
    for s in sheets:
        m = re.match(r"Config([0-9.]+)_([0-9]+)", s)
        cfg, rep = m.group(1), m.group(2)
        curve, ym, mts = parse_specimen(pd.read_excel(xlsx, sheet_name=s, header=None))
        curves[s] = curve
        curve.to_csv(curves_dir / f"Config{cfg}_{rep}.csv", index=False)
        rows.append({
            "config": cfg, "replicate": int(rep), "sheet": s, "n_points": len(curve),
            "young_modulus_MPa": ym, "max_tensile_stress_MPa": mts,
            "strain_at_break": curve["strain"].iloc[-1] if len(curve) else np.nan,
            "toughness_MJ_m3": toughness(curve),
        })
    return curves, pd.DataFrame(rows)


def build_summary(raw_summary: pd.DataFrame, config_meta: pd.DataFrame) -> pd.DataFrame:
    """Join per-specimen metrics with the composition/temperature metadata."""
    cols = ["config", "replicate", "pp_pct", "pal_pct", "extr_temp_C",
            "young_modulus_MPa", "max_tensile_stress_MPa", "strain_at_break",
            "toughness_MJ_m3", "n_points", "sheet"]
    return (raw_summary.merge(config_meta, on="config", how="left")
            .sort_values(["extr_temp_C", "config", "replicate"])
            .reset_index(drop=True)[cols])


def by_config(summary: pd.DataFrame) -> pd.DataFrame:
    """Replicate-averaged mechanical properties per configuration."""
    return (summary.groupby(["config", "pp_pct", "pal_pct", "extr_temp_C"], as_index=False)
            .agg(n_reps=("replicate", "count"),
                 young_modulus_MPa_mean=("young_modulus_MPa", "mean"),
                 young_modulus_MPa_std=("young_modulus_MPa", "std"),
                 UTS_MPa_mean=("max_tensile_stress_MPa", "mean"),
                 UTS_MPa_std=("max_tensile_stress_MPa", "std"),
                 toughness_MJ_m3_mean=("toughness_MJ_m3", "mean"),
                 toughness_MJ_m3_std=("toughness_MJ_m3", "std"))
            .sort_values("config").reset_index(drop=True))


def _cv(x: pd.Series) -> float:
    m = x.mean()
    return 100 * x.std(ddof=1) / m if m else np.nan


def cv_by_config(summary: pd.DataFrame) -> pd.DataFrame:
    """DoE-style table: mean UTS / toughness + coefficient of variation per config."""
    t = (summary.groupby(["pp_pct", "pal_pct", "extr_temp_C"], as_index=False)
         .agg(n=("replicate", "count"),
              CV_modulus_pct=("young_modulus_MPa", _cv),
              UTS_mean=("max_tensile_stress_MPa", "mean"),
              CV_UTS_pct=("max_tensile_stress_MPa", _cv),
              tough_mean=("toughness_MJ_m3", "mean"),
              CV_toughness_pct=("toughness_MJ_m3", _cv))
         .sort_values(["pp_pct", "extr_temp_C"]).reset_index(drop=True))
    t.columns = ["PP %", "PAL %", "Extr. temp (°C)", "n", "CV Young's modulus (%)",
                 "UTS mean (MPa)", "CV UTS (%)", "Toughness mean (MJ/m³)", "CV toughness (%)"]
    return t


def objective_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix among UTS, modulus, and toughness."""
    props = ["max_tensile_stress_MPa", "young_modulus_MPa", "toughness_MJ_m3"]
    labels = {"max_tensile_stress_MPa": "UTS", "young_modulus_MPa": "Modulus",
              "toughness_MJ_m3": "Toughness"}
    return summary.dropna(subset=props)[props].corr().rename(index=labels, columns=labels)


def uts_toughness_stats(summary: pd.DataFrame) -> dict:
    """Pearson r / Spearman rho between UTS and toughness."""
    from scipy import stats
    d = summary.dropna(subset=["max_tensile_stress_MPa", "toughness_MJ_m3"])
    r, p = stats.pearsonr(d["max_tensile_stress_MPa"], d["toughness_MJ_m3"])
    rho, _ = stats.spearmanr(d["max_tensile_stress_MPa"], d["toughness_MJ_m3"])
    return {"n": len(d), "pearson_r": float(r), "pearson_p": float(p), "spearman_rho": float(rho)}


def design_space(summary: pd.DataFrame) -> pd.DataFrame:
    """Per (PAL%, temperature) cell means for the design-space views."""
    return (summary.groupby(["pal_pct", "extr_temp_C"], as_index=False)
            .agg(max_stress_mean=("max_tensile_stress_MPa", "mean"),
                 max_stress_std=("max_tensile_stress_MPa", "std"),
                 modulus_mean=("young_modulus_MPa", "mean"),
                 toughness_mean=("toughness_MJ_m3", "mean"),
                 toughness_std=("toughness_MJ_m3", "std"),
                 n=("replicate", "count")))


def grid(design: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot a design-space column to a PAL% (rows, high->low) x temperature (cols) grid."""
    temps = sorted(design["extr_temp_C"].unique())
    pals = sorted(design["pal_pct"].unique(), reverse=True)
    return design.pivot(index="pal_pct", columns="extr_temp_C", values=value).reindex(index=pals, columns=temps)
