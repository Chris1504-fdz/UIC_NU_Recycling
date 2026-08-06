"""Parse the rheometer workbook (headerless flow-curve exports)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RATE_LO, RATE_HI

# Physical columns identified via viscosity = shear_stress / shear_rate.
_STRESS_COL, _RATE_COL, _VISC_COL = 6, 7, 8


def recycle_level(sheet: str) -> str:
    """'5x' / '1x' from the sheet name prefix, else '?'."""
    s = sheet.lower()
    return "5x" if s.startswith("5x") else ("1x" if s.startswith("1x") else "?")


def parse_flow_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose first cell is a measurement index -> tidy (rate, stress, viscosity)."""
    idx = pd.to_numeric(df[0], errors="coerce")
    block = df[idx.notna()]
    out = pd.DataFrame({
        "point": pd.to_numeric(block[0], errors="coerce"),
        "shear_stress_Pa": pd.to_numeric(block[_STRESS_COL], errors="coerce"),
        "shear_rate_1_s": pd.to_numeric(block[_RATE_COL], errors="coerce"),
        "viscosity_Pa_s": pd.to_numeric(block[_VISC_COL], errors="coerce"),
    }).dropna(subset=["shear_rate_1_s", "viscosity_Pa_s"])
    return out[out["shear_rate_1_s"] > 0].sort_values("shear_rate_1_s").reset_index(drop=True)


def powerlaw_fit(curve: pd.DataFrame, max_rate: float = RATE_HI) -> tuple[float, float]:
    """Fit viscosity = K * rate^(n-1) over the well-behaved region. Returns (K, n)."""
    reg = curve[curve["shear_rate_1_s"] <= max_rate]
    if len(reg) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(np.log(reg["shear_rate_1_s"]), np.log(reg["viscosity_Pa_s"]), 1)
    return float(np.exp(intercept)), float(slope + 1)   # K, flow index n


def load_rheology(xlsx: Path, curves_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Parse every flow-curve sheet, write a tidy CSV each, return ``(curves, summary)``.

    ``curves`` maps sheet name -> ``(recycle_level, DataFrame)``.
    """
    xl = pd.ExcelFile(xlsx)
    curves, rows = {}, []
    for s in xl.sheet_names:
        curve = parse_flow_curve(pd.read_excel(xlsx, sheet_name=s, header=None))
        if curve.empty:
            continue
        lvl = recycle_level(s)
        safe = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
        curve.to_csv(curves_dir / f"{safe}.csv", index=False)
        curves[s] = (lvl, curve)
        K, n = powerlaw_fit(curve)
        rows.append({
            "sample": s, "recycle_level": lvl, "file": f"{safe}.csv", "n_points": len(curve),
            "shear_rate_min": curve["shear_rate_1_s"].min(),
            "shear_rate_max": curve["shear_rate_1_s"].max(),
            "viscosity_at_lowest_shear_Pa_s": curve["viscosity_Pa_s"].iloc[0],
            "powerlaw_K": K, "powerlaw_n": n,
        })
    summary = pd.DataFrame(rows).sort_values(["recycle_level", "sample"]).reset_index(drop=True)
    return curves, summary


def window_summary(curves: dict, lo: float = RATE_LO, hi: float = RATE_HI) -> pd.DataFrame:
    """Geometric-mean viscosity inside the extrusion window, per recycle level."""
    rows = []
    for lvl in ("1x", "5x"):
        members = [c for (l, c) in curves.values() if l == lvl]
        if not members:
            continue
        big = pd.concat(members)
        win = big[(big["shear_rate_1_s"] >= lo) & (big["shear_rate_1_s"] <= hi)]
        rows.append({
            "recycle_level": lvl, "n_runs": len(members),
            "geo_mean_viscosity_Pa_s": float(np.exp(np.log(win["viscosity_Pa_s"]).mean())),
            "min_Pa_s": win["viscosity_Pa_s"].min(), "max_Pa_s": win["viscosity_Pa_s"].max(),
        })
    return pd.DataFrame(rows)
