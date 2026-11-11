"""Parse Instron (Bluehill) raw-data CSV exports and derive tensile metrics.

Used for the 2026-09-03 DoE delivery: one folder per formulation
(``nu_samples-f<NN>[_<k>].is_tens_RawData``) holding one ``Specimen_RawData_<i>.csv``
per replicate. Each CSV is a short metadata block (gauge length, diameter, area, length),
a blank line, a header row, a units row, then the time series
``Time, Extension, Load, Tensile strain (Extension), Tensile stress``.

Scalar metrics are *derived here* from the raw curve (the earlier xlsx delivery shipped
them pre-computed by the instrument software):

* **UTS** - peak stress.
* **Young's modulus** - least-squares chord slope over the first rise through
  ``MODULUS_WINDOW`` (10-40 % of UTS), i.e. above the toe and below yield.
* **Toe offset** - strain where that chord crosses zero stress; subtracted from the
  reported strains (ASTM D638-style toe compensation).
* **Break** - first post-peak sample whose stress drops below ``BREAK_FRAC`` x UTS
  (the machine keeps logging after fracture, so the last row is *not* the break).
* **Toughness** - area under the (stress >= 0) curve up to the break, MJ/m^3.
"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TENSILE_COLS

FOLDER_RE = re.compile(r"nu_samples-f(\d+)(?:r(\d+))?(?:-([a-z]+))?(?:_(\d+))?\.is_tens_RawData$", re.I)
FILE_RE = re.compile(r"Specimen_RawData_(\d+)\.csv$", re.I)
RECYCLED_RE = re.compile(r"(\d+)\s*x\s*recycled", re.I)   # lab's top folder, e.g. "0x recycled"

MODULUS_WINDOW = (0.10, 0.40)   # fraction of UTS bounding the chord fit
BREAK_FRAC = 0.05               # fraction of UTS below which (post-peak) = fractured

_META_KEYS = {
    "Strain : Tensile strain (Extension) gauge length": "gauge_length_mm",
    "Specimen properties : Diameter": "diameter_mm",
    "Specimen properties : Area": "area_mm2",
    "Specimen properties : Length": "length_mm",
}
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --- Inventory ---------------------------------------------------------------

def catalogue(extract_dir: Path, zip_path: Path | None = None) -> pd.DataFrame:
    """One row per raw specimen CSV found under ``extract_dir``.

    ``session`` is 0 for the main ``f<NN>`` folder and *k* for a follow-up ``f<NN>_k``
    folder (extra specimens run later in the day). ``batch`` is a tag after the number, e.g.
    ``new`` for ``f08-new`` (material remade), else empty. ``recycled_x`` is the number of
    recycling loops read from a parent folder named ``<N>x recycled`` (NaN when absent, as in
    the 2026-09 delivery) or else from an ``R<N>`` tag in the folder name (``F01R1``); that tag also goes in
    ``specimen_id`` (``f01r1_s1``), so recycled specimens never collide with cycle 0. If ``zip_path`` is given, the archive's per-file timestamps are
    attached as ``exported_at`` (one stamp per folder = when Bluehill exported that folder; it
    orders the sessions but is not the exact test time). Folders holding specimen CSVs that do
    not match the naming are reported, not silently dropped.
    """
    stamps = {}
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as zf:
            for zi in zf.infolist():
                stamps[Path(zi.filename).as_posix()] = datetime(*zi.date_time)

    rows, skipped = [], set()
    for f in sorted(Path(extract_dir).rglob("Specimen_RawData_*.csv")):
        m = FOLDER_RE.search(f.parent.name)
        if not m:
            skipped.add(f.parent.name)
            continue
        form, tag, batch, session = int(m.group(1)), m.group(2), (m.group(3) or "").lower(), int(m.group(4) or 0)
        spec = int(FILE_RE.search(f.name).group(1))
        sid = (f"f{form:02d}" + (f"r{int(tag)}" if tag else "") + (f"-{batch}" if batch else "")
               + (f"_{session}" if session else "") + f"_s{spec}")
        rel = f.relative_to(extract_dir).as_posix()
        rec = next((int(r.group(1)) for part in f.relative_to(extract_dir).parts[:-2]
                    if (r := RECYCLED_RE.search(part))), int(tag) if tag else np.nan)
        rows.append({"specimen_id": sid, "formulation": form, "batch": batch, "session": session,
                     "specimen": spec, "recycled_x": rec, "folder": f.parent.name, "file": f.name,
                     "exported_at": stamps.get(rel, pd.NaT), "path": f})
    if skipped:
        import warnings
        warnings.warn(f"specimen folders not matching the naming, skipped: {sorted(skipped)}")
    return (pd.DataFrame(rows).sort_values(["formulation", "recycled_x", "batch", "session", "specimen"])
            .reset_index(drop=True))


# --- One specimen ------------------------------------------------------------

def read_specimen(path: Path) -> tuple[pd.DataFrame, dict]:
    """Return ``(tidy_curve, meta)`` for one raw CSV. Curve columns = ``TENSILE_COLS``."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Time,"))
    meta = {v: np.nan for v in _META_KEYS.values()}
    for l in lines[:hdr]:
        m = re.match(r'^(.*?),"([^"]*)",(.*)$', l)
        if m and m.group(1).strip() in _META_KEYS:
            meta[_META_KEYS[m.group(1).strip()]] = float(m.group(2))
    curve = pd.read_csv(io.StringIO("\n".join(lines[hdr + 2:])), header=None, names=TENSILE_COLS)
    curve = (curve.apply(pd.to_numeric, errors="coerce")
                  .dropna(subset=["strain", "stress_MPa"]).reset_index(drop=True))
    return curve, meta


def metrics(curve: pd.DataFrame, window=MODULUS_WINDOW, break_frac=BREAK_FRAC) -> dict:
    """Derive the scalar tensile metrics from one tidy curve (see module docstring)."""
    e = curve["strain"].to_numpy(float)
    s = curve["stress_MPa"].to_numpy(float)
    ipk = int(np.argmax(s))
    uts = float(s[ipk])

    # Break: first post-peak sample below break_frac * UTS (exclusive end index).
    post = np.flatnonzero(s[ipk:] < break_frac * uts)
    ibrk = ipk + int(post[0]) if post.size else len(s)

    # Modulus: chord over the first rise through [lo, hi] * UTS (pre-peak).
    pre = s[:ipk + 1]
    i_lo = int(np.argmax(pre >= window[0] * uts))
    i_hi = int(np.argmax(pre >= window[1] * uts))
    i_hi = max(i_hi, i_lo + 2)
    x, y = e[i_lo:i_hi + 1], s[i_lo:i_hi + 1]
    E, b = np.polyfit(x, y, 1)
    ss_res = float(((y - (E * x + b)) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    toe = float(-b / E) if E else 0.0

    e_brk = float(e[ibrk - 1])
    tough = float(_trapz(np.clip(s[:ibrk], 0, None), e[:ibrk]))
    return {
        "young_modulus_MPa": float(E), "modulus_r2": r2, "modulus_fit_points": int(i_hi - i_lo + 1),
        "toe_offset_strain": toe,
        "max_tensile_stress_MPa": uts, "strain_at_uts": float(e[ipk]) - toe,
        "strain_at_break": e_brk - toe, "strain_at_break_raw": e_brk,
        "toughness_MJ_m3": tough,
        "n_points": int(len(s)), "n_points_to_break": int(ibrk),
        "i_modulus_lo": i_lo, "i_modulus_hi": i_hi, "i_peak": ipk, "i_break": ibrk,
    }


# --- Whole delivery ----------------------------------------------------------

def load_delivery(cat: pd.DataFrame, curves_dir: Path | None = None) -> tuple[dict, pd.DataFrame]:
    """Parse every catalogued CSV and return ``(curves, summary)``; ``curves`` maps
    specimen_id -> tidy DataFrame. With ``curves_dir`` given, a tidy curve CSV per specimen
    (``<specimen_id>.csv``) is also written there.
    """
    curves, rows = {}, []
    if curves_dir is not None:
        Path(curves_dir).mkdir(parents=True, exist_ok=True)
    for _, r in cat.iterrows():
        curve, meta = read_specimen(r["path"])
        curves[r["specimen_id"]] = curve
        if curves_dir is not None:
            curve.to_csv(Path(curves_dir) / f"{r['specimen_id']}.csv", index=False)
        dt = curve["time_s"].iloc[-1] - curve["time_s"].iloc[0]
        rate = (curve["extension_mm"].iloc[-1] - curve["extension_mm"].iloc[0]) / dt if dt else np.nan
        rows.append({**r.drop("path").to_dict(), **meta,
                     "crosshead_rate_mm_min": 60 * rate, **metrics(curve)})
    return curves, pd.DataFrame(rows)


def attach_doe(summary: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """Join the DoE formulation factors onto the per-specimen table.

    Column names follow the project's processed-tensile schema (``pal_pct``,
    ``extr_temp_C``). ``recycle_level`` is added but left empty: the delivery does not
    encode which stage of the 1/5/10x degradation series these specimens are.
    """
    d = design.rename(columns={"PAL_pct": "pal_pct", "extrusion_temp_C": "extr_temp_C"})
    out = summary.merge(d, on="formulation", how="left")
    out["recycle_level"] = pd.array([pd.NA] * len(out), dtype="Int64")
    front = ["specimen_id", "formulation", "polymer_system", "pal_pct", "extr_temp_C",
             "recycle_level", "session", "specimen",
             "diameter_mm", "area_mm2", "gauge_length_mm", "crosshead_rate_mm_min",
             "young_modulus_MPa", "max_tensile_stress_MPa", "strain_at_uts",
             "strain_at_break", "toughness_MJ_m3"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest].sort_values(["formulation", "session", "specimen"]).reset_index(drop=True)


# --- Aggregates --------------------------------------------------------------

_KEYS = ["formulation", "polymer_system", "pal_pct", "extr_temp_C"]


def by_formulation(summary: pd.DataFrame) -> pd.DataFrame:
    """Replicate-averaged properties per formulation (mean, std, n)."""
    return (summary.groupby(_KEYS, as_index=False)
            .agg(n_specimens=("specimen_id", "count"),
                 n_sessions=("session", "nunique"),
                 diameter_mm_mean=("diameter_mm", "mean"),
                 young_modulus_MPa_mean=("young_modulus_MPa", "mean"),
                 young_modulus_MPa_std=("young_modulus_MPa", "std"),
                 UTS_MPa_mean=("max_tensile_stress_MPa", "mean"),
                 UTS_MPa_std=("max_tensile_stress_MPa", "std"),
                 strain_at_break_mean=("strain_at_break", "mean"),
                 strain_at_break_std=("strain_at_break", "std"),
                 toughness_MJ_m3_mean=("toughness_MJ_m3", "mean"),
                 toughness_MJ_m3_std=("toughness_MJ_m3", "std"))
            .sort_values("formulation").reset_index(drop=True))


def _cv(x: pd.Series) -> float:
    m = x.mean()
    return 100 * x.std(ddof=1) / m if m else np.nan


def cv_by_formulation(summary: pd.DataFrame) -> pd.DataFrame:
    """Reproducibility table (mean + coefficient of variation) per formulation."""
    t = (summary.groupby(_KEYS, as_index=False)
         .agg(n=("specimen_id", "count"),
              modulus_mean=("young_modulus_MPa", "mean"),
              CV_modulus_pct=("young_modulus_MPa", _cv),
              UTS_mean=("max_tensile_stress_MPa", "mean"),
              CV_UTS_pct=("max_tensile_stress_MPa", _cv),
              tough_mean=("toughness_MJ_m3", "mean"),
              CV_toughness_pct=("toughness_MJ_m3", _cv))
         .sort_values("formulation").reset_index(drop=True))
    t.columns = ["Formulation", "Polymer", "PAL %", "Extr. temp (°C)", "n",
                 "Young's modulus mean (MPa)", "CV Young's modulus (%)", "UTS mean (MPa)", "CV UTS (%)",
                 "Toughness mean (MJ/m³)", "CV toughness (%)"]
    return t


# --- Received vs. planned ----------------------------------------------------

def coverage(design: pd.DataFrame, cat: pd.DataFrame, min_reps: int = 3) -> pd.DataFrame:
    """Per planned formulation: how many specimens / sessions arrived, and status."""
    got = (cat.groupby("formulation")
           .agg(n_specimens=("specimen_id", "count"),
                n_main=("session", lambda s: int((s == 0).sum())),
                n_followup=("session", lambda s: int((s != 0).sum())),
                first_export=("exported_at", "min"), last_export=("exported_at", "max"))
           .reset_index())
    cov = design.merge(got, on="formulation", how="left")
    for c in ("n_specimens", "n_main", "n_followup"):
        cov[c] = cov[c].fillna(0).astype(int)
    cov["status"] = np.where(cov["n_specimens"] == 0, "missing",
                     np.where(cov["n_specimens"] < min_reps, "received (< %d reps)" % min_reps,
                              "received"))
    extra = sorted(set(cat["formulation"]) - set(design["formulation"]))
    if extra:
        raise ValueError(f"delivery contains formulations not in the DoE: {extra}")
    return cov


def compare_to_plan(design: pd.DataFrame, plan: pd.DataFrame, cov: pd.DataFrame,
                    cat: pd.DataFrame) -> dict:
    """Headline numbers for the provided-vs-received comparison."""
    received = cov[cov["n_specimens"] > 0]
    missing = cov[cov["n_specimens"] == 0]
    return {
        "planned_formulations": int(len(design)),
        "received_formulations": int(len(received)),
        "missing_formulations": missing["formulation"].tolist(),
        "missing_by_polymer": missing["polymer_system"].value_counts().to_dict(),
        "planned_polymer_balance": design["polymer_system"].value_counts().to_dict(),
        "received_polymer_balance": received["polymer_system"].value_counts().to_dict(),
        "planned_recycle_levels": sorted(plan["recycle_level"].unique().tolist()),
        "received_recycle_levels": "not encoded in delivery",
        "planned_measurements": int(len(plan)),
        "received_specimens": int(len(cat)),
        "formulations_with_ge3_reps": int((cov["n_specimens"] >= 3).sum()),
        "formulations_with_followup_session": cov.loc[cov["n_followup"] > 0, "formulation"].tolist(),
        "pal_range_planned": (int(design["PAL_pct"].min()), int(design["PAL_pct"].max())),
        "pal_range_received": (int(received["PAL_pct"].min()), int(received["PAL_pct"].max())),
        "temp_range_planned": (int(design["extrusion_temp_C"].min()), int(design["extrusion_temp_C"].max())),
        "temp_range_received": (int(received["extrusion_temp_C"].min()), int(received["extrusion_temp_C"].max())),
    }
