"""Plotting helpers. Each returns a Matplotlib Figure (and optionally saves a PNG).

Encoding choices follow the data's job: sequential perceptually-uniform maps for
magnitudes (magma / cividis / viridis), a diverging map for correlation (RdBu_r).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from .config import RATE_LO, RATE_HI


def _save(fig, save: Path | None):
    if save is not None:
        fig.savefig(save, dpi=150)
    # Close so the inline backend doesn't ALSO auto-display it (avoids double render);
    # the returned Figure still renders once as the cell's result.
    plt.close(fig)
    return fig


# --- Tensile -----------------------------------------------------------------

def style_cv(cv: pd.DataFrame):
    """Return a Styler of the CV table with the CV columns shaded (red = noisier)."""
    cv_cols = [c for c in ["CV Young's modulus (%)", "CV UTS (%)", "CV toughness (%)"] if c in cv.columns]
    fmt = {c: "{:.1f}" for c in cv_cols}
    fmt.update({"PP %": "{:.0f}", "PAL %": "{:.0f}", "Extr. temp (°C)": "{:.0f}",
                "Young's modulus mean (MPa)": "{:.0f}", "UTS mean (MPa)": "{:.2f}",
                "Toughness mean (MJ/m³)": "{:.2f}"})
    fmt = {k: v for k, v in fmt.items() if k in cv.columns}   # keys differ per delivery
    return (cv.style.format(fmt)
            .background_gradient(cmap="Reds", subset=cv_cols)
            .set_caption("Mean + coefficient of variation (std/mean) across replicates "
                         "— higher CV = less reproducible"))


def curve_anatomy(curves: dict, summary: pd.DataFrame, pick: str | None = None,
                  save: Path | None = None):
    """Annotated single curve showing what's extracted: modulus, UTS, toughness, break."""
    if pick is None or pick not in curves:
        pick = "Config20_1" if "Config20_1" in curves else next(iter(curves))
    c = curves[pick].reset_index(drop=True)
    row = summary.set_index("sheet").loc[pick]
    E, uts, tough = (float(row["young_modulus_MPa"]), float(row["max_tensile_stress_MPa"]),
                     float(row["toughness_MJ_m3"]))
    ipk = int(c["stress_MPa"].idxmax())
    pos = c[c["stress_MPa"] > 0]
    strain0 = float(pos["strain"].iloc[0]) if len(pos) else 0.0

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(c["strain"], c["stress_MPa"], color="C0", lw=2, zorder=3, label="stress–strain curve")
    ax.fill_between(c["strain"], c["stress_MPa"], 0, where=(c["stress_MPa"] > 0),
                    color="C0", alpha=0.15, zorder=1,
                    label=f"area under curve = toughness ≈ {tough:.2f} MJ/m³")
    ax.scatter([c.loc[ipk, "strain"]], [c.loc[ipk, "stress_MPa"]], color="C3", s=55, zorder=5)
    ax.annotate(f"UTS ≈ {uts:.1f} MPa", (c.loc[ipk, "strain"], c.loc[ipk, "stress_MPa"]),
                textcoords="offset points", xytext=(10, -6), color="C3", fontsize=9)
    xt = np.array([strain0, strain0 + 1.1 * uts / E])
    ax.plot(xt, E * (xt - strain0), color="C2", ls="--", lw=1.8,
            label=f"initial slope = Young's modulus ≈ {E:.0f} MPa")
    xb = float(c["strain"].iloc[-1])
    ax.axvline(xb, color="grey", ls=":", lw=1.3, label=f"strain at break ≈ {xb:.2f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Tensile strain (mm/mm)"); ax.set_ylabel("Tensile stress (MPa)")
    ax.set_title(f"What we extract from each stress–strain curve  ({pick})")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save)


def stress_strain(curves: dict, summary: pd.DataFrame, config_meta: pd.DataFrame,
                  save: Path | None = None):
    """One representative curve per configuration, colored by extrusion temperature."""
    si = summary.set_index("sheet")
    order = sorted(curves, key=lambda s: si.loc[s, "config"])
    temps = config_meta.set_index("config")["extr_temp_C"]
    tmin, tmax = temps.min(), temps.max()
    cmap = plt.cm.viridis
    seen = set()
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in order:
        row = si.loc[s]
        if row["config"] in seen:
            continue
        seen.add(row["config"])
        c = curves[s]
        color = cmap((row["extr_temp_C"] - tmin) / (tmax - tmin + 1e-9))
        ax.plot(c["strain"], c["stress_MPa"], color=color, lw=1.4,
                label=f"Cfg {row['config']} ({int(row['pp_pct'])}/{int(row['pal_pct'])} PP/PAL, {int(row['extr_temp_C'])}C)")
    ax.set_xlabel("Tensile strain (mm/mm)"); ax.set_ylabel("Tensile stress (MPa)")
    ax.set_title("Stress–strain curves by configuration")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save)


def props_vs_temp(summary: pd.DataFrame, save: Path | None = None):
    """Young's modulus and UTS vs extrusion temperature, grouped by PP%."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, lab, title in [
        (axes[0], "young_modulus_MPa", "Young's modulus (MPa)", "Stiffness vs. extrusion temperature"),
        (axes[1], "max_tensile_stress_MPa", "Max tensile stress / UTS (MPa)", "Strength vs. extrusion temperature"),
    ]:
        for pp, grp in summary.groupby("pp_pct"):
            ax.scatter(grp["extr_temp_C"], grp[col], s=40, alpha=0.6, label=f"{int(pp)}% PP")
        means = summary.groupby("extr_temp_C")[col].mean()
        ax.plot(means.index, means.values, "k--", lw=1, marker="o", label="temp mean")
        ax.set_xlabel("Extrusion temperature (C)"); ax.set_ylabel(lab)
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, save)


def heatmap(grid: pd.DataFrame, title: str, cbar_label: str, cmap: str = "magma",
            fmt: str = ".1f", save: Path | None = None):
    """Annotated PAL% x temperature heatmap from a pivoted grid."""
    temps, pals, M = list(grid.columns), list(grid.index), grid.values
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(temps)), [f"{int(t)}" for t in temps])
    ax.set_yticks(range(len(pals)), [f"{int(p)}%" for p in pals])
    ax.set_xlabel("Extrusion temperature (°C)"); ax.set_ylabel("PAL content")
    ax.set_title(title)
    mid = (np.nanmin(M) + np.nanmax(M)) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=10,
                        color="white" if v < mid else "black")
    fig.colorbar(im, ax=ax).set_label(cbar_label)
    fig.tight_layout()
    return _save(fig, save)


def maxstress_3d(summary: pd.DataFrame, grid_stress: pd.DataFrame, save: Path | None = None):
    """Mean-strength surface over the design grid with specimens overlaid."""
    pals, temps = list(grid_stress.index), list(grid_stress.columns)
    Tg, Pg = np.meshgrid(temps, pals)
    Zg = grid_stress.values
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Tg, Pg, Zg, cmap="magma", alpha=0.55, edgecolor="none",
                    vmin=np.nanmin(Zg), vmax=np.nanmax(Zg))
    sc = ax.scatter(summary["extr_temp_C"], summary["pal_pct"], summary["max_tensile_stress_MPa"],
                    c=summary["max_tensile_stress_MPa"], cmap="magma", s=45,
                    edgecolor="k", linewidth=0.4, depthshade=False)
    ax.set_xlabel("Extrusion temp (°C)"); ax.set_ylabel("PAL content (%)")
    ax.set_zlabel("Max tensile stress (MPa)")
    ax.set_title("Max tensile stress vs. PAL% and extrusion temperature\n"
                 "(surface = per-cell mean, dots = specimens)")
    ax.view_init(elev=22, azim=-58)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1).set_label("Max tensile stress (MPa)")
    fig.tight_layout()
    return _save(fig, save)


def correlation(summary: pd.DataFrame, corr: pd.DataFrame, stats: dict, save: Path | None = None):
    """UTS-vs-toughness scatter + objective correlation matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    d = summary.dropna(subset=["max_tensile_stress_MPa", "toughness_MJ_m3"])
    sc = ax.scatter(d["max_tensile_stress_MPa"], d["toughness_MJ_m3"],
                    c=d["pal_pct"], cmap="viridis", s=45, edgecolor="k", lw=0.3)
    b1, b0 = np.polyfit(d["max_tensile_stress_MPa"], d["toughness_MJ_m3"], 1)
    xs = np.array([d["max_tensile_stress_MPa"].min(), d["max_tensile_stress_MPa"].max()])
    ax.plot(xs, b0 + b1 * xs, "r--", lw=1.5)
    ax.set_xlabel("UTS (MPa)"); ax.set_ylabel("Toughness (MJ/m³)")
    ax.set_title(f"UTS vs toughness (per specimen, n={stats['n']})\n"
                 f"Pearson r = {stats['pearson_r']:.2f} (p={stats['pearson_p']:.3f}) · "
                 f"Spearman ρ = {stats['spearman_rho']:.2f}")
    fig.colorbar(sc, ax=ax).set_label("PAL %"); ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns)
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title("Objective correlation matrix (Pearson)")
    fig.colorbar(im, ax=ax).set_label("r")
    fig.tight_layout()
    return _save(fig, save)


# --- Rheology ----------------------------------------------------------------

def flow_curves(curves: dict, save: Path | None = None):
    """All flow curves, colored by recycle level (red = 5x, blue = 1x)."""
    cmaps = {"5x": plt.cm.Reds, "1x": plt.cm.Blues}
    n_each = {k: sum(1 for lvl, _ in curves.values() if lvl == k) for k in ("5x", "1x")}
    cnt = {"5x": 0, "1x": 0}
    fig, ax = plt.subplots(figsize=(9, 6))
    for s, (lvl, c) in curves.items():
        cnt[lvl] = cnt.get(lvl, 0) + 1
        shade = 0.4 + 0.5 * cnt[lvl] / max(n_each.get(lvl, 1), 1)
        ax.loglog(c["shear_rate_1_s"], c["viscosity_Pa_s"], marker="o", ms=3, lw=1,
                  color=cmaps.get(lvl, plt.cm.Greys)(shade), label=s)
    ax.set_xlabel("Shear rate (1/s)"); ax.set_ylabel("Viscosity (Pa·s)")
    ax.set_title("Melt viscosity vs. shear rate (red = 5x recycled, blue = 1x)")
    ax.grid(which="both", alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, save)


def recycle_comparison(curves: dict, save: Path | None = None):
    """Geometric-mean viscosity per recycle level with a min-max spread band."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for lvl, color in [("5x", "C3"), ("1x", "C0")]:
        members = [c for (l, c) in curves.values() if l == lvl]
        if not members:
            continue
        big = pd.concat(members)
        big["rate_round"] = big["shear_rate_1_s"].round(3)
        grp = big.groupby("rate_round")["viscosity_Pa_s"]
        gmean = np.exp(grp.apply(lambda v: np.log(v).mean()))
        lo, hi = grp.min(), grp.max()
        x = gmean.index.values
        ax.fill_between(x, lo.reindex(gmean.index).values, hi.reindex(gmean.index).values,
                        color=color, alpha=0.18)
        ax.loglog(x, gmean.values, marker="o", ms=4, lw=2, color=color,
                  label=f"{lvl} recycled — geo. mean of {len(members)} runs")
    ax.set_xlabel("Shear rate (1/s)"); ax.set_ylabel("Viscosity (Pa·s)")
    ax.set_xlim(0, 10**.5)
    ax.set_title("Melt viscosity by recycle level: geometric mean ± run-to-run spread")
    ax.grid(which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    return _save(fig, save)


def extrusion_window(curves: dict, lo: float = RATE_LO, hi: float = RATE_HI, save: Path | None = None):
    """Flow curves with the extrusion window shaded and the high-shear artifact greyed."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for s, (lvl, c) in curves.items():
        ax.loglog(c["shear_rate_1_s"], c["viscosity_Pa_s"], lw=1, alpha=0.55,
                  color="C3" if lvl == "5x" else "C0")
    rate_max = max(c["shear_rate_1_s"].max() for _, c in curves.values())
    ax.axvspan(lo, hi, color="tab:green", alpha=0.12)
    ax.axvspan(hi, rate_max, color="grey", alpha=0.10)
    ax.axvline(hi, color="tab:green", ls="--", lw=1.2)
    ax.legend(handles=[
        Line2D([0], [0], color="C0", lw=2, label="1× recycled"),
        Line2D([0], [0], color="C3", lw=2, label="5× recycled"),
        Patch(facecolor="tab:green", alpha=0.12, label=f"extrusion window {lo}–{hi:.0f} s⁻¹ (Anupam)"),
        Patch(facecolor="grey", alpha=0.10, label="high-shear drop — artifact (redo)"),
    ], fontsize=8)
    ax.set_xlabel("Shear rate (1/s)"); ax.set_ylabel("Viscosity (Pa·s)")
    ax.set_title("Rheology region of interest: extrusion window 0.1–10 s⁻¹")
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save)


# --- DoE ---------------------------------------------------------------------

def doe_scatter(design: pd.DataFrame, save: Path | None = None):
    """Formulation design: PAL% vs temperature, one marker/color per polymer system.

    (Recycle level is not shown — it is a within-sample series applied to every point.)
    """
    styles = {"PP+PAL": ("o", "C0"), "HDPE+PAL": ("s", "C1"), "TPU+PAL": ("^", "C2")}
    fig, ax = plt.subplots(figsize=(6, 6))
    for poly, (mk, color) in styles.items():
        sub = design[design["polymer_system"] == poly]
        if sub.empty:
            continue
        ax.scatter(sub["PAL_pct"], sub["extrusion_temp_C"], marker=mk, s=90,
                   color=color, edgecolor="k", lw=0.4, label=poly)
    ax.set_xlabel("PAL content (%)", size=20) 
    ax.set_ylabel("Extrusion temperature (°C)", size=20)
    ax.set_title(f"DoE formulations, optimized LHS ({len(design)})", size = 20)
    ax.legend(title="polymer system"); ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=20)
    fig.tight_layout()
    return _save(fig, save)


def design_benchmark(bench: pd.DataFrame, save: Path | None = None):
    """One bar panel per diagnostic comparing optimized LHS vs baselines (arrows show 'good')."""
    panels = [("centered_discrepancy", "centered discrepancy  (↓ better)"),
              ("maximin", "maximin distance  (↑ better)"),
              ("max_abs_corr", "max |factor corr|  (↓ better)"),
              ("fill_distance", "fill distance  (↓ better)")]
    colors = ["C0", "C1", "C2"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (col, title) in zip(axes, panels):
        ax.bar(range(len(bench)), bench[col], color=colors)
        ax.set_xticks(range(len(bench)), bench.index, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Design diagnostics: optimized LHS vs. baselines")
    fig.tight_layout()
    return _save(fig, save)


# --- DoE tensile delivery (2026-09-03, Instron raw exports) -------------------
# Fixed categorical hues per polymer system (never re-assigned when a series is
# filtered out). Sequential magnitude maps use one hue, light -> dark.

POLY_COLORS = {"PP+PAL": "#2a78d6", "HDPE+PAL": "#eb6834", "TPU+PAL": "#1baf7a"}
POLY_MARKERS = {"PP+PAL": "o", "HDPE+PAL": "s", "TPU+PAL": "^"}
_INK, _INK2 = "#0b0b0b", "#52514e"


def _poly_handles(polys=None, mono=False, markersize=9):
    """Legend handles per polymer system; ``mono`` greys the fill when colour encodes something else."""
    polys = polys or list(POLY_COLORS)
    return [Line2D([0], [0], marker=POLY_MARKERS[p], color="none",
                   markerfacecolor="#9a9a96" if mono else POLY_COLORS[p],
                   markeredgecolor="white", markeredgewidth=0.8, markersize=markersize, label=p) for p in polys]


def _tidy(ax):
    ax.grid(alpha=0.25, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def doe_coverage(cov: pd.DataFrame, save: Path | None = None):
    """Planned formulations in PAL % x temperature space.

    Filled marker = received (area ~ number of specimens), hollow = not delivered.
    Marker shape / hue = polymer system. Labels give the formulation id and specimen count.
    """
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.add_patch(plt.Rectangle((10, 180), 40, 60, facecolor="#2a78d6", alpha=0.05, edgecolor="none"))
    for poly, color in POLY_COLORS.items():
        sub = cov[cov["polymer_system"] == poly]
        if sub.empty:
            continue
        got, miss = sub[sub["n_specimens"] > 0], sub[sub["n_specimens"] == 0]
        ax.scatter(got["PAL_pct"], got["extrusion_temp_C"], marker=POLY_MARKERS[poly],
                   s=70 + 40 * got["n_specimens"], color=color, edgecolor="white", lw=1.5, zorder=3)
        ax.scatter(miss["PAL_pct"], miss["extrusion_temp_C"], marker=POLY_MARKERS[poly],
                   s=150, facecolor="none", edgecolor=color, lw=1.8, zorder=3)
    for _, r in cov.iterrows():
        txt = f"f{int(r['formulation']):02d}" + (f"  n={int(r['n_specimens'])}" if r["n_specimens"] else "  —")
        ax.annotate(txt, (r["PAL_pct"], r["extrusion_temp_C"]), textcoords="offset points",
                    xytext=(9, 6), fontsize=8.5, color=_INK2)
    handles = _poly_handles() + [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9a9a96", markeredgecolor="white",
               markersize=10, label="received (area ∝ specimens)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#9a9a96",
               markeredgewidth=1.8, markersize=10, label="not delivered"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.set_xlim(2, 62); ax.set_ylim(172, 250)
    ax.set_xlabel("PAL content (%)"); ax.set_ylabel("Extrusion temperature (°C)")
    n_got, n_all = int((cov["n_specimens"] > 0).sum()), len(cov)
    ax.set_title(f"DoE coverage of the 2026-09-03 delivery: {n_got} of {n_all} formulations "
                 f"(shaded = design region)")
    _tidy(ax)
    fig.tight_layout()
    return _save(fig, save)


def instron_anatomy(curve: pd.DataFrame, row: pd.Series, save: Path | None = None):
    """One specimen, annotated: chord-modulus window, toe offset, UTS, break, toughness area.

    The post-fracture tail the machine kept logging is drawn in grey and excluded.
    """
    e, s = curve["strain"].to_numpy(float), curve["stress_MPa"].to_numpy(float)
    ib, ipk = int(row["i_break"]), int(row["i_peak"])
    ilo, ihi = int(row["i_modulus_lo"]), int(row["i_modulus_hi"])
    E, uts, tough, toe = (float(row["young_modulus_MPa"]), float(row["max_tensile_stress_MPa"]),
                          float(row["toughness_MJ_m3"]), float(row["toe_offset_strain"]))
    color = POLY_COLORS.get(row.get("polymer_system", ""), "#2a78d6")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(e, s, color="#b9b8b3", lw=1, label="raw export (incl. post-fracture tail)")
    ax.plot(e[:ib], s[:ib], color=color, lw=2, zorder=3, label="kept: start → break")
    ax.fill_between(e[:ib], np.clip(s[:ib], 0, None), 0, color=color, alpha=0.12, zorder=1,
                    label=f"area = toughness ≈ {tough:.2f} MJ/m³")
    from .instron import MODULUS_WINDOW
    ax.plot(e[ilo:ihi + 1], s[ilo:ihi + 1], color=_INK, lw=3.5, alpha=0.85, zorder=4,
            label=f"chord window {int(100*MODULUS_WINDOW[0])}–{int(100*MODULUS_WINDOW[1])} % of UTS "
                  f"→ E ≈ {E:.0f} MPa (R² = {row['modulus_r2']:.4f})")
    xt = np.array([toe, toe + 0.7 * uts / E])
    ax.plot(xt, E * (xt - toe), color=_INK, lw=1.2, ls=":", zorder=2)
    ax.scatter([e[ipk]], [uts], color="#e34948", s=60, zorder=5, edgecolor="white", lw=1)
    ax.annotate(f"UTS ≈ {uts:.2f} MPa", (e[ipk], uts), textcoords="offset points", xytext=(8, 6),
                fontsize=9, color=_INK2)
    ax.axvline(e[ib - 1], color=_INK2, lw=1, ls="--")
    ax.annotate(f"break: ε ≈ {e[ib-1]:.3f} (raw)\n= {row['strain_at_break']:.3f} toe-corrected",
                (e[ib - 1], 0.55 * uts), textcoords="offset points", xytext=(6, 0), fontsize=8.5, color=_INK2)
    ax.axvline(toe, color=_INK2, lw=0.8, ls=":")
    ax.annotate(f"toe offset {toe:.4f}", (toe, 0.05 * uts), textcoords="offset points", xytext=(4, 0),
                fontsize=8, color=_INK2)
    ax.axhline(0, color=_INK2, lw=0.6)
    ax.set_xlim(-0.002, e[min(len(e) - 1, ib + max(20, ib // 4))])
    ax.set_xlabel("Tensile strain, extension / gauge length (mm/mm)"); ax.set_ylabel("Tensile stress (MPa)")
    ax.set_title(f"What we extract from each Instron export — {row['specimen_id']} "
                 f"({row.get('polymer_system','')}, {int(row.get('pal_pct', 0))} % PAL, "
                 f"{int(row.get('extr_temp_C', 0))} °C)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    _tidy(ax)
    fig.tight_layout()
    return _save(fig, save)


def instron_curves_grid(curves: dict, summary: pd.DataFrame, save: Path | None = None):
    """Small multiples: one panel per formulation, replicates overlaid up to the break.

    Solid = main session, dashed = follow-up ``_1`` session. Shared stress axis; the strain
    axis is per panel (TPU stretches ~20x further than PP / HDPE).
    """
    forms = sorted(summary["formulation"].unique())
    ncol = 3; nrow = int(np.ceil(len(forms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, forms):
        sub = summary[summary["formulation"] == f]
        poly = sub["polymer_system"].iloc[0]; color = POLY_COLORS.get(poly, "#2a78d6")
        for _, r in sub.iterrows():
            c = curves[r["specimen_id"]]; ib = int(r["i_break"])
            ax.plot(c["strain"].iloc[:ib], c["stress_MPa"].iloc[:ib], color=color, lw=1.4,
                    ls="-" if r["session"] == 0 else "--", alpha=0.9)
        ax.set_title(f"f{int(f):02d} · {poly} · {int(sub['pal_pct'].iloc[0])} % PAL · "
                     f"{int(sub['extr_temp_C'].iloc[0])} °C   (n = {len(sub)})", fontsize=9)
        ax.axhline(0, color=_INK2, lw=0.5); _tidy(ax)
        ax.tick_params(labelsize=8)
    for ax in axes[len(forms):]:
        ax.set_visible(False)
    for ax in axes[:len(forms)]:
        ax.set_xlabel("strain (mm/mm)", fontsize=8.5)
    for i in range(0, len(forms), ncol):
        axes[i].set_ylabel("stress (MPa)", fontsize=8.5)
    handles = _poly_handles(sorted(summary["polymer_system"].unique(), key=list(POLY_COLORS).index)) + [
        Line2D([0], [0], color=_INK2, lw=1.4, label="main session"),
        Line2D([0], [0], color=_INK2, lw=1.4, ls="--", label="follow-up session (_1)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Stress–strain curves per DoE formulation (to break)", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    return _save(fig, save)


_PROP_PANELS = [("UTS_MPa", "UTS (MPa)"), ("young_modulus_MPa", "Young's modulus (MPa)"),
                ("toughness_MJ_m3", "Toughness (MJ/m³)")]


def _prop_panel(ax, by_form: pd.DataFrame, col: str, lab: str):
    """Mean ± std of one property per formulation on ``ax`` (marker / hue = polymer)."""
    for i, (_, r) in enumerate(by_form.iterrows()):
        poly = r["polymer_system"]
        ax.errorbar(i, r[f"{col}_mean"], yerr=r[f"{col}_std"] if pd.notna(r[f"{col}_std"]) else 0,
                    fmt=POLY_MARKERS[poly], color=POLY_COLORS[poly], ms=8, capsize=3, lw=1.2,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(range(len(by_form)), [f"f{int(f):02d}" for f in by_form["formulation"]], fontsize=8.5)
    ax.set_ylabel(lab); _tidy(ax)


def props_by_formulation(by_form: pd.DataFrame, save: Path | None = None, separate: bool = False):
    """Mean ± std of UTS, modulus, and toughness per formulation (marker = polymer).

    ``separate=False`` -> one figure with the three panels side by side (returned).
    ``separate=True``  -> one figure per property, each displayed with ``plt.show()``; with
    ``save`` given, each is written as ``<stem>_<property><suffix>``. Returns None.
    """
    polys = sorted(by_form["polymer_system"].unique(), key=list(POLY_COLORS).index)
    note = "single-specimen formulations show no bar"

    if separate:
        for col, lab in _PROP_PANELS:
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            _prop_panel(ax, by_form, col, lab)
            ax.legend(handles=_poly_handles(polys), fontsize=8.5)
            ax.set_title(f"{lab} — replicate mean ± std per formulation\n({note})", fontsize=10)
            fig.tight_layout()
            if save is not None:
                save = Path(save)
                fig.savefig(save.with_name(f"{save.stem}_{col}{save.suffix}"), dpi=150)
            plt.show()
            plt.close(fig)
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (col, lab) in zip(axes, _PROP_PANELS):
        _prop_panel(ax, by_form, col, lab)
    axes[0].legend(handles=_poly_handles(polys), fontsize=8.5)
    fig.suptitle(f"Replicate mean ± std per formulation ({note})")
    fig.tight_layout()
    return _save(fig, save)


def props_vs_factors(summary: pd.DataFrame, by_form: pd.DataFrame, save: Path | None = None):
    """Each property against PAL % (top row) and extrusion temperature (bottom row).

    Small translucent points = specimens; large markers = formulation means. Hue / shape = polymer.
    """
    props = [("max_tensile_stress_MPa", "UTS_MPa", "UTS (MPa)"),
             ("young_modulus_MPa", "young_modulus_MPa", "Young's modulus (MPa)"),
             ("toughness_MJ_m3", "toughness_MJ_m3", "Toughness (MJ/m³)")]
    factors = [("pal_pct", "PAL content (%)"), ("extr_temp_C", "Extrusion temperature (°C)")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, (fx, flab) in enumerate(factors):
        for j, (raw_col, agg_col, plab) in enumerate(props):
            ax = axes[i, j]
            for poly, color in POLY_COLORS.items():
                s = summary[summary["polymer_system"] == poly]
                b = by_form[by_form["polymer_system"] == poly]
                if s.empty:
                    continue
                ax.scatter(s[fx], s[raw_col], marker=POLY_MARKERS[poly], color=color, s=22, alpha=0.35,
                           edgecolor="none")
                ax.scatter(b[fx], b[f"{agg_col}_mean"], marker=POLY_MARKERS[poly], color=color, s=95,
                           edgecolor="white", lw=1, zorder=3)
            ax.set_xlabel(flab); ax.set_ylabel(plab); _tidy(ax)
    axes[0, 0].legend(handles=_poly_handles(sorted(summary["polymer_system"].unique(),
                                                    key=list(POLY_COLORS).index)), fontsize=8.5)
    fig.suptitle("Tensile properties vs. design factors (small = specimens, large = formulation mean)")
    fig.tight_layout()
    return _save(fig, save)


def design_space_bubbles(by_form: pd.DataFrame, save: Path | None = None):
    """PAL % x temperature maps, one per property: fill = magnitude (one-hue ramp), shape = polymer."""
    panels = [("UTS_MPa_mean", "mean UTS (MPa)", "Blues"),
              ("young_modulus_MPa_mean", "mean Young's modulus (MPa)", "Blues"),
              ("toughness_MJ_m3_mean", "mean toughness (MJ/m³)", "Blues")]
    from matplotlib.colors import LinearSegmentedColormap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, lab, cmap) in zip(axes, panels):
        vmin, vmax = by_form[col].min(), by_form[col].max()
        # one hue, light -> dark, but start at ~25 % so the smallest value stays visible on the surface
        cmap = LinearSegmentedColormap.from_list(f"{cmap}_trunc", plt.get_cmap(cmap)(np.linspace(0.25, 1, 256)))
        for poly in POLY_COLORS:
            b = by_form[by_form["polymer_system"] == poly]
            if b.empty:
                continue
            sc = ax.scatter(b["pal_pct"], b["extr_temp_C"], c=b[col], cmap=cmap, vmin=vmin, vmax=vmax,
                            marker=POLY_MARKERS[poly], s=210, edgecolor=_INK2, lw=0.5, zorder=3)
        for _, r in by_form.iterrows():
            ax.annotate(f"f{int(r['formulation']):02d}", (r["pal_pct"], r["extr_temp_C"]),
                        textcoords="offset points", xytext=(11, -3), fontsize=8, color=_INK2)
        ax.set_xlim(5, 58); ax.set_ylim(172, 246)
        ax.set_xlabel("PAL content (%)"); ax.set_ylabel("Extrusion temperature (°C)")
        ax.set_title(lab, fontsize=10); _tidy(ax)
        fig.colorbar(sc, ax=ax, shrink=0.85).set_label(lab)
    axes[0].legend(handles=_poly_handles(sorted(by_form["polymer_system"].unique(),
                                                key=list(POLY_COLORS).index), mono=True),
                   fontsize=8, loc="lower left", title="marker = polymer", title_fontsize=8)
    fig.suptitle("Where in the design space the received formulations sit, and what they measured")
    fig.tight_layout()
    return _save(fig, save)


def _front2d(x: np.ndarray, y: np.ndarray) -> list[int]:
    """Positions of the two-objective (max, max) non-dominated points, by increasing ``x``."""
    keep, best = [], -np.inf
    for i in np.lexsort((-y, -x)):          # x descending; a point survives if it raises the best y
        if y[i] > best:
            keep.append(i); best = y[i]
    return keep[::-1]


def _label_points(ax, x, y, texts, bold, fontsize: float = 8.5):
    """Annotate points, greedily choosing per label the corner (right / left, above / below) that
    overlaps the fewest markers and already-placed labels. ``bold`` flags the emphasised labels."""
    pt = ax.transData.transform(np.c_[x, y]) * 72 / ax.figure.dpi          # positions in points
    w, h, m = 0.62 * fontsize * max(map(len, texts)) + 2, fontsize + 1, 7  # label box, marker half-size
    boxes = [(px - m, py - m, px + m, py + m) for px, py in pt]
    for xi, yi, (px, py), t, b in zip(x, y, pt, texts, bold):
        best = None
        for dx, dy, ha in ((8, 5, "left"), (8, -12, "left"), (-8, 5, "right"), (-8, -12, "right"),
                           (-w / 2, 11, "left"), (-w / 2, -19, "left")):
            x0 = px + dx - (w if ha == "right" else 0); box = (x0, py + dy, x0 + w, py + dy + h)
            hits = sum(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3] for q in boxes)
            if best is None or hits < best[0]:
                best = (hits, dx, dy, ha, box)
        boxes.append(best[4])
        ax.annotate(t, (xi, yi), textcoords="offset points", xytext=best[1:3], ha=best[3],
                    fontsize=fontsize, color=_INK if b else _INK2,
                    fontweight="bold" if b else "normal", zorder=6)


def pareto_front(pareto: pd.DataFrame, log_toughness: bool = True, save: Path | None = None):
    """Pairwise projections of the UTS x Young's modulus x toughness Pareto front (all maximised).

    ``pareto`` is ``tensile.pareto_front(by_form, ...)``. Filled marker = non-dominated on the three
    objectives together, hollow = dominated; error bars = replicate std. The stepped line is the
    two-objective front of that panel (shaded = the region it dominates), so a filled marker that
    sits below the line is kept on the front by the third objective. Hue / shape = polymer.
    """
    labs = dict(_PROP_PANELS)
    pairs = [("UTS_MPa", "toughness_MJ_m3", "UTS vs. toughness"),
             ("UTS_MPa", "young_modulus_MPa", "UTS vs. Young's modulus"),
             ("young_modulus_MPa", "toughness_MJ_m3", "Young's modulus vs. toughness")]
    d = pareto.dropna(subset=[f"{c}_mean" for c in labs]).sort_values("pareto_rank")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (cx, cy, title) in zip(axes, pairs):
        x, y = d[f"{cx}_mean"].to_numpy(float), d[f"{cy}_mean"].to_numpy(float)
        for (_, r), xi, yi in zip(d.iterrows(), x, y):
            poly = r["polymer_system"]; color = POLY_COLORS[poly]
            ax.errorbar(xi, yi, xerr=np.nan_to_num(r[f"{cx}_std"]), yerr=np.nan_to_num(r[f"{cy}_std"]),
                        fmt="none", ecolor=color, elinewidth=0.9, alpha=0.45, zorder=2)
            if r["pareto_optimal"]:
                ax.scatter(xi, yi, marker=POLY_MARKERS[poly], s=120, color=color, edgecolor="white",
                           lw=1.2, zorder=4)
            else:       # truly hollow and drawn on top, so a coincident filled marker stays visible
                ax.scatter(xi, yi, marker=POLY_MARKERS[poly], s=120, facecolor="none", edgecolor=color,
                           lw=1.6, zorder=5)
        for c, axis, set_scale in ((cx, ax.xaxis, ax.set_xscale), (cy, ax.yaxis, ax.set_yscale)):
            if log_toughness and c == "toughness_MJ_m3":
                set_scale("log")
                axis.set_minor_locator(LogLocator(subs=(2, 5)))      # plain 0.1 / 0.2 / 0.5 / 1 ticks
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        k = _front2d(x, y)
        xs = np.r_[xlim[0], x[k], x[k][-1]]; ys = np.r_[y[k][0], y[k], ylim[0]]
        ax.step(xs, ys, where="pre", color=_INK, lw=1.3, zorder=3)
        ax.fill_between(xs, ys, ylim[0], step="pre", color=_INK, alpha=0.05, lw=0, zorder=1)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(labs[cx] + (" — log scale" if ax.get_xscale() == "log" else ""))
        ax.set_ylabel(labs[cy] + (" — log scale" if ax.get_yscale() == "log" else ""))
        ax.set_title(title, fontsize=10)
        _tidy(ax)
    handles = _poly_handles(sorted(d["polymer_system"].unique(), key=list(POLY_COLORS).index)) + [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9a9a96", markeredgecolor="white",
               markersize=10, label="on the 3-objective Pareto front"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#9a9a96",
               markeredgewidth=1.6, markersize=10, label="dominated"),
        Line2D([0], [0], color=_INK, lw=1.3, label="2-objective front of the panel (shaded = dominated)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    front = ", ".join(f"f{int(f):02d}" for f in sorted(d.loc[d["pareto_optimal"], "formulation"]))
    fig.suptitle(f"Pareto front of UTS × Young's modulus × toughness (formulation means ± std, "
                 f"all maximised): {front}")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ax, (cx, cy, _) in zip(axes, pairs):        # after the layout: placement works in display space
        _label_points(ax, d[f"{cx}_mean"].to_numpy(float), d[f"{cy}_mean"].to_numpy(float),
                      [f"f{int(f):02d}" for f in d["formulation"]], d["pareto_optimal"].to_numpy())
    return _save(fig, save)


# --- DoE augmentation (recycle level as a design factor) ------------------------------------

def doe_augmented(plan: pd.DataFrame, save: Path | None = None):
    """The augmented measurement plan, one panel per polymer, in PAL % x temperature.

    Filled marker = requested run (1x, in hand or pending); each proposed run at a higher
    recycle level is a ring around its formulation, larger for more loops, labelled ``5x`` / ``10x``.
    """
    polys = [p for p in POLY_COLORS if (plan["polymer_system"] == p).any()]
    fig, axes = plt.subplots(1, len(polys), figsize=(5 * len(polys), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    ring = lambda level: 200 + 32 * level                    # marker area of a proposed run's ring
    for ax, poly in zip(axes, polys):
        sub = plan[plan["polymer_system"] == poly]; color = POLY_COLORS[poly]; mk = POLY_MARKERS[poly]
        base = sub[sub["status"] == "requested"]
        ax.scatter(base["PAL_pct"], base["extrusion_temp_C"], marker=mk, s=90, color=color,
                   edgecolor="white", lw=1, zorder=4)
        for _, r in sub[sub["status"] == "proposed"].iterrows():
            ax.scatter(r["PAL_pct"], r["extrusion_temp_C"], marker=mk, s=ring(r["recycle_level"]),
                       facecolor="none", edgecolor=color, lw=1.6, zorder=3)
        for f, g in sub.groupby("formulation"):
            levels = sorted(int(x) for x in g.loc[g["status"] == "proposed", "recycle_level"])
            txt = f"f{int(f):02d}" + (" · " + ", ".join(f"{x}×" for x in levels) if levels else "")
            ax.annotate(txt, (g["PAL_pct"].iloc[0], g["extrusion_temp_C"].iloc[0]), textcoords="offset points",
                        xytext=(14, -4), fontsize=8.5, color=_INK if levels else _INK2,
                        fontweight="bold" if levels else "normal")
        n_req, n_new = int((sub["status"] == "requested").sum()), int((sub["status"] == "proposed").sum())
        ax.set_title(f"{poly}: {n_req} requested + {n_new} proposed", fontsize=10)
        ax.set_xlim(2, 66); ax.set_ylim(172, 248); ax.set_xlabel("PAL content (%)"); _tidy(ax)
    axes[0].set_ylabel("Extrusion temperature (°C)")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#9a9a96", markeredgecolor="white",
                      markersize=9, label="requested (1×)"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#9a9a96",
                      markeredgewidth=1.6, markersize=13, label="proposed run at a higher recycle level (ring ∝ loops)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Augmented plan: recycle level added as a design factor, requested 1× runs kept")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, save)


# --- Revised DoE (2026-10): recycle cycle 0-3 as a design factor -----------------------------

CYCLE_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal blue ramp, cycle 0 -> 3 (validated)


def doe_sliced(design: pd.DataFrame, pal_range=(10, 50), temp_range=(180, 240), save: Path | None = None):
    """Sliced-LHS design, one panel per polymer in PAL % x temperature; colour (light -> dark) and the
    label give the recycle cycle. Faint lines are the per-polymer strata: each band holds one run.
    """
    polys = [p for p in POLY_COLORS if (design["polymer_system"] == p).any()]
    cycles = sorted(int(c) for c in design["recycle_cycle"].unique())
    fig, axes = plt.subplots(1, len(polys), figsize=(4.6 * len(polys), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, poly in zip(axes, polys):
        sub = design[design["polymer_system"] == poly]
        m = len(sub)
        for e in np.linspace(*pal_range, m + 1)[1:-1]:
            ax.axvline(e, color="#e4e3df", lw=0.8, zorder=0)
        for e in np.linspace(*temp_range, m + 1)[1:-1]:
            ax.axhline(e, color="#e4e3df", lw=0.8, zorder=0)
        for _, r in sub.iterrows():
            c = int(r["recycle_cycle"])
            ax.scatter(r["PAL_pct"], r["extrusion_temp_C"], marker=POLY_MARKERS[poly], s=120,
                       color=CYCLE_COLORS[cycles.index(c)], edgecolor="white", lw=1.2, zorder=3)
            ax.annotate(f"C{int(r['configuration']):02d} · {c}", (r["PAL_pct"], r["extrusion_temp_C"]),
                        textcoords="offset points", xytext=(9, -3), fontsize=8.5, color=_INK)
        ax.set_title(f"{poly} · {m} configurations", fontsize=10, color=_INK)
        ax.set_xlim(pal_range[0] - 2, pal_range[1] + 6); ax.set_ylim(temp_range[0] - 5, temp_range[1] + 5)
        ax.set_xlabel("PAL content (%)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Extrusion temperature (°C)")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CYCLE_COLORS[i], markeredgecolor="white",
                      markersize=9, label=f"cycle {c}  ({int((design['recycle_cycle'] == c).sum())} configs)")
               for i, c in enumerate(cycles)]
    fig.legend(handles=handles, title="recycle cycle", loc="lower center", ncol=len(cycles), fontsize=8.5,
               title_fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Revised DoE: {len(design)} configurations, sliced Latin hypercube, recycle cycles "
                 f"{cycles[0]}–{cycles[-1]}", color=_INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, save)


def doe_correlation(design: pd.DataFrame, title: str | None = None, save: Path | None = None):
    """Pearson correlation of the numeric design factors (PAL %, temperature, recycle cycle): all
    configurations, then each polymer slice. Fixed -1..1 diverging scale; diagonal greyed out; dotted
    lines on the colour bar mark the |r| < 0.3 check.
    """
    cols = {"PAL_pct": "PAL %", "extrusion_temp_C": "Temp. (°C)", "recycle_cycle": "Recycle cycle"}
    groups = [(f"All {len(design)} configurations", design)] + [
        (f"{p} ({int((design['polymer_system'] == p).sum())})", design[design["polymer_system"] == p])
        for p in POLY_COLORS if (design["polymer_system"] == p).any()]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.3 * len(groups) + 0.8, 3.75))
    for k, (ax, (name, sub)) in enumerate(zip(np.atleast_1d(axes), groups)):
        corr = sub[list(cols)].corr().to_numpy()
        shown = np.where(np.eye(len(cols), dtype=bool), np.nan, corr)
        im = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.imshow(np.where(np.eye(len(cols), dtype=bool), 1.0, np.nan), cmap="Greys", vmin=0, vmax=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr[i, j]
                ax.text(j, i, "1" if i == j else ("0.00" if abs(v) < 0.005 else f"{v:+.2f}"), ha="center", va="center", fontsize=13,
                        color=_INK2 if i == j else ("white" if abs(v) > 0.6 else _INK))
        ax.set_xticks(range(len(cols)), cols.values(), rotation=25, ha="right", fontsize=12)
        ax.set_yticks(range(len(cols)), cols.values() if k == 0 else [""] * len(cols), fontsize=12)
        ax.set_title(name, fontsize=13.5, color=_INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Pearson r  (closer to 0 = better)", fontsize=11.5)
    cb.set_ticks([-1, -0.5, -0.3, 0, 0.3, 0.5, 1])
    for t in (-0.3, 0.3):                                     # the |r| < 0.3 check
        cb.ax.axhline(t, color=_INK, lw=1, ls=":")
    if title != "":
        fig.suptitle(title or "Correlation between design factors", color=_INK)
    return _save(fig, save)


_METRIC_PANELS = [("discrepancy", "Centered L2 discrepancy", "↓ lower = better"),
                  ("min_dist", "Min pairwise distance", "↑ higher = better"),
                  ("max_corr", "Max |factor correlation|", "↓ lower = better, < 0.3"),
                  ("fill_dist", "Fill distance (largest hole)", "↓ lower = better")]


def design_method_comparison(tbl: pd.DataFrame, used: dict | None = None, used_method: str = "Sliced LHS (CD only, w = 0)",
                             save: Path | None = None):
    """The four space-filling metrics per design method, mean +/- 1 std over seeds (``tbl`` from
    ``doe.compare_methods``). Top row: all runs in the (PAL, T, cycle) cube; bottom: the worst polymer.
    The method used is blue, the others grey; the star is the selected design (``used`` from
    ``doe.design_quality``). Dotted line: the r = 0.3 confounding threshold."""
    methods = list(dict.fromkeys(tbl["method"]))
    n_seeds = int(tbl.groupby("method").size().max())
    rows = [("all", "All runs, polymer ignored"), ("poly", "Within each polymer (worst of 3)")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharex=True)
    for r, (pre, row_lab) in enumerate(rows):
        for c, (key, lab, better) in enumerate(_METRIC_PANELS):
            ax, col = axes[r, c], f"{pre}_{key}"
            g = tbl.groupby("method", sort=False)[col]
            ax.bar(range(len(methods)), g.mean().reindex(methods), width=0.66,
                   yerr=g.std().reindex(methods), capsize=3, error_kw={"lw": 1, "ecolor": _INK2},
                   color=["#2a78d6" if m == used_method else "#c9c8c2" for m in methods])
            if used is not None:
                ax.scatter([methods.index(used_method)], [used[col]], marker="*", s=230, color=_INK,
                           edgecolor="white", lw=0.8, zorder=5, label="selected design")
            if key == "max_corr":
                ax.axhline(0.3, ls=":", color=_INK2, lw=1, label="r = 0.3")
            ax.set_title(f"{lab}\n({better})", fontsize=10, color=_INK)
            if c == 0:
                ax.set_ylabel(row_lab, fontsize=10, color=_INK)
            ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right", fontsize=8.5)
            ax.grid(axis="y", alpha=0.25, lw=0.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0 and key in ("discrepancy", "max_corr"):
                ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Design-method quality as fabricated: mean ± 1 std over {n_seeds} seeds", color=_INK)
    fig.tight_layout()
    return _save(fig, save)


def doe_pairs(design: pd.DataFrame, pal_range=(10, 50), temp_range=(180, 240), title: str | None = None,
              save: Path | None = None):
    """Pair plot of PAL %, temperature and recycle cycle, one colour/marker per polymer (lower triangle).
    Diagonal: runs per band (7 bands of PAL and temperature, the 4 cycles) for each polymer -- a sliced
    LHS puts exactly one run of each polymer in every band."""
    polys = [p for p in POLY_COLORS if (design["polymer_system"] == p).any()]
    m = int(design["polymer_system"].value_counts().max())
    cyc = sorted(design["recycle_cycle"].unique())
    fac = [("PAL_pct", "PAL %", np.linspace(*pal_range, m + 1)),
           ("extrusion_temp_C", "Temp. (°C)", np.linspace(*temp_range, m + 1)),
           ("recycle_cycle", "Recycle cycle", np.arange(len(cyc) + 1) - 0.5)]
    lims = [(pal_range[0] - 2, pal_range[1] + 2), (temp_range[0] - 4, temp_range[1] + 4), (-0.5, len(cyc) - 0.5)]
    fig, axes = plt.subplots(3, 3, figsize=(10, 9.4))
    for i, (yi, ylab, _) in enumerate(fac):
        for j, (xj, xlab, edges) in enumerate(fac):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:                                         # runs per band, grouped by polymer
                w = 0.8 / len(polys)
                for k, p in enumerate(polys):
                    cnt, _ = np.histogram(design.loc[design["polymer_system"] == p, xj], bins=edges)
                    ax.bar(np.arange(len(cnt)) + (k - (len(polys) - 1) / 2) * w, cnt, width=w * 0.9,
                           color=POLY_COLORS[p], label=p)
                ax.set_xticks(range(len(edges) - 1),
                              [f"{int(round(a))}" for a in edges[:-1]] if xj != "recycle_cycle" else cyc,
                              fontsize=7.5)
                ax.set_ylabel("runs per band", fontsize=8.5); ax.set_yticks([0, 1, 2])
            else:
                for p in polys:
                    s = design[design["polymer_system"] == p]
                    ax.scatter(s[xj], s[yi], marker=POLY_MARKERS[p], s=46, color=POLY_COLORS[p],
                               edgecolor="white", lw=0.8)
                ax.set_xlim(lims[j]); ax.set_ylim(lims[i])
                if yi == "recycle_cycle":
                    ax.set_yticks(cyc)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=9)
            if i == 2 or i == j:
                ax.set_xlabel(xlab if i != j else f"{xlab} band" if xj != "recycle_cycle" else xlab, fontsize=9)
            _tidy(ax); ax.tick_params(labelsize=8)
    fig.legend(handles=_poly_handles(polys), loc="upper right", bbox_to_anchor=(0.93, 0.9), frameon=False,
               fontsize=9, title="polymer system", title_fontsize=9)
    fig.suptitle(title or f"Pair plot of the design ({len(design)} configurations)", color=_INK)
    fig.tight_layout()
    return _save(fig, save)


WEIGHT_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # ordinal blue for weights > 0 (validated)


def corr_weight_tradeoff(sweep: pd.DataFrame, used: dict | None = None, save: Path | None = None):
    """Pareto view of the correlation-weight sweep (``doe.sweep_corr_weight``): worst within-polymer |r|
    (x, lower = better) against the all-runs fill distance (lower = better) and min distance (higher =
    better). Small dots = single designs (one per seed); large markers = mean +/- 1 std per weight, joined
    in weight order (grey = weight 0, discrepancy only; light -> dark blue = larger weights); dashed
    staircase = Pareto front of all single designs (no design is better on both axes); dotted line =
    |r| = 0.3; star = selected design."""
    weights = sorted(sweep["corr_weight"].unique())
    pos = [w for w in weights if w > 0]
    ramp = [WEIGHT_COLORS[int(round(i))] for i in np.linspace(0, len(WEIGHT_COLORS) - 1, len(pos))] if pos else []
    color = {0.0: _INK2, **dict(zip(pos, ramp))}
    panels = [("all_fill_dist", "Fill distance, all runs", "↓ lower = better", False),
              ("all_min_dist", "Min pairwise distance, all runs", "↑ higher = better", True)]
    x = sweep["poly_max_corr"].to_numpy()
    g = sweep.groupby("corr_weight")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (col, lab, better, maximize) in zip(axes, panels):
        y = sweep[col].to_numpy()
        ax.scatter(x, y, s=20, c=[color[w] for w in sweep["corr_weight"]], alpha=0.5, lw=0, zorder=2)
        front = _front2d(-x, y if maximize else -y)            # (max, max) on (-|r|, +/-y)
        fx, fy = x[front], y[front]
        order = np.argsort(fx)
        ax.plot(fx[order], fy[order], drawstyle="steps-post", ls="--", color=_INK, lw=1.1, zorder=3)
        ax.scatter(fx, fy, s=58, facecolor="none", edgecolor=_INK, lw=1, zorder=4)
        mx, my = g["poly_max_corr"].mean(), g[col].mean()
        ax.plot(mx, my, color=_INK2, lw=1, zorder=4)
        for w in weights:
            ax.errorbar(mx[w], my[w], xerr=g["poly_max_corr"].std()[w], yerr=g[col].std()[w], fmt="o", ms=8,
                        color=color[w], mec="white", mew=1, ecolor=color[w], elinewidth=1, capsize=2, zorder=5)
        ax.axvline(0.3, ls=":", color=_INK2, lw=1)
        if used is not None:
            ax.scatter(used["poly_max_corr"], used[col], marker="*", s=230, color=_INK, edgecolor="white",
                       lw=0.8, zorder=7)
        ax.set_xlim(-0.01, 0.33)
        ax.set_xlabel("Worst within-polymer max |r|  (↓ lower = better)", fontsize=9.5)
        ax.set_ylabel(f"{lab}  ({better})", fontsize=9.5)
        ax.set_title(f"{lab} vs. correlation", fontsize=10, color=_INK)
        _tidy(ax)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#9a9a96", markeredgecolor="none",
                      alpha=0.6, markersize=6, label="single design (one seed)"),
               Line2D([0], [0], marker="o", color="#9a9a96", markerfacecolor="#9a9a96", markeredgecolor="white",
                      markersize=8, label="mean ± 1 std per weight"),
               Line2D([0], [0], color=_INK, ls="--", lw=1.1, marker="o", markerfacecolor="none", markersize=7,
                      label="Pareto front (single designs)"),
               Line2D([0], [0], color=_INK2, ls=":", lw=1, label="|r| = 0.3 threshold")]
    if used is not None:
        handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor=_INK, markeredgecolor="white",
                              markersize=13, label="selected design"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    axes[0].legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=color[w], markeredgecolor="white",
                                   markersize=8, label=f"w = {w:g}" + ("  (discrepancy only)" if w == 0 else ""))
                            for w in weights], title="correlation weight", title_fontsize=8.5, fontsize=8.5,
                   loc="upper right", frameon=False)
    fig.suptitle(f"Correlation weight sweep: sliced LHS, {g.size().max()} seeds per weight "
                 f"(w = 0: discrepancy only)", color=_INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, save)


def _cycle_count(design: pd.DataFrame, existing: pd.DataFrame | None, c: int) -> str:
    """Legend text for one cycle: ``"5 configs"``, or ``"7 new, 4 existing"`` when existing runs are drawn."""
    n = int((design["recycle_cycle"] == c).sum())
    if existing is None:
        return f"{n} configs"
    e = int((existing["recycle_cycle"] == c).sum())
    return ", ".join(t for t in (f"{n} new" if n else "", f"{e} existing" if e else "") if t)


def doe_design_space(design: pd.DataFrame, pal_range=(10, 50), temp_range=(180, 240), color_by: str = "polymer",
                     existing: pd.DataFrame | None = None, title: str | None = None, save: Path | None = None):
    """All polymers in one PAL % x temperature panel, one marker per polymer (circle / square / triangle).
    ``color_by="polymer"``: one colour per polymer; ``"cycle"``: colour = recycle cycle (light -> dark blue).
    Faint lines are the bands per factor; each band holds one run of every polymer. ``existing`` (runs
    already measured, same columns) are drawn hollow, and the bands then count existing + new runs."""
    polys = [p for p in POLY_COLORS if (design["polymer_system"] == p).any()]
    allr = design if existing is None else pd.concat([design, existing], ignore_index=True)
    # bands count distinct PAL / temperature points: one formulation at several cycles takes one band
    m = int(allr.drop_duplicates(["polymer_system", "PAL_pct", "extrusion_temp_C"])["polymer_system"].value_counts().max())
    fig, ax = plt.subplots(figsize=(9, 6.6))
    for e in np.linspace(*pal_range, m + 1)[1:-1]:
        ax.axvline(e, color="#e4e3df", lw=0.8, zorder=0)
    for e in np.linspace(*temp_range, m + 1)[1:-1]:
        ax.axhline(e, color="#e4e3df", lw=0.8, zorder=0)
    cycles = sorted(int(c) for c in allr["recycle_cycle"].unique())
    for poly in polys:
        sub = design[design["polymer_system"] == poly]
        color = (POLY_COLORS[poly] if color_by == "polymer" else
                 [CYCLE_COLORS[cycles.index(int(c))] for c in sub["recycle_cycle"]])
        ax.scatter(sub["PAL_pct"], sub["extrusion_temp_C"], marker=POLY_MARKERS[poly], s=240,
                   color=color, edgecolor="white", lw=1.4, zorder=3)
        if existing is not None:
            ex = existing[existing["polymer_system"] == poly].sort_values("recycle_cycle")
            # the same point at a later cycle too: the earlier cycle is the outer ring
            size = np.where(ex.duplicated(["PAL_pct", "extrusion_temp_C"], keep="last"), 500, 240)
            edge = (POLY_COLORS[poly] if color_by == "polymer" else
                    [CYCLE_COLORS[cycles.index(int(c))] for c in ex["recycle_cycle"]])
            ax.scatter(ex["PAL_pct"], ex["extrusion_temp_C"], marker=POLY_MARKERS[poly], s=size,
                       facecolor="none", edgecolor=edge, lw=2.2, zorder=4)      # on top: rings stay visible
    ax.set_xlim(pal_range[0] - 2, pal_range[1] + 2); ax.set_ylim(temp_range[0] - 5, temp_range[1] + 5)
    ax.set_xlabel("PAL content (%)", fontsize=14); ax.set_ylabel("Extrusion temperature (°C)", fontsize=14)
    ax.tick_params(labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(handles=_poly_handles(polys, mono=color_by != "polymer", markersize=11), title="polymer system",
                    title_fontsize=9, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    if color_by != "polymer":
        ax.add_artist(leg)
        ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=CYCLE_COLORS[i],
                                  markeredgecolor="white", markersize=11,
                                  label=f"cycle {c}  ({_cycle_count(design, existing, c)})")
                           for i, c in enumerate(cycles)], title="recycle cycle", title_fontsize=9, fontsize=9,
                  loc="upper left", bbox_to_anchor=(1.01, 0.72), frameon=False)
    if existing is not None:
        ax.add_artist(ax.get_legend())
        ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#9a9a96", markeredgecolor="white",
                                  markersize=11, label=f"new configuration ({len(design)})"),
                           Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#9a9a96",
                                  markeredgewidth=2.2, markersize=11, label=f"existing data ({len(existing)})")],
                  title="runs", title_fontsize=9, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 0.42),
                  frameon=False)
    ax.set_title(title or "Configurations in design space", color=_INK, fontweight="bold")
    fig.tight_layout()
    if existing is not None:
        fig.subplots_adjust(right=0.73)       # room for the longer legend text
    return _save(fig, save)


# --- Analysis of the delivered tensile data (deck 2026-10) ------------------------------------

_RESP = {"max_tensile_stress_MPa": "UTS (MPa)", "young_modulus_MPa": "Young's modulus (MPa)",
         "strain_at_break": "Strain at break (–)", "toughness_MJ_m3": "Toughness (MJ/m³)"}


def tensile_curves_by_polymer(curves: dict, summary: pd.DataFrame, save: Path | None = None):
    """Stress-strain curves, one panel per polymer (own strain axis: TPU stretches ~100x further, shared
    stress axis); within a panel one colour per formulation, light -> dark with PAL %, labelled PAL % · temperature."""
    polys = [p for p in POLY_COLORS if (summary["polymer_system"] == p).any()]
    fig, axes = plt.subplots(1, len(polys), figsize=(15, 4.6), sharey=True)
    for ax, poly in zip(np.atleast_1d(axes), polys):
        sub = summary[summary["polymer_system"] == poly]
        forms = sub.groupby("formulation")[["pal_pct", "extr_temp_C"]].first().sort_values("pal_pct")
        shades = [CYCLE_COLORS[int(round(i))] for i in np.linspace(0, len(CYCLE_COLORS) - 1, len(forms))]
        for (f, row), c in zip(forms.iterrows(), shades):
            for k, (_, r) in enumerate(sub[sub["formulation"] == f].iterrows()):
                cv = curves[r["specimen_id"]].iloc[: int(r["i_break"]) + 5]
                ax.plot(cv["strain"] - r["toe_offset_strain"], cv["stress_MPa"], color=c, lw=1.4, alpha=0.9,
                        label=f"{int(row.pal_pct)} % PAL · {int(row.extr_temp_C)} °C" if k == 0 else None)
        ax.set_title(poly, fontsize=13, color=_INK); ax.set_xlabel("Strain (–)", fontsize=12)
        ax.set_xlim(left=0); ax.tick_params(labelsize=11)
        _tidy(ax)
    # after all panels, so the shared top covers every polymer; headroom for a two-column legend above the curves
    np.atleast_1d(axes)[0].set_ylim(0, 1.25 * np.atleast_1d(axes)[0].get_ylim()[1])
    np.atleast_1d(axes)[0].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    for ax in np.atleast_1d(axes):
        ax.legend(fontsize=9.5, loc="upper center", ncol=2, frameon=True, facecolor="white", framealpha=0.92,
                  edgecolor="none", columnspacing=1.2, handlelength=1.6)
    np.atleast_1d(axes)[0].set_ylabel("Stress (MPa)", fontsize=12)
    fig.tight_layout()
    return _save(fig, save)


def props_by_formulation_log(summary: pd.DataFrame, save: Path | None = None):
    """The four responses per formulation (log scale): replicates as dots, mean as a bar mark; formulations
    grouped by polymer and ordered by PAL %, labelled PAL % / temperature."""
    polys = [p for p in POLY_COLORS if (summary["polymer_system"] == p).any()]
    forms = (summary.groupby(["polymer_system", "formulation"])[["pal_pct", "extr_temp_C"]].first().reset_index())
    forms["_p"] = forms["polymer_system"].map({p: i for i, p in enumerate(polys)})
    forms = forms.sort_values(["_p", "pal_pct"]).reset_index(drop=True)
    xpos = {f: i + 0.6 * p for i, (f, p) in enumerate(zip(forms["formulation"], forms["_p"]))}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    for ax, (col, lab) in zip(axes, _RESP.items()):
        for _, fr in forms.iterrows():
            v = summary.loc[summary["formulation"] == fr["formulation"], col]
            x = xpos[fr["formulation"]]
            ax.scatter(np.full(len(v), x) + np.linspace(-0.12, 0.12, len(v)), v, s=26, marker=POLY_MARKERS[fr["polymer_system"]],
                       color=POLY_COLORS[fr["polymer_system"]], edgecolor="white", lw=0.6, zorder=3)
            ax.plot([x - 0.3, x + 0.3], [np.exp(np.log(v).mean())] * 2, color=_INK, lw=2, zorder=4)
        ax.set_yscale("log"); ax.set_title(lab, fontsize=13, color=_INK)
        ax.set_xticks(list(xpos.values()), [f"{int(a)}/{int(b)}" for a, b in zip(forms["pal_pct"], forms["extr_temp_C"])],
                      rotation=90, fontsize=9.5)
        ax.tick_params(axis="y", labelsize=11); _tidy(ax)
    axes[0].set_xlabel("PAL % / temperature (°C)", fontsize=11)
    fig.legend(handles=_poly_handles(polys) + [Line2D([0], [0], color=_INK, lw=2, label="geometric mean")],
               loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, save)


def props_vs_factors_log(summary: pd.DataFrame, connect: bool = False, save: Path | None = None):
    """The four responses (log scale) against PAL % (top row) and extrusion temperature (bottom row): small
    translucent points = specimens, large markers = formulation geometric means; hue / shape = polymer.
    Rows share each column's y axis. ``connect`` joins each polymer's means in order of the x factor (a guide
    for the eye: the other factor changes between neighbouring points too)."""
    polys = [p for p in POLY_COLORS if (summary["polymer_system"] == p).any()]
    d = summary.assign(_cycle=summary["recycled_x"].fillna(0).astype(int) if "recycled_x" in summary else 0)
    g = d.groupby(["formulation", "polymer_system", "pal_pct", "extr_temp_C", "_cycle"], as_index=False)
    gm = g[list(_RESP)].agg(lambda v: float(np.exp(np.log(v).mean())))
    cycles = sorted(gm["_cycle"].unique())
    factors = [("pal_pct", "PAL content (%)"), ("extr_temp_C", "Extrusion temperature (°C)")]
    fig, axes = plt.subplots(2, len(_RESP), figsize=(17, 7.6), sharey="col")
    for i, (fx, flab) in enumerate(factors):
        for j, (col, lab) in enumerate(_RESP.items()):
            ax = axes[i, j]
            for p in polys:
                s, b = summary[summary["polymer_system"] == p], gm[gm["polymer_system"] == p]
                ax.scatter(s[fx], s[col], marker=POLY_MARKERS[p], color=POLY_COLORS[p], s=26, alpha=0.35,
                           edgecolor="none")
                for c in cycles:
                    bc = b[b["_cycle"] == c]
                    if c == cycles[0]:
                        ax.scatter(bc[fx], bc[col], marker=POLY_MARKERS[p], color=POLY_COLORS[p], s=110, edgecolor="white",
                                   lw=1.2, zorder=3)
                    else:                                   # recycled: hollow, drawn over the cycle-0 marker it shares x with
                        ax.scatter(bc[fx], bc[col], marker=POLY_MARKERS[p], facecolor="white", edgecolor=POLY_COLORS[p],
                                   s=90, lw=1.8, zorder=4)
                    if connect:
                        bs = bc.sort_values(fx)
                        ax.plot(bs[fx], bs[col], color=POLY_COLORS[p], lw=1.6, alpha=0.75, zorder=2,
                                ls="-" if c == cycles[0] else "--")
            ax.set_yscale("log"); ax.tick_params(labelsize=11); _tidy(ax)
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))    # plain 1-2-5 ticks, not 10^n
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_xlabel(flab, fontsize=12)
            if i == 0:
                ax.set_title(lab, fontsize=13, color=_INK)
    handles = _poly_handles(polys, markersize=10) + [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9a9a96", markeredgecolor="white", markersize=11,
               label="formulation (geometric mean)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9a9a96", alpha=0.4, markeredgecolor="none",
               markersize=6, label="specimen")]
    if len(cycles) > 1:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#6f6f6b",
                              markeredgewidth=1.8, markersize=10, label=f"recycled, cycle {cycles[-1]}"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, save)


_RESP_SHORT = {"max_tensile_stress_MPa": "UTS", "young_modulus_MPa": "Modulus", "strain_at_break": "Strain",
               "toughness_MJ_m3": "Toughness"}


def response_correlation(corrs: dict, save: Path | None = None):
    """Correlation matrices of the responses (from :func:`utils.analysis.response_correlation`), one panel
    per group: diverging map (blue = negative, red = positive, white = 0), r printed in every cell."""
    fig, axes = plt.subplots(1, len(corrs), figsize=(4.1 * len(corrs) + 1.0, 4.4), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12)
    for ax, (label, c) in zip(np.atleast_1d(axes), corrs.items()):
        im = ax.imshow(c.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(c)):
            for j in range(len(c)):
                r = c.iat[i, j]
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if abs(r) > 0.6 else _INK)
        names = [_RESP_SHORT.get(k, k) for k in c.columns]
        ax.set_xticks(range(len(c)), names, rotation=35, ha="right", fontsize=11.5)
        ax.set_yticks(range(len(c)), names, fontsize=11.5)
        ax.set_title(f"{label} (n = {c.attrs.get('n', '?')})", fontsize=13, color=_INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Pearson r of log values", fontsize=11)
    return _save(fig, save)


def pair_plot_specimens(summary: pd.DataFrame, save: Path | None = None):
    """Corner pair plot of the two process factors and the four responses (log scale), one dot per specimen,
    colour and marker per polymer; diagonal: per-polymer histograms."""
    polys = [p for p in POLY_COLORS if (summary["polymer_system"] == p).any()]
    vars_ = [("pal_pct", "PAL %", False), ("extr_temp_C", "Temp. (°C)", False)] + \
            [(c, l.split(" (")[0], True) for c, l in _RESP.items()]
    n = len(vars_)
    fig, axes = plt.subplots(n, n, figsize=(13, 12))
    for i, (yc, yl, ylog) in enumerate(vars_):
        for j, (xc, xl, xlog) in enumerate(vars_):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            xv = np.log10(summary[xc]) if xlog else summary[xc]
            if i == j:
                bins = np.histogram_bin_edges(xv, 12)
                for p in polys:
                    ax.hist(xv[summary["polymer_system"] == p], bins=bins, color=POLY_COLORS[p], alpha=0.55)
                ax.set_yticks([])
            else:
                yv = np.log10(summary[yc]) if ylog else summary[yc]
                for p in polys:
                    m = summary["polymer_system"] == p
                    ax.scatter(xv[m], yv[m], s=16, marker=POLY_MARKERS[p], color=POLY_COLORS[p], edgecolor="white", lw=0.4)
            if i == n - 1:
                ax.set_xlabel(("log₁₀ " if xlog else "") + xl, fontsize=10.5)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(("log₁₀ " if ylog else "") + yl, fontsize=10.5)
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=9); _tidy(ax)
    fig.legend(handles=_poly_handles(polys, markersize=10), loc="upper right", bbox_to_anchor=(0.92, 0.9),
               frameon=False, fontsize=12, title="polymer system", title_fontsize=12)
    fig.tight_layout()
    return _save(fig, save)


def anova_effects(an: pd.DataFrame, save: Path | None = None):
    """Effect-size heatmap: rows = effects, columns = responses, colour = partial eta^2 (one hue, light ->
    dark), text = partial eta^2 and the p-value."""
    effects = list(dict.fromkeys(an["effect"])); resp = list(dict.fromkeys(an["response"]))
    E = an.pivot(index="effect", columns="response", values="partial_eta2").loc[effects, resp]
    P = an.pivot(index="effect", columns="response", values="p").loc[effects, resp]
    fig, ax = plt.subplots(figsize=(10, 0.95 * len(effects) + 1.4))
    im = ax.imshow(E.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(effects)):
        for j in range(len(resp)):
            e, p = E.iat[i, j], P.iat[i, j]
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"
            ax.text(j, i, f"{e:.2f}\n{ptxt}", ha="center", va="center", fontsize=12.5,
                    color="white" if e > 0.55 else _INK, fontweight="bold" if p < 0.05 else "normal")
    ax.set_xticks(range(len(resp)), [_RESP[r].split(" (")[0] for r in resp], fontsize=12.5)
    ax.set_yticks(range(len(effects)), effects, fontsize=12.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("partial η²", fontsize=11)
    fig.tight_layout()
    return _save(fig, save)


def partial_dependence_grid(pds: dict, save: Path | None = None):
    """GP partial dependence: columns = responses, rows = PAL % and temperature; one line per polymer with
    its 5-95 % band (log y)."""
    resp = list(pds)
    fig, axes = plt.subplots(2, len(resp), figsize=(17, 7.4), sharex="row")
    for j, r in enumerate(resp):
        d = pds[r]
        for i, (factor, xl) in enumerate((("pal_pct", "PAL content (%)"), ("extr_temp_C", "Extrusion temperature (°C)"))):
            ax = axes[i, j]
            for p in POLY_COLORS:
                s = d[(d["polymer"] == p) & (d["factor"] == factor)]
                if s.empty:
                    continue
                ax.fill_between(s["x"], s["lo"], s["hi"], color=POLY_COLORS[p], alpha=0.15, lw=0)
                ax.plot(s["x"], s["mean"], color=POLY_COLORS[p], lw=2.2)
            ax.set_yscale("log"); ax.tick_params(labelsize=10.5); _tidy(ax)
            if i == 0:
                ax.set_title(_RESP[r], fontsize=13, color=_INK)
            ax.set_xlabel(xl, fontsize=11.5)
            ax.tick_params(labelbottom=True)
            if j == 0:
                ax.set_ylabel("averaged over temperature" if i == 0 else "averaged over PAL %", fontsize=11)
    handles = [Line2D([0], [0], color=POLY_COLORS[p], lw=2.2, label=p) for p in POLY_COLORS]
    fig.legend(handles=handles + [Patch(color="#9a9a96", alpha=0.3, label="5–95 % band")], loc="lower center",
               ncol=4, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, save)


_SOBOL_TERMS = ["polymer", "PAL", "temperature", "polymer × PAL", "polymer × temperature", "PAL × temperature",
                "polymer × PAL × temperature"]
_SOBOL_COLOR = "#6250d6"   # violet: not a polymer colour


def sobol_indices(sob: pd.DataFrame, save: Path | None = None):
    """GP Sobol indices (from :func:`utils.analysis.gp_sobol`), one column per response. Top: across
    polymers, bars = share of variance per main effect and interaction (median, 5-95 % whisker), open
    diamond = total index of an input. Bottom: within each polymer, the PAL / temperature / PAL x
    temperature split of that polymer's surface (median, 5-95 %)."""
    resp = list(dict.fromkeys(sob["response"]))
    fig, axes = plt.subplots(2, len(resp), figsize=(17, 9.6), sharey="row", height_ratios=(7, 3.6))
    ys = np.array([0, 1, 2, 3.5, 4.5, 5.5, 6.5])
    for j, r in enumerate(resp):
        d = sob[(sob["response"] == r) & (sob["scope"] == "all")]
        ax = axes[0, j]
        for y, term in zip(ys, _SOBOL_TERMS):
            s = d[(d["term"] == term) & (d["order"] != "total")].iloc[0]
            ax.barh(y, s["median"], height=0.62, color=_SOBOL_COLOR, lw=0)
            ax.plot([s["lo"], s["hi"]], [y, y], color=_INK, lw=1.2, solid_capstyle="butt")
            tot = d[(d["term"] == term) & (d["order"] == "total")]
            if not tot.empty:
                ax.scatter(tot["median"], [y], marker="D", s=70, facecolor="white", edgecolor=_INK, lw=1.5, zorder=5)
            if s["median"] >= 0.02:     # values in one column at the right edge, clear of whiskers and diamonds
                ax.text(1.22, y, f"{s['median']:.2f}", ha="right", va="center", fontsize=11, color=_INK)
        ax.axhline(2.75, color="#c9c8c2", lw=0.8)
        if j == len(resp) - 1:
            for y, lab in ((1, "1st order\n(main effects)"), (4.5, "2nd order"), (6.5, "3rd order")):
                ax.text(1.03, y, lab, transform=ax.get_yaxis_transform(), va="center", fontsize=11.5,
                        color=_INK2, style="italic")
        ax.set_xlim(0, 1.24); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.tick_params(labelsize=10.5)
        ax.set_title(_RESP[r], fontsize=13, color=_INK); _tidy(ax); ax.grid(axis="y", visible=False)
        ax.set_xlabel("share of the variance of log(response)", fontsize=11)
        ax = axes[1, j]
        for i, term in enumerate(["PAL", "temperature", "PAL × temperature"]):
            for k, p in enumerate(POLY_COLORS):
                s = sob[(sob["response"] == r) & (sob["scope"] == p) & (sob["term"] == term) & (sob["order"] != "total")]
                if s.empty:
                    continue
                s = s.iloc[0]; y = i + (k - 1) * 0.25
                ax.plot([s["lo"], s["hi"]], [y, y], color=POLY_COLORS[p], lw=1.6)
                ax.scatter([s["median"]], [y], marker=POLY_MARKERS[p], s=70, color=POLY_COLORS[p],
                           edgecolor="white", lw=0.8, zorder=5)
        ax.set_xlim(0, 1.24); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.tick_params(labelsize=10.5)
        _tidy(ax); ax.grid(axis="y", visible=False)
        ax.set_xlabel("share of the variance within the polymer", fontsize=11)
    axes[0, 0].set_yticks(ys, _SOBOL_TERMS, fontsize=11.5); axes[0, 0].invert_yaxis()
    axes[1, 0].set_yticks(range(3), ["PAL", "temperature", "PAL × temperature"], fontsize=11.5)
    axes[1, 0].invert_yaxis()
    axes[0, 0].set_ylabel("across polymers", fontsize=12); axes[1, 0].set_ylabel("within a polymer", fontsize=12)
    handles = [Patch(color=_SOBOL_COLOR, label="Sobol index (main effect or interaction)"),
               Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=_INK,
                      markersize=8, label="total index (main effect + its interactions)"),
               Line2D([0], [0], color=_INK, lw=1.2, label="5–95 % over GP draws")] + _poly_handles()
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, save)


def curves_by_cycle(curves: dict, summary: pd.DataFrame, save: Path | None = None):
    """Stress-strain curves of every formulation measured at two cycles, one panel each (polymer, then PAL %): cycle 0
    light, cycle 1 dark, shared axes per polymer row."""
    d = summary.assign(_cycle=summary["recycled_x"].fillna(0).astype(int))
    both = d.groupby("formulation")["_cycle"].nunique()
    d = d[d["formulation"].isin(both[both > 1].index)]
    polys = [p for p in POLY_COLORS if (d["polymer_system"] == p).any()]
    forms = {p: d[d["polymer_system"] == p].groupby("formulation")[["pal_pct", "extr_temp_C"]].first().sort_values("pal_pct") for p in polys}
    ncol = max(len(f) for f in forms.values())
    fig, axes = plt.subplots(len(polys), ncol, figsize=(4.1 * ncol, 3.25 * len(polys)), sharex="row", sharey="row", squeeze=False)
    shade = {0: CYCLE_COLORS[0], 1: CYCLE_COLORS[3]}
    for i, p in enumerate(polys):
        for j, (f, row) in enumerate(forms[p].iterrows()):
            ax = axes[i, j]
            for _, r in d[d["formulation"] == f].sort_values("_cycle").iterrows():
                cv = curves[r["specimen_id"]].iloc[: int(r["i_break"]) + 5]
                ax.plot(100 * (cv["strain"] - r["toe_offset_strain"]), cv["stress_MPa"], color=shade[r["_cycle"]], lw=1.6)
            ax.set_title(f"{p.split('+')[0]} F{f:02d} · {int(row.pal_pct)} % PAL · {int(row.extr_temp_C)} °C", fontsize=12.5, color=_INK)
            ax.tick_params(labelsize=11, labelbottom=True); _tidy(ax)
            ax.set_xlabel("Strain (%)", fontsize=11.5)
        axes[i, 0].set_ylabel("Stress (MPa)", fontsize=11.5)
    for i in range(len(polys)):                        # after all panels: a row's shared limits must cover every curve in it
        axes[i, 0].set_xlim(left=0); axes[i, 0].set_ylim(bottom=0)
    fig.legend(handles=[Line2D([0], [0], color=shade[c], lw=2.2, label=f"cycle {c}") for c in shade], loc="lower center", ncol=2,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, save)


def cycle_ratio_forest(pairs: pd.DataFrame, effect: pd.DataFrame, save: Path | None = None):
    """Effect of the recycle step: per formulation the ratio cycle 1 / cycle 0 with its 95 % CI (log x), one panel per
    response; bottom row = average across formulations (paired, 95 % CI). Right of 1 = higher after recycling."""
    resp = list(dict.fromkeys(pairs["response"]))
    order = {p: i for i, p in enumerate(POLY_COLORS)}
    fs = (pairs[pairs["response"] == resp[0]].assign(_o=lambda d: d["polymer_system"].map(order))
          .sort_values(["_o", "pal_pct"]))
    ys = {f: i for i, f in enumerate(fs["formulation"])}
    fig, axes = plt.subplots(1, len(resp), figsize=(17, 2.6 + 0.37 * len(ys)), sharey=True)
    for ax, r in zip(axes, resp):
        d = pairs[pairs["response"] == r]
        ax.axvline(1, color="#8a8f98", lw=1.2, zorder=1)
        for _, x in d.iterrows():
            y, c = ys[x["formulation"]], POLY_COLORS[x["polymer_system"]]
            ax.plot([x["lo"], x["hi"]], [y, y], color=c, lw=1.8, zorder=2)
            ax.scatter([x["ratio"]], [y], marker=POLY_MARKERS[x["polymer_system"]], s=85, color=c, edgecolor="white", lw=0.9, zorder=3)
        e = effect[(effect["response"] == r) & (effect["group"] == "all")].iloc[0]
        y = len(ys) + 0.3
        ax.plot([e["lo"], e["hi"]], [y, y], color=_INK, lw=2.4, zorder=2)
        ax.scatter([e["ratio"]], [y], marker="D", s=95, color=_INK, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(_RESP[r].split(" (")[0], fontsize=13, color=_INK)
        ax.set_xlabel("cycle 1 / cycle 0", fontsize=12); ax.tick_params(labelsize=11); _tidy(ax); ax.grid(axis="y", visible=False)
    labels = [f"{p.split('+')[0]} F{f:02d} · {int(a)} % · {int(t)} °C" for f, p, a, t in
              zip(fs["formulation"], fs["polymer_system"], fs["pal_pct"], fs["extr_temp_C"])] + [f"average ({len(ys)} pairs)"]
    axes[0].set_yticks(list(ys.values()) + [len(ys) + 0.3], labels, fontsize=11.5)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return _save(fig, save)


def cycle_gain_loss(pairs: pd.DataFrame, effect: pd.DataFrame, means: pd.DataFrame, log_y: bool = False, per_polymer: bool = False,
                    save: Path | None = None):
    """Gain or loss from cycle 0 to cycle 1, one panel per response on its real scale: thin line = one formulation (colour
    = polymer); thick line = the average across formulations (geometric mean, 95 % CI whiskers), overall in black or, with
    ``per_polymer``, one per polymer in its colour. Title = the overall average change."""
    resp = list(dict.fromkeys(pairs["response"]))
    polys = [p for p in POLY_COLORS if (pairs["polymer_system"] == p).any()]
    fig, axes = plt.subplots(1, len(resp), figsize=(17, 5.6))
    groups = [(p, POLY_COLORS[p], POLY_MARKERS[p]) for p in polys] if per_polymer else [("all", _INK, "D")]
    for ax, r in zip(axes, resp):
        d = pairs[pairs["response"] == r]
        scale = 100 if r == "strain_at_break" else 1
        for _, x in d.iterrows():
            ax.plot([0, 1], [x["gm_0"] * scale, x["gm_1"] * scale], color=POLY_COLORS[x["polymer_system"]], lw=1.3, alpha=0.5, zorder=2,
                    marker=POLY_MARKERS[x["polymer_system"]], ms=5, mec="white", mew=0.5)
        big = d.assign(_m=np.abs(np.log(d["ratio"]))).nlargest(2, "_m")["formulation"].tolist()   # the two largest moves
        for _, x in d[d["formulation"].isin(big)].iterrows():
            ax.annotate(f"F{int(x['formulation']):02d}", (1, x["gm_1"] * scale), xytext=(7, 0), textcoords="offset points",
                        fontsize=10.5, color=_INK2, va="center")
        for k, (g, col, mk) in enumerate(groups):
            m = means[(means["response"] == r) & (means["group"] == g)].sort_values("cycle")
            off = 0.14 + 0.06 * k
            if per_polymer:                                 # whiskers would show the spread of levels, not the change: label the ratio instead
                ax.plot([-off, 1 + off], m["gm"] * scale, color=col, lw=3.4, marker=mk, ms=10, mec="white", mew=0.8, zorder=4)
                e = effect[(effect["response"] == r) & (effect["group"] == g)].iloc[0]
                ax.annotate(f"×{e['ratio']:.2f}", (1 + off, m["gm"].iloc[-1] * scale), xytext=(8, 0), textcoords="offset points", va="center",
                            fontsize=12, fontweight="bold", color=col)
            else:
                ax.errorbar([-off, 1 + off], m["gm"] * scale, yerr=[(m["gm"] - m["lo"]) * scale, (m["hi"] - m["gm"]) * scale], color=col, lw=0,
                            elinewidth=2, capsize=4, capthick=2, marker=mk, ms=9, mec="white", mew=0.8, zorder=4)
                ax.plot([-off, 1 + off], m["gm"] * scale, color=col, lw=3.2, zorder=3)
        e = effect[(effect["response"] == r) & (effect["group"] == "all")].iloc[0]
        chg = lambda v: f"{100 * (v - 1):+.0f}".replace("-", "−")
        ax.set_title(f"{_RESP[r].split(' (')[0]}: {chg(e['ratio'])} % on average\n95 % CI {chg(e['lo'])} to {chg(e['hi'])} %, p = {e['p']:.2f}",
                     fontsize=12.5, color=_INK)
        unit = {"max_tensile_stress_MPa": "MPa", "young_modulus_MPa": "MPa", "strain_at_break": "%", "toughness_MJ_m3": "MJ/m³"}[r]
        ax.set_ylabel(unit, fontsize=12); ax.set_xticks([0, 1], ["cycle 0", "cycle 1"], fontsize=12.5)
        ax.set_xlim(-0.5, 1.5 + (0.35 if per_polymer else 0)); ax.tick_params(axis="y", labelsize=11); _tidy(ax); ax.grid(axis="x", visible=False)
        if log_y:                                           # polymers 1-2 orders of magnitude apart: log axis keeps every line readable
            ax.set_yscale("log"); ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_ylim(bottom=0)
    n = pairs["formulation"].nunique()
    handles = [Line2D([0], [0], color=POLY_COLORS[p], lw=1.3, alpha=0.6, marker=POLY_MARKERS[p], ms=6, mec="white", label=f"{p} formulation") for p in polys]
    handles += ([Line2D([0], [0], color=_INK2, lw=3.2, marker="o", ms=8, label="thick: polymer average of its 4 formulations; × = its ratio cycle 1 / cycle 0")] if per_polymer
                else [Line2D([0], [0], color=_INK, lw=3.2, marker="D", ms=8, label=f"average of the {n} formulations, with 95 % CI")])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, save)


def cycle_trends(summary: pd.DataFrame, save: Path | None = None):
    """Same trend at both cycles? The formulations measured at both, one panel per response and factor: each response
    against PAL % (adjusted to the mean temperature) and against temperature (adjusted to the mean PAL %), with a straight
    fit per polymer; cycle 0 filled and solid, cycle 1 hollow and dashed (log y)."""
    import statsmodels.formula.api as smf
    d = summary.assign(_c=summary["recycled_x"].fillna(0).astype(int))
    both = d.groupby("formulation")["_c"].nunique()
    d = d[d["formulation"].isin(both[both > 1].index)]
    polys = [p for p in POLY_COLORS if (d["polymer_system"] == p).any()]
    fig, axes = plt.subplots(2, len(_RESP), figsize=(14.5, 6.3), sharey="col")
    for j, (r, lab) in enumerate(_RESP.items()):
        m = (d.assign(ly=np.log(d[r].astype(float))).groupby(["formulation", "_c", "polymer_system", "pal_pct", "extr_temp_C"], as_index=False)["ly"].mean()
             .rename(columns={"polymer_system": "poly", "pal_pct": "pal", "extr_temp_C": "temp"}))
        for i, (x, other, xl) in enumerate((("pal", "temp", "PAL content (%)"), ("temp", "pal", "Extrusion temperature (°C)"))):
            ax = axes[i, j]
            for c in (0, 1):
                mc = m[m["_c"] == c]
                f = smf.ols("ly ~ C(poly) + pal + temp", mc).fit()
                y = mc["ly"] - f.params[other] * (mc[other] - mc[other].mean())       # adjusted to the mean of the other factor
                for p in polys:
                    k = mc["poly"] == p
                    if c == 0:
                        ax.scatter(mc.loc[k, x], np.exp(y[k]), marker=POLY_MARKERS[p], s=70, color=POLY_COLORS[p], edgecolor="white", lw=0.8, zorder=3)
                    else:
                        ax.scatter(mc.loc[k, x], np.exp(y[k]), marker=POLY_MARKERS[p], s=62, facecolor="white", edgecolor=POLY_COLORS[p], lw=1.6, zorder=4)
                    xs = np.array([mc.loc[k, x].min(), mc.loc[k, x].max()])
                    level = y[k].mean() + f.params[x] * (xs - mc.loc[k, x].mean())   # the common slope of this cycle
                    ax.plot(xs, np.exp(level), color=POLY_COLORS[p], lw=2, ls="-" if c == 0 else "--", zorder=2)
            ax.set_yscale("log"); ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(labelsize=11); _tidy(ax); ax.set_xlabel(xl, fontsize=12)
            if i == 0:
                ax.set_title(lab, fontsize=13, color=_INK)
    handles = _poly_handles(polys, markersize=10) + [
        Line2D([0], [0], color=_INK2, lw=2, marker="o", ms=8, label="cycle 0"),
        Line2D([0], [0], color=_INK2, lw=2, ls="--", marker="o", ms=8, mfc="white", label="cycle 1")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, save)


def lvgp_surfaces(s: dict, table: pd.DataFrame, label: str, polymers=("PP+PAL", "HDPE+PAL"), other="TPU+PAL", cycles=(0, 1),
                  save: Path | None = None):
    """``other`` = a polymer with cycle 0 only (its own column); None when every polymer has both cycles."""
    if other is None:
        return _lvgp_surfaces_rows(s, table, label, polymers, cycles, save)
    """The LVGP's fitted surfaces for one response. For each polymer with both cycles (rows): PAL % x temperature at cycle 0,
    at cycle 1 (colour scale shared by the two), and their ratio (diverging around 1, dashed contour at 1, dotted where the
    90 % band excludes 1). Last column: ``other``, a polymer with no recycled data, cycle 0 only. Markers = the runs, numbers =
    their measured geometric means. ``s`` = lvgp_compare.cycle_surfaces, ``table`` = lvgp_compare.run_table."""
    from matplotlib.colors import LogNorm
    P, T = np.meshgrid(s["pal"], s["temp"], indexing="ij")
    fig, axes = plt.subplots(len(polymers), 4, figsize=(15, 7.5), sharex=True, sharey=True, layout="constrained")

    def panel(ax, p, c, vals):
        im = ax.contourf(P, T, s["surf"][(p, c)], levels=np.geomspace(vals.min(), vals.max(), 14), norm=LogNorm(vals.min(), vals.max()), cmap="Blues")
        r = table[(table["polymer"] == p) & (table["cycle"] == c)]
        ax.scatter(r["pal"], r["temp"], marker=POLY_MARKERS[p], s=75, color="white", edgecolor=_INK, lw=1.2, zorder=3, clip_on=False)
        for _, x in r.iterrows():
            v = f"{np.exp(x['mean_log']):.3g}"
            below = x["temp"] > 228
            ax.annotate(v, (x["pal"], x["temp"]), xytext=(0, -15 if below else 10), textcoords="offset points", ha="right" if x["pal"] > 42 else "left" if x["pal"] < 14 else "center",
                        va="center", fontsize=12, color=_INK, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
        return im

    for i, p in enumerate(polymers):
        vals = np.concatenate([s["surf"][(p, c)].ravel() for c in cycles])
        for j, c in enumerate(cycles):
            im = panel(axes[i, j], p, c, vals)
            if i == 0:
                axes[i, j].set_title(f"cycle {c}", fontsize=14, color=_INK)
        cb = fig.colorbar(im, ax=axes[i, :2], fraction=0.035, pad=0.012, format="%.3g"); cb.ax.tick_params(labelsize=11)
        axes[i, 0].set_ylabel(f"{p.split('+')[0]}: temperature (°C)", fontsize=13)
        ax = axes[i, 2]; R = s["ratio"][p]; lim = 2.0
        cf = ax.contourf(P, T, np.clip(R["ratio"], 1 / lim, lim), levels=np.geomspace(1 / lim, lim, 15), norm=LogNorm(1 / lim, lim), cmap="RdBu_r")
        ax.contour(P, T, R["ratio"], levels=[1], colors=_INK, linewidths=1.5, linestyles="--")
        sure = (R["lo"] > 1) | (R["hi"] < 1)
        if sure.any():
            ax.contourf(P, T, sure.astype(float), levels=[0.5, 1.5], colors="none", hatches=["...."])
        r = table[(table["polymer"] == p) & (table["cycle"] == cycles[1])]
        ax.scatter(r["pal"], r["temp"], marker=POLY_MARKERS[p], s=60, color="white", edgecolor=_INK, lw=1.2, zorder=3, clip_on=False)
        cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.02, ticks=[0.5, 0.75, 1, 1.5, 2], format="%g"); cb.ax.tick_params(labelsize=11)
        if i == 0:
            ax.set_title("ratio cycle 1 / cycle 0", fontsize=14, color=_INK)
    vals = s["surf"][(other, cycles[0])].ravel()
    im = panel(axes[0, 3], other, cycles[0], vals)
    axes[0, 3].set_title(f"{other.split('+')[0]}, cycle 0", fontsize=14, color=_INK)
    cb = fig.colorbar(im, ax=axes[0, 3], fraction=0.05, pad=0.02, format="%.3g"); cb.ax.tick_params(labelsize=11)
    axes[1, 3].axis("off")
    axes[1, 3].text(0.5, 0.55, f"{other.split('+')[0]}: no recycled\ndata yet", transform=axes[1, 3].transAxes, ha="center", va="center", fontsize=14, color=_INK2)
    for ax in axes.ravel():
        ax.tick_params(labelsize=11, labelbottom=True)
    for ax in axes[1, :3]:
        ax.set_xlabel("PAL content (%)", fontsize=13)
    axes[0, 3].set_xlabel("PAL content (%)", fontsize=13)
    return _save(fig, save)


def _lvgp_surfaces_rows(s, table, label, polymers, cycles, save):
    """One row per polymer: cycle 0, cycle 1 (shared colour scale per row) and their ratio."""
    from matplotlib.colors import LogNorm
    P, T = np.meshgrid(s["pal"], s["temp"], indexing="ij")
    fig, axes = plt.subplots(len(polymers), 3, figsize=(12.2, 3.35 * len(polymers)), sharex=True, sharey=True, layout="constrained")
    for i, p in enumerate(polymers):
        vals = np.concatenate([s["surf"][(p, c)].ravel() for c in cycles])
        for j, c in enumerate(cycles):
            ax = axes[i, j]
            im = ax.contourf(P, T, s["surf"][(p, c)], levels=np.geomspace(vals.min(), vals.max(), 14), norm=LogNorm(vals.min(), vals.max()), cmap="Blues")
            r = table[(table["polymer"] == p) & (table["cycle"] == c)]
            ax.scatter(r["pal"], r["temp"], marker=POLY_MARKERS[p], s=70, color="white", edgecolor=_INK, lw=1.2, zorder=3, clip_on=False)
            for _, x in r.iterrows():
                ax.annotate(f"{np.exp(x['mean_log']):.3g}", (x["pal"], x["temp"]), xytext=(0, -15 if x["temp"] > 228 else 10), textcoords="offset points",
                            ha="right" if x["pal"] > 42 else "left" if x["pal"] < 14 else "center", va="center", fontsize=11.5, color=_INK,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
            if i == 0:
                ax.set_title(f"cycle {c}", fontsize=14, color=_INK)
        cb = fig.colorbar(im, ax=axes[i, :2], fraction=0.035, pad=0.012, format="%.3g"); cb.ax.tick_params(labelsize=10.5)
        axes[i, 0].set_ylabel(f"{p.split('+')[0]}: temperature (°C)", fontsize=12.5)
        ax = axes[i, 2]; R = s["ratio"][p]; lim = 2.0
        cf = ax.contourf(P, T, np.clip(R["ratio"], 1 / lim, lim), levels=np.geomspace(1 / lim, lim, 15), norm=LogNorm(1 / lim, lim), cmap="RdBu_r")
        ax.contour(P, T, R["ratio"], levels=[1], colors=_INK, linewidths=1.5, linestyles="--")
        sure = (R["lo"] > 1) | (R["hi"] < 1)
        if sure.any():
            ax.contourf(P, T, sure.astype(float), levels=[0.5, 1.5], colors="none", hatches=["...."])
        r = table[(table["polymer"] == p) & (table["cycle"] == cycles[1])]
        ax.scatter(r["pal"], r["temp"], marker=POLY_MARKERS[p], s=55, color="white", edgecolor=_INK, lw=1.2, zorder=3, clip_on=False)
        cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.02, ticks=[0.5, 0.75, 1, 1.5, 2], format="%g"); cb.ax.tick_params(labelsize=10.5)
        if i == 0:
            ax.set_title("ratio cycle 1 / cycle 0", fontsize=14, color=_INK)
    for ax in axes.ravel():
        ax.tick_params(labelsize=10.5, labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel("PAL content (%)", fontsize=12.5)
    return _save(fig, save)


def lvgp_surface_3d(s: dict, table: pd.DataFrame, label: str, polymers=("PP+PAL", "HDPE+PAL"), cycles=(0, 1), save: Path | None = None):
    """The same surfaces as a 3-D view: per polymer the cycle 0 (light) and cycle 1 (dark) surfaces, with the runs as markers."""
    P, T = np.meshgrid(s["pal"], s["temp"], indexing="ij")
    fig = plt.figure(figsize=(6.2 * len(polymers), 5.2))
    col = {0: "#a9a8a2", 1: "#2a78d6"}
    for k, p in enumerate(polymers):
        ax = fig.add_subplot(1, len(polymers), k + 1, projection="3d")
        for c in cycles:
            ax.plot_surface(P, T, s["surf"][(p, c)], color=col[c], alpha=0.8 if c == 0 else 0.55, lw=0, antialiased=True, shade=False, rstride=2, cstride=2)
            ax.plot_wireframe(P, T, s["surf"][(p, c)], color=col[c], lw=0.35, rstride=5, cstride=5, alpha=0.9)
            r = table[(table["polymer"] == p) & (table["cycle"] == c)]
            ax.scatter(r["pal"], r["temp"], np.exp(r["mean_log"]), color=col[c], edgecolor=_INK, s=50, depthshade=False)
        ax.set_xlabel("PAL (%)", fontsize=11); ax.set_ylabel("Temperature (°C)", fontsize=11); ax.set_zlabel(label, fontsize=11)
        ax.set_title(p.split("+")[0], fontsize=13, color=_INK); ax.view_init(24, -58); ax.tick_params(labelsize=9)
    fig.legend(handles=[Patch(color=col[c], label=f"cycle {c}") for c in cycles], loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, save)


def read_ratio_example(save: Path | None = None):
    """Teaching figure: three invented example intervals on the ratio axis, showing how to read a ratio with its CI."""
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    rows = [(1.45, 1.15, 1.8, "#1c5cab", "Clear gain: all of it above 1"), (1.15, 0.8, 1.6, "#6f6f6b", "Not clear: the interval crosses 1"),
            (0.7, 0.55, 0.88, "#c7472f", "Clear loss: all of it below 1")]
    ax.axvline(1, color="#6f6f6b", lw=1.6, ls=":", zorder=1)
    for i, (m, lo, hi, c, t) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=c, lw=3, zorder=2)
        ax.plot([lo, lo], [i - 0.12, i + 0.12], color=c, lw=3); ax.plot([hi, hi], [i - 0.12, i + 0.12], color=c, lw=3)
        ax.scatter([m], [i], s=120, color=c, zorder=3, edgecolor="white")
        ax.text(2.25, i, t, va="center", fontsize=13, color=_INK)
    ax.set_xscale("log"); ax.set_xlim(0.45, 2.2); ax.set_ylim(2.6, -0.8)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(0.5, 1, 2))); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([]); ax.tick_params(labelsize=12)
    ax.set_xlabel("ratio cycle 1 / cycle 0   (example values)", fontsize=13)
    ax.text(1, -0.55, "1 = no change", ha="center", fontsize=13, color=_INK2)
    ax.text(0.47, -0.55, "← lost", fontsize=13, color=_INK2); ax.text(2.15, -0.55, "gained →", ha="right", fontsize=13, color=_INK2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.03, right=0.6, bottom=0.2, top=0.98)
    return _save(fig, save)


def warping_surfaces(s: dict, warp: pd.DataFrame, polymers=("PP+PAL", "HDPE+PAL", "TPU+PAL"), save: Path | None = None):
    """The LVGP's mean corner warping per polymer on PAL % x temperature (one shared colour scale), measured formulation
    means as markers with their value. ``s`` = dict(pal, temp, surf={polymer: array})."""
    from matplotlib.colors import Normalize
    P, T = np.meshgrid(s["pal"], s["temp"], indexing="ij")
    vals = np.concatenate([s["surf"][p].ravel() for p in polymers])
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), sharey=True, layout="constrained")
    g = warp.groupby(["formulation", "polymer_system", "pal_pct", "extr_temp_C"])["warping_cm"].mean().reset_index()
    for ax, p in zip(axes, polymers):
        im = ax.contourf(P, T, s["surf"][p], levels=np.linspace(vals.min(), vals.max(), 14), norm=Normalize(vals.min(), vals.max()), cmap="Oranges")
        r = g[g["polymer_system"] == p]
        ax.scatter(r["pal_pct"], r["extr_temp_C"], marker=POLY_MARKERS[p], s=70, color="white", edgecolor=_INK, lw=1.2, zorder=3)
        for _, x in r.iterrows():
            ax.annotate(f"{x['warping_cm']:.2f}", (x["pal_pct"], x["extr_temp_C"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10.5,
                        color=_INK, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
        ax.set_title(p.split("+")[0], fontsize=13, color=_INK); ax.set_xlabel("PAL content (%)", fontsize=12); ax.tick_params(labelsize=11)
    axes[0].set_ylabel("Extrusion temperature (°C)", fontsize=12)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015); cb.set_label("predicted corner warping (cm)", fontsize=11)
    return _save(fig, save)


def cycle_ratio_vs_temperature(pairs: pd.DataFrame, responses=("strain_at_break", "toughness_MJ_m3", "max_tensile_stress_MPa"),
                               save: Path | None = None):
    """Ratio cycle 1 / cycle 0 of each formulation against its extrusion temperature (log y), with its 95 % CI."""
    fig, axes = plt.subplots(1, len(responses), figsize=(5.2 * len(responses), 4.5))
    for ax, r in zip(np.atleast_1d(axes), responses):
        d = pairs[pairs["response"] == r]
        ax.axhline(1, color="#8a8f98", lw=1.2, zorder=1)
        for _, x in d.iterrows():
            c = POLY_COLORS[x["polymer_system"]]
            ax.plot([x["extr_temp_C"]] * 2, [x["lo"], x["hi"]], color=c, lw=1.4, alpha=0.55, zorder=2)
            ax.scatter([x["extr_temp_C"]], [x["ratio"]], marker=POLY_MARKERS[x["polymer_system"]], s=90, color=c, edgecolor="white", lw=0.9, zorder=3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(_RESP[r].split(" (")[0], fontsize=13, color=_INK)
        ax.set_xlabel("Extrusion temperature (°C)", fontsize=12); ax.tick_params(labelsize=11); _tidy(ax)
    np.atleast_1d(axes)[0].set_ylabel("cycle 1 / cycle 0", fontsize=12)
    np.atleast_1d(axes)[-1].legend(handles=_poly_handles(sorted(pairs["polymer_system"].unique(), key=list(POLY_COLORS).index), markersize=10),
                                   frameon=False, fontsize=11.5, loc="upper left")
    fig.tight_layout()
    return _save(fig, save)


def partial_dependence_cycles(pds: dict, save: Path | None = None):
    """GP partial dependence by cycle (from :func:`utils.analysis.gp_partial_dependence_cycles`): columns = responses,
    rows = PAL % and temperature; one colour per polymer, cycle 0 solid, cycle 1 dashed, 5-95 % bands (log y)."""
    resp = list(pds)
    fig, axes = plt.subplots(2, len(resp), figsize=(14.5, 6.3), sharex="row")
    for j, r in enumerate(resp):
        d = pds[r]
        for i, (factor, xl) in enumerate((("pal_pct", "PAL content (%)"), ("extr_temp_C", "Extrusion temperature (°C)"))):
            ax = axes[i, j]
            for p in POLY_COLORS:
                for c, ls in ((0, "-"), (1, "--")):
                    s = d[(d["polymer"] == p) & (d["factor"] == factor) & (d["cycle"] == c)]
                    if s.empty:
                        continue
                    ax.fill_between(s["x"], s["lo"], s["hi"], color=POLY_COLORS[p], alpha=0.11, lw=0)
                    ax.plot(s["x"], s["mean"], color=POLY_COLORS[p], lw=2.2, ls=ls)
            ax.set_yscale("log"); ax.tick_params(labelsize=10.5); _tidy(ax)
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            if i == 0:
                ax.set_title(_RESP[r], fontsize=13, color=_INK)
            ax.set_xlabel(xl, fontsize=11.5); ax.tick_params(labelbottom=True)
            if j == 0:
                ax.set_ylabel("averaged over temperature" if i == 0 else "averaged over PAL %", fontsize=11)
    handles = [Line2D([0], [0], color=POLY_COLORS[p], lw=2.2, label=p) for p in POLY_COLORS] + [
        Line2D([0], [0], color=_INK2, lw=2.2, label="cycle 0"), Line2D([0], [0], color=_INK2, lw=2.2, ls="--", label="cycle 1"),
        Patch(color="#9a9a96", alpha=0.3, label="5–95 % band")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, save)


def warping_by_formulation(warp: pd.DataFrame, save: Path | None = None):
    """Corner warping (cm). Left: every corner per formulation (polymer, then PAL %), bar = mean. Right: formulation means
    against PAL % and against extrusion temperature."""
    order = {p: i for i, p in enumerate(POLY_COLORS)}
    forms = (warp.groupby(["formulation", "polymer_system", "pal_pct", "extr_temp_C"], as_index=False)["warping_cm"].mean()
             .assign(_o=lambda d: d["polymer_system"].map(order)).sort_values(["_o", "pal_pct"]).reset_index(drop=True))
    xpos = {f: i + 0.6 * o for i, (f, o) in enumerate(zip(forms["formulation"], forms["_o"]))}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), width_ratios=(2.3, 1, 1), sharey=True)
    ax = axes[0]
    for _, fr in forms.iterrows():
        v = warp.loc[warp["formulation"] == fr["formulation"], "warping_cm"].sort_values().to_numpy()
        x, poly = xpos[fr["formulation"]], fr["polymer_system"]
        seen = {}
        off = []
        for val in v:                                   # spread corners that share a reading
            k = seen.get(val, 0); seen[val] = k + 1
            off.append(0.075 * ((k + 1) // 2) * (1 if k % 2 else -1))
        ax.scatter(x + np.array(off), v, s=30, marker=POLY_MARKERS[poly], color=POLY_COLORS[poly], alpha=0.5, edgecolor="none", zorder=2)
        ax.plot([x - 0.32, x + 0.32], [fr["warping_cm"]] * 2, color=_INK, lw=2.2, zorder=3)
    ax.set_xticks(list(xpos.values()), [f"F{f:02d}\n{int(a)} %" for f, a in zip(forms["formulation"], forms["pal_pct"])], fontsize=11)
    ax.set_ylabel("Corner warping (cm)", fontsize=12.5); ax.set_ylim(bottom=0); ax.tick_params(axis="y", labelsize=11); _tidy(ax)
    ax.grid(axis="x", visible=False)
    for ax, (col, lab) in zip(axes[1:], (("pal_pct", "PAL content (%)"), ("extr_temp_C", "Extrusion temperature (°C)"))):
        for p in POLY_COLORS:
            s = forms[forms["polymer_system"] == p]
            ax.scatter(s[col], s["warping_cm"], s=110, marker=POLY_MARKERS[p], color=POLY_COLORS[p], edgecolor="white", lw=1.1)
        ax.set_xlabel(lab, fontsize=12.5); ax.tick_params(labelsize=11); _tidy(ax)
    fig.legend(handles=_poly_handles(markersize=10) + [Line2D([0], [0], color=_INK, lw=2.2, label="formulation mean")], loc="lower center",
               ncol=4, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, save)


def tradeoffs(summary: pd.DataFrame, save: Path | None = None):
    """Two trade-off views per specimen (log-log): UTS vs toughness and modulus vs strain at break."""
    polys = [p for p in POLY_COLORS if (summary["polymer_system"] == p).any()]
    pairs = [("max_tensile_stress_MPa", "toughness_MJ_m3"), ("young_modulus_MPa", "strain_at_break")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (xc, yc) in zip(axes, pairs):
        for p in polys:
            m = summary["polymer_system"] == p
            ax.scatter(summary.loc[m, xc], summary.loc[m, yc], s=40, marker=POLY_MARKERS[p], color=POLY_COLORS[p],
                       edgecolor="white", lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(_RESP[xc], fontsize=12.5); ax.set_ylabel(_RESP[yc], fontsize=12.5)
        ax.tick_params(labelsize=11); _tidy(ax)
    axes[1].legend(handles=_poly_handles(polys, markersize=10), frameon=False, fontsize=11.5, loc="upper right")
    fig.tight_layout()
    return _save(fig, save)


def cv_heatmap(cv: pd.DataFrame, save: Path | None = None):
    """Replicate CV (%) per formulation (rows, grouped by polymer) and response (columns); one hue."""
    resp = [c for c in _RESP if c in cv.columns]
    M = cv[resp].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(cv) + 1.4))
    im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=60, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=11, color="white" if M[i, j] > 40 else _INK)
    ax.set_xticks(range(len(resp)), [_RESP[r].split(" (")[0] for r in resp], fontsize=11.5)
    ax.set_yticks(range(len(cv)), [f"{p.split('+')[0]}  {int(a)} % · {int(t)} °C  (n={n})"
                                   for p, a, t, n in zip(cv["polymer_system"], cv["pal_pct"], cv["extr_temp_C"], cv["n"])],
                  fontsize=10.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("CV (%)", fontsize=11)
    fig.tight_layout()
    return _save(fig, save)


def design_balance(new: pd.DataFrame, fixed: pd.DataFrame, pal_range=(10, 50), temp_range=(180, 240), cycles=(0, 1, 2, 3),
                   save: Path | None = None):
    """How balanced the data is, before and after the new runs. Left: runs per polymer and cycle, cell text = fabricated
    + new, colour = total. Middle and right: each polymer's PAL % and temperature settings, hollow = fabricated, filled =
    new, so the gaps being filled are visible."""
    polys = [p for p in POLY_COLORS if (fixed["polymer_system"] == p).any() or (new["polymer_system"] == p).any()]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), width_ratios=(1.15, 1, 1))
    ax = axes[0]
    F = fixed.groupby(["polymer_system", "recycle_cycle"]).size().unstack(fill_value=0).reindex(index=polys, columns=cycles, fill_value=0)
    N = new.groupby(["polymer_system", "recycle_cycle"]).size().unstack(fill_value=0).reindex(index=polys, columns=cycles, fill_value=0)
    T = F + N
    ax.imshow(T.to_numpy(float), cmap="Blues", vmin=0, vmax=max(6, float(T.to_numpy().max())), aspect="auto")
    for i in range(len(polys)):
        for j in range(len(cycles)):
            f, n = int(F.iat[i, j]), int(N.iat[i, j])
            txt = (f"{f}" if f else "–") + (f" + {n}" if n else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=13, fontweight="bold" if n else "normal",
                    color="white" if T.iat[i, j] > 3.5 else _INK)
    ax.set_xticks(range(len(cycles)), [f"cycle {c}" for c in cycles], fontsize=11.5)
    ax.set_yticks(range(len(polys)), [p.split("+")[0] for p in polys], fontsize=12)
    ax.set_title("Runs per polymer and cycle: fabricated + new", fontsize=12.5, color=_INK); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for ax, (col, lab, rng_) in zip(axes[1:], (("PAL_pct", "PAL content (%)", pal_range), ("extrusion_temp_C", "Extrusion temperature (°C)", temp_range))):
        for i, p in enumerate(polys):
            fv = fixed.loc[fixed["polymer_system"] == p, col].drop_duplicates()
            nv = new.loc[new["polymer_system"] == p, col]
            ax.scatter(fv, np.full(len(fv), i), marker=POLY_MARKERS[p], s=120, facecolor="white", edgecolor=POLY_COLORS[p], lw=2, zorder=3)
            ax.scatter(nv, np.full(len(nv), i), marker=POLY_MARKERS[p], s=120, color=POLY_COLORS[p], edgecolor="white", lw=1, zorder=4)
            ax.hlines(i, rng_[0], rng_[1], color="#e4e3df", lw=6, zorder=1)
        ax.set_yticks(range(len(polys)), [p.split("+")[0] for p in polys], fontsize=12); ax.invert_yaxis()
        ax.set_xlim(rng_[0] - 2, rng_[1] + 2); ax.set_xlabel(lab, fontsize=12); ax.tick_params(labelsize=11)
        ax.set_title(f"{lab.split(' (')[0]} settings per polymer", fontsize=12.5, color=_INK)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
    fig.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#6f6f6b", markeredgewidth=2, markersize=10, label="fabricated"),
                        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6f6f6b", markeredgecolor="white", markersize=10, label="new")],
               loc="lower center", ncol=2, frameon=False, fontsize=11.5, bbox_to_anchor=(0.66, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, save)


def design_size_tradeoff(nsweep: pd.DataFrame, chosen=(6, 7), col: str = "n_per_polymer", xlabel: str | None = None,
                         save: Path | None = None):
    """Coverage vs number of new runs (conditioned design): within-polymer fill distance and discrepancy,
    mean +/- 1 std over seeds; the counts in ``chosen`` are highlighted (first = selected). ``col`` is the
    size column, per polymer by default or a total."""
    g = nsweep.groupby(col)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (col, lab) in zip(axes, (("poly_fill_dist", "Largest gap within a polymer\n(fill distance, ↓ better)"),
                                     ("poly_discrepancy", "Unevenness within a polymer\n(discrepancy, ↓ better)"))):
        m, s = g[col].mean(), g[col].std()
        ax.errorbar(m.index, m, yerr=s, color=_INK2, lw=1.6, marker="o", ms=7, capsize=3)
        for n in chosen:
            ax.scatter([n], [m[n]], s=150, color="#2a78d6" if n == chosen[0] else "#9a9a96", zorder=5,
                       edgecolor="white", lw=1.2)
        ax.set_xlabel(xlabel or "new configurations per polymer", fontsize=12); ax.set_title(lab, fontsize=12.5, color=_INK)
        ax.set_xticks(list(m.index)); ax.tick_params(labelsize=11); _tidy(ax)
    fig.tight_layout()
    return _save(fig, save)
