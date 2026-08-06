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
    cv_cols = ["CV Young's modulus (%)", "CV UTS (%)", "CV toughness (%)"]
    fmt = {c: "{:.1f}" for c in cv_cols}
    fmt.update({"PP %": "{:.0f}", "PAL %": "{:.0f}", "Extr. temp (°C)": "{:.0f}",
                "UTS mean (MPa)": "{:.2f}", "Toughness mean (MJ/m³)": "{:.2f}"})
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
