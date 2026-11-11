"""Collect the assets of the project summary (run from anywhere: ``python build_assets.py``).

* redraws the figures of ``notebooks/data_processing_v3.ipynb`` from the saved tables with the same
  ``utils.plots`` helpers, but print-sized: larger text, flatter panels, no super-title (the LaTeX
  caption carries it). The notebook versions are drawn 14-15 in wide for the screen and would end up
  with ~4 pt text on a page;
* writes ``tables/cv_table.tex`` -- the per-formulation CV table, shaded like the notebook's
  (white -> red on one scale shared by the three CV columns; higher CV = less reproducible).

Only this directory is written to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from utils import instron, plots, tensile  # noqa: E402

CV_COLS = ["CV Young's modulus (%)", "CV UTS (%)", "CV toughness (%)"]
TABLES = ROOT / "processed_data" / "tensile_doe"


def _for_print(fig, size: tuple[float, float], scale: float, name: str, legend_below: bool = False,
               title_scale: float | None = None) -> None:
    """Resize ``fig``, scale every text in it by ``scale``, drop the super-title, and save it.

    ``legend_below`` moves the (enlarged) in-panel legend to a row under the panels, where it
    cannot cover data points. ``title_scale`` overrides ``scale`` for the panel titles (long ones
    would otherwise outgrow their panel).
    """
    fig.set_size_inches(*size)
    fig.suptitle("")
    bottom = 0.0
    if legend_below:
        old = next(ax.get_legend() for ax in fig.axes if ax.get_legend() is not None)
        handles, labels = old.legend_handles, [t.get_text() for t in old.get_texts()]
        old.remove()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
                   handletextpad=0.3, columnspacing=1.6)
        bottom = 0.085
    titles = {ax.title for ax in fig.axes}
    for t in fig.findobj(Text):
        t.set_fontsize(t.get_fontsize() * (title_scale if title_scale and t in titles else scale))
    for ax in fig.axes:          # ticks created later (new size -> new tick locations) get the scaled size too
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        if labels:
            ax.tick_params(labelsize=labels[0].get_fontsize())
    fig.canvas.draw()            # settle ticks and label positions before the layout pass
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(HERE / "figures" / name, dpi=200)


def figures() -> None:
    design = pd.read_csv(ROOT / "doe" / "doe_formulations.csv")
    summary = pd.read_csv(TABLES / "tensile_doe_summary.csv")
    by_form = pd.read_csv(TABLES / "tensile_doe_by_formulation.csv")
    rigid = summary[summary["polymer_system"] != "TPU+PAL"]          # as in the notebook, section 8

    plots.doe_scatter(design).savefig(HERE / "figures" / "doe_formulations.png", dpi=200, bbox_inches="tight")
    _for_print(plots.props_vs_factors(summary, by_form), (14, 6.6), 1.55, "doe_tensile_props_vs_factors.png",
               legend_below=True)
    space = plots.design_space_bubbles(by_form)
    for ax in space.axes[1:3]:       # same y-axis in the three panels: label it once, or it runs into the colorbars
        ax.set_ylabel("")
    _for_print(space, (15, 4.6), 1.5, "doe_tensile_design_space.png", legend_below=True)
    corr = plots.correlation(rigid, tensile.objective_correlations(rigid), tensile.uts_toughness_stats(rigid))
    corr.axes[0].set_title("UTS vs toughness")      # the statistics (r, n) are in the report's text and caption
    _for_print(corr, (13, 5.0), 1.6, "doe_tensile_uts_toughness_correlation.png", title_scale=1.3)


def _shade(v: float, lo: float, hi: float) -> str:
    """Cell with a white -> red background, linear between ``lo`` and ``hi``."""
    pct = round(80 * (v - lo) / (hi - lo))
    ink = r"\color{white}" if pct > 55 else ""
    return rf"\cellcolor{{cvred!{pct}}}{ink}{v:.1f}"


def cv_table() -> None:
    cv = instron.cv_by_formulation(pd.read_csv(TABLES / "tensile_doe_summary.csv"))
    rng = {c: (0.0, cv[CV_COLS].max().max()) for c in CV_COLS}   # one scale -> columns are comparable
    rows = []
    for _, r in cv.iterrows():
        cells = [f"f{int(r['Formulation']):02d}", r["Polymer"], f"{r['PAL %']:.0f}", f"{r['Extr. temp (°C)']:.0f}",
                 f"{int(r['n'])}"]
        for mean_col, fmt, cv_col in [("Young's modulus mean (MPa)", "{:.0f}", CV_COLS[0]),
                                      ("UTS mean (MPa)", "{:.1f}", CV_COLS[1]),
                                      ("Toughness mean (MJ/m³)", "{:.2f}", CV_COLS[2])]:
            cells.append(fmt.format(r[mean_col]))
            cells.append("--" if pd.isna(r[cv_col]) else _shade(r[cv_col], *rng[cv_col]))
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)
    (HERE / "tables" / "cv_table.tex").write_text(
        "% generated by build_assets.py from processed_data/tensile_doe/tensile_doe_summary.csv\n"
        r"\begin{tabular}{llrrr rr rr rr}" "\n" r"\toprule" "\n"
        r" & & PAL & $T_\mathrm{extr}$ & & \multicolumn{2}{c}{Young's modulus} & \multicolumn{2}{c}{UTS} & "
        r"\multicolumn{2}{c}{Toughness} \\" "\n"
        r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}" "\n"
        r"Form. & Polymer & (\%) & (\textdegree C) & $n$ & mean (MPa) & CV (\%) & mean (MPa) & CV (\%) & "
        r"mean (MJ/m\textsuperscript{3}) & CV (\%) \\" "\n" r"\midrule" "\n"
        f"{body}\n" r"\bottomrule" "\n" r"\end{tabular}" "\n", encoding="utf-8")


if __name__ == "__main__":
    figures()
    cv_table()
    print("assets written to", HERE)
