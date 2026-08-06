"""Project paths and shared constants.

Centralizes everything path- and constant-related so the notebook stays thin and
runs correctly whether launched from the repo root or from ``notebooks/``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

# First 5 columns of every Config* tensile sheet, in order.
TENSILE_COLS = ["time_s", "extension_mm", "load_N", "strain", "stress_MPa"]

# Extrusion-relevant shear-rate window (Anupam, 08-06 meeting), in 1/s.
RATE_LO, RATE_HI = 0.1, 10.0

# Design-of-experiments factor space (08-06 meeting).
#
# Formulation factors are what the LHS samples. Recycle level is NOT a design factor:
# recycling is a within-sample *degradation series* (the same material is recycled
# 0 -> 1 -> 5 -> 10 times and measured at each stage, esp. in the rheology test), so
# every formulation is swept through the whole series.
DOE = SimpleNamespace(
    polymers=["PP+PAL", "HDPE+PAL", "TPU+PAL"],   # categorical (>= 3 for LVGP)
    pal_range=(10.0, 50.0),                       # % PAL, continuous
    temp_range=(180.0, 240.0),                    # extrusion temp, degC, continuous
    n_formulations=12,                            # base formulations to make; multiple of len(polymers)
    recycles=[1, 5, 10],                          # within-sample degradation series (recycled loops)
    seed=42,
)


def project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: CWD) until a folder containing ``raw_data`` is found."""
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "raw_data").exists():
            return cand
    return p


def paths(root: Path | None = None) -> SimpleNamespace:
    """Return a namespace of all project directories."""
    root = project_root() if root is None else Path(root)
    proc = root / "processed_data"
    return SimpleNamespace(
        root=root,
        raw=root / "raw_data" / "data_06_29_26",
        proc=proc,
        tensile=proc / "tensile",
        tensile_curves=proc / "tensile" / "curves",
        rheology=proc / "rheology",
        rheology_curves=proc / "rheology" / "curves",
        figures=proc / "figures",
        doe=root / "doe",
    )


def ensure_dirs(p: SimpleNamespace) -> None:
    """Create the output directories if they don't exist."""
    for d in (p.tensile_curves, p.rheology_curves, p.figures, p.doe):
        d.mkdir(parents=True, exist_ok=True)
