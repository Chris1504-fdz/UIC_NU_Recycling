"""Design of experiments via an optimized Latin Hypercube.

The LHS samples only the **formulation** factors (polymer, PAL %, temperature).
Recycle level is a within-sample degradation series, not a design factor, so it is
applied to every formulation afterwards via :func:`expand_recycle`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .config import DOE


def optimized_lhs(n_formulations: int | None = None, seed: int | None = None,
                  polymers=None, pal_range=None, temp_range=None) -> tuple[pd.DataFrame, dict]:
    """Space-filling formulation design via an optimized LHS over 3 factors:
    polymer system (categorical), PAL % (continuous), extrusion temperature (continuous).

    Continuous factors are sampled continuously; the categorical polymer is assigned from
    LHS strata (balanced when ``n_formulations`` is a multiple of the polymer count).

    Returns ``(design_df, quality)``; quality has centered discrepancy and the max absolute
    pairwise factor correlation (both: lower is better).
    """
    n = DOE.n_formulations if n_formulations is None else n_formulations
    seed = DOE.seed if seed is None else seed
    polymers = polymers or DOE.polymers
    pal_range = pal_range or DOE.pal_range
    temp_range = temp_range or DOE.temp_range

    U = qmc.LatinHypercube(d=3, optimization="random-cd", seed=seed).random(n)
    poly = np.array(polymers)[np.clip((U[:, 0] * len(polymers)).astype(int), 0, len(polymers) - 1)]
    pal = np.round(pal_range[0] + U[:, 1] * (pal_range[1] - pal_range[0])).astype(int)
    temp = (np.round((temp_range[0] + U[:, 2] * (temp_range[1] - temp_range[0])) / 5) * 5).astype(int)

    design = pd.DataFrame({
        "formulation": np.arange(1, n + 1),
        "polymer_system": poly, "PAL_pct": pal, "extrusion_temp_C": temp,
    })
    quality = {
        "n_formulations": n,
        "discrepancy": float(qmc.discrepancy(U)),
        "max_factor_corr": float(np.abs(np.corrcoef(U, rowvar=False) - np.eye(3)).max()),
    }
    return design, quality


def expand_recycle(design: pd.DataFrame, recycles=None) -> pd.DataFrame:
    """Cross each formulation with the recycle degradation series -> the measurement plan.

    One row per (formulation, recycle level): the actual list of samples to characterize
    (each formulation recycled 0/1/5/10x and measured, especially in the rheology test).
    """
    recs = DOE.recycles if recycles is None else recycles
    plan = design.merge(pd.DataFrame({"recycle_level": recs}), how="cross")
    plan.insert(0, "run", np.arange(1, len(plan) + 1))
    return plan
