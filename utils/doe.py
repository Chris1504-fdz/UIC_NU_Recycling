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


def design_metrics(U: np.ndarray, n_probe: int = 20000, probe_seed: int = 0) -> dict:
    """Space-filling / independence diagnostics for a unit-cube design ``U`` (n x d in [0,1)).

    - centered_discrepancy : centered L2 discrepancy (deviation from uniform), lower is better.
    - maximin              : min pairwise Euclidean distance, higher is better.
    - max_abs_corr         : max |pairwise factor correlation|, lower is better.
    - fill_distance        : coverage radius (largest hole), lower is better; the sup is
                             Monte-Carlo estimated with ``n_probe`` probe points.
    """
    from scipy.spatial.distance import pdist, cdist
    d = U.shape[1]
    probe = np.random.default_rng(probe_seed).random((n_probe, d))
    return {
        "centered_discrepancy": float(qmc.discrepancy(U, method="CD")),
        "maximin": float(pdist(U).min()),
        "max_abs_corr": float(np.abs(np.corrcoef(U, rowvar=False) - np.eye(d)).max()),
        "fill_distance": float(cdist(probe, U).min(axis=1).max()),
    }


def benchmark_designs(n: int | None = None, d: int = 3, reps: int = 100, seed: int | None = None,
                      n_probe: int = 20000, probe_seed: int = 0) -> pd.DataFrame:
    """Compare the optimized LHS against plain-LHS and random baselines (baselines averaged
    over ``reps`` seeds) on all four metrics. One row per strategy.

    A single shared probe set is used for the fill distance so the strategies are compared
    on identical footing (and the optimized-LHS row matches :func:`design_metrics`).
    """
    from scipy.spatial.distance import pdist, cdist
    n = DOE.n_formulations if n is None else n
    seed = DOE.seed if seed is None else seed
    probe = np.random.default_rng(probe_seed).random((n_probe, d))

    def score(U):
        return (float(qmc.discrepancy(U, method="CD")),
                float(pdist(U).min()),
                float(np.abs(np.corrcoef(U, rowvar=False) - np.eye(d)).max()),
                float(cdist(probe, U).min(axis=1).max()))

    def avg(gen):
        return np.mean([score(gen(k)) for k in range(reps)], axis=0)

    opt = score(qmc.LatinHypercube(d=d, optimization="random-cd", seed=seed).random(n))
    plain = avg(lambda k: qmc.LatinHypercube(d=d, seed=k).random(n))
    rand = avg(lambda k: np.random.default_rng(10_000 + k).random((n, d)))
    return pd.DataFrame([opt, plain, rand],
                        columns=["centered_discrepancy", "maximin", "max_abs_corr", "fill_distance"],
                        index=["optimized LHS", "plain LHS (avg)", "random (avg)"])
