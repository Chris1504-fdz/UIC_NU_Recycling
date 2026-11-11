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


# --- Augmenting the plan: recycle level as a design factor ----------------------------------
# The requested formulations (measured at 1x) stay fixed; new (formulation, recycle level) runs
# are chosen so that the union is space-filling over (polymer, PAL %, temperature, recycle).

RECYCLE_RANGE = (1, 10)          # loops; 1x = first extrusion, r x = r - 1 further recycling loops


def code_runs(runs: pd.DataFrame, pal_range=None, temp_range=None, recycle_range=RECYCLE_RANGE) -> np.ndarray:
    """Unit-cube coordinates (PAL %, temperature, recycle level) of a run table."""
    pal_range = pal_range or DOE.pal_range
    temp_range = temp_range or DOE.temp_range
    return np.column_stack([
        (runs["PAL_pct"].to_numpy(float) - pal_range[0]) / (pal_range[1] - pal_range[0]),
        (runs["extrusion_temp_C"].to_numpy(float) - temp_range[0]) / (temp_range[1] - temp_range[0]),
        (runs["recycle_level"].to_numpy(float) - recycle_range[0]) / (recycle_range[1] - recycle_range[0]),
    ])


def _phi(X: np.ndarray, p: int = 15) -> float:
    """Morris-Mitchell phi_p = (sum d_ij^-p)^(1/p): lower = points more spread out (maximin-like)."""
    from scipy.spatial.distance import pdist
    d = pdist(X)
    return float((d ** -p).sum() ** (1 / p)) if len(d) else 0.0


def _max_abs_corr(X: np.ndarray) -> float:
    """Max |pairwise correlation| among the columns of ``X``; a constant column counts as 0."""
    if len(X) < 3:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.nan_to_num(np.corrcoef(X, rowvar=False))
    np.fill_diagonal(c, 0.0)
    return float(np.abs(c).max())


def plan_metrics(runs: pd.DataFrame, corr_weight: float = 0.1) -> dict:
    """Space-filling diagnostics of a run table over (polymer, PAL %, temperature, recycle).

    - cd_pooled   : centered L2 discrepancy of all runs in the coded (PAL, T, recycle) cube, lower is better
    - cd_slices   : mean centered discrepancy within each polymer (space-filling per polymer), lower is better
    - phi_pooled  : Morris-Mitchell phi_p of all runs (maximin-like, lower is better) -- diagnostic only:
                    it is set by the closest pair of the fixed runs, so it barely reacts to added runs
    - max_abs_corr: max |correlation| among the three coded factors, lower is better
    - objective   : cd_pooled + cd_slices + corr_weight * max_abs_corr (what :func:`augment_plan` minimises)
    - loops       : recycling loops the plan costs, sum over formulations of (max level - 1)
    """
    X = code_runs(runs)
    slices = [X[(runs["polymer_system"] == p).to_numpy()] for p in runs["polymer_system"].unique()]
    m = {"n_runs": int(len(runs)),
         "cd_pooled": float(qmc.discrepancy(X, method="CD")),
         "cd_slices": float(np.mean([qmc.discrepancy(Xp, method="CD") for Xp in slices])),
         "phi_pooled": _phi(X),
         "max_abs_corr": _max_abs_corr(X),
         "loops": int((runs.groupby("formulation")["recycle_level"].max() - 1).sum())}
    m["objective"] = m["cd_pooled"] + m["cd_slices"] + corr_weight * m["max_abs_corr"]
    return m


def candidate_runs(design: pd.DataFrame, levels=(5, 10), new_formulations: int = 0,
                   seed: int | None = None) -> pd.DataFrame:
    """Pool of runs :func:`augment_plan` may add: every formulation of ``design`` at each recycle
    level in ``levels``; with ``new_formulations`` > 0, also that many extra formulations from a
    fresh optimized LHS (numbered after the existing ones) at the same levels.
    """
    pool = design.merge(pd.DataFrame({"recycle_level": list(levels)}), how="cross")
    pool["source"] = "existing formulation"
    if new_formulations:
        extra, _ = optimized_lhs(new_formulations, seed=DOE.seed + 1 if seed is None else seed)
        extra["formulation"] += int(design["formulation"].max())
        extra = extra.merge(pd.DataFrame({"recycle_level": list(levels)}), how="cross")
        extra["source"] = "new formulation"
        pool = pd.concat([pool, extra], ignore_index=True)
    return pool


def augment_plan(design: pd.DataFrame, n_new: int, existing: pd.DataFrame | None = None,
                 pool: pd.DataFrame | None = None, levels=(5, 10), corr_weight: float = 0.1,
                 restarts: int = 30, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add ``n_new`` runs to ``existing`` (default: every formulation at 1x) so that the union is
    space-filling over (polymer, PAL %, temperature, recycle level).

    Runs are drawn from ``pool`` (default :func:`candidate_runs` at ``levels``), balanced across
    polymers, by an exchange search: from a random balanced start, swap a chosen run for an
    unchosen one of the same polymer whenever :func:`plan_metrics` ``objective`` of the union
    improves; best of ``restarts`` starts.

    Returns ``(plan, new_runs, quality)``: the full plan with a ``status`` column
    (``requested`` / ``proposed``), the proposed runs alone, and the metrics of the union.
    """
    rng = np.random.default_rng(DOE.seed if seed is None else seed)
    if existing is None:
        existing = design.assign(recycle_level=1)
    existing = existing.assign(source="existing formulation") if "source" not in existing else existing
    pool = candidate_runs(design, levels) if pool is None else pool
    pool = pool.reset_index(drop=True)
    by_poly = {p: pool.index[pool["polymer_system"] == p].to_numpy() for p in DOE.polymers if
               (pool["polymer_system"] == p).any()}
    polys = list(by_poly)

    def score(idx):
        return plan_metrics(pd.concat([existing, pool.loc[idx]], ignore_index=True), corr_weight)["objective"]

    best_idx, best = None, np.inf
    for _ in range(restarts):
        # balanced allocation; the remainder goes to randomly chosen polymers
        counts = dict(zip(polys, [n_new // len(polys)] * len(polys)))
        for p in rng.choice(polys, n_new % len(polys), replace=False):
            counts[p] += 1
        chosen = {p: list(rng.choice(by_poly[p], counts[p], replace=False)) for p in polys}
        idx = sum(chosen.values(), [])
        cur = score(idx)
        improved = True
        while improved:
            improved = False
            for p in polys:
                for i, run in enumerate(chosen[p]):
                    for cand in by_poly[p]:
                        if cand in chosen[p]:
                            continue
                        trial = chosen[p][:i] + [cand] + chosen[p][i + 1:]
                        s = score(sum([trial if q == p else chosen[q] for q in polys], []))
                        if s < cur - 1e-12:
                            chosen[p], cur, improved = trial, s, True
                            break
        if cur < best:
            best, best_idx = cur, sum(chosen.values(), [])

    new_runs = pool.loc[sorted(best_idx)].reset_index(drop=True)
    plan = pd.concat([existing.assign(status="requested"), new_runs.assign(status="proposed")],
                     ignore_index=True).sort_values(["formulation", "recycle_level"]).reset_index(drop=True)
    plan.insert(0, "run", np.arange(1, len(plan) + 1))
    return plan, new_runs, plan_metrics(plan, corr_weight)


def augment_lhs(design: pd.DataFrame, n_new: int, existing: pd.DataFrame | None = None,
                recycle_range=(5, 10), recycle_levels=None, corr_weight: float = 0.1,
                restarts: int = 12, sweeps: int = 30, seed: int | None = None,
                pal_range=None, temp_range=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Augmented Latin hypercube: ``n_new`` **new formulations** at recycle levels in ``recycle_range``,
    chosen so that, together with ``existing`` (default: every formulation of ``design`` at 1x), the
    plan is space-filling over (polymer, PAL %, temperature, recycle level).

    Latin structure. PAL % and temperature are split into ``n_existing + n_new`` strata; the new runs
    take the strata the existing runs leave free, so the **union** is itself a Latin hypercube in those
    two factors (exactly so when ``n_new == n_existing`` and the existing runs are an LHS). The recycle
    level gets ``n_new`` Latin strata over ``recycle_range`` (rounded to whole loops), or a balanced
    assignment of ``recycle_levels`` when that is given (e.g. ``(5, 10)``). Polymers are balanced.

    Optimisation. Column permutations only (swapping the values of two runs in one factor), which keep
    the Latin structure, accepted when :func:`plan_metrics` ``objective`` of the union improves; best of
    ``restarts`` random starts, each up to ``sweeps`` passes over all pairs.

    Returns ``(plan, new_runs, quality)`` like :func:`augment_plan`; new formulations are numbered
    after the existing ones.
    """
    rng = np.random.default_rng(DOE.seed if seed is None else seed)
    pal_range = pal_range or DOE.pal_range
    temp_range = temp_range or DOE.temp_range
    existing = design.assign(recycle_level=1) if existing is None else existing
    n_old, m = len(existing), len(existing) + n_new

    def free_strata(col, lo, hi):
        occupied = set(np.floor((existing[col].to_numpy(float) - lo) / (hi - lo) * m).astype(int).clip(0, m - 1))
        return np.array([s for s in range(m) if s not in occupied])

    def stratum_values(strata, lo, hi, step):
        return np.round((lo + (strata + 0.5) / m * (hi - lo)) / step) * step

    free_pal, free_temp = free_strata("PAL_pct", *pal_range), free_strata("extrusion_temp_C", *temp_range)
    if recycle_levels is not None:
        rec_pool = np.array([recycle_levels[i % len(recycle_levels)] for i in range(n_new)], float)
    else:
        lo, hi = recycle_range
        rec_pool = np.round(lo + (np.arange(n_new) + 0.5) / n_new * (hi - lo))
    poly_pool = np.array([DOE.polymers[i % len(DOE.polymers)] for i in range(n_new)])
    ids = np.arange(n_new) + int(existing["formulation"].max()) + 1

    def table(cols):
        return pd.DataFrame({"formulation": ids, "polymer_system": cols["poly"],
                             "PAL_pct": stratum_values(cols["pal"], *pal_range, 1).astype(int),
                             "extrusion_temp_C": stratum_values(cols["temp"], *temp_range, 5).astype(int),
                             "recycle_level": cols["rec"].astype(int)})

    def score(cols):
        return plan_metrics(pd.concat([existing, table(cols)], ignore_index=True), corr_weight)["objective"]

    best_cols, best = None, np.inf
    for _ in range(restarts):
        cols = {"pal": rng.permutation(rng.choice(free_pal, n_new, replace=False)),
                "temp": rng.permutation(rng.choice(free_temp, n_new, replace=False)),
                "rec": rng.permutation(rec_pool), "poly": rng.permutation(poly_pool)}
        cur = score(cols)
        for _ in range(sweeps):
            improved = False
            for key in cols:
                for i in range(n_new - 1):
                    for j in range(i + 1, n_new):
                        cols[key][[i, j]] = cols[key][[j, i]]          # a column permutation: Latin structure kept
                        s = score(cols)
                        if s < cur - 1e-12:
                            cur, improved = s, True
                        else:
                            cols[key][[i, j]] = cols[key][[j, i]]      # undo
            if not improved:
                break
        if cur < best:
            best, best_cols = cur, {k: v.copy() for k, v in cols.items()}

    new_runs = table(best_cols).assign(source="new formulation")
    plan = pd.concat([existing.assign(source="existing formulation", status="requested"),
                      new_runs.assign(status="proposed")], ignore_index=True)
    plan = plan.sort_values(["formulation", "recycle_level"]).reset_index(drop=True)
    plan.insert(0, "run", np.arange(1, len(plan) + 1))
    return plan, new_runs, plan_metrics(plan, corr_weight)


# --- Revised DoE (2026-10): recycle cycle 0-3 as a design factor ----------------------------
# Agreed with the lab on 2026-10-06: ~21 configurations, recycle cycles 0 / 1 / 2 / 3 treated as an
# integer factor. Every configuration is its own formulation (polymer, PAL %, temperature) taken to
# one recycle cycle.

RECYCLE_CYCLES = (0, 1, 2, 3)    # 0 = as extruded (no recycling), r = r recycling loops


def _sliced_objective(X: np.ndarray, slices: list, corr_weight: float) -> float:
    """Centered discrepancy of all runs + mean over the slices (+ ``corr_weight`` * max |factor corr|
    over all runs and every slice, when ``corr_weight`` > 0)."""
    return float(qmc.discrepancy(X, method="CD")
                 + np.mean([qmc.discrepancy(X[s], method="CD") for s in slices])
                 + corr_weight * max([_max_abs_corr(X)] + [_max_abs_corr(X[s]) for s in slices]))


def sliced_lhs(n_per_polymer: int = 7, recycle_cycles=RECYCLE_CYCLES, polymers=None, pal_range=None,
               temp_range=None, corr_weight: float = 0.0, restarts: int = 20, sweeps: int = 30,
               balance_overall: bool = True, seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Sliced Latin hypercube over (polymer, PAL %, temperature, recycle cycle), ``n_per_polymer``
    configurations per polymer; each configuration is a new formulation taken to one recycle cycle.

    Structure, on the grid the lab sets (1 % PAL, 5 degC):

    - PAL % is cut into ``n = n_per_polymer * n_polymers`` fine bands grouped into ``n_per_polymer``
      coarse bands; every polymer takes one fine band from each coarse band (its value = the band
      centre rounded to 1 %). Each polymer is Latin in PAL on its own and the pooled design is too.
    - Temperature: the 5 degC grid has only 13 settings for 21 runs, so the pooled design cannot be
      Latin, and fine-band centres would round across coarse-band edges (207.1 -> 205 degC). Instead
      every polymer takes one **grid value** inside each of the ``n_per_polymer`` coarse bands; polymers
      may share a value. Each polymer is Latin in temperature on the grid; the pooled spread is left to
      the discrepancy term.
    - Recycle cycles are balanced within each polymer (counts differ by at most one) and, with
      ``balance_overall``, over the whole design too; which cycles get the extra runs is searched over.

    Optimisation. Moves that keep the structure -- swap two runs' PAL % or recycle cycle within a
    polymer, swap two polymers' fine PAL bands within one coarse band, or move a polymer's temperature to
    another grid value in the same band -- accepted when :func:`_sliced_objective` improves (coded unit
    cube; the cycles are coded as the centres of ``len(recycle_cycles)`` equal cells). Best of
    ``restarts`` random starts for every allocation of the extra cycle runs. The objective is the
    centered discrepancy only (pooled + per polymer), as in the DMREF designs; factor correlation is
    checked afterwards against |r| < 0.3. ``corr_weight`` > 0 also penalises it (lower |r|, but a
    smaller min distance and larger fill distance -- see :func:`compare_methods`).

    Returns ``(design, quality)``; ``quality['objective_random_start']`` is the mean objective of the
    random (unoptimised) sliced starts, as a baseline.
    """
    from itertools import combinations, combinations_with_replacement
    rng = np.random.default_rng(DOE.seed if seed is None else seed)
    polymers = list(polymers or DOE.polymers)
    pal_range = pal_range or DOE.pal_range
    temp_range = temp_range or DOE.temp_range
    cycles = np.asarray(recycle_cycles, int)
    P, m, k = len(polymers), n_per_polymer, len(cycles)
    n = P * m

    # PAL: value of every fine band on the 1 % grid; temperature: the 5 degC grid and its coarse bands
    mid = (np.arange(n) + 0.5) / n
    pal_val = np.round(pal_range[0] + mid * (pal_range[1] - pal_range[0])).astype(int)
    pal_c = (pal_val - pal_range[0]) / (pal_range[1] - pal_range[0])
    temp_val = np.arange(temp_range[0], temp_range[1] + 1e-9, 5).astype(int)
    temp_c = (temp_val - temp_range[0]) / (temp_range[1] - temp_range[0])
    temp_band = np.minimum((temp_c * m).astype(int), m - 1)
    band_idx = [np.flatnonzero(temp_band == c) for c in range(m)]
    assert all(len(b) for b in band_idx), "a temperature band holds no 5 degC grid value"
    slices = [np.arange(p * m, (p + 1) * m) for p in range(P)]

    def coded(st):
        rec = (np.searchsorted(cycles, st["rec"].ravel()) + 0.5) / k
        return np.column_stack([pal_c[st["pal"].ravel()], temp_c[st["temp"].ravel()], rec])

    def score(st):                          # + 1 per repeated grid value of PAL % / temperature in a polymer
        dup = sum(m - len(np.unique(v[row])) for v, key in ((pal_val, "pal"), (temp_val, "temp"))
                  for row in st[key])
        return _sliced_objective(coded(st), slices, corr_weight) + dup

    def random_sliced():                    # (P, m) fine-band indices; fine f lies in coarse f // P
        fine = np.array([rng.permutation(np.arange(c * P, (c + 1) * P)) for c in range(m)]).T
        return np.array([rng.permutation(row) for row in fine])

    def random_temps():                     # (P, m) grid indices, one per coarse band, in random order
        return np.array([rng.permutation([rng.choice(b) for b in band_idx]) for _ in range(P)])

    extra_sets = list(combinations(range(k), m % k))          # cycles that get one run more
    allocs = [[np.full(k, m // k) + np.isin(np.arange(k), list(e)) for e in a]
              for a in combinations_with_replacement(extra_sets, P)]
    if balance_overall:
        allocs = [a for a in allocs if np.ptp(np.sum(a, axis=0)) <= 1]
    best, best_st, starts = np.inf, None, []
    for counts in allocs:
        for _ in range(restarts):
            st = {"pal": random_sliced(), "temp": random_temps(),
                  "rec": np.array([rng.permutation(np.repeat(cycles, c)) for c in counts])}
            cur = score(st); starts.append(_sliced_objective(coded(st), slices, corr_weight))
            for _ in range(sweeps):
                improved = False
                for key in ("pal", "rec"):                    # within a polymer: re-pair the factors
                    for p in range(P):
                        for i in range(m - 1):
                            for j in range(i + 1, m):
                                a = st[key][p]; a[[i, j]] = a[[j, i]]
                                s = score(st)
                                if s < cur - 1e-12:
                                    cur, improved = s, True
                                else:
                                    a[[i, j]] = a[[j, i]]
                a = st["pal"]                                 # across polymers: trade fine PAL bands
                for c in range(m):
                    pos = [int(np.flatnonzero(a[p] // P == c)[0]) for p in range(P)]
                    for p in range(P - 1):
                        for q in range(p + 1, P):
                            a[p, pos[p]], a[q, pos[q]] = a[q, pos[q]], a[p, pos[p]]
                            s = score(st)
                            if s < cur - 1e-12:
                                cur, improved = s, True
                            else:
                                a[p, pos[p]], a[q, pos[q]] = a[q, pos[q]], a[p, pos[p]]
                a = st["temp"]                                # another grid value in the same band
                for p in range(P):
                    for i in range(m):
                        old = a[p, i]
                        for g in band_idx[temp_band[old]]:
                            if g == a[p, i]:
                                continue
                            prev, a[p, i] = a[p, i], g
                            s = score(st)
                            if s < cur - 1e-12:
                                cur, improved = s, True
                            else:
                                a[p, i] = prev
                if not improved:
                    break
            if cur < best:
                best, best_st = cur, {key: v.copy() for key, v in st.items()}

    design = pd.DataFrame({"polymer_system": np.repeat(polymers, m),
                           "PAL_pct": pal_val[best_st["pal"].ravel()],
                           "extrusion_temp_C": temp_val[best_st["temp"].ravel()],
                           "recycle_cycle": best_st["rec"].ravel()})
    design["_p"] = design["polymer_system"].map({p: i for i, p in enumerate(polymers)})
    design = (design.sort_values(["_p", "recycle_cycle", "PAL_pct"]).drop(columns="_p")
              .reset_index(drop=True))
    design.insert(0, "configuration", np.arange(1, n + 1))

    X = coded(best_st)
    quality = {"n_configurations": n,
               "per_cycle": design["recycle_cycle"].value_counts().sort_index().to_dict(),
               "cd_pooled": float(qmc.discrepancy(X, method="CD")),
               "cd_slices": float(np.mean([qmc.discrepancy(X[s], method="CD") for s in slices])),
               "max_abs_corr": _max_abs_corr(X),
               "max_abs_corr_slices": max(_max_abs_corr(X[s]) for s in slices),
               "objective": best,
               "objective_random_start": float(np.mean(starts)),
               "recycling_loops": int(design["recycle_cycle"].sum()),
               "distinct_temps_per_polymer": design.groupby("polymer_system")["extrusion_temp_C"].nunique().to_dict()}
    return design, quality


def specimen_log(design: pd.DataFrame, replicates: int = 5) -> pd.DataFrame:
    """One row per tensile specimen (``C<NN>_<rep>``) with blank columns for the lab to fill in.

    The blanks are what decides whether an odd specimen is excluded for a physical reason: test date
    (session), data-acquisition rate, measured diameter, where it broke, and anything unusual.
    """
    log = design.loc[design.index.repeat(replicates)].reset_index(drop=True)
    log.insert(1, "replicate", np.tile(np.arange(1, replicates + 1), len(design)))
    log.insert(0, "specimen_id", [f"C{c:02d}_{r}" for c, r in zip(log["configuration"], log["replicate"])])
    for col in ("test_date", "data_rate_Hz", "diameter_mm", "break_location", "notes"):
        log[col] = ""
    return log


# --- Design quality: the four space-filling metrics, and a comparison of generators ----------
# Same four metrics as the DMREF DoE notebooks (experiment03_hfo2_batch1_analysis, section 7):
# centered L2 discrepancy, min pairwise distance (maximin), max |factor correlation|, fill distance.
# Adjusted to this setup: the polymer is categorical, so the metrics are taken in the
# (PAL %, temperature, cycle) cube twice -- all runs together (polymer ignored) and within each
# polymer (worst of the three). Every design is scored as fabricated, on the lab grid.

METHODS = ("Sobol", "Plain LHS", "Optimized LHS", "Sliced LHS (CD only, w = 0)", "Sliced LHS (CD + corr, w = 0.1)")


def unit_cube(design: pd.DataFrame, pal_range=None, temp_range=None, cycles=RECYCLE_CYCLES) -> np.ndarray:
    """(PAL %, temperature, recycle cycle) of ``design`` in [0, 1]; the cycles as i / (n_cycles - 1)."""
    pal_range = pal_range or DOE.pal_range
    temp_range = temp_range or DOE.temp_range
    return np.column_stack([
        (design["PAL_pct"].to_numpy(float) - pal_range[0]) / (pal_range[1] - pal_range[0]),
        (design["extrusion_temp_C"].to_numpy(float) - temp_range[0]) / (temp_range[1] - temp_range[0]),
        np.searchsorted(np.asarray(cycles), design["recycle_cycle"].to_numpy()) / (len(cycles) - 1)])


def probe_points(n: int = 100_000, cycles=RECYCLE_CYCLES, seed: int = 0) -> np.ndarray:
    """Dense probe set for the fill distance: PAL % and temperature continuous, the cycle on its levels."""
    rng = np.random.default_rng(seed)
    P = rng.random((n, 3))
    P[:, 2] = rng.integers(len(cycles), size=n) / (len(cycles) - 1)
    return P


def four_metrics(X: np.ndarray, probe: np.ndarray) -> dict:
    """Centered L2 discrepancy (lower better), min pairwise distance (higher better), max |factor
    correlation| (lower better), fill distance = largest distance from the space to its nearest run
    (lower better)."""
    from scipy.spatial.distance import pdist, cdist
    return {"discrepancy": float(qmc.discrepancy(X, method="CD")), "min_dist": float(pdist(X).min()),
            "max_corr": _max_abs_corr(X), "fill_dist": float(cdist(probe, X).min(axis=1).max())}


def design_quality(design: pd.DataFrame, probe: np.ndarray | None = None) -> dict:
    """:func:`four_metrics` of all runs (``all_*``) and of the worst polymer (``poly_*``: the highest
    discrepancy / correlation / fill distance and the lowest min distance of the three), plus balance:
    runs per polymer and how many polymers miss a recycle cycle."""
    probe = probe_points() if probe is None else probe
    X = unit_cube(design)
    per = [four_metrics(X[(design["polymer_system"] == p).to_numpy()], probe)
           for p in design["polymer_system"].unique()]
    worst = {"discrepancy": max, "min_dist": min, "max_corr": max, "fill_dist": max}
    counts = design["polymer_system"].value_counts()
    return {**{f"all_{k}": v for k, v in four_metrics(X, probe).items()},
            **{f"poly_{k}": f([m[k] for m in per]) for k, f in worst.items()},
            "runs_per_polymer": f"{counts.min()}-{counts.max()}",
            "polymers_missing_a_cycle": int((design.groupby("polymer_system")["recycle_cycle"].nunique()
                                             < len(RECYCLE_CYCLES)).sum())}


def from_unit(U: np.ndarray, polymers=None, pal_range=None, temp_range=None, cycles=RECYCLE_CYCLES) -> pd.DataFrame:
    """Map a 4-column unit sample (polymer, PAL %, temperature, cycle) to the lab grid, as fabricated."""
    polymers = list(polymers or DOE.polymers)
    pal_range = pal_range or DOE.pal_range
    temp_range = temp_range or DOE.temp_range
    cycles = np.asarray(cycles)
    return pd.DataFrame({
        "polymer_system": np.array(polymers)[np.minimum((U[:, 0] * len(polymers)).astype(int), len(polymers) - 1)],
        "PAL_pct": np.round(pal_range[0] + U[:, 1] * (pal_range[1] - pal_range[0])).astype(int),
        "extrusion_temp_C": (np.round((temp_range[0] + U[:, 2] * (temp_range[1] - temp_range[0])) / 5) * 5).astype(int),
        "recycle_cycle": cycles[np.minimum((U[:, 3] * len(cycles)).astype(int), len(cycles) - 1)]})


def generate(method: str, n_per_polymer: int = 7, seed: int = 0) -> pd.DataFrame:
    """One design from a generator in :data:`METHODS`, on the lab grid. Sobol and the two plain /
    optimized LHS sample (polymer, PAL, temperature, cycle) jointly in a 4-d unit cube (the August
    method is 'Optimized LHS'); a sliced one is :func:`sliced_lhs` with the correlation weight in its
    name, e.g. 'Sliced LHS (CD + corr, w = 0.02)' (no 'w = ...': 0, discrepancy only)."""
    import re
    n = n_per_polymer * len(DOE.polymers)
    if method.startswith("Sliced"):
        w = re.search(r"w = ([0-9.]+)", method)
        return sliced_lhs(n_per_polymer, corr_weight=float(w.group(1)) if w else 0.0, seed=seed)[0]
    if method == "Sobol":
        import warnings
        with warnings.catch_warnings():                       # n is not a power of 2
            warnings.simplefilter("ignore")
            U = qmc.Sobol(d=4, scramble=True, seed=seed).random(n)
    elif method == "Plain LHS":
        U = qmc.LatinHypercube(d=4, seed=seed).random(n)
    elif method == "Optimized LHS":
        U = qmc.LatinHypercube(d=4, optimization="random-cd", seed=seed).random(n)
    else:
        raise ValueError(f"unknown method {method!r}; one of {METHODS}")
    return from_unit(U)


def _quality_job(job):
    method, seed, n_per_polymer = job
    return {"method": method, "seed": seed, **design_quality(generate(method, n_per_polymer, seed))}


def _single_thread():
    """Worker initializer: one BLAS thread per process (dozens of spinning BLAS pools stall the pool)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _parallel(fn, jobs: list, workers: int | None) -> pd.DataFrame:
    """Run ``fn`` over ``jobs`` in worker processes (one BLAS thread each); one row per job."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers or min(len(jobs), 32), initializer=_single_thread) as ex:
        return pd.DataFrame(list(ex.map(fn, jobs)))


def compare_methods(methods=METHODS, n_seeds: int = 10, n_per_polymer: int = 7, workers: int | None = None) -> pd.DataFrame:
    """:func:`design_quality` of every method x seed (seeds 0..n_seeds-1), one row each, run in parallel
    processes (the sliced designs take ~30 s each)."""
    return _parallel(_quality_job, [(m, s, n_per_polymer) for m in methods for s in range(n_seeds)], workers)


CORR_WEIGHTS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)


def _weight_job(job):
    weight, seed, n_per_polymer = job
    design, q = sliced_lhs(n_per_polymer, corr_weight=weight, seed=seed)
    return {"corr_weight": weight, "seed": seed, **design_quality(design), "recycling_loops": q["recycling_loops"]}


def sweep_corr_weight(weights=CORR_WEIGHTS, n_seeds: int = 10, n_per_polymer: int = 7,
                      workers: int | None = None) -> pd.DataFrame:
    """:func:`design_quality` of the sliced LHS for every correlation weight x seed (``corr_weight`` of
    :func:`sliced_lhs`; 0 = discrepancy only), one row each -- the trade-off between factor correlation
    and spacing, for :func:`utils.plots.corr_weight_tradeoff`."""
    return _parallel(_weight_job, [(w, s, n_per_polymer) for w in weights for s in range(n_seeds)], workers)


_HIGHER_BETTER = {"min_dist"}             # every other metric: lower = better


def method_summary(tbl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per method (rows): ``mean ± std`` of every metric of :func:`compare_methods` (just the value for a
    method with one row, e.g. a single picked design), the runs per polymer seen, and the number of seeds
    where a polymer misses a cycle. Columns carry the better direction
    (↓ / ↑). Also returns ``best``, a boolean frame of the same shape marking the winning method of each
    column on the mean (ties all marked; balance = every seed has the same number of runs per polymer)."""
    metrics = [c for c in tbl.columns if c.startswith(("all_", "poly_"))]
    g = tbl.groupby("method", sort=False)
    mean, std = g[metrics].mean(), g[metrics].std()
    label = {c: f"{c} {'↑' if c.split('_', 1)[1] in _HIGHER_BETTER else '↓'}" for c in metrics}
    summary = (mean.round(4).astype(str) + (" ± " + std.round(4).astype(str)).where(std.notna(), "")
               ).rename(columns=label)
    best = pd.DataFrame({label[c]: (mean[c].round(4) == (mean[c].round(4).max() if c.split("_", 1)[1]
                                                         in _HIGHER_BETTER else mean[c].round(4).min()))
                         for c in metrics})
    runs = g["runs_per_polymer"].agg(lambda s: ", ".join(sorted(s.unique())))
    missing = g["polymers_missing_a_cycle"].agg(lambda s: int((s > 0).sum()))
    balanced = g["runs_per_polymer"].agg(lambda s: all(a == b for a, b in (x.split("-") for x in s)))
    summary["runs per polymer (min-max; equal = balanced)"] = runs
    summary["seeds with a polymer missing a cycle ↓"] = missing
    best["runs per polymer (min-max; equal = balanced)"] = balanced
    best["seeds with a polymer missing a cycle ↓"] = missing == missing.min()
    return summary, best


def pareto_mask(df: pd.DataFrame, minimize=(), maximize=()) -> np.ndarray:
    """True for the rows no other row beats: at least as good on every listed column, better on one."""
    O = np.column_stack([df[c].to_numpy(float) for c in minimize] + [-df[c].to_numpy(float) for c in maximize])
    return np.array([not any((O[j] <= O[i]).all() and (O[j] < O[i]).any() for j in range(len(O)) if j != i)
                     for i in range(len(O))])


# --- Conditioned design: new runs placed around the runs already measured ---------------------

def existing_runs(design: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Runs already measured: one row per (formulation, recycle cycle) present in a delivery catalogue
    (``instron.catalogue``: ``formulation``, ``recycled_x``), with the factors of ``design`` (the DoE
    the lab followed) and the specimen count."""
    got = (cat.dropna(subset=["recycled_x"]).groupby(["formulation", "recycled_x"]).size()
           .rename("specimens").reset_index().rename(columns={"recycled_x": "recycle_cycle"}))
    got["recycle_cycle"] = got["recycle_cycle"].astype(int)
    return (design.merge(got, on="formulation")
            [["formulation", "polymer_system", "PAL_pct", "extrusion_temp_C", "recycle_cycle", "specimens"]]
            .sort_values(["formulation", "recycle_cycle"]).reset_index(drop=True))


def conditioned_sliced_lhs(fixed: pd.DataFrame, n_per_polymer=7, recycle_cycles=RECYCLE_CYCLES,
                           polymers=None, pal_range=None, temp_range=None, corr_weight: float = 0.02,
                           restarts: int = 20, sweeps: int = 30, seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Sliced LHS of ``n_per_polymer`` new configurations per polymer (one int, or {polymer: int} when the
    total does not split evenly) **conditioned on** the runs in ``fixed`` (polymer_system, PAL_pct,
    extrusion_temp_C, recycle_cycle), which stay where they are.

    Structure, per polymer, on the lab grid (1 % PAL, 5 degC): PAL % and temperature are each cut into
    ``m = n_fixed_points + n_per_polymer`` bands (distinct PAL / temperature points of the fixed runs); the new runs take the bands the fixed runs leave free, one each,
    so the polymer's fixed + new runs cover every band once (when the fixed runs sit in distinct bands).
    Cycles: the new runs' counts make each polymer's fixed + new counts as even as possible, and the
    cycles of the new runs as even as possible overall; which allocation is searched over.

    Optimisation: :func:`_sliced_objective` of **fixed + new** (discrepancy of all runs + mean per polymer
    + ``corr_weight`` x max |r|), by moves that keep the structure -- swap two new runs' PAL % or cycle within
    a polymer, or move a new run to another grid value in a free band. Best of ``restarts`` per allocation.

    Returns ``(new_design, quality)``; quality is computed on fixed + new.
    """
    from itertools import product
    rng = np.random.default_rng(DOE.seed if seed is None else seed)
    polymers = [p for p in (polymers or DOE.polymers)]
    pal_range = pal_range or DOE.pal_range
    temp_range = temp_range or DOE.temp_range
    cycles = np.asarray(recycle_cycles, int)
    k = len(cycles)
    n_by = dict(n_per_polymer) if isinstance(n_per_polymer, dict) else {p: int(n_per_polymer) for p in polymers}
    pal_grid = np.arange(pal_range[0], pal_range[1] + 1e-9, 1).astype(int)
    temp_grid = np.arange(temp_range[0], temp_range[1] + 1e-9, 5).astype(int)
    code = lambda v, r: (np.asarray(v, float) - r[0]) / (r[1] - r[0])

    # per polymer: fixed coordinates, free bands and the grid values inside each band
    fx = {p: fixed[fixed["polymer_system"] == p] for p in polymers}
    setup = {}
    for p in polymers:
        # a formulation measured at several cycles sits at one PAL / temperature point: it takes one band
        n_new = n_by[p]
        m = len(fx[p][["PAL_pct", "extrusion_temp_C"]].drop_duplicates()) + n_new
        band = lambda v, r: np.minimum((code(v, r) * m).astype(int), m - 1)
        pb, tb = band(pal_grid, pal_range), band(temp_grid, temp_range)
        used_p, used_t = set(band(fx[p]["PAL_pct"], pal_range)), set(band(fx[p]["extrusion_temp_C"], temp_range))
        free_p, free_t = [b for b in range(m) if b not in used_p], [b for b in range(m) if b not in used_t]
        if len(free_p) < n_new or len(free_t) < n_new:
            raise ValueError(f"{p}: fixed runs share bands; fewer than {n_new} free bands")
        setup[p] = {"pal_in": {b: pal_grid[pb == b] for b in free_p}, "temp_in": {b: temp_grid[tb == b] for b in free_t},
                    "pal_band": dict(zip(pal_grid, pb)), "temp_band": dict(zip(temp_grid, tb)),
                    "free_p": free_p, "free_t": free_t,
                    "fixed_counts": np.array([(fx[p]["recycle_cycle"] == c).sum() for c in cycles])}

    # cycle allocations of the new runs: per polymer the most even fixed + new counts, then even overall
    def compositions(n, parts):
        if parts == 1:
            yield (n,); return
        for i in range(n + 1):
            for rest in compositions(n - i, parts - 1):
                yield (i,) + rest
    best_per = {}
    for p in polymers:
        opts = [np.array(c) for c in compositions(n_by[p], k)]
        spread = [np.ptp(setup[p]["fixed_counts"] + c) for c in opts]
        best_per[p] = [c for c, s in zip(opts, spread) if s == min(spread)]
    allocs = list(product(*[best_per[p] for p in polymers]))
    tot = [np.ptp(np.sum(a, axis=0)[np.sum(a, axis=0) > 0]) if np.sum(a) else 0 for a in allocs]
    allocs = [a for a, t in zip(allocs, tot) if t == min(tot)]

    fixed_X = np.column_stack([code(fixed["PAL_pct"], pal_range), code(fixed["extrusion_temp_C"], temp_range),
                               (np.searchsorted(cycles, fixed["recycle_cycle"]) + 0.5) / k])
    fixed_poly = fixed["polymer_system"].to_numpy()
    new_poly = np.repeat(polymers, [n_by[p] for p in polymers])
    all_poly = np.concatenate([fixed_poly, new_poly])
    slices = [np.flatnonzero(all_poly == p) for p in polymers]

    def coded(st):
        pal = np.concatenate([st[p]["pal"] for p in polymers]); tmp = np.concatenate([st[p]["temp"] for p in polymers])
        rec = np.concatenate([st[p]["rec"] for p in polymers])
        Xn = np.column_stack([code(pal, pal_range), code(tmp, temp_range), (np.searchsorted(cycles, rec) + 0.5) / k])
        return np.vstack([fixed_X, Xn])

    score = lambda st: _sliced_objective(coded(st), slices, corr_weight)

    def start(counts):
        st = {}
        for p, c in zip(polymers, counts):
            S, n_new = setup[p], n_by[p]
            pb = rng.permutation(rng.choice(S["free_p"], n_new, replace=False))
            tb = rng.permutation(rng.choice(S["free_t"], n_new, replace=False))
            st[p] = {"pal": np.array([rng.choice(S["pal_in"][b]) for b in pb]),
                     "temp": np.array([rng.choice(S["temp_in"][b]) for b in tb]),
                     "rec": rng.permutation(np.repeat(cycles, c))}
        return st

    best, best_st, starts = np.inf, None, []
    for counts in allocs:
        for _ in range(restarts):
            st = start(counts)
            cur = score(st); starts.append(cur)
            for _ in range(sweeps):
                improved = False
                for p in polymers:
                    S, n_new = setup[p], n_by[p]
                    for key in ("pal", "rec"):                    # re-pair the factors within a polymer
                        a = st[p][key]
                        for i in range(n_new - 1):
                            for j in range(i + 1, n_new):
                                a[[i, j]] = a[[j, i]]
                                s = score(st)
                                if s < cur - 1e-12:
                                    cur, improved = s, True
                                else:
                                    a[[i, j]] = a[[j, i]]
                    for key, inn, bmap, free in (("pal", "pal_in", "pal_band", "free_p"), ("temp", "temp_in", "temp_band", "free_t")):
                        a = st[p][key]                            # another grid value in a free band
                        for i in range(n_new):
                            taken = {S[bmap][v] for j, v in enumerate(a) if j != i}
                            for b in S[free]:
                                if b in taken:
                                    continue
                                for v in S[inn][b]:
                                    if v == a[i]:
                                        continue
                                    prev, a[i] = a[i], v
                                    s = score(st)
                                    if s < cur - 1e-12:
                                        cur, improved = s, True
                                    else:
                                        a[i] = prev
                if not improved:
                    break
            if cur < best:
                best, best_st = cur, {p: {kk: v.copy() for kk, v in d.items()} for p, d in st.items()}

    new = pd.DataFrame({"polymer_system": new_poly,
                        "PAL_pct": np.concatenate([best_st[p]["pal"] for p in polymers]),
                        "extrusion_temp_C": np.concatenate([best_st[p]["temp"] for p in polymers]),
                        "recycle_cycle": np.concatenate([best_st[p]["rec"] for p in polymers])})
    new["_p"] = new["polymer_system"].map({p: i for i, p in enumerate(polymers)})
    new = new.sort_values(["_p", "recycle_cycle", "PAL_pct"]).drop(columns="_p").reset_index(drop=True)
    new.insert(0, "configuration", np.arange(1, len(new) + 1))
    X = coded(best_st)
    quality = {"n_new": len(new), "n_fixed": len(fixed),
               "per_cycle_new": new["recycle_cycle"].value_counts().sort_index().to_dict(),
               "per_cycle_all": pd.concat([fixed["recycle_cycle"], new["recycle_cycle"]]).value_counts().sort_index().to_dict(),
               "cd_pooled": float(qmc.discrepancy(X, method="CD")),
               "cd_slices": float(np.mean([qmc.discrepancy(X[s], method="CD") for s in slices])),
               "max_abs_corr": _max_abs_corr(X), "max_abs_corr_slices": max(_max_abs_corr(X[s]) for s in slices),
               "objective": best, "objective_random_start": float(np.mean(starts)),
               "recycling_loops": int(new["recycle_cycle"].sum())}
    return new, quality


def _conditioned_job(job):
    fixed, seed, n_per_polymer, corr_weight = job
    new, q = conditioned_sliced_lhs(fixed, n_per_polymer, corr_weight=corr_weight, seed=seed)
    union = pd.concat([fixed[["polymer_system", "PAL_pct", "extrusion_temp_C", "recycle_cycle"]],
                       new.drop(columns="configuration")], ignore_index=True)
    return {"seed": seed, **design_quality(union), "recycling_loops": q["recycling_loops"], "objective": q["objective"]}


def conditioned_seeds(fixed: pd.DataFrame, seeds=range(10), n_per_polymer: int = 7, corr_weight: float = 0.02,
                      workers: int | None = None) -> pd.DataFrame:
    """:func:`design_quality` of fixed + new for :func:`conditioned_sliced_lhs` at every seed, one row each,
    in parallel processes (~2 min per design). Regenerate a seed's design with :func:`conditioned_sliced_lhs`."""
    return _parallel(_conditioned_job, [(fixed, s, n_per_polymer, corr_weight) for s in seeds], workers)


# --- Baselines for the conditioned design (framework of the DMREF DoE notebooks, section 8) ------
# Sobol and plain LHS ignore the runs already measured (a fresh sample, as the DMREF batches do); the
# optimized LHS is conditioned on them like DMREF's ``conditional_optimized_lhs``: column swaps of the new
# block that lower the centered discrepancy of [existing; new], with the existing runs frozen.

COND_BASELINES = ("Sobol", "Plain LHS", "Optimized LHS (conditioned)")


def _to_unit(runs: pd.DataFrame, polymers=None, pal_range=None, temp_range=None, cycles=RECYCLE_CYCLES) -> np.ndarray:
    """Runs in the 4-d unit cube of :func:`from_unit` (polymer and cycle at the centre of their cell)."""
    polymers = list(polymers or DOE.polymers)
    pal_range = pal_range or DOE.pal_range
    temp_range = temp_range or DOE.temp_range
    return np.column_stack([
        (runs["polymer_system"].map({p: i for i, p in enumerate(polymers)}).to_numpy(float) + 0.5) / len(polymers),
        (runs["PAL_pct"].to_numpy(float) - pal_range[0]) / (pal_range[1] - pal_range[0]),
        (runs["extrusion_temp_C"].to_numpy(float) - temp_range[0]) / (temp_range[1] - temp_range[0]),
        (np.searchsorted(np.asarray(cycles), runs["recycle_cycle"].to_numpy()) + 0.5) / len(cycles)])


def conditional_optimized_lhs(fixed: pd.DataFrame, n_new: int, seed: int = 0, n_iter: int = 5000) -> pd.DataFrame:
    """DMREF's batch extension: an optimized LHS of ``n_new`` runs in (polymer, PAL, temperature, cycle),
    refined by column swaps that lower the centered discrepancy of [fixed; new] (fixed frozen), on the lab grid."""
    rng = np.random.default_rng(seed)
    F = _to_unit(fixed)
    P = qmc.LatinHypercube(d=4, optimization="random-cd", seed=seed).random(n_new)
    best = qmc.discrepancy(np.vstack([F, P]))
    for _ in range(n_iter):
        c = rng.integers(4); a, b = rng.choice(n_new, 2, replace=False)
        P[[a, b], c] = P[[b, a], c]
        cand = qmc.discrepancy(np.vstack([F, P]))
        if cand < best:
            best = cand
        else:
            P[[a, b], c] = P[[b, a], c]
    return from_unit(P)


def _baseline_job(job):
    method, seed, fixed, n_new = job
    new = conditional_optimized_lhs(fixed, n_new, seed) if method.startswith("Optimized") else generate(method, n_new // len(DOE.polymers), seed)
    union = pd.concat([fixed[["polymer_system", "PAL_pct", "extrusion_temp_C", "recycle_cycle"]], new], ignore_index=True)
    return {"method": method, "seed": seed, **design_quality(union)}


def compare_conditioned_baselines(fixed: pd.DataFrame, n_seeds: int = 10, n_new: int = 21, methods=COND_BASELINES,
                                  workers: int | None = None) -> pd.DataFrame:
    """:func:`design_quality` of fixed + ``n_new`` new runs from each baseline in :data:`COND_BASELINES`,
    seeds 0..n_seeds-1, one row each (scored on all runs, like the conditioned sliced LHS)."""
    return _parallel(_baseline_job, [(m, s, fixed, n_new) for m in methods for s in range(n_seeds)], workers)


def write_plan_workbook(new: pd.DataFrame, fixed: pd.DataFrame, path, replicates: int = 5) -> None:
    """Hand-off workbook for the lab: sheets ``configurations`` (the new runs), ``specimen_log`` (one row per specimen, blanks
    to fill in), ``existing_runs`` (the fabricated runs the design was built around) and ``notes``."""
    per = new.groupby("polymer_system").size()
    notes = pd.DataFrame({"item": ["configuration", "existing runs", "polymer_system", "PAL_pct", "extrusion_temp_C", "recycle_cycle",
                                   "tensile_replicates", "file names", "specimen_log", "printability", "processing window"], "note": [
        f"C01-C{len(new):02d}; new numbering, independent of the August formulations f01-f12",
        f"designed around the {len(fixed)} runs already fabricated (sheet existing_runs); they are not repeated. If other runs exist, such as TPU at cycle 1, tell us and we will rebuild the design",
        "PP+PAL, HDPE+PAL or TPU+PAL; " + ", ".join(f"{n} {k.split('+')[0]}" for k, n in per.items()),
        "PAL content in %, range 10-50",
        "extrusion temperature in degC, range 180-240, 5 degC grid",
        "0 = as extruded (no recycling); 1-3 = number of recycling loops. We assume every loop uses the configuration's extrusion temperature; please tell us if not",
        f"{replicates} tensile specimens per configuration",
        "please name each test file by configuration and replicate, e.g. C07_3",
        "please fill in test date, data rate (Hz), measured diameter (mm), break location (gauge / near grip / in grip) and anything unusual (slip, voids)",
        "please also measure corner warping on every configuration, as in Sample Warping.xlsx, and note how many samples were usable",
        "if a configuration is outside what a polymer can be processed at, tell us and we will move it"]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        new.assign(tensile_replicates=replicates).to_excel(xw, sheet_name="configurations", index=False)
        specimen_log(new, replicates).to_excel(xw, sheet_name="specimen_log", index=False)
        fixed.to_excel(xw, sheet_name="existing_runs", index=False)
        notes.to_excel(xw, sheet_name="notes", index=False)

