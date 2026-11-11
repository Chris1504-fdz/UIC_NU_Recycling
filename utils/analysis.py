"""Statistical analysis of the tensile DoE data: screening, ANOVA, interaction tests, GP partial dependence
and GP Sobol indices.

Works on the per-specimen table of :func:`utils.instron.attach_doe` (factors ``polymer_system``,
``pal_pct``, ``extr_temp_C``; responses below). All responses are analysed on the log scale: they span
1-2 orders of magnitude across polymers (TPU vs PP/HDPE) and their replicate scatter grows with the level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESPONSES = {"max_tensile_stress_MPa": "UTS (MPa)",
             "young_modulus_MPa": "Young's modulus (MPa)",
             "strain_at_break": "Strain at break (-)",
             "toughness_MJ_m3": "Toughness (MJ/m³)"}
FACTORS = {"polymer_system": "polymer", "pal_pct": "PAL %", "extr_temp_C": "temperature"}


def drop_specimens(summary: pd.DataFrame, reasons: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the specimens in ``reasons`` (specimen_id -> why). Returns ``(kept, dropped)``."""
    out = summary["specimen_id"].isin(reasons)
    dropped = summary.loc[out, ["specimen_id", "polymer_system", "pal_pct", "extr_temp_C"]].copy()
    dropped["reason"] = dropped["specimen_id"].map(reasons)
    return summary.loc[~out].reset_index(drop=True), dropped.reset_index(drop=True)


def pooled_outliers(summary: pd.DataFrame, responses=tuple(RESPONSES), k: float = 1.5) -> pd.DataFrame:
    """Outlier screen that works with 3 replicates: the log residual of each specimen from its
    formulation median, pooled over all formulations (median points, residual 0, left out), then
    IQR fences ``k`` x IQR. Returns one row per specimen with the residual and flag per response."""
    summary = add_run(summary)
    out = summary[["specimen_id", "run", "polymer_system"]].copy()
    for r in responses:
        y = np.log(summary[r].astype(float))
        res = y - y.groupby(summary["run"]).transform("median")
        pool = res[res.abs() > 1e-12]
        q1, q3 = pool.quantile([0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        out[f"{r}_logres"] = res
        out[f"{r}_flag"] = (res < lo) | (res > hi)
    out["n_flags"] = out[[f"{r}_flag" for r in responses]].sum(axis=1)
    return out


def cv_table(summary: pd.DataFrame, responses=tuple(RESPONSES)) -> pd.DataFrame:
    """Replicate coefficient of variation (%) per run (formulation at one cycle) and response, with n and the
    mean (``<response>_mean``)."""
    g = add_run(summary).groupby(["formulation", "cycle", "polymer_system", "pal_pct", "extr_temp_C"])
    t = g.size().rename("n").to_frame()
    for r in responses:
        t[r] = 100 * g[r].std(ddof=1) / g[r].mean()
        t[f"{r}_mean"] = g[r].mean()
    return t.reset_index().sort_values(["cycle", "polymer_system", "pal_pct"]).reset_index(drop=True)


def anova_tables(summary: pd.DataFrame, responses=tuple(RESPONSES), interactions: bool = True) -> pd.DataFrame:
    """Type II ANOVA of log(response) on polymer (categorical), PAL % and temperature, specimen level
    (replicates give the pure error). With ``interactions`` each polymer gets its own PAL and temperature
    slope. Returns one row per (response, effect): df, F, p and partial eta^2 = SS / (SS + SS_residual)."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    d = summary.rename(columns={"polymer_system": "polymer", "pal_pct": "pal", "extr_temp_C": "temp"})
    rhs = "C(polymer) * (pal + temp)" if interactions else "C(polymer) + pal + temp"
    rows = []
    for r in responses:
        fit = smf.ols(f"np.log({r}) ~ {rhs}", data=d).fit()
        a = anova_lm(fit, typ=2)
        ss_res = a.loc["Residual", "sum_sq"]
        for eff, row in a.drop(index="Residual").iterrows():
            rows.append({"response": r, "effect": _effect_label(eff),
                         "df": int(row["df"]), "F": row["F"], "p": row["PR(>F)"],
                         "partial_eta2": row["sum_sq"] / (row["sum_sq"] + ss_res),
                         "r2_model": fit.rsquared, "df_resid": int(fit.df_resid)})
    return pd.DataFrame(rows)


def _effect_label(term: str) -> str:
    return term.replace("C(polymer)", "polymer").replace("pal", "PAL").replace("temp", "temperature").replace(":", " × ")


def _encode(polymer, pal, temp, polymers, pal_range, temp_range) -> np.ndarray:
    onehot = (np.asarray(polymer)[:, None] == np.asarray(polymers)[None, :]).astype(float)
    return np.column_stack([onehot, (np.asarray(pal, float) - pal_range[0]) / (pal_range[1] - pal_range[0]),
                            (np.asarray(temp, float) - temp_range[0]) / (temp_range[1] - temp_range[0])])


def _fit_gp(summary: pd.DataFrame, response: str, polymers, pal_range, temp_range, seed: int = 0):
    """GP on log(response): one-hot polymer, scaled PAL and temperature; Matern 5/2 with one length scale
    per input + white noise, fitted on the specimens."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    X = _encode(summary["polymer_system"], summary["pal_pct"], summary["extr_temp_C"], polymers, pal_range, temp_range)
    y = np.log(summary[response].to_numpy(float))
    kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.ones(X.shape[1]), (5e-2, 1e2), nu=2.5) + WhiteKernel(0.1, (1e-4, 1e1))
    return GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=10, random_state=seed).fit(X, y)


def gp_partial_dependence(summary: pd.DataFrame, response: str, polymers, pal_range=(10, 50), temp_range=(180, 240),
                          n_grid: int = 25, n_samples: int = 400, seed: int = 0) -> pd.DataFrame:
    """Partial dependence of ``response`` on PAL % and on temperature, per polymer, from the GP of
    :func:`_fit_gp`. For each polymer the GP is sampled jointly on a PAL x temperature grid; each sample is
    put back on the response scale and averaged over the other factor, so the band (5-95 %) is the
    uncertainty of the partial dependence itself.
    Returns long rows: polymer, factor, x, mean, lo, hi."""
    gp = _fit_gp(summary, response, polymers, pal_range, temp_range, seed)
    pal = np.linspace(*pal_range, n_grid)
    temp = np.linspace(*temp_range, n_grid)
    P, T = np.meshgrid(pal, temp, indexing="ij")
    rows = []
    for p in polymers:
        Xg = _encode(np.full(P.size, p), P.ravel(), T.ravel(), polymers, pal_range, temp_range)
        S = np.exp(gp.sample_y(Xg, n_samples, random_state=seed)).reshape(n_grid, n_grid, n_samples)
        for factor, axis, xs in (("pal_pct", 1, pal), ("extr_temp_C", 0, temp)):
            pdp = S.mean(axis=axis)                                   # (n_grid, n_samples)
            rows += [{"polymer": p, "factor": factor, "x": x, "mean": m, "lo": lo, "hi": hi}
                     for x, m, lo, hi in zip(xs, pdp.mean(1), np.percentile(pdp, 5, 1), np.percentile(pdp, 95, 1))]
    out = pd.DataFrame(rows)
    out.attrs["kernel"] = str(gp.kernel_)
    return out


def _latent_draws(gp, X: np.ndarray, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """GP mean and joint posterior draws (n_samples, len(X)) of the latent function at X. The white-noise
    term is left out, so specimen scatter is not counted as variation of the response surface."""
    import copy
    lat = copy.copy(gp)
    lat.kernel_ = gp.kernel_.k1
    mu, cov = lat.predict(X, return_cov=True)
    L = np.linalg.cholesky(cov + 1e-8 * np.mean(np.diag(cov)) * np.eye(len(mu)))
    return mu, mu + (L @ np.random.default_rng(seed).standard_normal((len(mu), n_samples))).T


def _sobol_all(F: np.ndarray) -> dict:
    """Functional ANOVA of F (..., polymer, PAL, temperature) on a uniform grid: the share of the variance
    of each main effect and interaction (they sum to 1)."""
    axes = {"polymer": -3, "PAL": -2, "temperature": -1}
    f0 = F.mean((-3, -2, -1), keepdims=True)
    main = {k: F.mean(tuple(a for a in axes.values() if a != i), keepdims=True) - f0 for k, i in axes.items()}
    pair = {}
    for a, b in (("polymer", "PAL"), ("polymer", "temperature"), ("PAL", "temperature")):
        other = ({-3, -2, -1} - {axes[a], axes[b]}).pop()
        pair[f"{a} × {b}"] = F.mean(other, keepdims=True) - f0 - main[a] - main[b]
    V = ((F - f0) ** 2).mean((-3, -2, -1))
    S = {k: (m ** 2).mean((-3, -2, -1)) / V for k, m in {**main, **pair}.items()}
    S["polymer × PAL × temperature"] = 1 - sum(S.values())
    return S


def _sobol_within(G: np.ndarray) -> dict:
    """Same for one polymer's surface G (..., PAL, temperature)."""
    g0 = G.mean((-2, -1), keepdims=True)
    V = ((G - g0) ** 2).mean((-2, -1))
    S = {"PAL": ((G.mean(-1, keepdims=True) - g0) ** 2).mean((-2, -1)) / V,
         "temperature": ((G.mean(-2, keepdims=True) - g0) ** 2).mean((-2, -1)) / V}
    S["PAL × temperature"] = 1 - S["PAL"] - S["temperature"]
    return S


def gp_sobol(summary: pd.DataFrame, response: str, polymers, pal_range=(10, 50), temp_range=(180, 240),
             n_grid: int = 21, n_samples: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Variance-based (Sobol) sensitivity indices of log(response) from the GP of :func:`_fit_gp`, with the
    inputs uniform over the polymers and the PAL and temperature ranges. The GP mean and joint posterior
    draws of the latent function are evaluated on a polymer x PAL x temperature grid and decomposed exactly
    (functional ANOVA): each index is the share of the variance due to one main effect (order 1) or one
    interaction (order 2, 3), and the total index of an input sums every term that contains it.
    ``scope`` 'all' decomposes across polymers; ``scope`` = a polymer decomposes that polymer's
    PAL x temperature surface. Returns rows: scope, term, order (1, 2, 3, 'total'), estimate (GP mean),
    median, lo, hi (5-95 % over the draws)."""
    gp = _fit_gp(summary, response, polymers, pal_range, temp_range, seed)
    Ip, Pa, Te = np.meshgrid(np.arange(len(polymers)), np.linspace(*pal_range, n_grid),
                             np.linspace(*temp_range, n_grid), indexing="ij")
    X = _encode(np.asarray(polymers)[Ip.ravel()], Pa.ravel(), Te.ravel(), polymers, pal_range, temp_range)
    mu, draws = _latent_draws(gp, X, n_samples, seed)
    shape = (len(polymers), n_grid, n_grid)
    F, M = draws.reshape(n_samples, *shape), mu.reshape(1, *shape)

    def rows(scope, S, Sm, inputs):
        for D in (S, Sm):
            D.update({f"total {x}": sum(v for k, v in D.items() if x in k.split(" × ")) for x in inputs})
        return [{"scope": scope, "term": k.removeprefix("total "),
                 "order": "total" if k.startswith("total") else k.count(" × ") + 1,
                 "estimate": float(Sm[k][0]), "median": float(np.median(v)),
                 "lo": float(np.percentile(v, 5)), "hi": float(np.percentile(v, 95))} for k, v in S.items()]

    out = rows("all", _sobol_all(F), _sobol_all(M), ["polymer", "PAL", "temperature"])
    for i, p in enumerate(polymers):
        out += rows(p, _sobol_within(F[:, i]), _sobol_within(M[:, i]), ["PAL", "temperature"])
    out = pd.DataFrame(out)
    out.insert(0, "response", response)
    out.attrs["kernel"] = str(gp.kernel_)
    return out


def lack_of_fit(summary: pd.DataFrame, responses=tuple(RESPONSES), rhs: str = "C(polymer) * (pal + temp)") -> pd.DataFrame:
    """Lack-of-fit F test of the model ``rhs`` (default: the interaction model) against one mean per run. Significant
    means the runs differ by more than replicate scatter explains (batch-to-batch variation or curvature), so
    specimen-level p-values are optimistic and inference should use the run means."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    d = add_run(summary).rename(columns={"polymer_system": "polymer", "pal_pct": "pal", "extr_temp_C": "temp"})
    rows = []
    for r in responses:
        t = anova_lm(smf.ols(f"np.log({r}) ~ {rhs}", d).fit(), smf.ols(f"np.log({r}) ~ C(run)", d).fit())
        rows.append({"response": r, "F": t["F"].iloc[1], "p": t["Pr(>F)"].iloc[1], "df_lof": int(t["df_diff"].iloc[1])})
    return pd.DataFrame(rows)


def add_run(summary: pd.DataFrame) -> pd.DataFrame:
    """Add ``cycle`` (recycle cycle, 0 when not recorded) and ``run`` (formulation at one cycle, e.g. F01·1): the
    unit that has replicates once one formulation is measured at several cycles."""
    cycle = summary["recycled_x"].fillna(0).astype(int) if "recycled_x" in summary else 0
    out = summary.assign(cycle=cycle)
    return out.assign(run=[f"F{f:02d}·{c}" for f, c in zip(out["formulation"], out["cycle"])])


def _formulation_means(summary: pd.DataFrame, response: str) -> pd.DataFrame:
    """One row per run (formulation at one cycle): polymer, pal, temp, cycle and ly = mean of log(response) over
    its specimens."""
    d = add_run(summary).rename(columns={"polymer_system": "polymer", "pal_pct": "pal", "extr_temp_C": "temp"})
    return (d.assign(ly=np.log(d[response].astype(float)))
            .groupby(["formulation", "cycle", "polymer", "pal", "temp"], as_index=False)["ly"].mean())


def anova_means(summary: pd.DataFrame, responses=tuple(RESPONSES), rhs: str = "C(polymer) + pal + temp") -> pd.DataFrame:
    """Type II ANOVA on the formulation means of log(response), by default polymer + PAL % + temperature
    (12 means, 7 residual df); ``rhs`` = "C(polymer) * pal + temp" gives each polymer its own PAL slope
    (5 residual df). The robust test when :func:`lack_of_fit` is significant. Same columns as
    :func:`anova_tables`."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows = []
    for r in responses:
        fit = smf.ols(f"ly ~ {rhs}", _formulation_means(summary, r)).fit()
        a = anova_lm(fit, typ=2)
        ss_res = a.loc["Residual", "sum_sq"]
        for eff, row in a.drop(index="Residual").iterrows():
            rows.append({"response": r, "effect": _effect_label(eff),
                         "df": int(row["df"]), "F": row["F"], "p": row["PR(>F)"],
                         "partial_eta2": row["sum_sq"] / (row["sum_sq"] + ss_res),
                         "r2_model": fit.rsquared, "df_resid": int(fit.df_resid)})
    return pd.DataFrame(rows)


def interaction_tests(summary: pd.DataFrame, responses=tuple(RESPONSES)) -> pd.DataFrame:
    """Each two-way interaction added on its own to the additive means model of :func:`anova_means`
    (polymer + PAL % + temperature, 7 residual df) and tested with a nested F test. Testing one at a time
    keeps 5-6 residual df; with all of them at once only 3 are left. Returns response, interaction, df, F, p
    and R^2 without and with the interaction."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    terms = {"polymer × PAL": "C(polymer):pal", "polymer × temperature": "C(polymer):temp", "PAL × temperature": "pal:temp"}
    rows = []
    for r in responses:
        fm = _formulation_means(summary, r)
        base = smf.ols("ly ~ C(polymer) + pal + temp", fm).fit()
        for name, t in terms.items():
            full = smf.ols(f"ly ~ C(polymer) + pal + temp + {t}", fm).fit()
            c = anova_lm(base, full).iloc[1]
            rows.append({"response": r, "interaction": name, "df": int(c["df_diff"]), "F": c["F"], "p": c["Pr(>F)"],
                         "r2_additive": base.rsquared, "r2_with": full.rsquared, "df_resid": int(full.df_resid)})
    return pd.DataFrame(rows)


def pal_effect_by_polymer(summary: pd.DataFrame, responses=tuple(RESPONSES), step: float = 10.0) -> pd.DataFrame:
    """Change (%) of each response for +``step`` PAL points within each polymer, temperature held fixed:
    one PAL slope per polymer in log(response) ~ polymer + polymer:PAL + temperature on the formulation
    means. Returns response, polymer, change_pct and its 95 % CI (lo, hi)."""
    import statsmodels.formula.api as smf
    rows = []
    for r in responses:
        fit = smf.ols("ly ~ 0 + C(polymer) + C(polymer):pal + temp", _formulation_means(summary, r)).fit()
        ci = fit.conf_int()
        for term in fit.params.index[fit.params.index.str.endswith(":pal")]:
            pct = lambda b: 100 * (np.exp(step * b) - 1)
            rows.append({"response": r, "polymer": term.split("[")[1].split("]")[0], "change_pct": pct(fit.params[term]),
                         "lo": pct(ci.loc[term, 0]), "hi": pct(ci.loc[term, 1]), "p": fit.pvalues[term]})
    return pd.DataFrame(rows)


def response_correlation(summary: pd.DataFrame, responses=tuple(RESPONSES), polymers=None,
                         method: str = "pearson") -> dict:
    """Correlation between the responses, on log values, over the specimens: all polymers pooled and each
    polymer on its own, in the order of ``polymers`` (pooled values are dominated by the differences between
    polymers). Returns {label: correlation DataFrame}, label = "All polymers" or the polymer;
    ``.attrs["n"]`` = specimens."""
    polymers = [p for p in (polymers or sorted(summary["polymer_system"].unique()))
                if (summary["polymer_system"] == p).any()]
    out = {}
    groups = [("All polymers", summary)] + [(p, summary[summary["polymer_system"] == p]) for p in polymers]
    for label, g in groups:
        c = np.log(g[list(responses)].astype(float)).corr(method=method)
        c.attrs["n"] = len(g)
        out[label] = c
    return out


# --- Recycling: the same formulation at two cycles --------------------------------------------------------------

def cycle_pairs(summary: pd.DataFrame, responses=tuple(RESPONSES), cycles=(0, 1)) -> pd.DataFrame:
    """Per formulation measured at both ``cycles``: the geometric means and their ratio (later / earlier) with a 95 %
    CI from the replicates (Welch on the logs). Long rows: response, formulation, polymer_system, pal_pct, extr_temp_C,
    gm_0, gm_1, ratio, lo, hi, p."""
    from scipy import stats
    d = add_run(summary)
    rows = []
    for (f, poly, pal, temp), g in d.groupby(["formulation", "polymer_system", "pal_pct", "extr_temp_C"]):
        a, b = g[g["cycle"] == cycles[0]], g[g["cycle"] == cycles[1]]
        if a.empty or b.empty:
            continue
        for r in responses:
            la, lb = np.log(a[r].astype(float)), np.log(b[r].astype(float))
            va, vb = la.var(ddof=1) / len(la), lb.var(ddof=1) / len(lb)
            df = (va + vb) ** 2 / (va ** 2 / (len(la) - 1) + vb ** 2 / (len(lb) - 1))
            dm, half = lb.mean() - la.mean(), stats.t.ppf(0.975, df) * np.sqrt(va + vb)
            rows.append({"response": r, "formulation": f, "polymer_system": poly, "pal_pct": pal, "extr_temp_C": temp,
                         "gm_0": np.exp(la.mean()), "gm_1": np.exp(lb.mean()), "ratio": np.exp(dm),
                         "lo": np.exp(dm - half), "hi": np.exp(dm + half),
                         "p": stats.ttest_ind(lb, la, equal_var=False).pvalue})
    return pd.DataFrame(rows)


def cycle_effect(pairs: pd.DataFrame) -> pd.DataFrame:
    """Average effect of the recycle step across formulations, from :func:`cycle_pairs`: the formulation is the unit
    (paired on the log ratios), overall and per polymer. Rows: response, group, n, ratio, lo, hi (95 % CI), p
    (paired t), p_wilcoxon (exact signed-rank, overall only is meaningful with n = 8)."""
    from scipy import stats
    rows = []
    for r, g in pairs.groupby("response", sort=False):
        for label, s in [("all", g)] + list(g.groupby("polymer_system")):
            x = np.log(s["ratio"].to_numpy())
            se = x.std(ddof=1) / np.sqrt(len(x))
            half = stats.t.ppf(0.975, len(x) - 1) * se
            rows.append({"response": r, "group": label, "n": len(x), "ratio": np.exp(x.mean()),
                         "lo": np.exp(x.mean() - half), "hi": np.exp(x.mean() + half),
                         "p": stats.ttest_1samp(x, 0).pvalue, "p_wilcoxon": stats.wilcoxon(x).pvalue})
    return pd.DataFrame(rows)


def cycle_interactions(pairs: pd.DataFrame) -> pd.DataFrame:
    """Does the recycle effect depend on polymer, PAL % or temperature? Each is regressed on its own against the
    formulations' log ratios from :func:`cycle_pairs`. Rows: response, factor, F, p, n."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    rows = []
    for r, g in pairs.groupby("response", sort=False):
        d = g.assign(lr=np.log(g["ratio"]))
        for name, rhs in (("polymer", "C(polymer_system)"), ("PAL %", "pal_pct"), ("temperature", "extr_temp_C")):
            a = anova_lm(smf.ols(f"lr ~ {rhs}", d).fit(), typ=2).iloc[0]
            rows.append({"response": r, "factor": name, "F": a["F"], "p": a["PR(>F)"], "n": len(d)})
    return pd.DataFrame(rows)


def gp_with_cycle(summary: pd.DataFrame, response: str, polymers, pal_range=(10, 50), temp_range=(180, 240), seed: int = 0):
    """GP of :func:`_fit_gp` with the recycle cycle (scaled by 1/3) as one more input."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    d = add_run(summary)
    X = np.column_stack([_encode(d["polymer_system"], d["pal_pct"], d["extr_temp_C"], polymers, pal_range, temp_range),
                         d["cycle"].to_numpy(float) / 3])
    y = np.log(d[response].to_numpy(float))
    kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(np.ones(X.shape[1]), (5e-2, 1e2), nu=2.5) + WhiteKernel(0.1, (1e-4, 1e1))
    return GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=10, random_state=seed).fit(X, y)


def gp_partial_dependence_cycles(summary: pd.DataFrame, response: str, polymers, cycles=(0, 1), pal_range=(10, 50),
                                 temp_range=(180, 240), n_grid: int = 25, n_samples: int = 400, seed: int = 0) -> pd.DataFrame:
    """Partial dependence as :func:`gp_partial_dependence`, from :func:`gp_with_cycle`, per polymer at each measured
    cycle (a polymer is never predicted at a cycle it has no data for). Long rows: polymer, cycle, factor, x, mean, lo, hi."""
    gp = gp_with_cycle(summary, response, polymers, pal_range, temp_range, seed)
    have = set(zip(add_run(summary)["polymer_system"], add_run(summary)["cycle"]))
    pal, temp = np.linspace(*pal_range, n_grid), np.linspace(*temp_range, n_grid)
    P, T = np.meshgrid(pal, temp, indexing="ij")
    rows = []
    for p in polymers:
        for c in cycles:
            if (p, c) not in have:
                continue
            Xg = np.column_stack([_encode(np.full(P.size, p), P.ravel(), T.ravel(), polymers, pal_range, temp_range),
                                  np.full(P.size, c / 3)])
            _, D = _latent_draws(gp, Xg, n_samples, seed)
            S = np.exp(D.T).reshape(n_grid, n_grid, n_samples)
            for factor, axis, xs in (("pal_pct", 1, pal), ("extr_temp_C", 0, temp)):
                pdp = S.mean(axis=axis)
                rows += [{"polymer": p, "cycle": c, "factor": factor, "x": x, "mean": m, "lo": lo, "hi": hi}
                         for x, m, lo, hi in zip(xs, pdp.mean(1), np.percentile(pdp, 5, 1), np.percentile(pdp, 95, 1))]
    out = pd.DataFrame(rows)
    out.attrs["kernel"] = str(gp.kernel_)
    return out


def gp_sobol_cycle(summary: pd.DataFrame, response: str, polymers=("PP+PAL", "HDPE+PAL"), cycles=(0, 1), pal_range=(10, 50),
                   temp_range=(180, 240), n_grid: int = 13, n_samples: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Sobol indices of log(response) over polymer x PAL x temperature x cycle from :func:`gp_with_cycle`, on the
    polymers measured at every cycle in ``cycles`` (TPU has no recycled data yet, so it is left out, not extrapolated).
    Exact functional ANOVA on the grid. Rows: response, term, order (1-4 or 'total'), estimate, median, lo, hi."""
    from itertools import combinations
    d = summary[summary["polymer_system"].isin(polymers)]
    gp = gp_with_cycle(d, response, list(polymers), pal_range, temp_range, seed)
    pal, temp = np.linspace(*pal_range, n_grid), np.linspace(*temp_range, n_grid)
    Ip, Pa, Te, Cy = np.meshgrid(np.arange(len(polymers)), pal, temp, np.asarray(cycles, float), indexing="ij")
    X = np.column_stack([_encode(np.asarray(polymers)[Ip.ravel()], Pa.ravel(), Te.ravel(), list(polymers), pal_range, temp_range),
                         Cy.ravel() / 3])
    mu, draws = _latent_draws(gp, X, n_samples, seed)
    shape = Ip.shape
    names = ("polymer", "PAL", "temperature", "cycle")

    def decompose(F):                       # F: (samples, polymer, PAL, temperature, cycle)
        axes = (1, 2, 3, 4)
        total = ((F - F.mean(axes, keepdims=True)) ** 2).mean(axes)
        effect, shares = {}, {}
        for order in range(1, 5):
            for sub in combinations(range(4), order):
                m = F.mean(tuple(axes[i] for i in range(4) if i not in sub), keepdims=True) - F.mean(axes, keepdims=True)
                for k in range(1, order):   # take out the lower-order terms it contains
                    for low in combinations(sub, k):
                        m = m - effect[low]
                effect[sub] = m
                shares[sub] = (m ** 2).mean(axes) / total
        return shares

    S, Sm = decompose(draws.reshape(n_samples, *shape)), decompose(mu.reshape(1, *shape))
    rows = []
    label = lambda sub: " × ".join(names[i] for i in sub)
    for sub, v in S.items():
        rows.append({"term": label(sub), "order": len(sub), "estimate": float(Sm[sub][0]), "median": float(np.median(v)),
                     "lo": float(np.percentile(v, 5)), "hi": float(np.percentile(v, 95))})
    for i, name in enumerate(names):
        tot = sum(v for sub, v in S.items() if i in sub)
        rows.append({"term": name, "order": "total", "estimate": float(sum(v[0] for sub, v in Sm.items() if i in sub)),
                     "median": float(np.median(tot)), "lo": float(np.percentile(tot, 5)), "hi": float(np.percentile(tot, 95))})
    out = pd.DataFrame(rows)
    out.insert(0, "response", response)
    out.attrs["kernel"] = str(gp.kernel_)
    return out


# --- Printability: corner warping -------------------------------------------------------------------------------

def load_warping(path, design: pd.DataFrame) -> pd.DataFrame:
    """The lab's ``Sample Warping.xlsx``: one column per formulation (header F01, F04(note), ...), corner warping in cm
    below it. Long rows: formulation, polymer_system, pal_pct, extr_temp_C, warping_cm, note."""
    import re
    raw = pd.read_excel(path, header=None)
    head = next(i for i, row in raw.iterrows() if row.astype(str).str.fullmatch(r"F\d+.*").sum() >= 3)
    rows = []
    for col in raw.columns:
        m = re.match(r"F(\d+)\s*(?:\((.*)\))?", str(raw.at[head, col]))
        if m:
            vals = pd.to_numeric(raw.loc[head + 1:, col], errors="coerce").dropna()
            rows += [{"formulation": int(m[1]), "warping_cm": float(v), "note": m[2] or ""} for v in vals]
    d = design.rename(columns={"PAL_pct": "pal_pct", "extrusion_temp_C": "extr_temp_C"})
    return pd.DataFrame(rows).merge(d, on="formulation", how="left").sort_values(["formulation"]).reset_index(drop=True)


def warping_summary(warp: pd.DataFrame) -> pd.DataFrame:
    """Per formulation: n corners, mean, sd, maximum and share of corners at 0.3 cm or more."""
    g = warp.groupby(["formulation", "polymer_system", "pal_pct", "extr_temp_C"])["warping_cm"]
    return g.agg(n="size", mean="mean", sd="std", max="max", share_ge_03=lambda v: float((v >= 0.3).mean())).reset_index()


def warping_tests(warp: pd.DataFrame, summary: pd.DataFrame, responses=tuple(RESPONSES)) -> dict:
    """Warping against the design and against the cycle-0 tensile properties. ``between``: Kruskal-Wallis across
    formulations on the corners (differences between formulations at all). ``anova``: Type II on the formulation means,
    mean warping ~ polymer + PAL + temperature (not log: zeros occur). ``tensile``: Spearman of mean warping with each
    response's geometric mean (formulation level)."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from scipy import stats
    ws = warping_summary(warp).rename(columns={"polymer_system": "polymer", "pal_pct": "pal", "extr_temp_C": "temp"})
    a = anova_lm(smf.ols("mean ~ C(polymer) + pal + temp", ws).fit(), typ=2)
    anova = pd.DataFrame({"effect": [_effect_label(i) for i in a.index[:-1]], "df": a["df"].iloc[:-1].astype(int).to_numpy(),
                          "F": a["F"].iloc[:-1].to_numpy(), "p": a["PR(>F)"].iloc[:-1].to_numpy()})
    kw = stats.kruskal(*[g["warping_cm"].to_numpy() for _, g in warp.groupby("formulation")])
    d = add_run(summary)
    d0 = d[d["cycle"] == 0].groupby("formulation")[list(responses)].agg(lambda v: float(np.exp(np.log(v).mean())))
    w = ws.set_index("formulation")["mean"]
    tensile = []
    for r in responses:
        rho = stats.spearmanr(w, d0.loc[w.index, r])
        tensile.append({"response": r, "spearman_rho": rho.statistic, "p": rho.pvalue, "n": len(w)})
    return {"between": {"H": kw.statistic, "p": kw.pvalue}, "anova": anova, "tensile": pd.DataFrame(tensile)}


def diameter_effect(summary: pd.DataFrame, responses=tuple(RESPONSES)) -> pd.DataFrame:
    """Do thicker specimens of one run read differently? Regression of log(response) on log(diameter) after removing
    each run's mean, so polymer, PAL, temperature and cycle play no part. Rows: response, slope, r, p, n."""
    from scipy import stats
    d = add_run(summary).dropna(subset=["diameter_mm"])      # a few files carry no diameter
    x = np.log(d["diameter_mm"].astype(float))
    xr = x - x.groupby(d["run"]).transform("mean")
    rows = []
    for r in responses:
        y = np.log(d[r].astype(float))
        fit = stats.linregress(xr, y - y.groupby(d["run"]).transform("mean"))
        rows.append({"response": r, "slope": fit.slope, "r": fit.rvalue, "p": fit.pvalue, "n": len(d)})
    return pd.DataFrame(rows)


def cycle_vs_factors(pairs: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Across the paired formulations, the log ratio of :func:`cycle_pairs` against extrusion temperature and against the
    change in mean diameter between the cycles (Pearson and Spearman), and temperature again without each formulation
    in turn (largest leave-one-out p). Exploratory: n = number of pairs. Rows: response, x, slope, r, p, spearman, p_spearman, p_loo."""
    from scipy import stats
    dm = np.log(add_run(summary).groupby(["formulation", "cycle"])["diameter_mm"].mean().unstack())
    rows = []
    for r, g in pairs.groupby("response", sort=False):
        g = g.assign(lr=np.log(g["ratio"]), dlog=[dm.at[f, 1] - dm.at[f, 0] for f in g["formulation"]])
        for name, col in (("extrusion temperature", "extr_temp_C"), ("diameter change", "dlog")):
            f, sp = stats.linregress(g[col], g["lr"]), stats.spearmanr(g[col], g["lr"])
            loo = max(stats.linregress(g.drop(i)[col], g.drop(i)["lr"]).pvalue for i in g.index)
            rows.append({"response": r, "x": name, "slope": f.slope, "r": f.rvalue, "p": f.pvalue, "spearman": sp.statistic,
                         "p_spearman": sp.pvalue, "p_loo": loo, "n": len(g)})
    return pd.DataFrame(rows)


def cycle_means(pairs: pd.DataFrame) -> pd.DataFrame:
    """Average level at each cycle over the paired formulations, overall and per polymer: geometric mean across
    formulations with a 95 % CI (t, formulations as the unit), and the change. Rows: response, group, cycle, gm, lo, hi."""
    from scipy import stats
    rows = []
    for r, g in pairs.groupby("response", sort=False):
        for label, s in [("all", g)] + list(g.groupby("polymer_system")):
            for c, col in ((0, "gm_0"), (1, "gm_1")):
                x = np.log(s[col].to_numpy())
                half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
                rows.append({"response": r, "group": label, "cycle": c, "n": len(x), "gm": np.exp(x.mean()),
                             "lo": np.exp(x.mean() - half), "hi": np.exp(x.mean() + half)})
    return pd.DataFrame(rows)


def cycle_trend_table(summary: pd.DataFrame, responses=tuple(RESPONSES)) -> pd.DataFrame:
    """Is the trend the same after recycling? On the formulations measured at both cycles, the slopes in
    log(response) ~ polymer + PAL + temperature at each cycle (% change per +10 PAL points and per +10 degC, adjusted for
    the other factor) and the test that a slope changed (same model on the log ratios). Rows: response, factor, slope_0,
    lo_0, hi_0, slope_1, lo_1, hi_1, p_change."""
    import statsmodels.formula.api as smf
    d = add_run(summary)
    both = d.groupby("formulation")["cycle"].nunique()
    d = d[d["formulation"].isin(both[both > 1].index)]
    pct = lambda b: 100 * (np.exp(10 * b) - 1)
    rows = []
    for r in responses:
        fm = _formulation_means(d, r)
        fits = {c: smf.ols("ly ~ C(polymer) + pal + temp", fm[fm["cycle"] == c]).fit() for c in (0, 1)}
        w = fm.pivot(index=["formulation", "polymer", "pal", "temp"], columns="cycle", values="ly").reset_index()
        diff = smf.ols("lr ~ C(polymer) + pal + temp", w.assign(lr=w[1] - w[0])).fit()
        for name, term in (("PAL", "pal"), ("temperature", "temp")):
            row = {"response": r, "factor": name, "p_change": diff.pvalues[term]}
            for c, f in fits.items():
                ci = f.conf_int().loc[term]
                row.update({f"slope_{c}": pct(f.params[term]), f"lo_{c}": pct(ci[0]), f"hi_{c}": pct(ci[1])})
            rows.append(row)
    return pd.DataFrame(rows)


def warping_slopes(warp: pd.DataFrame) -> pd.DataFrame:
    """Does adding PAL or changing temperature move warping? mean warping ~ polymer + PAL + temperature on the formulation
    means, in cm per +10 PAL points and per +10 degC, with 95 % CI; ``within_polymer`` rows give each polymer's own PAL and
    temperature correlation (Spearman, n = 3 or 4: descriptive only)."""
    import statsmodels.formula.api as smf
    from scipy import stats
    ws = warping_summary(warp).rename(columns={"polymer_system": "polymer", "pal_pct": "pal", "extr_temp_C": "temp"})
    f = smf.ols("mean ~ C(polymer) + pal + temp", ws).fit()
    rows = []
    ci = f.conf_int()
    for name, term in (("PAL", "pal"), ("temperature", "temp")):
        rows.append({"scope": "all", "factor": name, "cm_per_10": 10 * f.params[term], "lo": 10 * ci.loc[term, 0],
                     "hi": 10 * ci.loc[term, 1], "p": f.pvalues[term], "n": len(ws)})
    for poly, g in ws.groupby("polymer"):
        for name, term in (("PAL", "pal"), ("temperature", "temp")):
            sp = stats.spearmanr(g[term], g["mean"])
            rows.append({"scope": poly, "factor": name, "spearman": sp.statistic, "p": sp.pvalue, "n": len(g)})
    return pd.DataFrame(rows)

