"""
lvgp_compare.py -- data preparation, fitting and leave-one-formulation-out comparison of the LVGP and lvgp_mixed on the
tensile data of this project (used by lvgp_vs_mixed.ipynb and check_same_as_cat_kernel.py).

Modeling unit: the formulation. Each of the 12 formulations is one design point (polymer system = categorical level;
PAL % and extrusion temperature = continuous inputs); its response is the mean of log(response) over its specimens, and the
variance of that mean (sample variance / n) is the fixed noise. Log scale because the replicate scatter grows with the level
and the responses span one to two orders of magnitude across polymers (see utils/analysis.py).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

POLYMERS = ["PP+PAL", "HDPE+PAL", "TPU+PAL"]                  # categorical levels 1, 2, 3 (utils.config.DOE.polymers)
BOUNDS = [(10.0, 50.0), (180.0, 240.0)]                       # PAL %, extrusion temperature (degC): the DoE ranges
RESPONSES = {"max_tensile_stress_MPa": "UTS (MPa)", "young_modulus_MPa": "Young's modulus (MPa)",
             "strain_at_break": "Strain at break (-)", "toughness_MJ_m3": "Toughness (MJ/m³)"}
DROP = {"f11_s3": "failed test: no load recorded (UTS 0.12 MPa, modulus fit R² 0.001)",
        "f08_s1": "superseded by the f08 remake (f08-new): 119 % strain vs 7 % for the 3 remade specimens"}
LABELS = {"lvgp": "LVGP", "lvgp_mixed": "lvgp_mixed", "polymer_mean": "polymer mean", "global_mean": "global mean"}


def load_tensile(project_root, delivery="10_26_26"):
    """Per-specimen tensile properties of a delivery, parsed with the project's utils (same steps and the same two
    specimens set aside as notebooks/data_processing_v4.ipynb)."""
    from pathlib import Path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils import config, io, instron, analysis
    p = config.paths(Path(project_root))
    design = pd.read_csv(p.doe / "doe_formulations.csv")
    cat = instron.catalogue(io.extract_zip(p.root / "raw_data" / delivery))
    _, raw = instron.load_delivery(cat)
    summary = instron.attach_doe(raw, design)
    kept, _ = analysis.drop_specimens(summary, DROP)
    return kept


def formulation_table(kept, response):
    """One row per formulation: inputs, level code, number of specimens, mean and variance of log(response), and the
    variance of the mean (the fixed noise)."""
    d = kept.assign(logy=np.log(kept[response].astype(float)))
    t = d.groupby("formulation").agg(polymer=("polymer_system", "first"), pal=("pal_pct", "first"), temp=("extr_temp_C", "first"),
                                     n=("logy", "size"), mean_log=("logy", "mean"), var_log=("logy", "var"),
                                     mean_raw=(response, "mean")).reset_index()
    t["level"] = [POLYMERS.index(p) + 1 for p in t["polymer"]]
    t["var_mean"] = t["var_log"] / t["n"]
    return t.sort_values(["level", "pal"]).reset_index(drop=True)


def to_data(table):
    """{level: dict(X, y_mean, y_var)} for the models; every level appears, even with no point."""
    data = {}
    for lv in range(1, len(POLYMERS) + 1):
        r = table[table["level"] == lv]
        data[lv] = dict(X=r[["pal", "temp"]].to_numpy(float), y_mean=r["mean_log"].to_numpy(float),
                        y_var=r["var_mean"].to_numpy(float))
    return data


def formulation_data(kept, response):
    return to_data(formulation_table(kept, response)), BOUNDS


def fit(key, data, seed=0):
    import torch
    from models import MODELS
    warnings.filterwarnings("ignore")
    torch.manual_seed(seed)
    return MODELS[key].fit(data, needs_r=False, bounds=BOUNDS)


def _fit_predict(task):
    """One left-out fit (top level so a process pool can run it): returns (index, mean, variance)."""
    import torch
    torch.set_num_threads(1)
    idx, key, seed, data, level, x = task
    m = fit(key, data, seed)
    mu, var = m.predict(level, x)
    return idx, float(np.ravel(mu)[0]), float(np.ravel(var)[0])


def lofo(kept, response, keys=("lvgp", "lvgp_mixed"), seeds=(0,), workers=1):
    """Leave-one-formulation-out: fit on the other 11 formulation means, predict the left-out mean (log scale).
    Rows: formulation x model x seed with the error, the predictive variance and the standardized residual
    z = error / sqrt(predictive variance + variance of the observed mean). Two baselines without a GP: the mean of the
    same polymer's other formulations, and the mean of all other formulations."""
    t = formulation_table(kept, response)
    rows, tasks = [], []
    for i, r in t.iterrows():
        train = t.drop(index=i)
        data = to_data(train)
        x = np.array([[r["pal"], r["temp"]]])
        for key in keys:
            for seed in seeds:
                rows.append(dict(formulation=r["formulation"], polymer=r["polymer"], model=key, seed=seed, y=r["mean_log"],
                                 var_obs=r["var_mean"], mu=np.nan, var_pred=np.nan))
                tasks.append((len(rows) - 1, key, seed, data, int(r["level"]), x))
        same = train[train["level"] == r["level"]]["mean_log"]
        for key, vals in (("polymer_mean", same), ("global_mean", train["mean_log"])):
            rows.append(dict(formulation=r["formulation"], polymer=r["polymer"], model=key, seed=0, y=r["mean_log"],
                             var_obs=r["var_mean"], mu=float(vals.mean()), var_pred=float(vals.var() * (1 + 1 / len(vals)))))
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_fit_predict, tasks))
    else:
        results = [_fit_predict(tk) for tk in tasks]
    for idx, mu, var in results:
        rows[idx]["mu"], rows[idx]["var_pred"] = mu, var
    out = pd.DataFrame(rows)
    out["err"] = out["mu"] - out["y"]
    out["z"] = out["err"] / np.sqrt(out["var_pred"] + out["var_obs"])
    out["nlpd"] = 0.5 * np.log(2 * np.pi * (out["var_pred"] + out["var_obs"])) + 0.5 * out["z"] ** 2
    out["response"] = response
    return out


def summarize(cv):
    """Per response and model: RMSE of log(response) (≈ relative error), the median absolute relative error in %,
    the mean negative log predictive density, and the share of |z| <= 2 (calibration; about 95 % if calibrated)."""
    g = cv.groupby(["response", "model", "seed"])
    s = pd.DataFrame({"rmse_log": g["err"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
                      "median_abs_rel_err_pct": g["err"].apply(lambda e: float(np.median(np.expm1(np.abs(e))) * 100)),
                      "nlpd": g["nlpd"].mean(),
                      "within_2sd": g["z"].apply(lambda z: float(np.mean(np.abs(z) <= 2)))}).reset_index()
    return s.groupby(["response", "model"]).mean(numeric_only=True).drop(columns="seed").reset_index()


def latent_positions(model):
    """The LVGP's (or lvgp_mixed's) 2-D latent coordinates of the polymer levels, one row per level."""
    z = model.model.lv_mapping_layers[0].latents.detach().cpu().numpy()
    return pd.DataFrame(z, columns=["z1", "z2"], index=POLYMERS)


def mixed_shares(model):
    """lvgp_mixed's three output scales and the variance shares of its trend, offset and level-specific parts."""
    from models.mixed_versions.lvgp_blocks import mixed_scale_shares
    return mixed_scale_shares(model.model)


def joint_draws(model, level, X, n_samples=400, seed=0):
    """Joint draws of the fitted model's latent function (log response, no specimen noise) at the points X of one
    polymer level: returns (posterior mean, draws of shape (n_samples, len(X)))."""
    import torch
    with torch.no_grad():
        post = model.model(model._encode(level, X))
        mu = post.mean.numpy().ravel()
        C = post.covariance_matrix.numpy()
    ys, ym = float(model.model.y_std), float(model.model.y_mean)
    mu, C = ym + ys * mu, C * ys * ys
    C = 0.5 * (C + C.T)
    L = np.linalg.cholesky(C + 1e-9 * np.mean(np.diag(C)) * np.eye(len(mu)))
    Z = np.random.default_rng(seed).standard_normal((len(mu), n_samples))
    return mu, (mu[:, None] + L @ Z).T


def partial_dependence(model, n_grid=25, n_samples=400, seed=0):
    """Partial dependence on PAL % and on temperature, per polymer, the same way as utils.analysis.gp_partial_dependence:
    joint draws on a PAL x temperature grid, put back on the response scale (exp), averaged over the other input; the band
    (5-95 %) is the uncertainty of the partial dependence itself. Long rows: polymer, factor, x, mean, lo, hi."""
    pal = np.linspace(*BOUNDS[0], n_grid)
    temp = np.linspace(*BOUNDS[1], n_grid)
    P, T = np.meshgrid(pal, temp, indexing="ij")
    rows = []
    for lv, p in enumerate(POLYMERS, start=1):
        _, D = joint_draws(model, lv, np.column_stack([P.ravel(), T.ravel()]), n_samples, seed)
        S = np.exp(D.T).reshape(n_grid, n_grid, n_samples)
        for factor, axis, xs in (("pal_pct", 1, pal), ("extr_temp_C", 0, temp)):
            pdp = S.mean(axis=axis)
            rows += [{"polymer": p, "factor": factor, "x": x, "mean": m, "lo": lo, "hi": hi}
                     for x, m, lo, hi in zip(xs, pdp.mean(1), np.percentile(pdp, 5, 1), np.percentile(pdp, 95, 1))]
    return pd.DataFrame(rows)


POLY_COLORS = {"PP+PAL": "#2a78d6", "HDPE+PAL": "#eb6834", "TPU+PAL": "#1baf7a"}   # as utils/plots.py


def plot_partial_dependence(pds, title=None, log_y=False, save=None):
    """The layout of utils.plots.partial_dependence_grid (columns = responses, rows = PAL % and temperature, one line per
    polymer with its 5-95 % band) with a title, for one model. Linear y from 0 by default; ``log_y`` for a log axis."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    resp = list(pds)
    fig, axes = plt.subplots(2, len(resp), figsize=(17, 7.4), sharex="row")
    for j, r in enumerate(resp):
        d = pds[r]
        for i, (factor, xl) in enumerate((("pal_pct", "PAL content (%)"), ("extr_temp_C", "Extrusion temperature (°C)"))):
            ax = axes[i, j]
            for p, col in POLY_COLORS.items():
                s = d[(d["polymer"] == p) & (d["factor"] == factor)]
                ax.fill_between(s["x"], s["lo"], s["hi"], color=col, alpha=0.15, lw=0)
                ax.plot(s["x"], s["mean"], color=col, lw=2.2)
            if log_y:
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=10.5); ax.grid(alpha=.3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(RESPONSES[r], fontsize=13)
            ax.set_xlabel(xl, fontsize=11.5)
            if j == 0:
                ax.set_ylabel("averaged over temperature" if i == 0 else "averaged over PAL %", fontsize=11)
    handles = [Line2D([0], [0], color=c, lw=2.2, label=p) for p, c in POLY_COLORS.items()]
    fig.legend(handles=handles + [Patch(color="#9a9a96", alpha=0.3, label="5–95 % band")], loc="lower center",
               ncol=4, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97 if title else 1))
    if save:
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# --- The LVGP with the recycle cycle as an input -----------------------------------------------------------------------
# Modeling unit: the run (one formulation at one cycle), so the 8 recycled formulations are 2 design points each.
# The cycle is a third continuous input, not a level: cycle 2 should be closer to cycle 1 than to cycle 0.

BOUNDS_CYCLE = BOUNDS + [(0.0, 3.0)]                          # PAL %, temperature, recycle cycle


def run_table(kept, response):
    """One row per run: inputs, level code, number of specimens, mean and variance of log(response), and the variance of
    the mean (the fixed noise)."""
    d = kept.assign(logy=np.log(kept[response].astype(float)), cycle=kept["recycled_x"].fillna(0).astype(int))
    t = d.groupby(["formulation", "cycle"]).agg(polymer=("polymer_system", "first"), pal=("pal_pct", "first"),
                                                temp=("extr_temp_C", "first"), n=("logy", "size"), mean_log=("logy", "mean"),
                                                var_log=("logy", "var")).reset_index()
    t["level"] = [POLYMERS.index(p) + 1 for p in t["polymer"]]
    t["var_mean"] = t["var_log"] / t["n"]
    return t.sort_values(["level", "pal", "cycle"]).reset_index(drop=True)


def to_data_cycle(table):
    """{level: dict(X, y_mean, y_var)} with X = (PAL %, temperature, cycle)."""
    data = {}
    for lv in range(1, len(POLYMERS) + 1):
        r = table[table["level"] == lv]
        data[lv] = dict(X=r[["pal", "temp", "cycle"]].to_numpy(float), y_mean=r["mean_log"].to_numpy(float),
                        y_var=r["var_mean"].to_numpy(float))
    return data


def fit_cycle(kept, response, key="lvgp", seed=0):
    import torch
    from models import MODELS
    warnings.filterwarnings("ignore")
    torch.manual_seed(seed)
    return MODELS[key].fit(to_data_cycle(run_table(kept, response)), needs_r=False, bounds=BOUNDS_CYCLE)


def length_scales(model, names=("PAL %", "temperature", "cycle")):
    """The fitted length scale of each continuous input, in units of its scaled [0, 1] range: small = the response changes
    quickly along that input, large (> about 2) = nearly flat along it. For lvgp_mixed, the scales of its LVGP part."""
    import torch
    for m in model.model.modules():                      # the RBF over the continuous inputs: one scale per input
        ls = getattr(m, "lengthscale", None)
        if isinstance(ls, torch.Tensor) and ls.numel() == len(names):
            return dict(zip(names, ls.detach().cpu().numpy().ravel()))
    return {}


def cycle_surfaces(model, n_grid=41, cycles=(0, 1), n_samples=400, seed=0):
    """Fitted surfaces on a PAL x temperature grid. Long dict {(polymer, cycle): array of the response} from the posterior
    mean (exp of the log mean), and {polymer: dict(ratio, lo, hi, prob_gain)} for the ratio of the last to the first cycle
    from joint posterior draws (5-95 % band; prob_gain = probability that the ratio exceeds 1). ``pal``, ``temp`` = grids."""
    pal, temp = np.linspace(*BOUNDS[0], n_grid), np.linspace(*BOUNDS[1], n_grid)
    P, T = np.meshgrid(pal, temp, indexing="ij")
    surf, ratio = {}, {}
    for lv, p in enumerate(POLYMERS, start=1):
        X = np.vstack([np.column_stack([P.ravel(), T.ravel(), np.full(P.size, c, float)]) for c in cycles])
        mu, D = joint_draws(model, lv, X, n_samples, seed)
        mu, D = mu.reshape(len(cycles), n_grid, n_grid), D.reshape(n_samples, len(cycles), n_grid, n_grid)
        for i, c in enumerate(cycles):
            surf[(p, c)] = np.exp(mu[i])
        d = D[:, -1] - D[:, 0]
        ratio[p] = dict(ratio=np.exp(mu[-1] - mu[0]), lo=np.exp(np.percentile(d, 5, axis=0)), hi=np.exp(np.percentile(d, 95, axis=0)),
                        prob_gain=(d > 0).mean(axis=0))
    return dict(pal=pal, temp=temp, surf=surf, ratio=ratio)


def average_ratio(model, polymers=("PP+PAL", "HDPE+PAL"), n_grid=21, n_samples=1000, seed=0, cycles=(0, 1)):
    """The model's average gain over the whole PAL x temperature range: per polymer the geometric-mean ratio of the last
    to the first cycle, with its 5-95 % band. Rows: polymer, ratio, lo, hi."""
    pal, temp = np.linspace(*BOUNDS[0], n_grid), np.linspace(*BOUNDS[1], n_grid)
    P, T = np.meshgrid(pal, temp, indexing="ij")
    rows = []
    for p in polymers:
        X = np.vstack([np.column_stack([P.ravel(), T.ravel(), np.full(P.size, c, float)]) for c in cycles])
        _, D = joint_draws(model, POLYMERS.index(p) + 1, X, n_samples, seed)
        d = (D[:, P.size:] - D[:, :P.size]).mean(axis=1)
        rows.append(dict(polymer=p, ratio=float(np.exp(d.mean())), lo=float(np.exp(np.percentile(d, 5))), hi=float(np.exp(np.percentile(d, 95)))))
    return pd.DataFrame(rows)


def partial_dependence_cycle(model, cycles=(0, 1), have=None, n_grid=25, n_samples=400, seed=0):
    """Partial dependence as :func:`partial_dependence` at each cycle, for the (polymer, cycle) pairs in ``have`` (default:
    all). Long rows: polymer, cycle, factor, x, mean, lo, hi."""
    pal, temp = np.linspace(*BOUNDS[0], n_grid), np.linspace(*BOUNDS[1], n_grid)
    P, T = np.meshgrid(pal, temp, indexing="ij")
    rows = []
    for lv, p in enumerate(POLYMERS, start=1):
        for c in cycles:
            if have is not None and (p, c) not in have:
                continue
            _, D = joint_draws(model, lv, np.column_stack([P.ravel(), T.ravel(), np.full(P.size, c, float)]), n_samples, seed)
            S = np.exp(D.T).reshape(n_grid, n_grid, n_samples)
            for factor, axis, xs in (("pal_pct", 1, pal), ("extr_temp_C", 0, temp)):
                pdp = S.mean(axis=axis)
                rows += [{"polymer": p, "cycle": c, "factor": factor, "x": x, "mean": m, "lo": lo, "hi": hi}
                         for x, m, lo, hi in zip(xs, pdp.mean(1), np.percentile(pdp, 5, 1), np.percentile(pdp, 95, 1))]
    return pd.DataFrame(rows)


def _fit_predict_cycle(task):
    import torch
    from models import MODELS
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    idx, key, seed, data, level, x, bounds = task
    torch.manual_seed(seed)
    m = MODELS[key].fit(data, needs_r=False, bounds=bounds)
    mu, var = m.predict(level, x)
    return idx, float(np.ravel(mu)[0]), float(np.ravel(var)[0])


def leave_one_run_out(kept, response, seed=0, workers=1, use_cycle=True, key="lvgp"):
    """Fit on the other runs, predict the left-out run (log scale). Partner run of the same formulation stays in the
    training data: the question is whether the model can predict a run it has not seen. ``use_cycle=False`` fits the model
    without the cycle input, for comparison; ``key`` picks the model ("lvgp" or "lvgp_mixed"). Rows per run: formulation, cycle, polymer, y, var_obs, mu, var_pred, err, z,
    and partner_err = the error of simply taking the same formulation's other cycle."""
    t = run_table(kept, response)
    bounds = BOUNDS_CYCLE if use_cycle else BOUNDS
    rows, tasks = [], []
    for i, r in t.iterrows():
        train = t.drop(index=i)
        data = to_data_cycle(train)
        if not use_cycle:
            data = {k: dict(v, X=v["X"][:, :2]) for k, v in data.items()}
        x = np.array([[r["pal"], r["temp"], r["cycle"]]])[:, :len(bounds)]
        other = train[train["formulation"] == r["formulation"]]["mean_log"]
        rows.append(dict(formulation=r["formulation"], cycle=r["cycle"], polymer=r["polymer"], y=r["mean_log"], var_obs=r["var_mean"],
                         partner_err=float(other.iloc[0] - r["mean_log"]) if len(other) else np.nan))
        tasks.append((len(rows) - 1, key, seed, data, int(r["level"]), x, bounds))
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_fit_predict_cycle, tasks))
    else:
        results = [_fit_predict_cycle(tk) for tk in tasks]
    for idx, mu, var in results:
        rows[idx]["mu"], rows[idx]["var_pred"] = mu, var
    out = pd.DataFrame(rows)
    out["err"] = out["mu"] - out["y"]
    out["z"] = out["err"] / np.sqrt(out["var_pred"] + out["var_obs"])
    out["response"] = response
    return out


def leave_one_formulation_out(kept, response, seed=0, workers=1, key="lvgp"):
    """Stricter folds than :func:`leave_one_run_out`: both cycles of one formulation are held out together (12 folds), so the
    partner run cannot help. Mirrors predicting a new formulation (Option 2). Same columns as leave_one_run_out, plus ``model``."""
    t = run_table(kept, response)
    rows, tasks = [], []
    for f, grp in t.groupby("formulation"):
        data = to_data_cycle(t[t["formulation"] != f])
        X = grp[["pal", "temp", "cycle"]].to_numpy(float)
        for _, r in grp.iterrows():
            rows.append(dict(formulation=f, cycle=r["cycle"], polymer=r["polymer"], y=r["mean_log"], var_obs=r["var_mean"]))
        tasks.append((len(rows) - len(grp), key, seed, data, int(grp["level"].iloc[0]), X, BOUNDS_CYCLE))
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_fit_predict_many, tasks))
    else:
        results = [_fit_predict_many(tk) for tk in tasks]
    for idx, mu, var in results:
        for j in range(len(mu)):
            rows[idx + j]["mu"], rows[idx + j]["var_pred"] = float(mu[j]), float(var[j])
    out = pd.DataFrame(rows)
    out["err"] = out["mu"] - out["y"]
    out["z"] = out["err"] / np.sqrt(out["var_pred"] + out["var_obs"])
    out["response"], out["model"] = response, key
    return out


def _fit_predict_many(task):
    import torch
    from models import MODELS
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")
    idx, key, seed, data, level, X, bounds = task
    torch.manual_seed(seed)
    m = MODELS[key].fit(data, needs_r=False, bounds=bounds)
    mu, var = m.predict(level, X)
    return idx, np.ravel(mu), np.ravel(var)


def nlpd(cv):
    """Negative log predictive density of each left-out run mean (log scale), under the model's predictive variance plus the
    run's replicate variance of the mean. The proper score for a probabilistic prediction: it penalises a wrong mean and a
    wrong (over- or under-confident) variance. Lower is better; 0.5 * ln(2 pi e sigma^2) for a perfectly calibrated model."""
    v = cv["var_pred"] + cv["var_obs"]
    return 0.5 * np.log(2 * np.pi * v) + 0.5 * cv["err"] ** 2 / v


def paired_scores(a, b, names=("LVGP", "lvgp_mixed")):
    """Compare two cross-validation tables fold by fold (same rows, same order). Per response: mean NLPD of each, the mean
    and standard error of the per-fold difference (b - a: positive favours ``a``), the median difference, the number of folds
    ``a`` wins, RMSE and +-2 SD coverage of each. The paired difference is the honest comparison: fold-to-fold scatter is
    large and shared, and one confident miss can dominate a mean."""
    rows = []
    for r in a["response"].unique():
        x, y = a[a["response"] == r].reset_index(drop=True), b[b["response"] == r].reset_index(drop=True)
        assert (x["formulation"].values == y["formulation"].values).all() and (x["cycle"].values == y["cycle"].values).all()
        dx, dy = nlpd(x), nlpd(y)
        d = dy - dx
        rows.append({"response": r, f"nlpd {names[0]}": dx.mean(), f"nlpd {names[1]}": dy.mean(), "diff": d.mean(), "diff se": d.std(ddof=1) / np.sqrt(len(d)),
                     "diff median": d.median(), f"folds {names[0]} wins": int((d > 0).sum()), "folds": len(d),
                     f"rmse {names[0]}": float(np.sqrt((x["err"] ** 2).mean())), f"rmse {names[1]}": float(np.sqrt((y["err"] ** 2).mean())),
                     f"in 2sd {names[0]}": float((x["z"].abs() <= 2).mean()), f"in 2sd {names[1]}": float((y["z"].abs() <= 2).mean())})
    return pd.DataFrame(rows).set_index("response")


def log_evidence(model):
    """Log marginal likelihood of a fitted torch model at its MAP hyperparameters (GPyTorch adds the hyperprior terms, so this
    is the log joint of data and hyperparameters). In-sample and prior-sensitive: a side check, not the verdict."""
    from models.lvgp_torch import _mll_value
    return _mll_value(model.model)


def warping_data(warp):
    """{level: dict(X, y_mean, y_var)} for mean corner warping per formulation, X = (PAL %, temperature), on the log of the
    mean (variance of the mean by the delta method). Corners of one sample are not independent, so the noise is optimistic."""
    g = warp.groupby(["formulation", "polymer_system", "pal_pct", "extr_temp_C"])["warping_cm"]
    t = g.agg(n="size", m="mean", v="var").reset_index()
    t["level"] = [POLYMERS.index(p) + 1 for p in t["polymer_system"]]
    data = {}
    for lv in range(1, len(POLYMERS) + 1):
        r = t[t["level"] == lv]
        data[lv] = dict(X=r[["pal_pct", "extr_temp_C"]].to_numpy(float), y_mean=np.log(r["m"].to_numpy(float)),
                        y_var=(r["v"] / r["n"] / r["m"] ** 2).to_numpy(float))
    return data


def warping_lofo(warp, seed=0):
    """Leave one formulation out on mean warping (log of the mean): does knowing PAL % and temperature predict a left-out
    formulation better than its polymer's other means? Rows: formulation, polymer, y, lvgp, polymer_mean, err_lvgp, err_polymer."""
    import torch
    from models import MODELS
    warnings.filterwarnings("ignore")
    t = warp.groupby(["formulation", "polymer_system", "pal_pct", "extr_temp_C"])["warping_cm"].agg(["size", "mean", "var"]).reset_index()
    rows = []
    for f in t["formulation"]:
        data = warping_data(warp[warp["formulation"] != f])
        r = t[t["formulation"] == f].iloc[0]
        lv = POLYMERS.index(r["polymer_system"]) + 1
        torch.manual_seed(seed)
        m = MODELS["lvgp"].fit(data, needs_r=False, bounds=BOUNDS)
        mu, _ = m.predict(lv, np.array([[r["pal_pct"], r["extr_temp_C"]]]))
        same = t[(t["polymer_system"] == r["polymer_system"]) & (t["formulation"] != f)]["mean"]
        rows.append(dict(formulation=f, polymer=r["polymer_system"], y=float(np.log(r["mean"])), lvgp=float(np.ravel(mu)[0]), polymer_mean=float(np.log(same).mean())))
    out = pd.DataFrame(rows)
    out["err_lvgp"], out["err_polymer"] = out["lvgp"] - out["y"], out["polymer_mean"] - out["y"]
    return out

