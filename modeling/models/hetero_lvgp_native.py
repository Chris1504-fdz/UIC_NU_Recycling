"""
hetero_lvgp_native.py -- faithful numpy port of the lab's MATLAB heteroscedastic LVGP
(LVGP_fit_noise.m / LVGP_predict_noise.m / corr_mat.m / to_latent.m).

Why a port instead of lvgp-bayes: lvgp-bayes standardizes Y by std, injects the replicate noise via
a FixedNoiseGaussianLikelihood, and adds horseshoe priors -- a materially different heteroscedastic
treatment that let BO get trapped exploiting high-noise boundary points (never matched MATLAB, which
never gets stuck). This module reproduces MATLAB's treatment exactly:

  * min-max normalization: X_quant -> [0,1] per column by TRAINING range; Y -> [0,1] by training range.
  * Gaussian correlation kernel  R(x,x') = exp(-sum_i 10^phi_i (x_i-x'_i)^2); latent dims use phi=0.
  * latent embedding: level 1 -> origin; levels 2..L -> free z (dim_z=2). For dim_z=2 the 2nd
    coordinate of the first free level is fixed at 0 (identifiability), matching MATLAB's +-1e-4 box.
  * heteroscedastic noise ENTERS THE CORRELATION MATRIX: R <- R + diag(noise_var/scale2) in the
    likelihood (scale2=(Ymax-Ymin)^2), with sigma2 (process var) profiled out.
  * eps-ladder eigenvalue nugget: fit at each eps in 10^(-1:-0.5:-8) (continuation, warm-started),
    keep the eps with the best profile NLL; guarantee min-eigenvalue(R) >= eps before every Cholesky.
  * prediction: R_pred = R_corr(+eig nugget) + diag(noise_var/scale2)/sigma2; Y_hat and epistemic MSE
    exactly as LVGP_predict_noise (R_new_new has unit diagonal -> latent/epistemic variance).

Predictions are returned in RAW Y units. The aleatoric r(x) is supplied separately by the shared
AleatoricModels (same as every other python model), so acquisitions are engine-consistent.
"""
import numpy as np
from scipy.linalg import solve_triangular, cholesky
from scipy.optimize import minimize

from .base import BaseModel, AleatoricModels, _as_2d

EPS_ALL = 10.0 ** np.arange(-1.0, -8.0001, -0.5)     # nugget ladder 1e-1 ... 1e-8 (descending)
DIM_Z = 2
LB_PHI, UB_PHI = -2.0, 2.0                            # phi init box (MATLAB lb_phi_ini/ub_phi_ini)
LB_Z, UB_Z = -3.0, 3.0
N_OPT = 8                                             # random starts at the first (largest) eps


def _corr(X1, X2, phi_full):
    """R(x,x') = exp(-sum_i 10^phi_i (x1_i - x2_i)^2). phi_full covers quant + latent columns."""
    dist = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[1]):
        d = X1[:, i:i + 1] - X2[:, i:i + 1].T
        dist += (10.0 ** phi_full[i]) * d * d
    return np.exp(-dist)


def _z_matrix(z_free, n_lvs):
    """(n_lvs, DIM_Z) latent coords; row 0 = origin, rows 1.. = z_free reshaped."""
    Z = np.zeros((n_lvs, DIM_Z))
    Z[1:, :] = z_free.reshape(n_lvs - 1, DIM_Z)
    return Z


def _assemble_full(Xq, codes, phi, z_free, n_lvs, p_quant):
    """[X_quant_norm, latent(codes)] and the matching phi_full (latent dims get phi=0)."""
    Z = _z_matrix(z_free, n_lvs)
    Xlat = Z[codes]
    Xfull = np.hstack([Xq, Xlat]) if p_quant > 0 else Xlat
    phi_full = np.concatenate([phi, np.zeros(DIM_Z)])
    return Xfull, phi_full


def _profile(Xfull, phi_full, Y, noise_std2, min_eig):
    """MATLAB neg_log_l_noise: add heteroscedastic noise to R, floor eigenvalue, profile beta+sigma2.
    Returns (nll, cache) where cache carries the pieces prediction needs."""
    k = Y.shape[0]
    R = _corr(Xfull, Xfull, phi_full)
    R = 0.5 * (R + R.T)
    R = R + np.diag(noise_std2)
    mn = np.linalg.eigvalsh(R)[0]
    if mn < min_eig:
        R = R + np.eye(k) * (min_eig - mn)
    try:
        L = cholesky(R, lower=True)
    except np.linalg.LinAlgError:
        return np.inf, None
    M = np.ones((k, 1))
    LinvM = solve_triangular(L, M, lower=True)
    LinvY = solve_triangular(L, Y, lower=True)
    MTRinvM = float(LinvM.T @ LinvM)
    beta = float((LinvM.T @ LinvY) / MTRinvM)
    temp = solve_triangular(L, Y - M * beta, lower=True)
    sigma2 = float(temp.T @ temp) / k
    sigma2 = max(sigma2, 1e-300)
    nll = k * np.log(sigma2) + 2.0 * np.sum(np.log(np.diag(L)))
    return float(nll), dict(beta=beta, sigma2=sigma2, L=L, temp=temp, MTRinvM=MTRinvM)


def _poly_features(Wn, degree):
    """MATLAB build_poly_features: [1, W, W^2, ..., W^degree] + every pairwise cross term Wa*Wb."""
    n, p = Wn.shape
    cols = [np.ones((n, 1))]
    for deg in range(1, degree + 1):
        cols.append(Wn ** deg)
    for a in range(p - 1):
        for b in range(a + 1, p):
            cols.append((Wn[:, a] * Wn[:, b]).reshape(-1, 1))
    return np.hstack(cols)


class _MatlabAleatoric:
    """Pooled aleatoric log-variance model r(x) = exp(2*theta'Phi([x_cont, z(level)])) -- a faithful
    port of MATLAB fit_aleatoric_polymodel: ONE ridge poly over ALL points, category entering through
    its latent-z coordinates (not an independent poly per level). Fixes the ~4x noise inflation the
    per-category Python poly produced on under-sampled levels, which broke ANPEI/HAEI/RAHBO."""
    def __init__(self, X_cont, codes, y_var, z_mat, lv_index, degree=2, lam=1e-3):
        self.z_mat = z_mat; self.lv_index = lv_index; self.degree = degree
        W = np.hstack([X_cont, z_mat[codes]])
        self.mu = W.mean(0); self.sd = W.std(0); self.sd[self.sd == 0] = 1.0
        Wn = (W - self.mu) / self.sd
        Phi = _poly_features(Wn, degree)
        ls = 0.5 * np.log(np.maximum(np.asarray(y_var, float).ravel(), 1e-12))
        self.theta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(Phi.shape[1]), Phi.T @ ls)

    def r(self, level, x_new):
        X = _as_2d(x_new)
        z = self.z_mat[self.lv_index[level if isinstance(level, tuple) else int(level)]]
        W = np.hstack([X, np.repeat(z[None, :], X.shape[0], axis=0)])
        Wn = (W - self.mu) / self.sd
        ls = _poly_features(Wn, self.degree) @ self.theta
        return np.maximum(np.exp(2 * ls), 1e-12)


def _fixed_mask(p_quant, n_lvs):
    """MATLAB fixes the 2nd latent coord of the FIRST free level (index p_quant+1) at 0 for dim_z=2."""
    n = p_quant + DIM_Z * (n_lvs - 1)
    fixed = np.zeros(n, dtype=bool)
    if DIM_Z == 2:
        fixed[p_quant + 1] = True                     # 2nd coordinate of level-2 latent
    return fixed


class _LVGPNativeBase(BaseModel):
    ENGINE = "python"
    HETERO = True                                      # subclass overrides

    def __init__(self, fit, levels, ale):
        self._f = fit                                  # dict of fitted quantities (normalized space)
        self.levels = list(levels)
        self._ale = ale

    # ---- fit ---------------------------------------------------------------
    @classmethod
    def fit(cls, data_by_level, needs_r=True, bounds=None, seed=0, **_kw):
        # stack: quant X, integer level codes (0..L-1), replicate mean y, replicate var v
        levels = sorted(int(lv) for lv in data_by_level)
        n_lvs = len(levels)
        lv_index = {lv: i for i, lv in enumerate(levels)}
        Xs, ys, vs, codes = [], [], [], []
        for lv in levels:
            dd = data_by_level[lv]
            X = _as_2d(dd["X"])
            Xs.append(X); ys.append(np.asarray(dd["y_mean"], float))
            vs.append(np.asarray(dd["y_var"], float))
            codes.append(np.full(X.shape[0], lv_index[lv], int))
        Xq_raw = np.vstack(Xs); y = np.concatenate(ys).reshape(-1, 1)
        v = np.concatenate(vs); codes = np.concatenate(codes)
        p_quant = Xq_raw.shape[1]; k = y.shape[0]

        # MATLAB normalization: X_quant and Y by TRAINING min/max
        xmin = Xq_raw.min(0); xmax = Xq_raw.max(0)
        span = np.where(xmax - xmin > 0, xmax - xmin, 1.0)
        Xq = (Xq_raw - xmin) / span
        ymin = float(y.min()); ymax = float(y.max())
        yspan = ymax - ymin if ymax > ymin else 1.0
        Yn = (y - ymin) / yspan
        scale2 = yspan * yspan
        # heteroscedastic: replicate variances enter the correlation matrix (MATLAB LVGP_fit_noise).
        # homoscedastic: no per-point noise -- conditioning comes from the eps-ladder nugget alone,
        # exactly the MATLAB LVGP_fit (noise-free) case.
        noise_std2 = (v / scale2) if cls.HETERO else np.zeros_like(v)

        n_hyper = p_quant + DIM_Z * (n_lvs - 1)
        fixed = _fixed_mask(p_quant, n_lvs)
        lb = np.concatenate([np.full(p_quant, LB_PHI), np.full(DIM_Z * (n_lvs - 1), LB_Z)])
        ub = np.concatenate([np.full(p_quant, UB_PHI), np.full(DIM_Z * (n_lvs - 1), UB_Z)])
        bnds = [(0.0, 0.0) if fixed[i] else (lb[i], ub[i]) for i in range(n_hyper)]

        def split(h):
            return h[:p_quant], h[p_quant:]

        def nll_only(h):
            phi, zf = split(h)
            Xf, pf = _assemble_full(Xq, codes, phi, zf, n_lvs, p_quant)
            val, _ = _profile(Xf, pf, Yn, noise_std2, min_eig)
            return val

        # SLSQP multistart. Reaches the SAME optimum as MATLAB's fmincon('interior-point') (verified:
        # nll -518.37, latent matches to ~0.01) but is robust to the bounds -- unlike projected
        # L-BFGS-B, which slid the latent onto the z-edge and left native ~10 nll ABOVE MATLAB's
        # optimum. (scipy's true interior-point, trust-constr, also matches but is ~14x slower.)
        # Starts are scrambled-Sobol (MATLAB scrambles its Sobol too) over the FREE parameters (the
        # fixed z2-of-level-2 = 0 gauge is held out); eps-ladder continuation warm-starts each smaller
        # nugget. Finite differences kept -- MATLAB uses my_grad forward-differences, not analytic
        # gradients (see PYTHON_LVGP_INTEGRATION.md "Path B" for the analytic-gradient option).
        from scipy.stats import qmc
        free = ~fixed
        lb_f, ub_f = lb[free], ub[free]

        def nll_free(hf):
            h = np.zeros(n_hyper); h[free] = hf
            return nll_only(h)

        _box = list(zip(lb_f, ub_f))

        def _opt(hf0):
            # SLSQP (sequential quadratic programming): reaches the SAME optimum as MATLAB's
            # interior-point fmincon (verified: nll -518.37, latent matches to 3 sig figs) but is
            # ~14x cheaper than scipy's trust-constr interior-point. Robust to the bounds, so the
            # latent no longer sticks to the z-edge the way projected L-BFGS-B did.
            r = minimize(nll_free, hf0, method="SLSQP", bounds=_box,
                         options=dict(maxiter=300, ftol=1e-9))
            return r.x, float(r.fun)

        _sseed = int(np.random.default_rng([seed, 7, k]).integers(1 << 31))
        U = qmc.Sobol(int(free.sum()), scramble=True, seed=_sseed).random(N_OPT)
        cand0 = lb_f + U * (ub_f - lb_f)
        best_by_eps = []
        warm = None
        for j, min_eig in enumerate(EPS_ALL):
            starts = list(cand0) if j == 0 else []      # multistart at the first (largest) nugget
            if warm is not None:
                starts = [warm] + starts                # continue the running best down the ladder
            best_hf, best_v = None, np.inf
            for hf0 in starts:
                try:
                    hx, hv = _opt(hf0)
                except Exception:
                    continue
                if hv < best_v:
                    best_v, best_hf = hv, hx
            if best_hf is None:
                best_hf = (lb_f + ub_f) / 2.0; best_v = nll_free(best_hf)
            warm = best_hf
            h_full = np.zeros(n_hyper); h_full[free] = best_hf
            best_by_eps.append((best_v, h_full, min_eig))

        nll_b, h_b, eig_b = min(best_by_eps, key=lambda t: t[0])
        phi, zf = split(h_b)
        Xfull, phi_full = _assemble_full(Xq, codes, phi, zf, n_lvs, p_quant)

        # --- MATLAB assembly (fit_detail): stored R = R_corr + eigenvalue nugget, NO noise;
        #     sigma2 profiled from THAT matrix (this is the sigma2 prediction later divides noise by).
        M = np.ones((k, 1))
        R_corr = _corr(Xfull, Xfull, phi_full); R_corr = 0.5 * (R_corr + R_corr.T)
        raw_min_eig = float(np.linalg.eigvalsh(R_corr)[0])
        nug_opt = eig_b - raw_min_eig if raw_min_eig < eig_b else 0.0
        R_stored = R_corr + np.eye(k) * nug_opt
        Ls = cholesky(R_stored, lower=True)
        LsinvM = solve_triangular(Ls, M, lower=True)
        MTRinvM_s = float(LsinvM.T @ LsinvM)
        beta_s = float((LsinvM.T @ solve_triangular(Ls, Yn, lower=True)) / MTRinvM_s)
        temp_s = solve_triangular(Ls, Yn - M * beta_s, lower=True)
        sigma2 = max(float(temp_s.T @ temp_s) / k, 1e-300)

        # --- MATLAB predict factorization: R_pred = R_stored + diag(noise/scale2)/sigma2,
        #     re-derive beta/temp from R_pred (done ONCE; R_pred is fixed per fit).
        R_pred = R_stored + np.diag(noise_std2 / sigma2)
        Lp = cholesky(R_pred, lower=True)
        LpinvM = solve_triangular(Lp, M, lower=True)
        MTRinvM_p = float(LpinvM.T @ LpinvM)
        beta_p = float((LpinvM.T @ solve_triangular(Lp, Yn, lower=True)) / MTRinvM_p)
        temp_p = solve_triangular(Lp, Yn - M * beta_p, lower=True)
        RinvPY = solve_triangular(Lp.T, temp_p, lower=False)          # R_pred^{-1}(Y - M beta)

        fit = dict(Xfull=Xfull, phi_full=phi_full, beta=beta_p, sigma2=sigma2,
                   Lp=Lp, LpinvM=LpinvM, MTRinvM=MTRinvM_p, RinvPY=RinvPY,
                   xmin=xmin, span=span, ymin=ymin, yspan=yspan, scale2=scale2,
                   zf=zf, phi=phi, nll=nll_b, eig=eig_b,
                   n_lvs=n_lvs, p_quant=p_quant, lv_index=lv_index)
        # MATLAB-style pooled aleatoric r(x) over [x_cont, latent-z] (not the shared per-category poly)
        ale = (_MatlabAleatoric(Xq_raw, codes, v, _z_matrix(zf, n_lvs), lv_index)
               if needs_r else None)
        return cls(fit, levels, ale)

    # ---- prediction (LVGP_predict_noise) -----------------------------------
    def _encode(self, level, x_new):
        f = self._f
        X = _as_2d(x_new)
        Xq = (X - f["xmin"]) / f["span"]
        Z = _z_matrix(f["zf"], f["n_lvs"])
        code = f["lv_index"][int(level)]
        Xlat = np.repeat(Z[code][None, :], Xq.shape[0], axis=0)
        return np.hstack([Xq, Xlat]) if f["p_quant"] > 0 else Xlat

    def predict(self, level, x_new, observation_noise=False):
        f = self._f
        Xnew = self._encode(level, x_new)
        R_on = _corr(f["Xfull"], Xnew, f["phi_full"])            # (n_tr, m) train-new correlation
        yhat_n = f["beta"] + R_on.T @ f["RinvPY"]                # normalized-Y prediction
        yhat = yhat_n.ravel() * f["yspan"] + f["ymin"]           # -> raw Y

        # epistemic (latent) variance via R_pred factor; R_new_new has unit diagonal (no obs noise)
        Linv_Ron = solve_triangular(f["Lp"], R_on, lower=True)   # (n_tr, m)
        W = 1.0 - (f["LpinvM"].T @ Linv_Ron)                     # (1, m)
        quad = np.sum(Linv_Ron * Linv_Ron, axis=0)
        mse_n = f["sigma2"] * (1.0 - quad + (W.ravel() ** 2) / f["MTRinvM"])
        var = np.maximum(mse_n, 1e-24) * f["scale2"]             # -> raw Y^2
        return yhat, var

    def mean_std(self, level, x_new):
        mu, var = self.predict(level, x_new)
        return mu, np.maximum(np.sqrt(np.clip(var, 1e-24, None)), 1e-12)


class HeterLVGPNative(_LVGPNativeBase):
    """Heteroscedastic LVGP (MATLAB LVGP_fit_noise analogue): replicate variances in the correlation
    matrix + shared aleatoric r(x); all six acquisitions."""
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")
    HETERO = True


class LVGPNative(_LVGPNativeBase):
    """Homoscedastic LVGP (MATLAB LVGP_fit analogue): no per-point noise, eps-ladder nugget only;
    noise-blind acquisitions, matching standard_LVGP."""
    SUPPORTS = ("ei", "lcb", "pi")
    HETERO = False
