"""
base.py -- the model interface used by the model-agnostic BO loop, plus the shared aleatoric
r(x) (predicted aleatoric VARIANCE) used by haei/anpei/rahbo.

A Python model exposes:
    ENGINE   = "python"
    SUPPORTS = tuple of acquisition names it can run (noise-blind-only models list ei/lcb/pi)
    classmethod fit(data_by_level, needs_r=True) -> instance
    .levels                      : sorted 1-based levels
    .mean_std(level, x)          -> (mu, epistemic_std)   # the (mu, s) the acquisitions use
    .r(level, x)                 -> aleatoric variance     # only if SUPPORTS the hetero acqs

The aleatoric model is the per-category degree-2 ridge log-variance poly ported from
study_v2_gp/utils/aleatoric.py (same target/degree/ridge as Heter_BO_GF's poly).
"""
import numpy as np

POLY_DEGREE = 2
POLY_LAMBDA = 1e-3


def _phi(Wn, degree):
    """Polynomial features over standardized columns: [1, W, W^2, ..., W^degree] -- per-COLUMN powers,
    NO cross terms, exactly mirroring the MATLAB build_poly_features (acquisition_func.m /
    bayesian_optimizer.m). Wn is (n, d); at d=1 this reduces to [1, w, w^2] bit-exactly."""
    Wn = np.asarray(Wn, float)
    if Wn.ndim == 1:
        Wn = Wn.reshape(-1, 1)
    cols = [np.ones((Wn.shape[0], 1))]
    for deg in range(1, degree + 1):
        cols.append(Wn ** deg)
    return np.hstack(cols)


def _as_2d(X):
    X = np.asarray(X, float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def level_codes(level):
    """0-based categorical codes of a level key: an int (one categorical variable, 1-based) or a
    tuple of 1-based ints (several categorical variables, e.g. (l1, l2))."""
    if isinstance(level, tuple):
        return [int(l) - 1 for l in level]
    return [int(level) - 1]


def norm_level(level):
    """Canonical dict key: tuple stays a tuple of ints, anything else becomes an int."""
    return tuple(int(l) for l in level) if isinstance(level, tuple) else int(level)


def levels_per_var(level_keys):
    """Number of levels of each categorical variable, from the (complete) set of level keys."""
    codes = np.array([level_codes(k) for k in level_keys])
    return [int(c) + 1 for c in codes.max(axis=0)]


class CategoryAleatoric:
    """Degree-2 ridge log-variance model r(x) for ONE category, over the d continuous dims."""

    def __init__(self, X, y_var, degree=POLY_DEGREE, lam=POLY_LAMBDA):
        X = _as_2d(X)
        self.degree = degree
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
        self.sd = np.where(sd > 0, sd, 1.0)
        Wn = (X - self.mu) / self.sd
        Phi = _phi(Wn, degree)
        log_sigma = 0.5 * np.log(np.maximum(np.asarray(y_var, float).ravel(), 1e-12))
        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        self.theta = np.linalg.solve(A, Phi.T @ log_sigma)

    def predict(self, x_new):
        Wn = (_as_2d(x_new) - self.mu) / self.sd
        log_sigma = _phi(Wn, self.degree) @ self.theta
        return np.maximum(np.exp(2 * log_sigma), 1e-12)


class AleatoricModels:
    """Container of per-category aleatoric models."""

    def __init__(self, models_by_level):
        self.models = dict(models_by_level)

    @classmethod
    def fit(cls, data_by_level, degree=POLY_DEGREE, lam=POLY_LAMBDA):
        return cls({lv: CategoryAleatoric(d["X"], d["y_var"], degree, lam)
                    for lv, d in data_by_level.items()})

    def r(self, level, x_new):
        return self.models[level].predict(x_new)


class BaseModel:
    """Mix-in providing r(x) from a fitted AleatoricModels stored on self._ale (or None)."""
    ENGINE = "python"
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")

    def r(self, level, x_new):
        if getattr(self, "_ale", None) is None:
            raise RuntimeError(f"{type(self).__name__} was fit without the aleatoric model (needs_r=False)")
        return self._ale.r(level, x_new)
