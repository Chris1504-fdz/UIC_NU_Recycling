"""
lvgp_blocks.py -- building blocks for the VERSIONED LVGP kernel investigation (2026-09-30).

VERSIONING CONVENTION (user decision 2026-09-30): every kernel idea is a new file
`models/lvgp_v<N>.py` defining exactly one model; a version is NEVER edited once benchmarked -- a
change is a new version. This blocks file is APPEND-ONLY: blocks may be added, existing ones must not
change (versions import them). The reference models stay untouched: `lvgp_torch.py` (original LVGP)
and `lvgp_torch_mixed.py` (mixed kernel; frozen, see snapshots/2026-09-30_mixed_kernel/).

Motivation (notebook §12, and the separable-term analysis of the test functions): the original LVGP
kernel  k = s^2 k_x(x, x') exp(-||z_l - z_l'||^2)  has three limitations --
  (1) one x-structure for everything (level-common and level-specific parts share lengthscales),
  (2) every level has the SAME prior variance (the level correlation has a unit diagonal),
  (3) level offsets can only come from latent distance (single constant mean: the "offset tax").
Blocks here address (3) LevelMean, (2) LevelAmplitudeKernel, (1) second_lvgp_term.

All blocks follow the lvgp-bayes conventions so the multistart (restarts sampled from the priors)
and MAP fit work unchanged: every trainable parameter carries a prior; outputscales exp-positive
with NormalPrior(0, 1) on the raw value; x-lengthscales exp-positive with
MollifiedUniform(ln 0.1, ln 10) on the raw log-lengthscale (== LVGPR's own quant kernel).
"""
import math

import gpytorch
import torch
from gpytorch import kernels as gk
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import NormalPrior

from lvgp_bayes.models import LVGPR
from lvgp_bayes.priors.mollified_uniform import MollifiedUniformPrior


# ------------------------------------------------------------------------------------------ model
class CodedLVGPR(LVGPR):
    """LVGPR whose forward passes the raw 0-based level codes on to the mean and kernel:
        u = [ z (lat_dims) | x (q_dims) | codes (k) ]
    LVGPR's own kernels have active_dims on the first lat_dims + q_dims columns and the default
    ConstantMean ignores its input, so with no other change this is EXACTLY LVGPR. Level-indexed
    blocks (LevelMean, LevelAmplitudeKernel) read the trailing code columns."""

    def forward(self, x):
        codes = x[..., self.qual_index].long()                       # (..., k)
        emb = torch.cat([e(codes[..., i]) for i, e in enumerate(self.lv_mapping_layers)], -1)
        parts = [emb] + ([x[..., self.quant_index]] if len(self.quant_index) > 0 else [])
        u = torch.cat(parts + [codes.to(x.dtype)], -1)
        return MultivariateNormal(self.mean_module(u), self.covar_module(u))

    # layout helpers for the blocks
    @property
    def lat_dims(self):
        return int(sum(int(l.lv_dim) for l in self.lv_mapping_layers))

    @property
    def code_start(self):
        return self.lat_dims + len(self.quant_index)

    @property
    def n_levels_per_var(self):
        return [int(l.num_levels) for l in self.lv_mapping_layers]

    def diagnostics(self):
        """Fitted quantities worth recording per cell (overridden/extended by versions)."""
        return {}


# ------------------------------------------------------------------------------------------ means
class LevelMean(gpytorch.means.Mean):
    """m(l) = c + sum_j beta_j[l_j]: a constant plus one fixed effect per level of each categorical
    variable (standardized-y units; all ~ Normal(0, 1)). One categorical -> a per-level mean.
    Several -> additive main effects. Offsets then live in the mean instead of in latent distance."""

    def __init__(self, code_start, n_levels_per_var):
        super().__init__()
        self.code_start = int(code_start)
        self.n_vars = len(n_levels_per_var)
        self.register_parameter("constant", torch.nn.Parameter(torch.zeros(1)))
        self.register_prior("constant_prior", NormalPrior(0.0, 1.0), "constant")
        for j, L in enumerate(n_levels_per_var):
            self.register_parameter(f"level_effect_{j}", torch.nn.Parameter(torch.zeros(int(L))))
            self.register_prior(f"level_effect_{j}_prior", NormalPrior(0.0, 1.0), f"level_effect_{j}")

    def forward(self, u):
        out = self.constant.expand(u.shape[:-1])
        for j in range(self.n_vars):
            out = out + getattr(self, f"level_effect_{j}")[u[..., self.code_start + j].long()]
        return out

    def effects(self):
        return [getattr(self, f"level_effect_{j}").detach() for j in range(self.n_vars)]


# ---------------------------------------------------------------------------------------- kernels
class LevelAmplitudeKernel(gk.Kernel):
    """a(l) a(l') * base(u, u'),  a(l) = exp( sum_j rho_j[l_j] ),  rho ~ Normal(0, 1).
    Lets each level have its own prior variance: the level covariance becomes D R D (D = diag(a)),
    still PSD, instead of the unit-diagonal correlation R of the original LVGP."""

    def __init__(self, base_kernel, code_start, n_levels_per_var, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.code_start = int(code_start)
        self.n_vars = len(n_levels_per_var)
        for j, L in enumerate(n_levels_per_var):
            self.register_parameter(f"log_amp_{j}", torch.nn.Parameter(torch.zeros(int(L))))
            self.register_prior(f"log_amp_{j}_prior", NormalPrior(0.0, 1.0), f"log_amp_{j}")

    def amplitude(self, u):
        s = 0.0
        for j in range(self.n_vars):
            s = s + getattr(self, f"log_amp_{j}")[u[..., self.code_start + j].long()]
        return torch.exp(s)

    def forward(self, x1, x2, diag=False, **params):
        K = self.base_kernel(x1, x2, diag=diag, **params)
        a1, a2 = self.amplitude(x1), self.amplitude(x2)
        if diag:
            return K * a1 * a2
        return K.to_dense() * a1.unsqueeze(-1) * a2.unsqueeze(-2)


def scaled(base):
    """ScaleKernel with the lvgp-bayes outputscale convention (exp-positive, NormalPrior(0,1) raw)."""
    s = gk.ScaleKernel(base, outputscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
    s.register_prior("raw_outputscale_prior", NormalPrior(0.0, 1.0), "raw_outputscale")
    return s


def lvgp_x_rbf(q_dims, active_dims):
    """LVGPR's own continuous kernel: ARD RBF, exp-positive lengthscale,
    MollifiedUniform(ln 0.1, ln 10) on the raw log-lengthscale."""
    k = gk.RBFKernel(ard_num_dims=q_dims, active_dims=active_dims,
                     lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
    k.register_prior("raw_lengthscale_prior", MollifiedUniformPrior(math.log(0.1), math.log(10)),
                     "raw_lengthscale")
    return k


def latent_rbf_learnable(lat_dims):
    """RBF over the latent columns with its OWN learnable decay length (prior as for every learnable
    lengthscale in lvgp-bayes). Long length -> levels share this term almost fully (trend-like);
    short -> nearly independent per level."""
    k = gk.RBFKernel(active_dims=torch.arange(lat_dims),
                     lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
    k.register_prior("raw_lengthscale_prior", MollifiedUniformPrior(math.log(0.1), math.log(10)),
                     "raw_lengthscale")
    return k


def second_lvgp_term(lat_dims, q_dims):
    """A second LVGP product term  s2^2 k_x2(x, x') exp(-||dz||^2 / (2 l2^2))  sharing the latent
    map with the first term but with its OWN x-lengthscales and its OWN cross-level decay l2.
    (The mixed kernel's trend channel is the special case l2 -> infinity; its offset channel is the
    special case of very long x-lengthscales.)"""
    return scaled(lvgp_x_rbf(q_dims, lat_dims + torch.arange(q_dims)) * latent_rbf_learnable(lat_dims))


# =========================== blocks appended 2026-09-30 (round 2: v6-v10) ===========================
from lvgp_bayes.kernels import Matern52Kernel
from ..lvgp_torch_mixed import LVGPRMixed, _hvarfner_rbf      # frozen file: read-only reuse


def latent_only_term(lat_dims):
    """s3^2 exp(-||dz||^2 / (2 l3^2)): depends on the level ONLY (constant in x) -> per-level
    OFFSETS as random effects, correlated through the latent map (similar levels share offsets).
    Same construction as the mixed kernel's offset channel, as a separate additive term."""
    return scaled(latent_rbf_learnable(lat_dims))


class LatentAmplitudeKernel(gk.Kernel):
    """a(l) a(l') * base(u, u'),  log a(l) = w^T z_l  (z = the embedded latent columns of u).
    A per-level amplitude that varies SMOOTHLY over the latent map: lat_dims parameters however
    many levels there are (vs one per level in LevelAmplitudeKernel). w ~ Normal(0, 1)."""

    def __init__(self, base_kernel, lat_dims, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.lat_dims = int(lat_dims)
        self.register_parameter("amp_w", torch.nn.Parameter(torch.zeros(self.lat_dims)))
        self.register_prior("amp_w_prior", NormalPrior(0.0, 1.0), "amp_w")

    def amplitude(self, u):
        return torch.exp(u[..., :self.lat_dims] @ self.amp_w)

    def forward(self, x1, x2, diag=False, **params):
        K = self.base_kernel(x1, x2, diag=diag, **params)
        a1, a2 = self.amplitude(x1), self.amplitude(x2)
        if diag:
            return K * a1 * a2
        return K.to_dense() * a1.unsqueeze(-1) * a2.unsqueeze(-2)


def latent_rbf_fixed(lat_dims):
    """RBF over the latent columns with lengthscale FIXED at 1 (== LVGPR's qual_kernel, the gauge)."""
    k = gk.RBFKernel(active_dims=torch.arange(lat_dims))
    k.initialize(lengthscale=1.0)
    k.raw_lengthscale.requires_grad_(False)
    return k


def hvarfner_x_rbf(q_dims, active_dims):
    """botorch's dimension-scaled x-kernel exactly as the mixed kernel uses it (LogNormal(sqrt2 +
    0.5 ln d, sqrt3) on the lengthscale, 0.025 floor) -- imported from the frozen lvgp_torch_mixed."""
    return _hvarfner_rbf(q_dims, active_dims)


# reset_parameters with the below-floor clamp the Hvarfner prior needs (reused from the frozen file)
hvarfner_safe_reset = LVGPRMixed.reset_parameters


def lvgp_x_matern52(q_dims, active_dims):
    """Matern-5/2 x-kernel with LVGPR's conventions (== LVGPR(quant_correlation_class=
    'Matern52Kernel')): exp-positive lengthscale, MollifiedUniform(ln 0.1, ln 10) raw prior."""
    k = Matern52Kernel(ard_num_dims=q_dims, active_dims=active_dims,
                       lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log))
    k.register_prior("raw_lengthscale_prior", MollifiedUniformPrior(math.log(0.1), math.log(10)),
                     "raw_lengthscale")
    return k


def lvgp_term(x_kernel, latent_kernel):
    """Generic LVGP product term  s^2 * x_kernel * latent_kernel  (lvgp outputscale convention)."""
    return scaled(x_kernel * latent_kernel)


# ========================= blocks appended 2026-10-08 (round 3: v11-v14, fitting) =========================
def log_prior_total(model):
    """Sum of all registered log priors (what lvgp-bayes adds to the marginal likelihood for MAP)."""
    return sum(pr.log_prob(cl(mod)).sum() for _, mod, pr, cl, _ in model.named_priors())


def loo_lpd_torch(model):
    """Closed-form leave-one-out log predictive density of the training targets (Rasmussen &
    Williams 2006, sec. 5.4.2), DIFFERENTIABLE in the hyperparameters. Fixed observation noise
    included (predicts the noisy target y_i from the other n-1). Standardized-y units.
    Call in train mode."""
    X, y = model.train_inputs[0], model.train_targets
    marg = model.likelihood(model(X))
    L = torch.linalg.cholesky(marg.covariance_matrix)
    Kinv = torch.cholesky_inverse(L)
    alpha = Kinv @ (y - marg.mean)
    dinv = torch.diagonal(Kinv)
    r, v = alpha / dinv, 1.0 / dinv
    return (-0.5 * torch.log(2 * math.pi * v) - 0.5 * r ** 2 / v).sum()


def gp_criteria(model):
    """LML (fixed noise included), log prior, MAP objective, LOO log predictive density -- evaluated
    without gradients, model left in its previous train/eval mode."""
    was = model.training
    model.train()
    try:
        with torch.no_grad():
            marg = model.likelihood(model(model.train_inputs[0]))
            lml = float(marg.log_prob(model.train_targets))
            lp = float(log_prior_total(model))
            loo = float(loo_lpd_torch(model))
    finally:
        model.train(was)
    return {"lml": lml, "logprior": lp, "map": lml + lp, "loo_lpd": loo}


def mixed_scale_shares(model):
    """sigma_s^2, sigma_z^2, sigma_p^2 of an LVGPRMixed-structured kernel and the variance shares of
    its three channels (trend, offsets, level-specific)."""
    cov = model.covar_module
    s_s = float(cov.kernels[0].outputscale)
    s_z = float(cov.kernels[0].base_kernel.kernels[1].outputscale)
    s_p = float(cov.kernels[1].outputscale)
    tot = s_s + s_s * s_z + s_p
    return {"sigma_s2": s_s, "sigma_z2": s_z, "sigma_p2": s_p,
            "share_trend": s_s / tot, "share_offset": s_s * s_z / tot, "share_level": s_p / tot}


def mixed_scale_kernels(model):
    """The three ScaleKernels of an LVGPRMixed covariance: (sum part = sigma_s^2,
    inner offset scale = sigma_z^2, product part = sigma_p^2)."""
    cov = model.covar_module
    return cov.kernels[0], cov.kernels[0].base_kernel.kernels[1], cov.kernels[1]


def drop_prior(module, name):
    """Remove a registered gpytorch prior (from the objective AND from restart sampling)."""
    module._priors.pop(name, None)
    module._modules.pop(name, None)


# ========================= blocks appended 2026-10-15 (round 4: v15-v16, selection) =========================
def n_free_hyperparameters(model):
    """Free hyperparameters for BIC/AIC: trainable parameter count minus, per latent map, the gauge
    (translation + rotation for a 2-D map) and the redundant latent precision (structure-search convention)."""
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for layer in model.lv_mapping_layers:
        k -= (3 if int(layer.lv_dim) == 2 else 0) + 1
    return int(k)


def bic_criteria(model):
    """gp_criteria + k, n, BIC = -2 LML + k ln n, AIC (LML at the fitted hyperparameters, fixed noise included)."""
    c = gp_criteria(model)
    k, n = n_free_hyperparameters(model), int(model.train_targets.numel())
    return {**c, "k": k, "n": n, "bic": -2.0 * c["lml"] + k * math.log(n), "aic": -2.0 * c["lml"] + 2.0 * k}


# ========================= blocks appended 2026-10-26 (v20: Matern-5/2 in the mixed kernel) =========================
def hvarfner_x_matern52(q_dims, active_dims):
    """BoTorch's dimension-scaled x-kernel with its Matern option (get_covar_module_with_dim_scaled_prior(...,
    use_rbf_kernel=False)): Matern-5/2, LogNormal(sqrt2 + 0.5 ln d, sqrt3) prior on the lengthscale, 0.025 floor --
    the same prior, floor and parameterization (value = 0.025 + exp(raw)) as the frozen _hvarfner_rbf, only the
    kernel family differs."""
    import math as _m
    from gpytorch import kernels as _gk
    from gpytorch.constraints import GreaterThan as _GT
    from gpytorch.priors import LogNormalPrior as _LN
    prior = _LN(loc=_m.sqrt(2.0) + _m.log(q_dims) * 0.5, scale=_m.sqrt(3.0))
    return _gk.MaternKernel(nu=2.5, ard_num_dims=q_dims, active_dims=active_dims, lengthscale_prior=prior,
                            lengthscale_constraint=_GT(2.5e-2, transform=torch.exp, inv_transform=torch.log,
                                                       initial_value=prior.mode))


# ========================= blocks appended 2026-10-27 (v21: Tao et al. 2021 LV-mix level correlation) =========================
class LVMixMapping(gpytorch.Module):
    """LV-mix latent map of one categorical variable (Tao, Apley, Plumlee & Chen 2021, IJNME 122:4007, Sec. 4, d = 2):
    level t -> (z1(t), z2(t)), z1 a Cartesian coordinate in R and z2 an angle in [0, pi]. Indeterminacy is removed as in
    the paper (the d = 1 rules of LV-Car and LV-sph combined): level 1 is fixed at (0, 0); the other levels are free,
    z2 = pi * sigmoid(raw). 2L - 2 free coordinates (LV-Car with d = 2: 2L - 3). Priors (needed by lvgp-bayes's restart
    sampler; the paper fits by maximum likelihood): z1 ~ Normal(0, 1), as the LVGP's latents effectively are, and
    raw z2 ~ Normal(0, 1.5), which spreads the angles over (0, pi)."""

    def __init__(self, num_levels):
        super().__init__()
        from gpytorch.priors import NormalPrior as _NP
        self.register_buffer("num_levels", torch.tensor(int(num_levels)))
        self.register_buffer("lv_dim", torch.tensor(2))
        self.register_parameter("raw_z1", torch.nn.Parameter(torch.randn(int(num_levels) - 1) * .5))
        self.register_parameter("raw_z2", torch.nn.Parameter(torch.randn(int(num_levels) - 1) * .5))
        self.register_prior("raw_z1_prior", _NP(0.0, 1.0), "raw_z1")
        self.register_prior("raw_z2_prior", _NP(0.0, 1.5), "raw_z2")

    @property
    def latents(self):
        zero = torch.zeros(1, dtype=self.raw_z1.dtype, device=self.raw_z1.device)
        z1 = torch.cat([zero, self.raw_z1])
        z2 = torch.cat([zero, math.pi * torch.sigmoid(self.raw_z2)])
        return torch.stack([z1, z2], -1)                                   # (L, 2)

    def forward(self, x):
        return torch.nn.functional.embedding(x, self.latents)


class LVMixKernel(gpytorch.kernels.Kernel):
    """Level correlation of LV-mix, over the latent columns [z1_1, z2_1, z1_2, z2_2, ...] (one pair per categorical
    variable): prod_j exp(-(z1_j - z1_j')^2 / 2) cos(z2_j - z2_j'). The 1/2 follows gpytorch's RBF convention used by the
    LVGP's own level kernel (the paper writes exp(-(.)^2); the free z1 absorbs the factor). PSD: a Gaussian kernel times
    cos(a - b) = cos a cos b + sin a sin b. Can be negative, unlike the LVGP's exp(-||dz||^2 / 2)."""
    has_lengthscale = False

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        z1a, z1b = x1[..., 0::2], x2[..., 0::2]
        z2a, z2b = x1[..., 1::2], x2[..., 1::2]
        if diag:
            return torch.exp(-0.5 * ((z1a - z1b) ** 2).sum(-1)) * torch.cos(z2a - z2b).prod(-1)
        d1 = ((z1a.unsqueeze(-2) - z1b.unsqueeze(-3)) ** 2).sum(-1)
        c2 = torch.cos(z2a.unsqueeze(-2) - z2b.unsqueeze(-3)).prod(-1)
        return torch.exp(-0.5 * d1) * c2
