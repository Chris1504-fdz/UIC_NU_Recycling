"""
lvgp_torch_mixed.py -- LVGP with botorch's MixedSingleTaskGP sum-plus-product kernel algebra,
the latent-variable kernel taking the categorical slot (2026-07-31, user request).

RIGOROUS ADAPTATION of botorch v0.14.0 `MixedSingleTaskGP.__init__`
(botorch/models/gp_regression_mixed.py, L109-152), which builds

    sum_kernel  = ScaleKernel( RBF_1(ord dims)  +  ScaleKernel( Cat_1(cat dims) ) )   # L124-138
    prod_kernel = ScaleKernel( RBF_2(ord dims)  *   Cat_2(cat dims) )                 # L139-151
    covar       = sum_kernel + prod_kernel                                            # L152
    # NO global ScaleKernel; ConstantMean; Standardize outcome; train_Yvar fixed noise

with, per the class docstring L41-43, INDEPENDENT kernel instances (separate lengthscales)
for the sum and product terms. The continuous factory (get_covar_module_with_dim_scaled_prior,
botorch/models/utils/gpytorch_modules.py L100-133) is an ARD RBFKernel with the Hvarfner-2024
dim-scaled lengthscale prior LogNormal(loc=sqrt(2)+0.5*ln d, scale=sqrt(3)) and constraint
GreaterThan(0.025, transform=None, initial_value=prior.mode). CategoricalKernel is
exp(-Hamming/ell) with per-dim ell >= 1e-6 and no prior.

MAPPING onto the LVGP (embedded input u = [z_latent (lv_dim cols), x_quant (q cols)],
exactly LVGPR.forward's concatenation order):

  botorch construct                  -> this adaptation
  ---------------------------------------------------------------------------------------
  RBF_1 / RBF_2 over ord dims        -> two INDEPENDENT ARD RBFKernels over the quant dims,
                                        Hvarfner prior + constraint copied VERBATIM
                                        (inputs are unit-cube scaled by the wrapper, so the
                                        prior's scaling assumptions hold, as in botorch).
  Cat_1 (sum slot, own lengthscale)  -> RBFKernel over the latent dims with its OWN learnable
                                        scalar lengthscale (the analogue of Cat_1's independent
                                        ell), exp-Positive constraint.
  Cat_2 (product slot)               -> RBFKernel over the latent dims, lengthscale FIXED at
                                        1.0 -- the LVGP gauge (LVGPR's own qual_kernel,
                                        lvgp.py L187-191): the latent scale lives in z itself,
                                        so the product part identifies z exactly as in the
                                        original LVGP.
  inner ScaleKernel(Cat_1)  (L130)   -> inner ScaleKernel on the sum-part latent kernel.
  outer ScaleKernels + sum  (L152)   -> covar_module REPLACED wholesale after GPR's __init__
                                        (GPR wraps a single global ScaleKernel, botorch does
                                        not -- we follow botorch and discard the global wrap).

DOCUMENTED DEVIATIONS from botorch (all forced by lvgp-bayes's MAP machinery: multistart
restarts sample initial values FROM THE PRIORS -- GPR.reset_parameters, gpregression.py
L143-149 -- so every trainable parameter must carry a prior; botorch leaves its ScaleKernel
outputscales and categorical lengthscales prior-less):
  * raw_outputscale of all three ScaleKernels: NormalPrior(0,1) -- the same weakly-informative
    prior GPR itself puts on its outputscale (gpregression.py L99-101).
  * the sum-part latent lengthscale (raw): MollifiedUniformPrior(ln 0.1, ln 10) -- the same
    prior lvgp-bayes puts on every learnable lengthscale (lvgp.py L208-210).
Everything else is inherited unchanged from LVGPR/GPR: LVMapping latents + their priors,
ConstantMean with NormalPrior(0,1), internal y-standardization (== botorch Standardize),
noise handled by the wrapper (FixedNoiseGaussianLikelihood carrying replicate variances,
== botorch train_Yvar).

NOTE: LVGPR.to_pyro_random_module assumes the covar tree is ScaleKernel(Product(...)) and is
only used by the (optional) HMC path -- not supported for this class.
"""
import math

import torch
from gpytorch import kernels as gk
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors import LogNormalPrior, NormalPrior

from lvgp_bayes.models import LVGPR
from lvgp_bayes.priors.mollified_uniform import MollifiedUniformPrior

from .lvgp_torch import _LVGPTorchBase

_SQRT2, _SQRT3 = math.sqrt(2.0), math.sqrt(3.0)


def _hvarfner_rbf(q_dims, active_dims):
    """botorch get_covar_module_with_dim_scaled_prior (RBF branch), with ONE reparameterization:
    botorch uses GreaterThan(0.025, transform=None) -- raw == lengthscale -- because ITS fitter
    enforces constraint bounds inside the optimizer. lvgp-bayes's fit_model_scipy optimizes raw
    parameters UNBOUNDED (mll_scipy.py: bounds=None), so a raw==value parameterization walks
    negative and the LogNormal prior's support check raises (verified). We therefore keep the
    IDENTICAL induced model -- same prior on the lengthscale VALUE, same 0.025 hard floor --
    but parameterize value = 0.025 + exp(raw), raw unconstrained, per the package convention.
    (Rare edge: a prior draw below 0.025 during reset_parameters gives raw=nan and that restart
    fails gracefully; P ~ 0.2% per draw under the Hvarfner prior.)"""
    prior = LogNormalPrior(loc=_SQRT2 + math.log(q_dims) * 0.5, scale=_SQRT3)
    return gk.RBFKernel(
        ard_num_dims=q_dims,
        active_dims=active_dims,
        lengthscale_prior=prior,
        lengthscale_constraint=GreaterThan(2.5e-2, transform=torch.exp,
                                           inv_transform=torch.log,
                                           initial_value=prior.mode),
    )


def _latent_rbf(lat_dims, learn_lengthscale):
    """The categorical-slot kernel: RBF over the latent embedding.
    Product slot: lengthscale FIXED at 1.0 (LVGP gauge, == LVGPR qual_kernel).
    Sum slot: own learnable scalar lengthscale (== botorch Cat_1's independent ell)."""
    k = gk.RBFKernel(
        active_dims=torch.arange(lat_dims),
        lengthscale_constraint=Positive(transform=torch.exp, inv_transform=torch.log),
    )
    if learn_lengthscale:
        k.register_prior("raw_lengthscale_prior",
                         MollifiedUniformPrior(math.log(0.1), math.log(10)),
                         "raw_lengthscale")
    else:
        k.initialize(lengthscale=1.0)
        k.raw_lengthscale.requires_grad_(False)
    return k


def _scaled(base):
    """ScaleKernel with the GPR-convention outputscale prior (deviation from botorch's
    prior-less ScaleKernel; required for prior-sampled multistart restarts)."""
    s = gk.ScaleKernel(base, outputscale_constraint=Positive(transform=torch.exp,
                                                             inv_transform=torch.log))
    s.register_prior("raw_outputscale_prior", NormalPrior(0.0, 1.0), "raw_outputscale")
    return s


class LVGPRMixed(LVGPR):
    """LVGPR whose covariance is the botorch MixedSingleTaskGP composition with the latent
    kernel in the categorical slot. See module docstring for the line-by-line mapping."""

    def reset_parameters(self) -> None:
        """GPR.reset_parameters with one hardening: a Hvarfner prior draw below the 0.025
        lengthscale floor makes setting_closure raise (gpytorch bounds check) -- and
        fit_model_scipy calls reset_parameters OUTSIDE its per-restart try, so one bad draw
        (P ~ 0.2%/draw) kills the whole fit (observed: 1 crash in a 210-cell campaign).
        Clamp such draws just above the floor instead."""
        for _, module, prior, closure, setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            sample = prior.expand(closure(module).shape).sample()
            try:
                setting_closure(module, sample)
            except RuntimeError:                       # below a hard-floor constraint
                setting_closure(module, torch.clamp(sample, min=3e-2))

    def __init__(self, *args, lv_dim: int = 2, **kwargs):
        super().__init__(*args, lv_dim=lv_dim, **kwargs)
        n_qual = len(self.qual_index)
        lat_dims = n_qual * lv_dim
        q_dims = len(self.quant_index)
        quant_active = lat_dims + torch.arange(q_dims)

        # botorch L124-138: sum part = Scale( RBF_1(x) + Scale( Cat_1 ) )
        sum_kernel = _scaled(
            _hvarfner_rbf(q_dims, quant_active)
            + _scaled(_latent_rbf(lat_dims, learn_lengthscale=True))
        )
        # botorch L139-151: product part = Scale( RBF_2(x) * Cat_2 )
        prod_kernel = _scaled(
            _hvarfner_rbf(q_dims, quant_active)
            * _latent_rbf(lat_dims, learn_lengthscale=False)
        )
        # botorch L152: covar = sum + product, replacing GPR's global ScaleKernel wrap
        self.covar_module = sum_kernel + prod_kernel


class HeterLVGPTorchMixed(_LVGPTorchBase):
    """Heteroscedastic mixed-kernel LVGP: LVGPRMixed + replicate variances as fixed noise
    + the MATLAB-parity pooled aleatoric poly (all inherited from _LVGPTorchBase.fit)."""
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")
    HETERO = True
    MODEL_CLS = LVGPRMixed
