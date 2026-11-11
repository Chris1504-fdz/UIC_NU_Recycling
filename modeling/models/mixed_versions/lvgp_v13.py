"""
lvgp_v13.py -- LVGP kernel VERSION 13 (2026-10-08): the mixed kernel with BOTORCH-LIKE output scales

Kernel: the frozen mixed kernel (LVGPRMixed), with ONE change in the objective: the three output
scales sigma_s^2, sigma_z^2, sigma_p^2 carry NO prior (maximum likelihood for them), as in botorch's
MixedSingleTaskGP. Everything else keeps its prior (MAP). The mixed kernel's Normal(0,1) prior on
log sigma^2 is centered at sigma^2 = 1 and costs ~10.6 nats to switch a channel off to 0.01 -- this
version tests whether that prior keeps unneeded channels alive (e.g. on Hartmann).
Restart starting points for the scales are still drawn from the old Normal(0,1) on log sigma^2 (the
prior is only removed from the OBJECTIVE; lvgp-bayes samples restarts from priors).

Frozen once benchmarked: changes go into a new version file (see lvgp_blocks.py).
"""
import torch

from .lvgp_blocks import drop_prior, mixed_scale_kernels, mixed_scale_shares
from ..lvgp_torch import _LVGPTorchBase
from ..lvgp_torch_mixed import LVGPRMixed


class LVGPRv13(LVGPRMixed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for s in mixed_scale_kernels(self):
            drop_prior(s, "raw_outputscale_prior")

    def reset_parameters(self):
        LVGPRMixed.reset_parameters(self)                     # every parameter that still has a prior
        with torch.no_grad():
            for s in mixed_scale_kernels(self):               # scales: start from the old prior
                s.raw_outputscale.copy_(torch.randn_like(s.raw_outputscale))

    def diagnostics(self):
        return mixed_scale_shares(self)


class HeterLVGPv13(_LVGPTorchBase):
    """Version 13 + fixed replicate noise + MATLAB-parity aleatoric poly (inherited)."""
    SUPPORTS = ("ei", "lcb", "pi", "haei", "anpei", "rahbo")
    HETERO = True
    MODEL_CLS = LVGPRv13
