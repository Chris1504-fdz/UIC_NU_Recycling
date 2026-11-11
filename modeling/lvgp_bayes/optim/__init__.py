from .mll_scipy import fit_model_scipy
try:
    from .numpyro_hmc import run_hmc_numpyro   # optional MCMC backend (needs jax+numpyro);
except ImportError:                            # the scipy MAP path above does not require it
    run_hmc_numpyro = None
