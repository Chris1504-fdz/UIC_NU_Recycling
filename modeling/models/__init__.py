"""
Two latent-variable GP models for mixed continuous + categorical inputs, copied verbatim from the lvgp_kernel study
(IDEAL lab, cat_kernel repository: models/lvgp_torch.py, models/lvgp_torch_mixed.py, models/mixed_versions/lvgp_v13.py and
their dependencies). Only this file and models/mixed_versions/__init__.py are new: they register just the two models.

    MODELS["lvgp"]        the LVGP (Zhang, Tao, Chen & Apley 2020), via the lab's lvgp-bayes package (../lvgp_bayes)
    MODELS["lvgp_mixed"]  the mixed LVGP: the LVGP kernel plus a shared trend and level offsets, no prior on the three
                          output scales (version 13 of the study; called lvgp_mixed everywhere)

Both take the data as {level: dict(X=(n, d) array, y_mean=(n,), y_var=(n,))}: per design point the mean of its replicates
and the variance of that mean, used as fixed noise (floored at 1e-4 of the variance of y). Fit with
MODELS[key].fit(data, needs_r=False, bounds=[(lo, hi), ...]); predict with model.predict(level, X) -> (mean, variance).
check_same_as_cat_kernel.py verifies that both give the same predictions as the cat_kernel originals.
"""
# Importing BoTorch sets GPyTorch's global numerical defaults (6 instead of 3 Cholesky retries with growing jitter, exact
# solves up to 4096 points). The cat_kernel study always imported it (its registry includes BoTorch's GP), so every result
# there was produced with these settings; importing it here reproduces them exactly. Without it, fits of nearly singular
# kernel matrices (noiseless data) differ in the 4th to 5th digit. check_same_as_cat_kernel.py tests the equality.
import botorch  # noqa: F401  (side effect: the settings above)

from .lvgp_torch import HeterLVGPTorch
from .mixed_versions.lvgp_v13 import HeterLVGPv13

MODELS = {
    "lvgp": HeterLVGPTorch,
    "lvgp_mixed": HeterLVGPv13,
}
