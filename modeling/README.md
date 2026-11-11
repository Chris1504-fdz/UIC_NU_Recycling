# modeling — LVGP and mixed LVGP for the tensile data

This folder holds the two Gaussian-process models that are candidates for this project's property models. It also holds a
notebook that compares them on the October delivery (`raw_data/10_26_26/`).

| Key | Model | In one line |
|---|---|---|
| `lvgp` | **LVGP** (latent-variable GP; Zhang, Tao, Chen & Apley, *Technometrics* 2020) | The standard GP for continuous + categorical inputs. |
| `lvgp_mixed` | **mixed LVGP** (the lab's lvgp_kernel study, version 13) | The LVGP plus a trend shared by all levels and per-level offsets, with no prior on the output scales. |

The code is copied verbatim from the lvgp_kernel study (the `cat_kernel` repository), so that results here match the study.
`check_same_as_cat_kernel.py` verifies this: both models give bit-identical predictions to the originals.

**Preliminary choice (2026-10-28): use the LVGP** for these data. See [Preliminary result](#preliminary-result-2026-10-28).

## Contents

| Path | What it is |
|---|---|
| `lvgp_vs_mixed.ipynb` | The analysis: data, both models fitted to all 12 formulations (latent maps, partial dependence), a leave-one-formulation-out comparison, and the verdict. |
| `lvgp_compare.py` | Helpers for the notebook: data loading, the formulation table, fitting, leave-one-formulation-out cross-validation, scores, partial dependence, plots. |
| `check_same_as_cat_kernel.py` | Checks that the models here behave exactly like the originals in cat_kernel. |
| `models/` | The two models. Every file is a verbatim copy except `__init__.py` and `mixed_versions/__init__.py`. |
| `lvgp_bayes/` | The lab's PyTorch LVGP package (lvgp-bayes v0.3.0, [syerramilli/lvgp-bayes](https://github.com/syerramilli/lvgp-bayes)), vendored as in cat_kernel; see `lvgp_bayes/VENDORED.md`. |

## The two models

Both are GPs on $(x, c)$:

* $x$ holds the continuous inputs (here PAL % and extrusion temperature), scaled to $[0, 1]$.
* $c$ is the categorical level (here the polymer system: PP+PAL, HDPE+PAL, TPU+PAL).

The categorical input enters through a **latent map**. Each level gets a learned point $z(c)$ in a 2-D latent space, and
levels that behave alike end up close together. To fix rotation and translation, the first level sits at the origin and the
second on the first axis.

### LVGP

One product kernel:

$$
k\big((x,c),(x',c')\big) \;=\; \sigma^2\; k_x(x,x')\; \exp\!\big(-\tfrac12\lVert z(c)-z(c')\rVert^2\big),
\qquad k_x(x,x') = \exp\!\Big(-\tfrac12\sum_j \frac{(x_j-x'_j)^2}{\ell_j^2}\Big).
$$

Each level has its own response surface. The latent distance between two levels sets how much they share. This is a
**multiplicative** (second-order) interaction: the level changes the whole shape of the response in $x$.

### lvgp_mixed

The same product part, plus an **additive** (first-order) part. The kernel algebra follows BoTorch's `MixedSingleTaskGP`:

$$
k \;=\; \underbrace{\sigma_s^2\big[k_x^{(1)}(x,x') + \sigma_z^2\, k_z^{(1)}(c,c')\big]}_{\text{additive: a trend shared by all levels, plus level offsets}}
\;+\; \underbrace{\sigma_p^2\; k_x^{(2)}(x,x')\; k_z^{(2)}(c,c')}_{\text{multiplicative: level-specific shapes (the LVGP)}}
$$

* **Kernels.** The two x-kernels are independent, each with its own lengthscales. Both latent kernels use the same latent
  map $z(c)$.
* **Nesting.** With $\sigma_s^2 = 0$ the model is the LVGP. lvgp_mixed can therefore represent everything the LVGP can, at the
  price of more hyperparameters: 16 against 11 here (2 continuous inputs, 3 levels).
* **No scale priors.** The three output scales $\sigma_s^2, \sigma_z^2, \sigma_p^2$ have no prior, so they are fitted by
  maximum likelihood. The first mixed kernel (`models/lvgp_torch_mixed.py`) put a Normal(0, 1) prior on their logs. That
  prior made it costly to switch a part off, and removing it was more accurate in the study. Every other parameter keeps the
  LVGP's priors.
* **What the study found.** Noiseless benchmark, 24 tests: where the levels share structure, lvgp_mixed has about 0.4 of the
  LVGP's error. It loses slightly (0–7 %) on Hartmann, and it has rare large failures where it misses the level structure.

### What the two models share

* **Data format.** `{level: dict(X=(n, d) array, y_mean=(n,), y_var=(n,))}`, with one row per design point. `y_mean` is the
  mean of its replicates. `y_var` is the variance of that mean, used as **fixed, known noise**, floored at $10^{-4}$ in
  standardized units. A level is an integer (1, 2, …) or, for several categorical inputs, a tuple.
* **Fitting.** MAP with L-BFGS-B (scipy) from 3 starts: the default start plus 2 drawn from the priors. y is standardized
  internally. Results are deterministic given `torch.manual_seed(seed)`.

```python
import sys, torch
sys.path.insert(0, "modeling")            # the folder that contains models/ and lvgp_bayes/
from models import MODELS

torch.manual_seed(0)
m = MODELS["lvgp_mixed"].fit(data, needs_r=False, bounds=[(10, 50), (180, 240)])
mean, var = m.predict(level, X_new)       # var: variance of the latent function (no observation noise)
```

`needs_r=False` skips the replicate-noise model that the study used for its acquisition functions; it is not needed here.

## Where the code comes from

All source paths are relative to `/data/zhq7531/IDEAL/lvgp_kernel/cat_kernel`. Every file is copied unchanged and checked
with `cmp`.

| Here | Source in cat_kernel | Why it is needed |
|---|---|---|
| `models/lvgp_torch.py` | `models/lvgp_torch.py` | The LVGP (`HeterLVGPTorch`, `heter_lvgp_torch` in the study). |
| `models/mixed_versions/lvgp_v13.py` | `models/mixed_versions/lvgp_v13.py` | lvgp_mixed (`HeterLVGPv13`, `lvgp_v13` in the study). |
| `models/mixed_versions/lvgp_blocks.py` | `models/mixed_versions/lvgp_blocks.py` | Building blocks of the mixed versions; v13 uses `drop_prior` and the variance shares. |
| `models/lvgp_torch_mixed.py` | `models/lvgp_torch_mixed.py` | The mixed kernel itself (`LVGPRMixed`), which v13 subclasses. **Frozen in the study; do not edit.** |
| `models/hetero_lvgp_native.py`, `models/base.py` | same paths | The base classes and the replicate-noise model. |
| `lvgp_bayes/` | `lvgp_bayes/` | lvgp-bayes v0.3.0. |

Only two files are new:

* `models/__init__.py` registers just the two keys `lvgp` and `lvgp_mixed`. The study's registry holds every model it compared.
* `models/mixed_versions/__init__.py` is a one-line docstring.

**Rule for changes.** Do not edit the copied files here. A model change belongs in cat_kernel, as a new version file
(`lvgp_v<N>.py`), and is copied over afterwards. Run the check again after every copy.

### The BoTorch import

`models/__init__.py` starts with `import botorch`, even though neither model uses BoTorch. Importing BoTorch changes
GPyTorch's global numerical settings:

* 6 instead of 3 Cholesky retries, with growing jitter;
* exact solves up to 4096 points instead of 800.

The study's registry always imported BoTorch, so every study result was produced with these settings. Without the import,
fits of nearly singular kernel matrices (for example on noiseless data) differ from the study's in the 4th to 5th digit.
With it, they are identical.

## Is it the same code? `check_same_as_cat_kernel.py`

The check runs both code bases, this folder and cat_kernel, in separate processes. Each process uses one thread, because
multithreaded BLAS sums are not bit-reproducible. Both get the same environment, data, seeds and test points, and both models
are fitted. The check then compares every predicted mean and variance.

It uses four datasets, all small and mixed continuous + categorical:

| Dataset | Description |
|---|---|
| `smooth3` | 2 continuous inputs, 3 levels, 6 points per level, noiseless |
| `noisy4` | 3 continuous inputs, 4 levels, 8 points per level, with replicate noise |
| `twofactor` | 2 continuous inputs, two categoricals (3 × 3 levels), noiseless |
| `tensile` | this project's log UTS, 12 formulation means with their noise |

```bash
/data/zhq7531/envs/ml_gp_env/bin/python check_same_as_cat_kernel.py   # about 3 min; exit code 0 = PASS
#   --cat-kernel PATH   another cat_kernel checkout     --seeds 0 1 2   fitting seeds
#   --tol 1e-10         allow a tolerance (default 0)   --against-self  run this folder twice (determinism)
```

**Result (2026-10-28):** PASS. All 114 predictions per code base (4 datasets × 3 seeds × 2 models × levels) are identical,
with a largest difference of exactly 0. It also passes in the base anaconda environment.

| Environment | Python | torch | gpytorch | botorch | numpy | scipy |
|---|---|---|---|---|---|---|
| `ml_gp_env` | 3.13.5 | 2.7.1 | 1.14 | 0.14.0 | 2.2.6 | 1.16.0 |

## Running the notebook

* **Kernel.** Open `lvgp_vs_mixed.ipynb` with the `ml_gp_env` kernel.
* **Imports.** It imports the project's `utils/` (parser, specimen exclusions) and this folder.
* **Run time.** About 6 minutes, dominated by the cross-validation (12 folds × 4 responses × 3 seeds × 2 models = 288 fits in 24 processes).
* **Saving.** Like the project's other notebooks, it writes nothing by default. Each cell that can save sets a switch:
  * `SAVE_FIG` writes the figure to `modeling/figures/`;
  * `SAVE_CV` writes the cross-validation results to `modeling/results/lofo_cv.csv`;
  * `LOAD_CV` reads those results back instead of refitting.

## Preliminary result (2026-10-28)

**Data.** 12 formulations from the August DoE, all at recycle cycle 0:

* 4 formulations per polymer;
* 42 specimens, after setting aside the same two as `notebooks/data_processing_v4.ipynb`.

**Modeling unit.** One point per formulation: the mean of log(response) over its specimens, with the variance of that mean as
known noise.

**Test.** Leave one formulation out, fit to the other 11, and predict it. Each fit is repeated with 3 seeds.

The table gives the RMSE of log(response); 0.1 is about 10 % error.

| Response | LVGP | lvgp_mixed | Polymer mean | Global mean | LVGP closer (of 12) | Wilcoxon p |
|---|---|---|---|---|---|---|
| UTS | **0.34** | 0.36 | 0.55 | 0.85 | 4 | 0.52 |
| Young's modulus | **0.26** | 0.51 | 0.67 | 1.89 | 8 | 0.20 |
| Strain at break | **0.92** | 1.16 | 1.21 | 1.85 | 7 | 0.30 |
| Toughness | **1.42** | 1.50 | 1.65 | 1.65 | 9 | 0.15 |

* **Accuracy.** The LVGP has the lower error on all four responses. Neither model is better formulation by formulation, and
  no paired test is significant.
* **Where the gap comes from.** lvgp_mixed's higher RMSE comes from a few large misses on a left-out formulation: f07's
  modulus (error 1.40 against the LVGP's 0.02) and f04's strain at break (2.04 against 0.03).
* **The fitted structure.** Fitted to all 12 formulations, the two models give nearly the same surfaces. lvgp_mixed switches
  its shared trend off. For UTS and modulus it puts about two thirds of the variance into per-polymer offsets. For strain and
  toughness it is essentially the LVGP.
* **Uncertainty.** Both models are over-confident: only 50–70 % of left-out formulations fall within ±2 sd, against 95 %
  expected. The formulations differ in ways the three inputs do not explain, beyond the scatter of their means.

**Caveat.** This verdict is preliminary: 12 points, one recycle level. Re-run the notebook when the 5× / 10× recycle data or
the second DoE arrive.

## Notes for the GitHub version

* `.gitignore` excludes `__pycache__/` only.
* `figures/` and `results/` are written only when a save switch is on. Commit them or not, following the project's
  convention for `processed_data/`.
* The notebook is stored executed, with its outputs, so it can be read without running it.
* `lvgp_bayes/` is a vendored third-party package. Keep its `VENDORED.md`, and send fixes upstream rather than editing it here.
