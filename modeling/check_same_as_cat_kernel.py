"""
check_same_as_cat_kernel.py -- verify that the two models in this folder behave exactly like the originals in the
lvgp_kernel study (cat_kernel repository), before this folder is shared or pushed.

Both code bases are run in separate processes (they use the same package names, `models` and `lvgp_bayes`) with the same
Python environment, the same datasets, the same seeds and the same test points; the predicted means and variances must be
identical. Datasets (all small, mixed continuous + categorical, the shape of this project's data):

  smooth3     2 continuous inputs, 1 categorical with 3 levels, 6 points per level, noiseless
  noisy4      3 continuous inputs, 1 categorical with 4 levels, 8 points per level, replicate noise (fixed y_var)
  twofactor   2 continuous inputs, 2 categoricals (3 x 3 levels), 3 points per combination, noiseless
  tensile     this project's tensile data (log UTS per formulation, 12 means, replicate variances), if ../utils loads

Run (about 3 minutes, one thread per process):  python check_same_as_cat_kernel.py [--cat-kernel /path/to/cat_kernel] [--seeds 0 1 2]
Exit code 0 when every prediction matches to the bit (or within --tol), 1 otherwise.
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAT = "/data/zhq7531/IDEAL/lvgp_kernel/cat_kernel"
KEYS = {"here": {"lvgp": "lvgp", "lvgp_mixed": "lvgp_mixed"},
        "cat_kernel": {"lvgp": "heter_lvgp_torch", "lvgp_mixed": "lvgp_v13"}}

WORKER = r'''
import os, pickle, sys, warnings
import numpy as np, torch
torch.set_num_threads(1)                      # one thread: multithreaded BLAS sums are not bit-reproducible
warnings.filterwarnings("ignore")
root, which, inp, out = sys.argv[1:5]
sys.path.insert(0, root)
from models import MODELS
keys = %r[which]
cases = pickle.load(open(inp, "rb"))
res = {}
for name, case in cases.items():
    for seed in case["seeds"]:
        for model in ("lvgp", "lvgp_mixed"):
            torch.manual_seed(seed)
            m = MODELS[keys[model]].fit(case["data"], needs_r=False, bounds=case["bounds"])
            for lv in case["data"]:
                mu, var = m.predict(lv, case["Xtest"])
                res[(name, seed, model, lv)] = (np.ravel(mu), np.ravel(var))
pickle.dump(res, open(out, "wb"))
'''


def _levels(sizes):
    import itertools
    combos = list(itertools.product(*[range(1, s + 1) for s in sizes]))
    return [c[0] for c in combos] if len(sizes) == 1 else [tuple(c) for c in combos]


def make_cases(seeds):
    rng = np.random.default_rng(2026)
    cases = {}
    # smooth3: levels differ in amplitude and shape
    data = {}
    for lv in _levels([3]):
        X = rng.uniform(0, 1, (6, 2))
        data[lv] = dict(X=X, y_mean=np.sin(3 * X[:, 0] * lv) + X[:, 1] ** 2 * lv, y_var=np.zeros(6))
    cases["smooth3"] = dict(data=data, bounds=[(0, 1), (0, 1)], Xtest=rng.uniform(0, 1, (50, 2)), seeds=seeds)
    # noisy4: a shared trend plus a level-specific term, replicate noise
    data = {}
    for lv in _levels([4]):
        X = rng.uniform(-1, 1, (8, 3))
        f = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * lv * np.cos(2 * X[:, 2])
        v = (0.05 * (1 + lv / 4)) ** 2
        data[lv] = dict(X=X, y_mean=f + rng.normal(0, np.sqrt(v), 8), y_var=np.full(8, v))
    cases["noisy4"] = dict(data=data, bounds=[(-1, 1)] * 3, Xtest=rng.uniform(-1, 1, (50, 3)), seeds=seeds)
    # twofactor: two categoricals with 3 levels each
    data = {}
    for lv in _levels([3, 3]):
        X = rng.uniform(0, 2, (3, 2))
        data[lv] = dict(X=X, y_mean=lv[0] * np.exp(-X[:, 0]) + 0.5 * lv[1] * X[:, 1], y_var=np.zeros(3))
    cases["twofactor"] = dict(data=data, bounds=[(0, 2), (0, 2)], Xtest=rng.uniform(0, 2, (40, 2)), seeds=seeds)
    # tensile: this project's data, if available
    try:
        sys.path.insert(0, os.path.dirname(HERE))
        from lvgp_compare import load_tensile, formulation_data
        kept = load_tensile(os.path.dirname(HERE))
        data, bounds = formulation_data(kept, "max_tensile_stress_MPa")
        Xt = np.column_stack([np.linspace(10, 50, 30), np.linspace(180, 240, 30)])
        cases["tensile"] = dict(data=data, bounds=bounds, Xtest=Xt, seeds=seeds)
    except Exception as e:                                          # the check does not depend on the project's data
        print(f"(tensile case skipped: {type(e).__name__}: {e})")
    return cases


def run(root, which, cases_path, out_path):
    script = WORKER % (KEYS,)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", script, root, which, cases_path, out_path], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{which} worker failed:\n{r.stderr[-3000:]}")
    return pickle.load(open(out_path, "rb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-kernel", default=DEFAULT_CAT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tol", type=float, default=0.0, help="largest allowed absolute difference (default: identical)")
    ap.add_argument("--against-self", action="store_true", help="run this folder twice instead (tests determinism)")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    cases = make_cases(a.seeds)
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "cases.pkl"); pickle.dump(cases, open(cp, "wb"))
        here = run(HERE, "here", cp, os.path.join(d, "here.pkl"))
        orig = (run(HERE, "here", cp, os.path.join(d, "orig.pkl")) if a.against_self else
                run(os.path.abspath(a.cat_kernel), "cat_kernel", cp, os.path.join(d, "orig.pkl")))
    assert here.keys() == orig.keys()
    print(f"python {sys.version.split()[0]} · {len(here)} predictions per code base ({len(cases)} datasets x "
          f"{len(a.seeds)} seeds x 2 models x levels)\n")
    worst = 0.0
    print(f"{'dataset':10s} {'model':11s} {'max |d mean|':>13s} {'max |d var|':>12s}  identical")
    for name in cases:
        for model in ("lvgp", "lvgp_mixed"):
            ks = [k for k in here if k[0] == name and k[2] == model]
            dm = max(float(np.max(np.abs(here[k][0] - orig[k][0]))) for k in ks)
            dv = max(float(np.max(np.abs(here[k][1] - orig[k][1]))) for k in ks)
            same = all(np.array_equal(here[k][0], orig[k][0]) and np.array_equal(here[k][1], orig[k][1]) for k in ks)
            worst = max(worst, dm, dv)
            print(f"{name:10s} {model:11s} {dm:13.3g} {dv:12.3g}  {'yes' if same else 'NO'}")
    ok = worst <= a.tol
    print(f"\n{'PASS' if ok else 'FAIL'}: largest difference {worst:.3g} (allowed {a.tol:g})")
    sys.exit(0 if ok else 1)
