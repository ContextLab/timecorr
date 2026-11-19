"""
Order-2 recovery of nested structure with the squared-correlation summaries.

Data: ``tc.simulate_nested`` (K = 108: 6 supergroups x 3 modules x 6 nodes
unless noted).  The subjects share only the order-2 signal ``truth["h"]``
(T, 6), except in the isfc case (``shared="order1"``).  Features come from
``tc.features(order=2)`` with the explicit arguments of the future defaults:
``cfun=tc.autofc`` (unless noted), a Gaussian analysis kernel with standard
deviation ``sd2 = max(10, round(tau3 / 3.75))`` (``tau3 = T // 8`` unless
given; 33 at T 1000), ``rfun="auto"`` (PCA at order 1,
``WeightedStrength(signed=True)`` at order 2), ``delta_rfun="auto"``
(``"communities"``, automatic count) or ``"weighted_strength"``, and the
Gaussian feeding kernel (variance 25).

Decode protocol (``decode_score``): features subsampled every 10th
timepoint, columns z-scored per subject, ridge map with penalty
trace(Gram) / n fitted on subject A's order-2 features to ``h``, applied to
subject B's features; score = mean over the columns of ``h`` of the
correlation between prediction and ``h``.  The control applies the same map
to subjects from independent calls of ``simulate_nested`` (each with its own
``h``), scored against the first call's ``h``.  For the autofc cases the
control is the mean over 4 control subjects (one call each, seeds
seed + 1,000,000 + j), which go through the same ``features`` call as A and
B (so the fitted communities and PCA space are shared by all six); one
control subject per seed gave control scores with a standard deviation of
about 0.2 over seeds (0.3 at T 3000), too variable for a 4-seed bound of
0.15 (3 of 6 T 1000 cases exceeded it on the test seeds).  In the isfc case
the control group is computed in its own call, with the communities fitted
on the data group (frozen, as in the spike).

Development table (``_development_table()``; seeds 20000 + i; mean and
standard error over seeds; the seed counts are given per row):

| case | seeds | score | control (mean of 4 subjects) |
|-|-|-|-|
| sustained, communities (defaults-to-be) | 24 | .753 +- .019 | -.025 +- .018 |
| sustained, weighted strength | 24 | .645 +- .028 | -.025 +- .017 |
| sign-flip, communities | 24 | .690 +- .018 | .000 +- .025 |
| sign-flip, weighted strength | 24 | .563 +- .026 | .005 +- .020 |
| global 0.5, communities | 24 | .677 +- .021 | -.035 +- .018 |
| global 0.5, weighted strength | 24 | .549 +- .035 | -.039 +- .016 |
| tau2 15, T 3000, communities (slow) | 12 | .844 +- .016 | -.031 +- .050 |
| tau2 15, T 3000, weighted strength (slow) | 12 | .804 +- .022 | -.035 +- .048 |
| isfc, shared="order1", T 3000, tau3 40 (sd 11), held out, communities (slow) | 12 | .366 +- .027 | .053 +- .025 |
| same, weighted strength (slow) | 12 | .261 +- .029 | .042 +- .018 |

Standard deviation of the control over seeds: .08-.12 at T 1000, .17 at
T 3000.  The isfc rows use one control subject per seed (a group of 4
simulated in its own call).  Controls from the inputs (sustained, T 1000,
24 seeds): decode from the raw data .023 +- .011 (sd .056); from the
order-1 features (``order1_score``: ``features(order=1)``, PCA fitted on
the two subjects) .001 +- .046 (sd .228), asserted on 16 seeds
(30000-30015; measured -.098 +- .064) because 4 seeds were too few (4-seed
mean -.28 when the order-1 features came from the 6-subject call).
Adjusted Rand index of the order-1 communities against the modules: 1.000
on all seeds (sustained, sign-flip, T 3000), .956 +- .005 with global
signal 0.5 (not asserted).  Automatic count on independent autocorrelated
noise (K 108, T 1000, two subjects, Gaussian smoothing sd 5, 8 seeds): one
community (no eigenvalue above the null) on 7 of 8.  The T 3000 and isfc
rows used 12 seeds to keep the development run near 15 minutes (6
processes); the T 1000 rows used 24.

Assertions run on the fixed seeds 30000-30003: mean score above half the
development mean and |mean control| below 0.15.
"""

import multiprocessing
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import adjusted_rand_score

import timecorr as tc

STEP = 10
DEV_SEED = 20000
TEST_SEEDS = (30000, 30001, 30002, 30003)
CONTROL_OFFSET = 1_000_000
N_CONTROLS = 4


def wide_sd(T, tau3=None):
    """Standard deviation of the analysis kernel used in the spikes."""
    tau3 = T // 8 if tau3 is None else tau3
    return max(10, round(tau3 / 3.75))


def _ridge_predict(FA, FB, target):
    G = FA @ FA.T
    alpha = np.linalg.solve(
        G + np.trace(G) / len(G) * np.eye(len(G)), target - target.mean(0)
    )
    return FB @ (FA.T @ alpha)


def _zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def _score(pred, target):
    return float(
        np.mean([np.corrcoef(pred[:, c], target[:, c])[0, 1] for c in range(target.shape[1])])
    )


def decode_score(XA, XB, target, step=STEP):
    """
    Ridge map from subject A's features to ``target``, applied to subject B.

    Rows are subsampled every ``step`` timepoints; columns are z-scored per
    subject; the penalty is trace(Gram) / n.  Returns the mean over the
    columns of ``target`` of corr(prediction, target).
    """
    XA, XB, y = (np.asarray(a, dtype=float)[::step] for a in (XA, XB, target))
    return _score(_ridge_predict(_zscore(XA), _zscore(XB), y), y)


def held_out_decode_score(XA, XB, target, step=STEP):
    """``decode_score`` fitted on the first 40% of the (subsampled) timepoints
    of subject A and scored on the last 40% of subject B."""
    XA, XB, y = (np.asarray(a, dtype=float)[::step] for a in (XA, XB, target))
    n = len(y)
    a, b = int(0.4 * n), int(0.6 * n)
    pred = _ridge_predict(_zscore(XA[:a]), _zscore(XB[b:]), y[:a])
    return _score(pred, y[b:])


def _features(data, family, T, tau3=None, cfun=tc.autofc, delta_rfun=None):
    sd2 = wide_sd(T, tau3)
    if delta_rfun is None:
        delta_rfun = tc.Communities() if family == "communities" else "weighted_strength"
    return tc.features(
        data,
        order=2,
        cfun=cfun,
        weights_params={"var": sd2**2},
        rfun="auto",
        delta_rfun=delta_rfun,
        delta_weights_function=tc.gaussian_weights,
        delta_weights_params={"var": 25},
    )


# name: (simulate_nested arguments, D_1 family)
CASES = {
    "sustained, communities": (dict(T=1000), "communities"),
    "sustained, weighted strength": (dict(T=1000), "weighted_strength"),
    "sign-flip, communities": (dict(T=1000, coupling="signflip"), "communities"),
    "sign-flip, weighted strength": (dict(T=1000, coupling="signflip"), "weighted_strength"),
    "global 0.5, communities": (dict(T=1000, global_signal=0.5), "communities"),
    "global 0.5, weighted strength": (dict(T=1000, global_signal=0.5), "weighted_strength"),
    "tau2 15, T 3000, communities": (dict(T=3000, tau2=15), "communities"),
    "tau2 15, T 3000, weighted strength": (dict(T=3000, tau2=15), "weighted_strength"),
}


def case_scores(name, seed):
    """Scores of one autofc case on one seed (see the module docstring)."""
    gen, family = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, truth = tc.simulate_nested(S=2, set_random_seed=seed, **gen)
        others = [
            tc.simulate_nested(S=1, set_random_seed=seed + CONTROL_OFFSET + j, **gen)[0][0]
            for j in range(N_CONTROLS)
        ]
        delta_rfun = tc.Communities() if family == "communities" else None
        out = _features([data[0], data[1]] + others, family, gen["T"], delta_rfun=delta_rfun)
    h = truth["h"]
    X1, X2 = out[1], out[2]
    res = {
        "score": decode_score(X2[0], X2[1], h),
        "control": float(np.mean([decode_score(X2[0], N, h) for N in X2[2:]])),
        "raw": decode_score(data[0], data[1], h),
        "order1": decode_score(X1[0], X1[1], h),
    }
    if family == "communities":
        res["ari"] = adjusted_rand_score(truth["module"], delta_rfun.fits_[0]["labels"])
    return res


def isfc_scores(family, seed, T=3000, tau3=40, S=4):
    """Time-held-out decode with ``cfun=isfc`` and ``shared="order1"``."""
    gen = dict(S=S, T=T, tau3=tau3, shared="order1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, truth = tc.simulate_nested(set_random_seed=seed, **gen)
        other, _ = tc.simulate_nested(set_random_seed=seed + CONTROL_OFFSET, **gen)
        if family == "communities":
            comm = tc.Communities()
            X2 = _features(data, family, T, tau3, cfun=tc.isfc, delta_rfun=comm)[2]
            N2 = _features(
                other, family, T, tau3, cfun=tc.isfc, delta_rfun=comm.transform
            )[2]
        else:
            X2 = _features(data, family, T, tau3, cfun=tc.isfc)[2]
            N2 = _features(other, family, T, tau3, cfun=tc.isfc)[2]
    h = truth["h"]
    return {
        "score": held_out_decode_score(X2[0], X2[1], h),
        "control": held_out_decode_score(X2[0], N2[0], h),
    }


def noise_count(seed, T=1000, K=108, tau=5):
    """Number of communities from the shift null on autocorrelated noise
    (1 means no eigenvalue above the null)."""
    rng = np.random.default_rng(seed)
    data = [
        gaussian_filter1d(rng.standard_normal((T, K)), tau, axis=0) for _ in range(2)
    ]
    comm = tc.Communities()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tc.features(
            data,
            order=1,
            cfun=tc.autofc,
            rfun="auto",
            delta_rfun=comm,
            delta_weights_function=tc.gaussian_weights,
            delta_weights_params={"var": 25},
            return_delta=True,
        )
    return comm.fits_[0]["n"]


# tests -----------------------------------------------------------------------

_CACHE = {}


def _cached_case(name, seed):
    if (name, seed) not in _CACHE:
        _CACHE[(name, seed)] = case_scores(name, seed)
    return _CACHE[(name, seed)]


def _t3000(name):
    marks = [pytest.mark.slow] if CASES[name][0]["T"] > 1000 else []
    return pytest.param(name, marks=marks, id=name)


@pytest.mark.parametrize("name", [_t3000(n) for n in CASES])
def test_order2_signal_is_decoded(name):
    rows = [_cached_case(name, s) for s in TEST_SEEDS]
    score = np.mean([r["score"] for r in rows])
    control = np.mean([r["control"] for r in rows])
    assert score > DEV_MEAN[name] / 2, (score, DEV_MEAN[name])
    assert abs(control) < 0.15, control


def order1_score(seed, T=1000):
    """Decode of ``h`` from the order-1 features (PCA, fitted on the two
    subjects) of ``simulate_nested`` data (sustained coupling)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, truth = tc.simulate_nested(S=2, T=T, set_random_seed=seed)
        X1 = tc.features(data, order=1, cfun=tc.autofc,
                         weights_params={"var": wide_sd(T) ** 2}, rfun="auto")[1]
    return decode_score(X1[0], X1[1], truth["h"])


def test_raw_input_does_not_carry_h():
    rows = [_cached_case("sustained, communities", s) for s in TEST_SEEDS]
    assert abs(np.mean([r["raw"] for r in rows])) < 0.15


def test_order1_features_do_not_carry_h():
    # the order-1 decode varies more over seeds (sd about 0.25), so this
    # control uses 16 seeds
    scores = [order1_score(s) for s in range(TEST_SEEDS[0], TEST_SEEDS[0] + 16)]
    assert abs(np.mean(scores)) < 0.15, scores


def test_communities_recover_the_modules():
    for s in TEST_SEEDS:
        assert _cached_case("sustained, communities", s)["ari"] >= 0.9


@pytest.mark.slow
@pytest.mark.parametrize("family", ["communities", "weighted_strength"])
def test_isfc_time_held_out(family):
    rows = [isfc_scores(family, s) for s in TEST_SEEDS]
    score = np.mean([r["score"] for r in rows])
    control = np.mean([r["control"] for r in rows])
    assert score > DEV_MEAN["isfc, " + family] / 2, (score, control)
    assert abs(control) < 0.15, control


def test_automatic_count_on_noise():
    counts = [noise_count(30000 + i) for i in range(8)]
    assert sum(c == 1 for c in counts) >= 6, counts


_K702 = """
import json, resource, sys, time, warnings
import numpy as np
sys.path.insert(0, {tests!r})
import timecorr as tc
from test_nested_recovery import decode_score, wide_sd, CONTROL_OFFSET
warnings.simplefilter("ignore")
seed = {seed}
data, truth = tc.simulate_nested(S=2, T=1000, nodes_per_module=39, set_random_seed=seed)
other, _ = tc.simulate_nested(S=2, T=1000, nodes_per_module=39,
                              set_random_seed=seed + CONTROL_OFFSET)
units = [data[0], data[1], other[0]]
scale = 1 if sys.platform.startswith("linux") else 1 / 1024
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
t = time.time()
X = tc.features(units, order=2, cfun=tc.autofc,
                weights_params={{"var": wide_sd(1000) ** 2}},
                rfun=tc.WeightedStrength(signed=True), delta_rfun="weighted_strength",
                delta_weights_function=tc.gaussian_weights,
                delta_weights_params={{"var": 25}}, max_memory_gb=4)
wall = time.time() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
h = truth["h"]
print(json.dumps(dict(score=decode_score(X[2][0], X[2][1], h),
                      control=decode_score(X[2][0], X[2][2], h),
                      growth_gb=(peak - base) / 1e6, wall_s=wall)))
"""


@pytest.mark.slow
@pytest.mark.parametrize("seed", TEST_SEEDS[:2])
def test_k702_streamed(seed):
    import json

    code = _K702.format(tests=os.path.dirname(os.path.abspath(__file__)), seed=seed)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    print(f"K 702 seed {seed}: {res}")
    assert res["score"] > 0.2, res
    assert res["growth_gb"] < 6, res


# development table ---------------------------------------------------------

DEV_MEAN = {
    "sustained, communities": 0.753,
    "sustained, weighted strength": 0.645,
    "sign-flip, communities": 0.690,
    "sign-flip, weighted strength": 0.563,
    "global 0.5, communities": 0.677,
    "global 0.5, weighted strength": 0.549,
    "tau2 15, T 3000, communities": 0.844,
    "tau2 15, T 3000, weighted strength": 0.804,
    "isfc, communities": 0.366,
    "isfc, weighted_strength": 0.261,
}


def _dev_job(job):
    kind, name, seed = job
    if kind == "case":
        return job, case_scores(name, seed)
    if kind == "isfc":
        return job, isfc_scores(name, seed)
    if kind == "order1":
        return job, {"score": order1_score(seed)}
    return job, {"n": noise_count(seed)}


def _development_table(n_seeds=24, n_seeds_t3000=12, processes=6):  # pragma: no cover
    """Print the development table (not collected by pytest)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    jobs = []
    for name, (gen, _) in CASES.items():
        n = n_seeds if gen["T"] <= 1000 else n_seeds_t3000
        jobs += [("case", name, DEV_SEED + i) for i in range(n)]
    jobs += [("isfc", f, DEV_SEED + i) for f in ("communities", "weighted_strength")
             for i in range(n_seeds_t3000)]
    jobs += [("noise", "noise", DEV_SEED + i) for i in range(8)]
    jobs += [("order1", "order1", DEV_SEED + i) for i in range(n_seeds)]
    with multiprocessing.Pool(processes) as pool:
        results = pool.map(_dev_job, jobs, chunksize=1)
    table = {}
    for (kind, name, _), res in results:
        table.setdefault((kind, name), []).append(res)
    for (kind, name), rows in table.items():
        for key in rows[0]:
            v = np.array([r[key] for r in rows], dtype=float)
            se = v.std(ddof=1) / np.sqrt(len(v))
            print(f"{kind:5s} | {name:36s} | {key:7s} | n {len(v):2d} | "
                  f"{v.mean():+.3f} +- {se:.3f}", flush=True)
        if "control" in rows[0]:
            d = np.array([r["score"] - r["control"] for r in rows])
            print(f"{kind:5s} | {name:36s} | diff    | n {len(d):2d} | "
                  f"{d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    _development_table()
