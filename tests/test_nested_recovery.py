"""
Order-2 recovery of nested structure with the squared-correlation summaries.

Data: ``tc.simulate_nested`` (K = 108: 6 supergroups x 3 modules x 6 nodes
unless noted).  The subjects share only the order-2 signal ``truth["h"]``
(T, 6), except in the isfc case (``shared="order1"``).  Features come from
``tc.features(order=2)`` with these explicit arguments:
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
correlation between prediction and ``h``.  The control applies the same
map to subjects from independent calls of ``simulate_nested`` (each with its
own ``h``), scored against the first call's ``h``.  For the autofc cases the
control is the mean over 4 control subjects (one call each, seeds
seed + 1,000,000 + j), which go through the same ``features`` call as A and
B, so the fitted communities and PCA space are shared by all six.
Including the control subjects in the call changes the fitted communities
and therefore the score: with global signal 0.5 and communities, the score
was .617 +- .023 (adjusted Rand index .887) with one control subject in the
call (3 subjects) and .677 +- .021 (index .956) with four (6 subjects), on
the same 24 development seeds.  The first design used one control subject
per seed; its control bound failed on asserted seeds (3 of 6 T 1000 cases
on 30000-30003), as its development standard deviation predicts (about .2
per seed at T 1000 and .3 at T 3000, so a 4-seed mean has a standard error
of about .1 to .15), and it was replaced by the mean of 4.  In the isfc case
the control group is computed in its own call, with the communities fitted
on the data group (frozen, as in the spike).

Development table (``_development_table()``; seeds 20000 + i; mean and
standard error over seeds; the seed counts are given per row):

| case | seeds | score | control (mean of 4 subjects) |
|-|-|-|-|
| sustained, communities (`rfun="auto"`, `delta_rfun="auto"`) | 24 | .753 +- .019 | -.025 +- .018 |
| sustained, weighted strength | 24 | .645 +- .028 | -.025 +- .017 |
| sign-flip, communities | 24 | .690 +- .018 | .000 +- .025 |
| sign-flip, weighted strength | 24 | .563 +- .026 | .005 +- .020 |
| global 0.5, communities | 24 | .677 +- .021 | -.035 +- .018 |
| global 0.5, weighted strength | 24 | .549 +- .035 | -.039 +- .016 |
| tau2 15, T 3000, communities (slow) | 12 | .844 +- .016 | -.031 +- .050 |
| tau2 15, T 3000, weighted strength (slow) | 12 | .804 +- .022 | -.035 +- .048 |
| isfc, shared="order1", T 3000, tau3 40 (sd 11), transductive time split, communities (slow) | 12 | .366 +- .027 | .053 +- .025 |
| same, weighted strength (slow) | 12 | .261 +- .029 | .042 +- .018 |
| isfc, same data, held out with training fits, communities (slow) | 12 | .337 +- .029 | .044 +- .023 |
| same, weighted strength (slow) | 12 | .235 +- .030 | .038 +- .018 |

The transductive isfc rows (``isfc_scores``) fit the ridge map on the first
40% of the timepoints and score it on the last 40%, but the summaries
(communities, pair weights, signed means) are fitted on the whole
recording, the scored part included.  The held-out rows
(``held_out_isfc_scores``) split the recording first (first 40% train,
last 40% test, the middle 20% dropped), fit every summary on the training
part only, and compute the test part's features with those frozen fits
(``transform``, step by step); on the development seeds the frozen route
reproduced the pipeline's training features to within 1.3e-12.  The isfc
development controls (.053 +- .025 and .042 +- .018 transductive, .044 +-
.023 and .038 +- .018 held out) are about 2 standard errors above 0.

Standard deviation of the control over seeds: .08-.12 at T 1000, .17 at
T 3000.  The isfc rows use one control subject per seed (a group of 4
simulated in its own call).  Controls from the inputs (sustained, T 1000,
24 seeds): decode from the raw data .023 +- .011 (sd .056); from the
order-1 features (``order1_score``: ``features(order=1)``, PCA fitted on
the two subjects) .001 +- .046 (sd .228), asserted on 16 seeds
(30000-30015; measured -.098, standard error .064, sd .256) because 4 seeds were too few (4-seed
mean -.28 when the order-1 features came from the 6-subject call).  The
6-subject order-1 decode is the ``"order1"`` entry of ``case_scores``
(printed in the case rows of ``_development_table``; the same features as
``_order1_predictions``): .119 +- .052, sd .255, over the 24 development
seeds; the two-subject value above (.001 +- .046) is the ``order1`` job of
``_development_table`` (``order1_score``).  Both were re-measured on
20000-20023 with the same result.  Magnitude check
(``order1_magnitudes``, 6-subject call, 24 development seeds): mean
|score| .240 +- .028 for the shared-h subject and .213 +- .017 for the
independent-h control subjects (difference .027 +- .034, sd .167); asserted
on 30000-30015 that the difference is below 0.13 (about 3 standard errors
at 16 seeds; measured .031).  That statistic takes the absolute value after
the mean over the 6 columns of ``h``, so ``h`` carried with opposite signs
in different columns cancels.  With the absolute value taken per column
before the mean (``order1_target_magnitudes``, ``_magnitude``): development
seeds 20000-20023, shared .308 +- .022, control .331 +- .010, difference
-.022 +- .025 (sd .122); asserted on 30000-30015 that the difference is
below 0.09 (about 3 standard errors at 16 seeds; measured .074 +- .026,
shared .381, control .307); 24 further seeds (40000-40023, not asserted):
difference .001 +- .031 (shared .331, control .331).  What is asserted is
a bound on the mean decoding advantage of the shared-h subject over the
independent-h controls under this protocol (cross-subject ridge map,
order-1 PCA features of one 6-subject call), not that the order-1
features hold no information about h.
Adjusted Rand index of the order-1 communities against the modules: 1.000
on all seeds (sustained, sign-flip, T 3000), .956 +- .005 with global
signal 0.5 (not asserted).  Automatic count on independent autocorrelated
noise (K 108, T 1000, two subjects, Gaussian smoothing sd 5, 8 seeds): one
community (no eigenvalue above the null) on 7 of 8.  The T 3000 and isfc
rows used 12 seeds to keep the development run near 15 minutes (6
processes); the T 1000 rows used 24.

Assertions run on the fixed seeds 30000-30003 (30000-30007 for the T 3000
cases; 30000-30003 for the isfc rows): mean score above half the
development mean and |mean control| below 0.15.  Held-out isfc rows,
measured on 30000-30003: communities score .387 +- .038, control -.026 +-
.073; weighted strength score .293 +- .031, control -.026 +- .064.  The T 3000 control has a development standard deviation of about .17
per seed (12 seeds, 4 control subjects each), so its standard error is about
.085 at 4 seeds and .061 at 8 seeds.  Measured on 30000-30007: communities
score .858 +- .016, control -.104 +- .046 (sd .131); weighted strength score
.819 +- .029, control -.092 +- .056 (sd .158).

Control of the T 3000 cases pooled over the development and asserted seeds
(20000-20011 and 30000-30007, 20 seeds; mean and standard error over seeds):

| case | seeds | control (mean of 4 subjects) |
|-|-|-|
| tau2 15, T 3000, communities | 20 | -.060 +- .035 (sd .158) |
| tau2 15, T 3000, weighted strength | 20 | -.058 +- .036 (sd .162) |

Time-reversed control (``"reversed"`` in ``case_scores``; not asserted): the
prediction for subject B (the map fitted on A to ``h``) scored against the
subsampled ``h`` reversed in time.  Development seeds (20000-20023): sustained,
communities -.231 +- .052 (sd .257); sustained, weighted strength -.246 +-
.049 (sd .242).  This is not centered on 0: ``h`` itself correlates with its
own time reversal at -.213 +- .048 over the same seeds (mean over the
columns of ``h``, subsampled every 10th timepoint; T 1000), and the
prediction follows ``h`` (score .753 and .645), so the reversed score is
expected near that value rather than near 0.  Measured on 30000-30003:
-.293 +- .157 (communities) and -.233 +- .161 (weighted strength).

K 702 (slow, 2 seeds, ``max_memory_gb=4``, OMP 2; one control subject, not
asserted): seed 30000 score .825 (control -.014), peak resident memory
growth 1.08 GB, wall 91 s; seed 30001 score .741 (control .320), growth
1.09 GB, wall 91 s (250 s and 263 s in an earlier run alongside other jobs).
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
# T 3000 cases: the control varies more over seeds (sd about .17), so 8 seeds
TEST_SEEDS_T3000 = tuple(range(30000, 30008))
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


def _per_target(pred, target):
    """corr(prediction, target) of each column of ``target``."""
    return np.array(
        [np.corrcoef(pred[:, c], target[:, c])[0, 1] for c in range(target.shape[1])]
    )


def _score(pred, target):
    """Mean over the columns of ``target`` of the per-target correlation."""
    return float(np.mean(_per_target(pred, target)))


def _magnitude(pred, target):
    """Mean over the columns of ``target`` of the absolute per-target
    correlation (signs of different targets do not cancel)."""
    return float(np.mean(np.abs(_per_target(pred, target))))


def decode_prediction(XA, XB, target, step=STEP):
    """``decode_score``'s prediction for subject B and the subsampled target."""
    XA, XB, y = (np.asarray(a, dtype=float)[::step] for a in (XA, XB, target))
    return _ridge_predict(_zscore(XA), _zscore(XB), y), y


def decode_score(XA, XB, target, step=STEP):
    """
    Ridge map from subject A's features to ``target``, applied to subject B.

    Rows are subsampled every ``step`` timepoints; columns are z-scored per
    subject; the penalty is trace(Gram) / n.  Returns the mean over the
    columns of ``target`` of corr(prediction, target).
    """
    return _score(*decode_prediction(XA, XB, target, step))


def reversed_decode_score(XA, XB, target, step=STEP):
    """``decode_score``'s prediction for subject B scored against the
    subsampled ``target`` reversed in time."""
    XA, XB, y = (np.asarray(a, dtype=float)[::step] for a in (XA, XB, target))
    return _score(_ridge_predict(_zscore(XA), _zscore(XB), y), y[::-1])


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
        "reversed": reversed_decode_score(X2[0], X2[1], h),
        "control": float(np.mean([decode_score(X2[0], N, h) for N in X2[2:]])),
        "raw": decode_score(data[0], data[1], h),
        "order1": decode_score(X1[0], X1[1], h),
    }
    if family == "communities":
        res["ari"] = adjusted_rand_score(truth["module"], delta_rfun.fits_[0]["labels"])
    return res


def isfc_scores(family, seed, T=3000, tau3=40, S=4):
    """Transductive time split with ``cfun=isfc`` and ``shared="order1"``:
    the ridge map is fitted on the first 40% of subject A's features and
    scored on the last 40% of subject B's, but the features (communities,
    pair weights, signed means) are fitted on the whole recording, the
    scored part included.  ``held_out_isfc_scores`` fits them on the
    training part only."""
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


def split_decode_score(XA, XB, yA, yB, step=STEP):
    """Ridge map fitted on subject A's features ``XA`` to ``yA`` and applied
    to subject B's features ``XB``, scored against ``yB`` (rows subsampled
    every ``step`` timepoints, columns z-scored per subject and segment)."""
    XA, XB, yA, yB = (np.asarray(a, dtype=float)[::step] for a in (XA, XB, yA, yB))
    return _score(_ridge_predict(_zscore(XA), _zscore(XB), yA), yB)


def frozen_order2(subjects, feeding, analysis, sd2):
    """Order-2 features of ``subjects`` with frozen fits, step by step: the
    feeding kernel (Gaussian, variance 25) and ``autofc``, then
    ``feeding.transform``; the analysis kernel (standard deviation ``sd2``)
    and ``isfc``, then ``analysis.transform``.  Each summary transforms with
    its most recent fit (unit u with the fit's unit u when the number of
    units matches)."""
    c1 = tc.timecorr(subjects, weights_function=tc.gaussian_weights,
                     weights_params={"var": 25}, cfun=tc.autofc, rfun=None)
    D1 = feeding.transform(c1)
    c2 = tc.timecorr(D1, weights_function=tc.gaussian_weights,
                     weights_params={"var": sd2**2}, cfun=tc.isfc, rfun=None)
    return analysis.transform(c2)


def held_out_isfc_scores(family, seed, T=3000, tau3=40, S=4, check=False):
    """
    Time-held-out decode in which every learned summary is fitted on the
    training part only.

    The recording is split first: training rows are the first 40%, test rows
    the last 40%, and the middle 20% is dropped (a guard of 600 samples at
    T 3000, against kernel standard deviations of 5 and 11).  ``features``
    runs on the training rows with summary instances (feeding: communities
    with the automatic count, or weighted strength; analysis at order 2:
    signed weighted strength), so their fits are recorded; the test rows'
    order-2 features are computed with those frozen fits
    (``frozen_order2``).  Ridge is fitted on subject 0's training features
    and scored on subject 1's test features.  Control: the same map applied
    to the test features (frozen fits) of subject 0 of an independent call
    of ``simulate_nested`` (its own ``h``), scored against the first call's
    ``h``.  With ``check``, also returns the largest difference between
    ``frozen_order2`` on the training rows and the training features of
    ``features`` (the frozen route reproduces the pipeline).
    """
    gen = dict(S=S, T=T, tau3=tau3, shared="order1")
    sd2 = wide_sd(T, tau3)
    a, b = int(0.4 * T), int(0.6 * T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, truth = tc.simulate_nested(set_random_seed=seed, **gen)
        other, _ = tc.simulate_nested(set_random_seed=seed + CONTROL_OFFSET, **gen)
        feeding = tc.Communities() if family == "communities" else tc.WeightedStrength()
        analysis = tc.WeightedStrength(signed=True)
        train = [x[:a] for x in data]
        X2_train = tc.features(
            train, order=2, cfun=tc.isfc, weights_params={"var": sd2**2},
            rfun={1: "PCA", 2: analysis}, delta_rfun=feeding,
            delta_weights_function=tc.gaussian_weights,
            delta_weights_params={"var": 25},
        )[2]
        X2_test = frozen_order2([x[b:] for x in data], feeding, analysis, sd2)
        N2_test = frozen_order2([x[b:] for x in other], feeding, analysis, sd2)
        if check:
            again = frozen_order2(train, feeding, analysis, sd2)
    h = truth["h"]
    res = {
        "score": split_decode_score(X2_train[0], X2_test[1], h[:a], h[b:]),
        "control": split_decode_score(X2_train[0], N2_test[0], h[:a], h[b:]),
    }
    if check:
        res["max_diff"] = float(max(np.max(np.abs(x - y)) for x, y in zip(again, X2_train)))
    return res


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
    seeds = TEST_SEEDS if CASES[name][0]["T"] <= 1000 else TEST_SEEDS_T3000
    rows = [_cached_case(name, s) for s in seeds]
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


_ORDER1_CACHE = {}


def _order1_predictions(seed, T=1000):
    """Order-1 decode predictions (PCA fitted on all six subjects of one
    call) of ``h`` for the shared-h subject B and for the 4 independent-h
    control subjects, with the subsampled ``h``: ``(pred_B, [pred_N_j], y)``."""
    if (seed, T) in _ORDER1_CACHE:
        return _ORDER1_CACHE[(seed, T)]
    gen = dict(T=T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, truth = tc.simulate_nested(S=2, set_random_seed=seed, **gen)
        others = [
            tc.simulate_nested(S=1, set_random_seed=seed + CONTROL_OFFSET + j, **gen)[0][0]
            for j in range(N_CONTROLS)
        ]
        X1 = tc.features([data[0], data[1]] + others, order=1, cfun=tc.autofc,
                         weights_params={"var": wide_sd(T) ** 2}, rfun="auto")[1]
    h = truth["h"]
    pred_b, y = decode_prediction(X1[0], X1[1], h)
    preds_n = [decode_prediction(X1[0], N, h)[0] for N in X1[2:]]
    _ORDER1_CACHE[(seed, T)] = (pred_b, preds_n, y)
    return _ORDER1_CACHE[(seed, T)]


def order1_magnitudes(seed, T=1000):
    """|score| of the order-1 decode (see ``_order1_predictions``) for the
    shared-h subject B and for the 4 independent-h control subjects:
    ``(|score(A -> B)|, mean_j |score(A -> N_j)|)``.  The absolute value is
    taken after the mean over the columns of ``h``, so h carried with
    opposite signs in different columns cancels; ``order1_target_magnitudes``
    takes it per column."""
    pred_b, preds_n, y = _order1_predictions(seed, T)
    return (abs(_score(pred_b, y)),
            float(np.mean([abs(_score(p, y)) for p in preds_n])))


def order1_target_magnitudes(seed, T=1000):
    """As ``order1_magnitudes`` with ``_magnitude``: the mean over the
    columns of ``h`` of the absolute per-column correlation."""
    pred_b, preds_n, y = _order1_predictions(seed, T)
    return (_magnitude(pred_b, y),
            float(np.mean([_magnitude(p, y) for p in preds_n])))


def test_magnitude_statistic_does_not_cancel_across_targets():
    # predictions match three target columns and negate the other three:
    # every column carries its target perfectly
    rng = np.random.default_rng(0)
    target = rng.standard_normal((100, 6))
    pred = target * np.array([1, 1, 1, -1, -1, -1])
    np.testing.assert_allclose(_per_target(pred, target), [1, 1, 1, -1, -1, -1])
    assert abs(_score(pred, target)) < 1e-12
    assert abs(abs(_score(pred, target)) - 0) < 1e-12  # the old statistic
    assert _magnitude(pred, target) == pytest.approx(1.0)


def test_raw_input_does_not_carry_h():
    rows = [_cached_case("sustained, communities", s) for s in TEST_SEEDS]
    assert abs(np.mean([r["raw"] for r in rows])) < 0.15


def test_order1_features_do_not_carry_h():
    # the order-1 decode varies more over seeds (sd about 0.25), so this
    # control uses 16 seeds
    scores = [order1_score(s) for s in range(TEST_SEEDS[0], TEST_SEEDS[0] + 16)]
    assert abs(np.mean(scores)) < 0.15, scores


def test_order1_features_do_not_carry_h_in_magnitude():
    # development (24 seeds): |shared| .240 +- .028, |control| .213 +- .017,
    # difference .027 +- .034 (sd .167, so se .042 at 16 seeds); margin 0.13
    # is about 3 standard errors at 16 seeds
    rows = np.array([order1_magnitudes(s)
                     for s in range(TEST_SEEDS[0], TEST_SEEDS[0] + 16)])
    diff = np.mean(rows[:, 0] - rows[:, 1])
    assert diff < 0.13, (diff, rows.mean(0))


def test_order1_per_target_magnitude_advantage_is_bounded():
    # Asserted: the mean over 16 seeds of (shared-h minus control) decoding
    # magnitude from the order-1 features, under this protocol (ridge map
    # fitted on subject A, applied to subject B and to 4 independent-h
    # subjects; |corr| per column of h, then the mean over columns), is
    # below 0.09.  This bounds the average advantage; it does not show that
    # the order-1 features hold no information about h.
    # Development (20000-20023): shared .308 +- .022, control .331 +- .010,
    # difference -.022 +- .025 (sd .122, so se .031 at 16 seeds); margin
    # 0.09 is about 3 standard errors at 16 seeds.  Measured on the asserted
    # seeds 30000-30015: .074 +- .026 (shared .381, control .307).
    rows = np.array([order1_target_magnitudes(s)
                     for s in range(TEST_SEEDS[0], TEST_SEEDS[0] + 16)])
    diff = np.mean(rows[:, 0] - rows[:, 1])
    assert diff < 0.09, (diff, rows.mean(0))


def test_communities_recover_the_modules():
    for s in TEST_SEEDS:
        assert _cached_case("sustained, communities", s)["ari"] >= 0.9


@pytest.mark.slow
@pytest.mark.parametrize("family", ["communities", "weighted_strength"])
def test_isfc_transductive_time_split(family):
    # transductive: the summaries are fitted on the whole recording (see
    # isfc_scores); only the ridge map is fitted on the first 40%
    rows = [isfc_scores(family, s) for s in TEST_SEEDS]
    score = np.mean([r["score"] for r in rows])
    control = np.mean([r["control"] for r in rows])
    assert score > DEV_MEAN["isfc, " + family] / 2, (score, control)
    assert abs(control) < 0.15, control


@pytest.mark.slow
@pytest.mark.parametrize("family", ["communities", "weighted_strength"])
def test_isfc_held_out_with_training_fits(family):
    # every summary is fitted on the first 40% only; the last 40% gets its
    # features from the frozen fits (held_out_isfc_scores)
    rows = [held_out_isfc_scores(family, s, check=s == TEST_SEEDS[0]) for s in TEST_SEEDS]
    # the frozen route reproduces the pipeline on the training rows
    assert rows[0]["max_diff"] < 1e-9, rows[0]["max_diff"]
    score = np.mean([r["score"] for r in rows])
    control = np.mean([r["control"] for r in rows])
    assert score > DEV_MEAN["held-out isfc, " + family] / 2, (score, control)
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
other, _ = tc.simulate_nested(S=1, T=1000, nodes_per_module=39,
                              set_random_seed=seed + CONTROL_OFFSET)
units = [data[0], data[1], other[0]]
# ru_maxrss is the process's high-water mark (KB on Linux, bytes on macOS),
# taken here after the simulation, so the growth below is that of features
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

    from timecorr.higher_order import estimate_memory_gb

    # the property the test exists for: these steps fit max_memory_gb=4 only
    # when streamed, so completing under that budget implies they streamed
    K, T, S = 702, 1000, 3
    assert max(estimate_memory_gb(K, 2, T, S, streamed=True)) < 4
    assert estimate_memory_gb(K, 1, T, S)[1] > 4
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
    "held-out isfc, communities": 0.337,
    "held-out isfc, weighted_strength": 0.235,
}


def _dev_job(job):
    kind, name, seed = job
    if kind == "case":
        return job, case_scores(name, seed)
    if kind == "isfc":
        return job, isfc_scores(name, seed)
    if kind == "heldout":
        return job, held_out_isfc_scores(name, seed)
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
    jobs += [(kind, f, DEV_SEED + i) for kind in ("isfc", "heldout")
             for f in ("communities", "weighted_strength")
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
