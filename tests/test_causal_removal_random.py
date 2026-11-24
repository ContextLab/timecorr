"""Removal changes no bit: configurations drawn at random.

Each draw is a causal ``timecorr`` or ``features`` call on a series and on
its first ``cut`` rows; the first ``cut`` rows of every output must be equal
bit for bit (``np.array_equal`` with NaN equal to NaN).  The draws cover T
from 1 to 300, 1 to 40 features, 1 to 3 subjects, every built-in kernel,
``min_samples`` at effective sizes that occur at the cut, events, NaN and
infinite samples, the correlation and combine functions, reducers with and
without a fit, ``refit_every``, orders 1 to 3 and small memory limits.
"""

import functools
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr import helpers

KERNELS = [
    (tc.gaussian_weights, {"var": 9}),
    (tc.gaussian_weights, {"var": 100}),
    (tc.gaussian_weights, {"var": 150.0 ** 2}),
    (tc.laplace_weights, {"scale": 4}),
    (tc.t_weights, {"df": 3}),
    (tc.mexican_hat_weights, {"sigma": 5}),
    (tc.boxcar_weights, {"width": 11}),
    (tc.uniform_weights, {}),
]
MEASURES = ["strength", "eigenvector_centrality", "pagerank_centrality",
            "pagerank_centrality_abs"]
MODELS = ["PCA", "IncrementalPCA", "TruncatedSVD", "FastICA", "FactorAnalysis"]


def _data(rng, S, T, K):
    kind = rng.choice(["white", "walk", "offset", "shared"])
    data = []
    shared = rng.standard_normal((T, K))
    for _ in range(S):
        x = rng.standard_normal((T, K))
        if kind == "walk":
            x = np.cumsum(x, axis=0)
        elif kind == "offset":
            x = x + 1e4
        elif kind == "shared":
            x = x + shared
        if rng.random() < 0.4:
            x[rng.random(x.shape) < rng.choice([0.002, 0.03])] = np.nan
        if rng.random() < 0.15 and T > 2:
            x[rng.integers(T), rng.integers(K)] = rng.choice([np.inf, -np.inf])
        data.append(x)
    return data


def _min_samples(rng, kernel, cut):
    """None, or a value at or next to an effective size of a kernel row of
    the series cut at ``cut``."""
    if rng.random() < 0.5:
        return None
    lag = helpers._lag_kernel(cut, kernel[0], kernel[1], "backward", True, None)
    sizes = np.unique(helpers._kernel_samples(lag)[1])
    sizes = sizes[sizes >= 1]
    if not sizes.size:
        return None
    size = float(rng.choice(sizes))
    value = rng.choice([size, np.nextafter(size, 0), np.nextafter(size, np.inf),
                        size * (1 + 1e-12), size * (1 - 1e-12), np.ceil(size)])
    return float(max(1.0, value))


def _events(rng, S, T):
    if rng.random() < 0.7 or T < 8:
        return None
    one = lambda: sorted(int(b) for b in rng.choice(np.arange(1, T + 20),  # noqa: E731
                                                    size=rng.integers(1, 4),
                                                    replace=False))
    if S > 1 and rng.random() < 0.5:
        return [one() for _ in range(S)]
    return one()


def draw(rng):
    """``(label, run, data, cut, with_model)``: ``run(data)`` is the list
    of the output arrays of the call drawn; with ``with_model``,
    ``run(data, True)`` is that of ``transform`` of the model the first
    call returned."""
    T = int(rng.choice([rng.integers(1, 20), rng.integers(1, 301), rng.integers(120, 301)]))
    cut = int(rng.choice([rng.integers(1, T + 1), max(1, T - int(rng.integers(0, 3)))]))
    S = int(rng.choice([1, 1, 2, 3]))
    K = int(rng.choice([rng.integers(1, 7), rng.integers(1, 15), rng.integers(1, 41)]))
    kernel = KERNELS[rng.integers(len(KERNELS))]
    kw = dict(causal=True, weights_function=kernel[0], weights_params=kernel[1])
    kw["min_samples"] = _min_samples(rng, kernel, cut)
    events = _events(rng, S, T)
    if events is not None:
        kw["events"] = events
    if rng.random() < 0.15:
        kw["include_zero"] = False
    cfuns = [tc.autofc, None] if S == 1 else [tc.autofc, tc.isfc, None, functools.partial(
        tc.wisfc, subject_weights=np.ones((S, S)) - np.eye(S))]
    cfun = cfuns[rng.integers(len(cfuns))]
    kw["cfun"] = cfun
    if S > 1 and cfun is not None and rng.random() < 0.6:
        kw["combine"] = [tc.corrmean_combine, tc.mean_combine, tc.tstat_combine][
            rng.integers(3)]
    fit = int(rng.integers(max(1, cut // 3), cut + 1))
    fitted = False
    if rng.random() < 0.45:
        function = tc.timecorr
        choice = rng.random()
        if cfun is None or choice < 0.25:
            K = min(K, 12)
        elif choice < 0.5:
            kw["rfun"] = MEASURES[rng.integers(len(MEASURES))]
            K = min(K, 15)
        elif choice < 0.75:
            kw["rfun"] = MODELS[rng.integers(len(MODELS))]
            fitted = True
            K = min(K, 15)
        else:
            kw["rfun"] = ["weighted_strength", tc.WeightedStrength(signed=True),
                          tc.Communities(n=2), "communities"][rng.integers(4)]
            fitted = True
        if fitted:
            kw["fit_timepoints"] = fit
    else:
        function = tc.features
        delta = KERNELS[rng.integers(len(KERNELS))]
        kw.update(order=int(rng.integers(1, 4)), delta_weights_function=delta[0],
                  delta_weights_params=delta[1], return_delta=True)
        choice = rng.random()
        if cfun is None or choice < 0.15:
            kw.update(rfun=None, delta_rfun=None)
            K = min(K, 3 if kw["order"] == 3 else 4)
        elif choice < 0.35:
            kw.update(rfun=MEASURES[rng.integers(len(MEASURES))], delta_rfun="strength")
            K = min(K, 15)
        elif choice < 0.55:
            fitted = True  # the defaults
        elif choice < 0.75:
            fitted = True
            kw.update(rfun=MODELS[rng.integers(len(MODELS))],
                      delta_rfun=["PCA", "weighted_strength"][rng.integers(2)])
            if rng.random() < 0.5 and kw["delta_rfun"] == "PCA":
                kw["ndims"] = int(rng.integers(2, 5))
            K = min(K, 15)
        else:
            fitted = True
            kw.update(rfun=["weighted_strength", tc.WeightedStrength(signed=True)][
                rng.integers(2)], delta_rfun="weighted_strength")
        if fitted:
            kw["fit_timepoints"] = fit
            if rng.random() < 0.35 and cut > fit:
                kw["refit_every"] = int(rng.integers(1, max(2, (cut - fit) // 2 + 1)) + 5)
        if rng.random() < 0.4:
            kw["max_memory_gb"] = float(rng.choice([0.004, 0.01, 0.05]))
    data = _data(rng, S, T, K)

    # the fitted pipeline of the whole series, applied to the shorter one
    with_model = function is tc.features and fitted and "refit_every" not in kw
    models = []

    def run(arrays, transform=False):
        arrays = [np.array(a) for a in arrays]
        given = arrays[0] if S == 1 else arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if transform:
                out = models[0].transform(given, return_delta=True,
                                          events=kw.get("events"))
            elif with_model:
                out, model = function(given, return_model=True, **kw)
                models.append(model)
            else:
                out = function(given, **kw)
        flat = []
        for part in (out if isinstance(out, tuple) else (out,)):
            for value in (part if isinstance(part, list) else [part]):
                if isinstance(value, list):
                    flat += value
                elif value is not None:
                    flat.append(value)
        return [np.asarray(value) for value in flat]

    label = dict(kw, function=function.__name__, T=T, cut=cut, S=S, K=K)
    return label, run, data, cut, with_model


def _same(whole, part, cut):
    return len(whole) == len(part) and all(
        a[:cut].shape == b.shape and np.array_equal(a[:cut], b, equal_nan=True)
        for a, b in zip(whole, part))


# Draws that are not valid calls, by the message of the error they raise:
# only these are left out; any other exception is a failure of the test.
INVALID = {
    "min_samples no kernel row reaches":
        r"min_samples=\S+ is more than a row of the kernel can have",
    "fit rows without defined correlations":
        r"fit_timepoints\)[^.]* hold (no row|no timepoint|fewer than 2 rows) with defined",
    "too few features or rows for communities":
        r"needs at least 4 features|Communities needs K >= 4|communities exceed K // 2"
        r"|needs at least \d+ timepoints|Communities needs the number of communities n"
        r"|the shift null needs T >= 40 timepoints",
    "FastICA on correlations without a varying column":
        r"FastICA cannot be fitted: no column of its input varies",
    "more memory than max_memory_gb":
        r"more than max_memory_gb",
}
MEMORY = "more memory than max_memory_gb"


def _raised(run, data):
    """The exception ``run(data)`` raises, or None."""
    try:
        run(data)
    except Exception as error:
        return error
    return None


def _invalid(error):
    """The key of ``INVALID`` that the message of ``error`` matches, or None."""
    import re

    if error is None:
        return None
    for reason, pattern in INVALID.items():
        if re.search(pattern, str(error)):
            return reason
    return None


def check(seed, draws):
    """Run ``draws`` draws of ``seed``: ``(compared, skipped, failures)``,
    ``skipped`` the number of draws left out by reason (``INVALID``)."""
    import collections

    rng = np.random.default_rng(seed)
    compared = 0
    skipped = collections.Counter()
    failures = []
    for index in range(draws):
        label, run, data, cut, with_model = draw(rng)
        try:
            whole = run(data)
        except Exception as error:
            reason = _invalid(error)
            if reason is None:
                failures.append((seed, index, label,
                                 f"{type(error).__name__}: {error}"))
            else:
                skipped[reason] += 1
                # the shorter series is the same invalid call (it holds the
                # rows of the fit, and no check reads the length); only a
                # memory limit can be enough for it
                again = _raised(run, [x[:cut] for x in data])
                if _invalid(again) != reason and reason != MEMORY:
                    failures.append((seed, index, label,
                                     f"the series raises ({error}); the shorter one: "
                                     f"{again!r}"))
            continue
        try:
            part = run([x[:cut] for x in data])
        except Exception as error:
            failures.append((seed, index, label, f"the shorter series raises: {error}"))
            continue
        compared += 1
        if not _same(whole, part, cut):
            failures.append((seed, index, label, "different"))
        elif with_model and not _same(whole, run([x[:cut] for x in data], True), cut):
            failures.append((seed, index, label, "transform different"))
    return compared, skipped, failures


def test_removal_changes_no_bit_in_configurations_drawn_at_random():
    compared, skipped, failures = check(0, 300)
    assert failures == []
    assert compared >= 150


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_removal_changes_no_bit_in_many_configurations_drawn_at_random(seed):
    compared, skipped, failures = check(seed, 500)
    assert failures == []
    assert compared >= 250
