"""
The causal estimator: ``causal=True`` in ``timecorr``, ``wcorr``, ``autofc``,
``isfc`` and ``wisfc``.

The guarantee: output row t is unchanged (to 1e-9, NaN rows included) when
the input rows after t are replaced by other finite values, by NaN, by inf,
or removed; with ``include_zero=False`` also when row t itself changes.  The
tests check every row before several cut points of series of 300
timepoints, with an offset column, a column constant up to round-off, a
random walk and NaN samples.  Real computations; nothing is mocked.
"""

import functools
import re
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr import helpers

from .test_estimator_causality import _data as _conditioned_data
from .test_estimator_causality import _good

ATOL = 1e-9
T = 300
CUTS = (3, 40, 127, 128, 129, 200, 257, 299)

# name: (kernel function, parameters, a min_samples that its rows reach).
# The t kernel has weight at every lag but is narrow whatever its df: a
# backward row has at most 2.6 effective samples (2.1 without the diagonal).
KERNELS = {
    "gaussian": (tc.gaussian_weights, {"var": 25}, 6),
    "laplace": (tc.laplace_weights, {"scale": 8}, 6),
    "boxcar": (tc.boxcar_weights, {"width": 30}, 6),
    "t": (tc.t_weights, {"df": 3}, 2),
}
SUBJECT_WEIGHTS = np.array([[0.0, 1.0, 0.5], [0.3, 0.0, 1.0], [1.0, 2.0, 0.0]])
WISFC = functools.partial(tc.wisfc, subject_weights=SUBJECT_WEIGHTS)
# name: (cfun, number of subjects, combine)
CASES = {
    "autofc": (tc.autofc, 1, tc.null_combine),
    "isfc-null": (tc.isfc, 3, tc.null_combine),
    "isfc-corrmean": (tc.isfc, 3, tc.corrmean_combine),
    "wisfc-null": (WISFC, 3, tc.null_combine),
    "wisfc-corrmean": (WISFC, 3, tc.corrmean_combine),
    "smooth": (None, 1, tc.null_combine),
    "smooth-subjects": (None, 3, tc.null_combine),
}
MODES = ("finite", "nan", "inf", "remove")


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def _series(seed, n=T, K=5, missing=0.02):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, K))
    x[:, 0] += 1e6  # an offset
    x[:, 2] = 2.0 * (1 + 1e-16 * rng.standard_normal(n))  # constant up to round-off
    x[:, 3] = np.cumsum(rng.standard_normal(n))  # a random walk
    x[rng.random((n, K)) < missing] = np.nan
    return x


def _subjects(n_subjects, seed=0, missing=0.02):
    return [_series(seed + s, missing=missing) for s in range(n_subjects)]


def _perturb(x, n, mode, rng):
    """``x`` with rows ``n:`` replaced (finite values, NaN, inf) or removed."""
    if mode == "remove":
        return x[:n].copy()
    y = x.copy()
    if mode == "finite":
        y[n:] = rng.standard_normal(y[n:].shape) * 50 - 20
    elif mode == "nan":
        y[n:] = np.nan
    else:
        y[n:] = np.where(rng.random(y[n:].shape) < 0.5, np.inf, -np.inf)
        y[n:, 1] = 1e200  # squares overflow
    return y


def _call(data, case, kernel, include_zero=True, **kwargs):
    cfun, n_subjects, combine = CASES[case]
    weights_function, weights_params, _ = KERNELS[kernel]
    x = data[0] if n_subjects == 1 else data
    return _quiet(tc.timecorr, x, causal=True, cfun=cfun, combine=combine,
                  weights_function=weights_function, weights_params=weights_params,
                  include_zero=include_zero, **kwargs)


def _as_list(out):
    return out if isinstance(out, list) else [out]


def _max_difference(full, part, rows):
    """Largest absolute difference over rows ``:rows``; NaN must match NaN."""
    worst = 0.0
    for f, p in zip(_as_list(full), _as_list(part)):
        f, p = np.asarray(f)[:rows], np.asarray(p)[:rows]
        assert f.shape == p.shape
        assert np.array_equal(np.isnan(f), np.isnan(p))
        both = ~np.isnan(f)
        if both.any():
            worst = max(worst, float(np.max(np.abs(f[both] - p[both]))))
    return worst


# ----------------------------------------------------------------------------
# the guarantee
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("case", sorted(CASES))
def test_rows_do_not_depend_on_later_rows(case, include_zero, kernel, mode):
    data = _subjects(CASES[case][1])
    full = _call(data, case, kernel, include_zero)
    rng = np.random.default_rng(7)
    defined = 0
    for n in CUTS:
        part = _call([_perturb(x, n, mode, rng) for x in data], case, kernel,
                     include_zero)
        assert _max_difference(full, part, n) <= ATOL
        defined += int(np.sum(~np.isnan(np.asarray(_as_list(part)[0])[:n])))
    assert defined > 0  # the comparison is not between NaN rows only


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("case", ["autofc", "isfc-null", "wisfc-corrmean", "smooth"])
def test_rows_do_not_depend_on_later_rows_with_min_samples(case, include_zero, kernel,
                                                           mode):
    min_samples = KERNELS[kernel][2]
    data = _subjects(CASES[case][1], seed=40)
    full = _call(data, case, kernel, include_zero, min_samples=min_samples)
    plain = _call(data, case, kernel, include_zero)
    nan_rows = np.isnan(np.asarray(_as_list(full)[0])).all(axis=1)
    # the number asks for more than the default rule (except for the narrow
    # t kernel, whose third row already has 2 effective samples), and rows
    # reach it
    default = np.isnan(np.asarray(_as_list(plain)[0])).all(axis=1).sum()
    assert nan_rows.sum() > default or (kernel == "t" and nan_rows.sum() == default)
    assert not nan_rows[30:].all()
    rng = np.random.default_rng(27)
    for n in CUTS:  # (the first 3 rows alone cannot reach the number: NaN rows)
        part = _call([_perturb(x, n, mode, rng) for x in data], case, kernel,
                     include_zero, min_samples=min_samples)
        assert _max_difference(full, part, n) <= ATOL
        if mode != "remove":
            # (rows a later NaN leaves undefined do not change the columns,
            # and so the sums, of the chunk of kernel rows they are in)
            for f, p in zip(_as_list(full), _as_list(part)):
                assert np.array_equal(np.asarray(f)[:n], np.asarray(p)[:n],
                                      equal_nan=True)


@pytest.mark.parametrize("missing", [0, 0.002])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("case", ["autofc", "isfc-corrmean", "wisfc-null", "smooth"])
def test_rows_do_not_depend_on_later_rows_with_few_missing_samples(
        case, include_zero, kernel, mode, missing):
    # without NaN samples (and with NaN samples in some chunks of kernel rows
    # only) the estimator takes its path for complete data, and later NaN or
    # inf rows change the path of the chunk they are in
    data = _subjects(CASES[case][1], seed=30, missing=missing)
    assert bool(missing) == any(np.isnan(x).any() for x in data)
    full = _call(data, case, kernel, include_zero)
    rng = np.random.default_rng(17)
    for n in CUTS:
        part = _call([_perturb(x, n, mode, rng) for x in data], case, kernel,
                     include_zero)
        rows = n if include_zero or mode == "remove" else n + 1
        assert _max_difference(full, part, rows) <= ATOL
    assert not np.isnan(np.asarray(_as_list(full)[0])[20:]).any()


@pytest.mark.parametrize("mode", ["finite", "nan", "inf"])
@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_without_the_diagonal_row_t_does_not_depend_on_row_t(case, kernel, mode):
    data = _subjects(CASES[case][1])
    full = _call(data, case, kernel, include_zero=False)
    rng = np.random.default_rng(8)
    for n in CUTS:
        # rows n: change; rows :n + 1 (row n included) must not
        part = _call([_perturb(x, n, mode, rng) for x in data], case, kernel,
                     include_zero=False)
        assert _max_difference(full, part, n + 1) <= ATOL


@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("case", sorted(CASES))
def test_replacing_later_rows_with_finite_values_changes_no_bit(case, include_zero,
                                                               kernel):
    data = _subjects(CASES[case][1])
    full = _call(data, case, kernel, include_zero)
    rng = np.random.default_rng(9)
    for n in CUTS:
        part = _call([_perturb(x, n, "finite", rng) for x in data], case, kernel,
                     include_zero)
        rows = n if include_zero else n + 1
        for f, p in zip(_as_list(full), _as_list(part)):
            assert np.array_equal(np.asarray(f)[:rows], np.asarray(p)[:rows],
                                  equal_nan=True)


def test_rows_do_not_depend_on_later_rows_with_event_boundaries():
    x = _series(3)
    kw = dict(causal=True, cfun=tc.autofc, weights_params={"var": 25},
              events=[90, 210])
    full = _quiet(tc.timecorr, x, **kw)
    rng = np.random.default_rng(10)
    for mode in ("finite", "nan", "inf"):
        for n in CUTS:
            part = _quiet(tc.timecorr, _perturb(x, n, mode, rng), **kw)
            assert _max_difference(full, part, n) <= ATOL
    # warm-up starts again at each block
    nan_rows = np.flatnonzero(np.isnan(full).all(axis=1))
    assert {0, 1, 90, 91, 210, 211} <= set(nan_rows.tolist())
    assert not np.isnan(full[[89, 209, 299]]).any()


@pytest.mark.parametrize("mode", MODES)
def test_wcorr_rows_do_not_depend_on_later_rows(mode):
    a, b = _series(11), _series(12)
    rng = np.random.default_rng(13)

    def run(a, b):
        w = tc.gaussian_weights(len(a), {"var": 25}, direction="backward")
        return _quiet(tc.wcorr, a, b, w, causal=True)

    full = run(a, b)
    for n in CUTS:
        part = run(_perturb(a, n, mode, rng), _perturb(b, n, mode, rng))
        f, p = full[:, :, :n], part[:, :, :n]
        assert np.array_equal(np.isnan(f), np.isnan(p))
        assert np.allclose(f, p, atol=ATOL, rtol=0, equal_nan=True)
    assert np.sum(~np.isnan(full)) > 0


def test_non_finite_and_overflowing_samples_are_missing():
    x = np.random.default_rng(0).standard_normal((120, 4))
    holes = [(5, 0), (40, 1), (41, 1), (90, 3), (119, 2)]
    values = [np.inf, -np.inf, 1e200, -1e151, np.inf]
    y, z = x.copy(), x.copy()
    for (t, k), v in zip(holes, values):
        y[t, k] = v
        z[t, k] = np.nan
    for cfun in (tc.autofc, None):
        kw = dict(causal=True, cfun=cfun, weights_params={"var": 20})
        out = _quiet(tc.timecorr, y, **kw)
        assert np.array_equal(out, _quiet(tc.timecorr, z, **kw), equal_nan=True)
        assert not np.isnan(out[10:]).any()


# ----------------------------------------------------------------------------
# agreement with the estimator of causal=False
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("cfun", ["autofc", "isfc", "wisfc"])
def test_agrees_with_the_backward_kernel_of_causal_false(cfun):
    subjects = [_conditioned_data(seed=s) for s in range(3)]
    function = {"autofc": tc.autofc, "isfc": tc.isfc,
                "wisfc": functools.partial(tc.wisfc, subject_weights=1.0)}[cfun]
    kw = dict(cfun=function, combine=tc.null_combine, weights_params={"var": 30})
    causal = _quiet(tc.timecorr, subjects, causal=True, **kw)
    plain = _quiet(tc.timecorr, subjects, include_timepoints="backward", **kw)
    for s in range(3):
        # entries that are well conditioned for the estimator of causal=False
        # (the measure of test_estimator_causality.py)
        good = _good(subjects, s, cfun, "backward")
        defined = ~np.isnan(causal[s]).all(axis=1)
        assert not np.isnan(causal[s][defined]).any()
        assert defined.sum() >= 147
        good &= defined[:, np.newaxis]
        assert good.mean() > 0.5
        assert np.max(np.abs(causal[s][good] - plain[s][good])) <= ATOL


def test_smoothing_agrees_with_the_backward_kernel_of_causal_false():
    x = np.random.default_rng(1).standard_normal((200, 4))
    x[30:33, 1] = np.nan
    kw = dict(cfun=None, weights_params={"var": 30})
    causal = _quiet(tc.timecorr, x, causal=True, **kw)
    plain = _quiet(tc.timecorr, x, include_timepoints="backward", **kw)
    defined = ~np.isnan(causal)
    assert defined.all()  # one weighted sample defines a smoothed value
    assert np.max(np.abs(causal[defined] - plain[defined])) <= 1e-12


def test_default_is_not_causal():
    x = np.random.default_rng(2).standard_normal((80, 4))
    kw = dict(cfun=tc.autofc, weights_params={"var": 20})
    assert np.array_equal(_quiet(tc.timecorr, x, **kw),
                          _quiet(tc.timecorr, x, causal=False, min_samples=None, **kw))
    w = tc.gaussian_weights(80, {"var": 20})
    assert np.array_equal(_quiet(tc.wcorr, x, x, w),
                          _quiet(tc.wcorr, x, x, w, causal=False))


# ----------------------------------------------------------------------------
# warm-up rows
# ----------------------------------------------------------------------------


def _expected_undefined(w, observed, min_samples, min_count=3):
    """Rows of the kernel ``w`` below the rules, from the observed timepoints."""
    a = np.abs(w) * observed
    count = (a != 0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ess = a.sum(axis=1) ** 2 / (a**2).sum(axis=1)
    ess = np.nan_to_num(ess)
    if min_samples is None:
        return count < min_count
    return (count < min_count) | (ess < min_samples * (1 - 1e-12))


@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("include_zero", [True, False])
def test_default_nan_rows_are_the_zero_rows_of_causal_false(kernel, include_zero):
    x = np.random.default_rng(14).standard_normal((120, 4))
    weights_function, weights_params, _ = KERNELS[kernel]
    kw = dict(cfun=tc.autofc, weights_function=weights_function,
              weights_params=weights_params, include_zero=include_zero)
    causal = _quiet(tc.timecorr, x, causal=True, **kw)
    plain = _quiet(tc.timecorr, x, include_timepoints="backward", **kw)
    nan_rows = np.isnan(causal).all(axis=1)
    assert np.array_equal(nan_rows, (plain == 0).all(axis=1))
    assert np.array_equal(np.flatnonzero(nan_rows),
                          np.arange(2 if include_zero else 3))
    assert not np.isnan(causal[~nan_rows]).any()
    assert np.max(np.abs(causal[~nan_rows] - plain[~nan_rows])) <= ATOL
    # smoothing: a value is undefined only without any weighted sample
    smooth = _quiet(tc.timecorr, x, causal=True, **{**kw, "cfun": None})
    assert np.array_equal(np.flatnonzero(np.isnan(smooth).all(axis=1)),
                          np.arange(0 if include_zero else 1))
    assert not np.isnan(smooth[1:]).any()


def _reach(weights_function, weights_params, include_zero=True, n=4000):
    """The largest effective sample size of a row with ``n`` earlier timepoints."""
    w = weights_function(n, weights_params, direction="backward",
                         include_zero=include_zero)[-1]
    return float(np.abs(w).sum() ** 2 / (w**2).sum())


@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_a_min_samples_that_no_row_reaches_raises(kernel, include_zero):
    n = 120
    x = np.random.default_rng(15).standard_normal((n, 4))
    weights_function, weights_params, reached = KERNELS[kernel]
    # what a row reaches with any number of earlier timepoints
    most = _reach(weights_function, weights_params, include_zero)
    assert helpers._kernel_reach(helpers._lag_kernel(
        7, weights_function, weights_params, "backward", include_zero
    )) == pytest.approx(most, rel=1e-9)
    kw = dict(causal=True, weights_function=weights_function,
              weights_params=weights_params, include_zero=include_zero)
    shown = re.escape(f"{most:.4g}")
    for cfun in (tc.autofc, tc.isfc, None):
        with pytest.raises(ValueError, match=rf"min_samples.*{shown}"):
            tc.timecorr(x, cfun=cfun, min_samples=most * 1.01, **kw)
        low = min(reached, most * 0.99)
        assert not np.isnan(_quiet(tc.timecorr, x, cfun=cfun, min_samples=low,
                                   **kw)[-1]).any()
    # missing data may leave every row below a number the kernel reaches
    y = x.copy()
    y[::2] = np.nan
    assert np.isnan(_quiet(tc.timecorr, y, cfun=tc.autofc, min_samples=most * 0.99,
                           **kw)).all()


@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("cfun", [tc.autofc, None])
def test_whether_min_samples_raises_does_not_depend_on_the_length(kernel, cfun):
    weights_function, weights_params, reached = KERNELS[kernel]
    most = _reach(weights_function, weights_params)
    kw = dict(causal=True, cfun=cfun, weights_function=weights_function,
              weights_params=weights_params)
    x = np.random.default_rng(16).standard_normal((400, 3))
    for n in (3, 4, 10, 60, 400):
        # a number the kernel reaches, though not within n rows: no error, and
        # the rows that do not reach it are NaN
        out = _quiet(tc.timecorr, x[:n], min_samples=most * 0.999, **kw)
        assert out.shape[0] == n
        if n <= 10 and kernel != "t":
            assert np.isnan(out).all()
        # a number no row of any series reaches: the same error at every length
        with pytest.raises(ValueError, match="min_samples"):
            tc.timecorr(x[:n], min_samples=most * 1.001, **kw)


def test_three_rows_cannot_reach_six_samples_and_are_nan():
    x = np.random.default_rng(17).standard_normal((300, 4))
    kw = dict(causal=True, weights_params={"var": 25}, min_samples=6)
    for cfun in (tc.autofc, None):
        full = _quiet(tc.timecorr, x, cfun=cfun, **kw)
        short = _quiet(tc.timecorr, x[:3], cfun=cfun, **kw)
        assert np.isnan(short).all() and np.isnan(full[:3]).all()
        assert not np.isnan(full[100:]).any()
    w = tc.gaussian_weights(3, {"var": 25}, direction="backward")
    assert np.isnan(_quiet(tc.wcorr, x[:3], x[:3], w, causal=True, min_samples=6)).all()


def test_kernels_without_a_lag_profile_never_raise_for_min_samples():
    n = 60
    x = np.random.default_rng(18).standard_normal((n, 4))
    w = tc.gaussian_weights(n, {"var": 9}, direction="backward")
    most = float(np.max(w.sum(axis=1) ** 2 / (w**2).sum(axis=1)))
    # an array: every row is below the number, so every row is NaN
    assert np.isnan(_quiet(tc.wcorr, x, x, w, causal=True, min_samples=most + 1)).all()
    assert np.isnan(_quiet(tc.autofc, x, w, causal=True, min_samples=most + 1)).all()
    assert helpers._kernel_reach(helpers.as_kernel(w)) is None
    # a kernel function of the user's
    for cfun in (tc.autofc, None):
        out = _quiet(
            tc.timecorr, x, causal=True, cfun=cfun, weights_params={"var": 9},
            min_samples=most + 1,
            weights_function=lambda T, p: tc.gaussian_weights(T, p, direction="backward"))
        assert np.isnan(out).all()
    # the same kernel, built in: the error
    with pytest.raises(ValueError, match="min_samples"):
        tc.timecorr(x, causal=True, cfun=tc.autofc, weights_params={"var": 9},
                    min_samples=most + 1)


def test_reach_of_the_built_in_kernels():
    def reach(function, params, **kwargs):
        return helpers._kernel_reach(
            helpers._lag_kernel(5, function, params, "backward", **kwargs))

    # equal weights at every lag: no limit
    assert reach(tc.uniform_weights, None) is None
    x = np.random.default_rng(19).standard_normal((40, 3))
    out = _quiet(tc.timecorr, x, causal=True, cfun=tc.autofc, min_samples=30,
                 weights_function=tc.uniform_weights, weights_params={})
    assert np.array_equal(np.flatnonzero(np.isnan(out).all(axis=1)), np.arange(29))
    # a boxcar over the lags 0..14, and without the diagonal
    assert reach(tc.boxcar_weights, {"width": 30}) == pytest.approx(15)
    assert reach(tc.boxcar_weights, {"width": 30}, include_zero=False) == pytest.approx(14)
    # exclude_timepoints: the lags 0..4, and the lags from 5 on
    assert reach(tc.boxcar_weights, {"width": 30}, exclude=4) == pytest.approx(5)
    assert reach(tc.boxcar_weights, {"width": 30}, exclude=-4) == pytest.approx(10)
    # a wide Gaussian: sd * 2 * sqrt(pi) / 2 of a half kernel, about 1.77 sd
    wide = reach(tc.gaussian_weights, {"var": 1e6})
    assert wide == pytest.approx(_reach(tc.gaussian_weights, {"var": 1e6}, n=9000),
                                 rel=1e-9)
    assert 1700 < wide < 1800
    # event boundaries do not change what a kernel can reach
    kernel = helpers._lag_kernel(50, tc.gaussian_weights, {"var": 25}, "backward")
    labels = np.repeat([0, 1], 25)
    assert helpers._kernel_reach(kernel.with_events(labels)) == helpers._kernel_reach(kernel)


def _t_sums(df, start, n=4_000_000):
    """Sums of the t density and of its square over the lags ``start..n``."""
    d = np.arange(start, n, dtype=np.float64)
    w = (1 + d * d / df) ** (-(df + 1) / 2)
    return w.sum(), (w * w).sum()


@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("df", [1, 2, 5, 30])
def test_heavy_tails_have_a_known_reach(df, include_zero):
    kernel = helpers._lag_kernel(2, tc.t_weights, {"df": df}, "backward", include_zero)
    limit = helpers._kernel_reach(kernel)
    start = 0 if include_zero else 1
    if df == 1:
        # the Cauchy sums in closed form: sum 1 / (1 + d**2) and
        # sum 1 / (1 + d**2)**2 over d >= 0
        first = (1 + np.pi / np.tanh(np.pi)) / 2
        second = (1 + np.pi / 2 * (1 / np.tanh(np.pi) + np.pi / np.sinh(np.pi) ** 2)) / 2
        expected = (first - start) ** 2 / (second - start)
        assert limit == pytest.approx(expected, rel=1e-12)
    else:
        # added up: the lags from 4 million on hold less than 1e-12 of the sums
        first, second = _t_sums(df, start)
        assert limit == pytest.approx(first**2 / second, rel=1e-10)
    # decisive on both sides of the limit, in timecorr and in features,
    # whatever the length of the series
    for n in (20, 3):
        x = np.random.default_rng(21).standard_normal((n, 3))
        calls = (
            lambda m: tc.timecorr(
                x, causal=True, cfun=tc.autofc, weights_function=tc.t_weights,
                weights_params={"df": df}, include_zero=include_zero, min_samples=m),
            lambda m: tc.features(
                x, order=1, causal=True, cfun=tc.autofc, rfun=None, delta_rfun=None,
                weights_function=tc.t_weights, weights_params={"df": df},
                delta_weights_params={"var": 400}, include_zero=include_zero,
                min_samples=m),
        )
        for call in calls:
            if limit * (1 - 1e-6) >= 1:
                _quiet(call, limit * (1 - 1e-6))
            with pytest.raises(ValueError, match="min_samples.*more than a row"):
                call(max(1.0, limit) * (1 + 1e-6))


def test_an_unreachable_min_samples_never_gives_silent_nan_rows():
    # the case of a Cauchy kernel (at most 3.3 effective samples)
    x = np.random.default_rng(22).standard_normal((20, 3))
    with pytest.raises(ValueError, match=r"min_samples=4 .*is 3\.3"):
        tc.features(x, order=1, causal=True, cfun=tc.autofc, rfun=None,
                    delta_rfun=None, weights_function=tc.t_weights,
                    weights_params={"df": 1}, delta_weights_params={"var": 25},
                    min_samples=4)

    def reach(function, params, include_zero=True):
        return helpers._kernel_reach(
            helpers._lag_kernel(5, function, params, "backward", include_zero))

    # kernels far wider than the lags that are added up
    assert reach(tc.boxcar_weights, {"width": 1e9}) == 5e8
    assert reach(tc.boxcar_weights, {"width": 1e9}, False) == 5e8 - 1
    for scale in (3.0, 3000.0, 1e9):  # (1 + q) / (1 - q), q = exp(-1 / scale)
        q = np.exp(-1 / scale)
        assert reach(tc.laplace_weights, {"scale": scale}) == pytest.approx(
            (1 + q) / -np.expm1(-1 / scale), rel=1e-9)
    # sqrt(pi var) for the half of a wide Gaussian (plus half the lag-0 term)
    assert reach(tc.gaussian_weights, {"var": 1e14}) == pytest.approx(
        np.sqrt(np.pi * 1e14), rel=1e-6)
    # an upper bound for a wide Mexican hat
    a = np.abs(helpers.mexican_hat_profile(np.arange(100_000), {"sigma": 5000.0}))
    most = np.nanmax(np.cumsum(a) ** 2 / np.cumsum(a * a))
    assert most <= reach(tc.mexican_hat_weights, {"sigma": 5000.0}) <= most * 1.01
    # every one of these decides: more than the reach raises
    for function, params in ((tc.boxcar_weights, {"width": 1e9}),
                             (tc.laplace_weights, {"scale": 1e9}),
                             (tc.gaussian_weights, {"var": 1e14}),
                             (tc.mexican_hat_weights, {"sigma": 5000.0}),
                             (tc.t_weights, {"df": 0.5})):
        most = reach(function, params)
        kw = dict(causal=True, cfun=tc.autofc, weights_function=function,
                  weights_params=params)
        with pytest.raises(ValueError, match="min_samples"):
            tc.timecorr(x, min_samples=most * 1.02, **kw)
        assert _quiet(tc.timecorr, x, min_samples=min(5, most * 0.9),
                      **kw).shape == (20, 6)
    # equal weights at every lag have no limit: never an error
    for m in (5, 1e6, 1e300):
        out = _quiet(tc.timecorr, x, causal=True, cfun=tc.autofc, min_samples=m,
                     weights_function=tc.uniform_weights, weights_params={})
        assert np.isnan(out[:4]).all() and np.isnan(out[5:]).all() == (m > 20)


def test_three_equal_weights_are_three_samples():
    x = np.random.default_rng(3).standard_normal((60, 4))
    # a backward boxcar over the lags 0, 1, 2: three weights of 1 from row 2 on
    kw = dict(causal=True, cfun=tc.autofc, weights_function=tc.boxcar_weights,
              weights_params={"width": 5})
    out = _quiet(tc.timecorr, x, min_samples=3, **kw)
    nan_rows = np.isnan(out).all(axis=1)
    assert np.array_equal(np.flatnonzero(nan_rows), [0, 1])
    assert not np.isnan(out[2:]).any()
    # the default is 3
    assert np.array_equal(out, _quiet(tc.timecorr, x, **kw), equal_nan=True)
    # a little more than 3 samples asks for more than the window has
    with pytest.raises(ValueError, match="min_samples"):
        tc.timecorr(x, min_samples=3.001, **kw)
    # the same kernel as an array, in wcorr
    w = np.tril(np.ones((60, 60))) - np.tril(np.ones((60, 60)), -3)
    r = _quiet(tc.wcorr, x, x, w, causal=True)
    assert np.array_equal(np.flatnonzero(np.isnan(r).all(axis=(0, 1))), [0, 1])
    assert not np.isnan(r[:, :, 2:]).any()
    assert np.allclose(tc.mat2vec(r)[2:], out[2:], atol=1e-12, rtol=0)


@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("var, min_samples", [(25, None), (25, 1), (25, 3), (25, 8),
                                              (100, 10)])
def test_warm_up_rows_are_those_below_the_rules(include_zero, var, min_samples):
    n = 150
    x = np.random.default_rng(4).standard_normal((n, 4))
    out = _quiet(tc.timecorr, x, causal=True, cfun=tc.autofc, min_samples=min_samples,
                 weights_params={"var": var}, include_zero=include_zero)
    w = tc.gaussian_weights(n, {"var": var}, direction="backward",
                            include_zero=include_zero)
    expected = _expected_undefined(w, np.ones(n), min_samples)
    assert np.array_equal(np.isnan(out).all(axis=1), expected)
    assert not np.isnan(out[~expected]).any()
    first = int(np.flatnonzero(~expected)[0])
    assert np.array_equal(np.flatnonzero(expected), np.arange(first))
    if min_samples in (None, 1):
        assert first == (2 if include_zero else 3)  # fewer than 3 samples
    elif min_samples == 3:
        # three unequal weights are fewer than 3 effective samples
        assert first == (3 if include_zero else 4)
    else:
        assert first > 5  # more leading rows than the three-sample rule gives


def test_rows_without_samples_do_not_count_as_observed():
    n = 120
    x = np.random.default_rng(5).standard_normal((n, 4))
    x[40:52] = np.nan  # whole rows missing
    x[80, 2] = np.nan  # a single sample: the row is observed
    out = _quiet(tc.timecorr, x, causal=True, cfun=tc.autofc, min_samples=6,
                 weights_params={"var": 25})
    w = tc.gaussian_weights(n, {"var": 25}, direction="backward")
    observed = (~np.isnan(x)).any(axis=1).astype(float)
    expected = _expected_undefined(w, observed, 6)
    assert np.array_equal(np.isnan(out).all(axis=1), expected)
    assert not np.isnan(out[~expected]).any()
    assert expected[45:55].all() and not expected[70:].any()


def test_zero_variance_entries_stay_zero():
    x = np.random.default_rng(6).standard_normal((100, 4))
    x[:, 1] = 7.0
    out = _quiet(tc.timecorr, x, causal=True, cfun=tc.autofc,
                 weights_params={"var": 25})
    assert np.isnan(out[:2]).all() and not np.isnan(out[2:]).any()
    full = tc.vec2mat(out[2:])  # (K, K, T - 2)
    assert np.all(full[1, :, :] == 0) and np.all(full[:, 1, :] == 0)
    assert np.all(full[[0, 2, 3], [0, 2, 3], :] == 1)
    assert np.any(full[0, 2, :] != 0)


def test_warning_names_the_undefined_rows():
    x = np.random.default_rng(7).standard_normal((90, 3))
    w = tc.gaussian_weights(90, {"var": 25}, direction="backward")
    n8 = int(_expected_undefined(w, np.ones(90), 8).sum())
    assert n8 > 5
    with pytest.warns(UserWarning, match=rf"^{n8} of 90 timepoints .* NaN"):
        out = tc.timecorr(x, causal=True, cfun=tc.autofc, min_samples=8,
                          weights_params={"var": 25})
    assert np.array_equal(np.flatnonzero(np.isnan(out).all(axis=1)), np.arange(n8))
    with pytest.warns(UserWarning, match=r"^2 of 90 timepoints .* NaN") as record:
        tc.wcorr(x, x, w, causal=True)
    assert "min_samples" not in str(record[0].message)
    with pytest.warns(UserWarning, match=r"^3 of 90 timepoints .*min_samples=3.* NaN"):
        tc.autofc(x, w, causal=True, min_samples=3)
    with pytest.warns(UserWarning, match=r"[Ss]moothing .* 3 timepoint"):
        tc.timecorr(x, causal=True, cfun=None, weights_params={"var": 25},
                    min_samples=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # every smoothed value is defined
        tc.timecorr(x, causal=True, cfun=None, weights_params={"var": 25})
    # one warning per call
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tc.timecorr([x, x + 1, x * 2], causal=True, weights_params={"var": 25})
    assert len(caught) == 1


def test_smoothing_warm_up_is_per_feature():
    n = 80
    x = np.random.default_rng(8).standard_normal((n, 3))
    x[20:40, 1] = np.nan
    out = _quiet(tc.timecorr, x, causal=True, cfun=None, min_samples=4,
                 weights_params={"var": 25})
    w = tc.gaussian_weights(n, {"var": 25}, direction="backward")
    for k in range(3):
        observed = (~np.isnan(x[:, k])).astype(float)
        expected = _expected_undefined(w, observed, 4, min_count=0)
        assert np.array_equal(np.isnan(out[:, k]), expected)
    assert np.isnan(out[30, 1]) and not np.isnan(out[30, [0, 2]]).any()


# ----------------------------------------------------------------------------
# the correlation functions, called directly
# ----------------------------------------------------------------------------


def test_correlation_functions_called_directly():
    data = [np.random.default_rng(s).standard_normal((100, 4)) for s in range(3)]
    w = tc.gaussian_weights(100, {"var": 16}, direction="backward")
    kw = dict(causal=True, weights_params={"var": 16}, combine=tc.null_combine)
    for cfun, direct in (
        (tc.autofc, lambda: tc.autofc(data, w, causal=True)),
        (tc.isfc, lambda: tc.isfc(data, w, causal=True)),
        (WISFC, lambda: tc.wisfc(data, w, subject_weights=SUBJECT_WEIGHTS,
                                 causal=True)),
    ):
        expected = _quiet(tc.timecorr, data, cfun=cfun, **kw)
        got = _quiet(direct)
        for e, g in zip(expected, got):
            assert np.isnan(g[:2]).all() and not np.isnan(g[2:]).any()
            assert np.allclose(e, g, atol=1e-12, rtol=0, equal_nan=True)
    # a single array, and wcorr
    one = _quiet(tc.autofc, data[0], w, causal=True, min_samples=6)
    r = _quiet(tc.wcorr, data[0], data[0], w, causal=True, min_samples=6)
    assert one.shape == (100, 10) and r.shape == (4, 4, 100)
    undefined = np.isnan(r).all(axis=(0, 1))
    assert 3 < undefined.sum() < 50 and not np.isnan(r[:, :, ~undefined]).any()
    assert np.array_equal(np.isnan(one).all(axis=1), undefined)
    # (mat2vec itself sets NaN above the diagonal to 0)
    assert np.array_equal(one[~undefined], tc.mat2vec(r)[~undefined])
    assert np.all(r[[0, 1, 2, 3], [0, 1, 2, 3], 50] == 1)


def test_streamed_rows_equal_wcorr():
    x = _series(20, n=200, K=6)
    w = helpers._lag_kernel(200, tc.gaussian_weights, {"var": 25}, "backward")
    for a, b, symmetric in ((x, x, True), (x, _series(21, n=200, K=6), False)):
        full = _quiet(helpers._wcorr, a, b, w, warn=False, causal=True, min_samples=5,
                      autocorrelation=symmetric)
        got = np.empty_like(full)
        for i0, i1, R, _, _ in helpers._corr_row_blocks(a, b, w, symmetric, 2,
                                                        causal=True, min_samples=5):
            got[i0:i1] = R
        assert np.array_equal(np.isnan(full), np.isnan(got))
        assert np.isnan(full[:, :, 0]).all() and not np.isnan(full[:, :, 20:]).any()
        assert np.allclose(full, got, atol=1e-12, rtol=0, equal_nan=True)


# ----------------------------------------------------------------------------
# arguments
# ----------------------------------------------------------------------------

X = np.random.default_rng(0).standard_normal((60, 4))
OK = dict(causal=True, cfun=tc.autofc, weights_params={"var": 9})


def test_backward_kernel_is_implied():
    out = _quiet(tc.timecorr, X, **OK)
    for side in ("backward", "pre", "BACKWARD"):
        assert np.array_equal(out, _quiet(tc.timecorr, X, include_timepoints=side, **OK),
                              equal_nan=True)


@pytest.mark.parametrize("side", ["forward", "post", "Forward"])
def test_forward_kernels_raise(side):
    with pytest.raises(ValueError, match="backward"):
        tc.timecorr(X, include_timepoints=side, **OK)


def test_kernel_parameters_are_required():
    with pytest.raises(ValueError, match="weights_params"):
        tc.timecorr(X, causal=True, cfun=tc.autofc)


def test_delta_kernel_raises():
    with pytest.raises(ValueError, match="eye_weights"):
        tc.timecorr(X, causal=True, cfun=tc.autofc, weights_function=tc.eye_weights,
                    weights_params={})
    with pytest.raises(ValueError, match="delta"):
        tc.wcorr(X, X, np.eye(60), causal=True)
    with pytest.raises(ValueError, match="delta"):
        tc.autofc(X, np.eye(60), causal=True)


def test_wisfc_needs_fixed_subject_weights():
    data = [X, X[::-1].copy(), X**2]
    with pytest.raises(ValueError, match="subject_weights"):
        tc.timecorr(data, causal=True, cfun=tc.wisfc, weights_params={"var": 9})
    with pytest.raises(ValueError, match="subject_weights"):
        tc.timecorr(data, causal=True, cfun=functools.partial(tc.wisfc),
                    weights_params={"var": 9})
    w = tc.gaussian_weights(60, {"var": 9}, direction="backward")
    with pytest.raises(ValueError, match="subject_weights"):
        tc.wisfc(data, w, causal=True)
    # fixed weights: the built-in function, without the warning of a callable
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tc.timecorr(data, causal=True, weights_params={"var": 9},
                          cfun=functools.partial(tc.wisfc, subject_weights=1.0))
    assert not [c for c in caught if "guarantee" in str(c.message)]
    direct = _quiet(tc.wisfc, data, w, subject_weights=1.0, causal=True)
    for o, d in zip(out, direct):
        assert np.allclose(o, d, atol=1e-12, rtol=0, equal_nan=True)


def test_events_must_be_boundaries():
    features = np.random.default_rng(1).standard_normal((60, 3))
    with pytest.raises(ValueError, match="boundaries"):
        tc.timecorr(X, events=features, **OK)
    with pytest.raises(ValueError, match="boundaries"):
        tc.timecorr([X, X + 1], events=[[30], features], **OK)
    out = _quiet(tc.timecorr, X, events=[30], **OK)
    assert out.shape == (60, 10)
    out = _quiet(tc.timecorr, [X, X + 1], events=[[30], None], **OK)
    assert len(out) == 2


def test_kernels_with_weight_on_later_timepoints_raise():
    two_sided = tc.gaussian_weights(60, {"var": 9})
    with pytest.raises(ValueError, match="later timepoints"):
        tc.wcorr(X, X, two_sided, causal=True)
    with pytest.raises(ValueError, match="later timepoints"):
        tc.autofc(X, two_sided, causal=True)
    with pytest.raises(ValueError, match="later timepoints"):
        tc.isfc([X, X + 1], two_sided, causal=True)
    with pytest.raises(ValueError, match="later timepoints"):
        tc.wcorr(X, X, helpers._lag_kernel(60, tc.gaussian_weights, {"var": 9}),
                 causal=True)
    for cfun in (tc.autofc, None):
        with pytest.raises(ValueError, match="later timepoints"):
            tc.timecorr(X, causal=True, cfun=cfun, weights_params={},
                        weights_function=lambda T, p: tc.gaussian_weights(T, {"var": 9}))
    # a function of the user's that returns a backward kernel is accepted
    out = _quiet(
        tc.timecorr, X, causal=True, cfun=tc.autofc, weights_params={"var": 9},
        weights_function=lambda T, p: tc.gaussian_weights(T, p, direction="backward"))
    assert np.allclose(out, _quiet(tc.timecorr, X, **OK), atol=1e-12, rtol=0,
                       equal_nan=True)


def test_a_correlation_function_of_the_users_warns_once():
    def mine(data, weights):
        return tc.autofc(data, weights)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tc.timecorr(X, causal=True, cfun=mine, weights_params={"var": 9})
    messages = [str(c.message) for c in caught if "guarantee" in str(c.message)]
    assert len(messages) == 1 and "cfun" in messages[0]
    assert issubclass(
        [c for c in caught if "guarantee" in str(c.message)][0].category, UserWarning)
    # it received the backward kernel as a T x T array
    plain = _quiet(tc.timecorr, X, cfun=tc.autofc, weights_params={"var": 9},
                   include_timepoints="backward")
    assert np.array_equal(out, plain)


def test_a_reducer_of_the_users_warns_once():
    def mine(arrays):
        return [np.nan_to_num(a[:, :2]) for a in arrays]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tc.timecorr(X, rfun=mine, **OK)
    messages = [str(c.message) for c in caught if "guarantee" in str(c.message)]
    assert len(messages) == 1 and "rfun" in messages[0]
    assert out.shape == (60, 2)


@pytest.mark.parametrize("rfun", [
    "PCA", {"model": "PCA"}, "weighted_strength", "weighted_strength_split",
    "communities", tc.WeightedStrength(), tc.Communities(n=2),
], ids=str)
def test_fitted_reducers_need_fit_timepoints(rfun):
    with pytest.raises(ValueError, match="fit_timepoints"):
        tc.timecorr(X, rfun=rfun, **OK)


@pytest.mark.parametrize("rfun", ["strength", "eigenvector_centrality",
                                  "pagerank_centrality_abs"])
def test_graph_measures_run_on_the_defined_rows(rfun):
    out = _quiet(tc.timecorr, X, rfun=rfun, **OK)
    corrs = _quiet(tc.timecorr, X, **OK)
    defined = ~np.isnan(corrs).all(axis=1)
    assert out.shape == (60, 4)
    assert np.array_equal(np.isnan(out).all(axis=1), ~defined)
    assert not np.isnan(out[defined]).any()
    assert np.allclose(out[defined], _quiet(tc.reduce, corrs[defined], rfun=rfun),
                       atol=1e-12, rtol=0)
    # and in a list of subjects
    outs = _quiet(tc.timecorr, [X, X[::-1].copy()], rfun=rfun, **OK)
    assert len(outs) == 2 and np.isnan(outs[1][:2]).all()
    assert not np.isnan(outs[1][2:]).any()


@pytest.mark.parametrize("bad", [-1, 0, 0.5, np.nan, np.inf, "3", True])
def test_min_samples_must_be_none_or_a_number_from_1(bad):
    with pytest.raises(ValueError, match="min_samples"):
        tc.timecorr(X, min_samples=bad, **OK)
    w = tc.gaussian_weights(60, {"var": 9}, direction="backward")
    with pytest.raises(ValueError, match="min_samples"):
        tc.wcorr(X, X, w, causal=True, min_samples=bad)


def test_min_samples_needs_causal():
    with pytest.raises(ValueError, match="causal=True"):
        tc.timecorr(X, cfun=tc.autofc, weights_params={"var": 9}, min_samples=5)
    with pytest.raises(ValueError, match="causal=True"):
        tc.wcorr(X, X, tc.gaussian_weights(60, {"var": 9}), min_samples=5)


def test_new_arguments_are_keyword_only():
    w = tc.gaussian_weights(60, {"var": 9}, direction="backward")
    with pytest.raises(TypeError):
        tc.wcorr(X, X, w, True)
    with pytest.raises(TypeError):
        tc.autofc(X, w, True)
    with pytest.raises(TypeError):
        tc.isfc(X, w, True)
    with pytest.raises(TypeError):
        tc.wisfc(X, w, 1.0, True)


# ----------------------------------------------------------------------------
# the arithmetic of row t does not depend on later rows
# ----------------------------------------------------------------------------


def _level_step(where, seed=0, n=120, K=3, step=1e4):
    """Rows before ``where`` (at least the first) near 0, the others near ``step``."""
    x = np.random.default_rng(seed).normal(size=(n, K))
    if where == 0:
        x[0] = 0
        x[1:] += step
    else:
        x[where:] += step
    return x


# The estimator centers the data anew for every group of 16 kernel rows, on
# the kernel-weighted mean at the group's first row, so rows long after a
# step are as well conditioned as rows before it (centered once per chunk of
# 128 rows, removing later rows moved a row after a step of 1e4 standard
# deviations by up to 1.6e-7; now at most 6e-13 up to steps of 1e8).


def test_a_later_nan_does_not_change_an_ill_conditioned_row():
    # the case that showed two arithmetic paths: a chunk of kernel rows took
    # the path for NaN samples when any of its rows had one
    x = _level_step(0)
    y = x.copy()
    y[100:] = np.nan
    w = tc.gaussian_weights(120, {"var": 25}, direction="backward")
    a = _quiet(tc.wcorr, x, x, w, causal=True)
    b = _quiet(tc.wcorr, y, y, w, causal=True)
    assert np.array_equal(a[:, :, :100], b[:, :, :100], equal_nan=True)
    # without the diagonal, row 40 as NaN changes no row up to 40
    y = x.copy()
    y[40] = np.nan
    kw = dict(causal=True, cfun=tc.autofc, weights_params={"var": 25},
              include_zero=False)
    assert np.array_equal(_quiet(tc.timecorr, x, **kw)[:41],
                          _quiet(tc.timecorr, y, **kw)[:41], equal_nan=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("where", [0, 60, 150])
@pytest.mark.parametrize("case", ["autofc", "isfc-null", "wisfc-corrmean", "smooth"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("step", [1e4, 1e8])
def test_rows_of_a_level_step_do_not_depend_on_later_rows(case, where, include_zero,
                                                          mode, missing, step):
    data = [_level_step(where, seed=s, n=T, K=4, step=step)
            for s in range(CASES[case][1])]
    if missing:
        for s, x in enumerate(data):
            x[np.random.default_rng(50 + s).random(x.shape) < 0.01] = np.nan
    full = _call(data, case, "gaussian", include_zero)
    rng = np.random.default_rng(31)
    for n in (1, 2, 3, 40, 61, 100, 127, 128, 129, 151, 200, 257, 299):
        part = _call([_perturb(x, n, mode, rng) for x in data], case, "gaussian",
                     include_zero)
        rows = n if include_zero or mode == "remove" else n + 1
        if mode == "remove":
            # (smoothed values are as large as the step: a few of their ulps)
            atol = max(ATOL, 8 * np.spacing(step)) if case == "smooth" else ATOL
            assert _max_difference(full, part, rows) <= atol
        else:
            # replacement changes no bit of an earlier row
            for f, p in zip(_as_list(full), _as_list(part)):
                assert np.array_equal(np.asarray(f)[:rows], np.asarray(p)[:rows],
                                      equal_nan=True)


@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("case", sorted(CASES))
def test_replacing_later_rows_with_nan_or_inf_changes_no_bit(case, include_zero, kernel):
    data = _subjects(CASES[case][1])
    full = _call(data, case, kernel, include_zero)
    rng = np.random.default_rng(19)
    for mode in ("nan", "inf"):
        for n in CUTS:
            part = _call([_perturb(x, n, mode, rng) for x in data], case, kernel,
                         include_zero)
            rows = n if include_zero else n + 1
            for f, p in zip(_as_list(full), _as_list(part)):
                assert np.array_equal(np.asarray(f)[:rows], np.asarray(p)[:rows],
                                      equal_nan=True)


# ----------------------------------------------------------------------------
# every prefix of a series runs
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_prefix_runs_and_equals_the_first_rows(case, include_zero, kernel):
    data = _subjects(CASES[case][1], seed=60)
    full = _call(data, case, kernel, include_zero)
    for n in list(range(1, 13)) + [40, 128, 129, 200]:
        part = _call([x[:n] for x in data], case, kernel, include_zero)
        assert all(np.asarray(p).shape[0] == n for p in _as_list(part))
        assert _max_difference(full, part, n) <= ATOL
    one = _as_list(_call([x[:1] for x in data], case, kernel, include_zero))
    if CASES[case][0] is not None:
        assert all(np.isnan(p).all() for p in one)


# name: (number of subjects, events).  Boundaries at or beyond the end of a
# prefix are times it has not seen: the same ``events`` are passed to every
# prefix.
EVENTS = {
    "shared": (1, [1, 5, 90, 210]),
    "short-blocks": (1, [1, 2, 4, 7, 8, 150]),
    "per-subject": (3, [[1, 60], None, [130, 131, 250]]),
}


@pytest.mark.parametrize("include_zero", [True, False])
@pytest.mark.parametrize("cfun", ["autofc", "isfc", "smooth"])
@pytest.mark.parametrize("events", sorted(EVENTS))
def test_every_prefix_runs_with_the_events_of_the_whole_series(events, cfun,
                                                               include_zero):
    n_subjects, spec = EVENTS[events]
    data = _subjects(n_subjects, seed=61)
    data = data[0] if n_subjects == 1 else data
    kw = dict(causal=True, cfun={"autofc": tc.autofc, "isfc": tc.isfc,
                                 "smooth": None}[cfun],
              weights_params={"var": 25}, include_zero=include_zero, events=spec)
    full = _quiet(tc.timecorr, data, **kw)
    for n in list(range(1, 13)) + [60, 91, 131, 211, 251]:
        part = _quiet(tc.timecorr, [x[:n] for x in _as_list(data)]
                      if n_subjects > 1 else data[:n], **kw)
        assert all(np.asarray(p).shape[0] == n for p in _as_list(part))
        assert _max_difference(full, part, n) <= ATOL
    # the boundaries the series has reached restrict its kernel
    last = np.asarray(_as_list(full)[0])
    if cfun != "smooth":
        assert np.isnan(last[1:3]).all()


def test_events_beyond_the_series_are_an_error_without_causal():
    x = np.random.default_rng(62).standard_normal((40, 3))
    with pytest.raises(ValueError, match=r"boundaries must lie in \[0, 40\)"):
        tc.timecorr(x, events=[10, 40], weights_params={"var": 25})
    with pytest.raises(ValueError, match="no block is long enough"):
        tc.timecorr(x[:2], events=[1], weights_params={"var": 25},
                    include_timepoints="backward")
    # with causal=True: ignored, and rows without 3 samples are NaN
    out = _quiet(tc.timecorr, x, causal=True, events=[10, 40, 4000],
                 weights_params={"var": 25})
    same = _quiet(tc.timecorr, x, causal=True, events=[10], weights_params={"var": 25})
    assert np.array_equal(out, same, equal_nan=True)
    assert np.isnan(_quiet(tc.timecorr, x[:2], causal=True, events=[1],
                           weights_params={"var": 25})).all()
    # negative boundaries are an error either way
    with pytest.raises(ValueError, match="boundaries must lie in"):
        tc.timecorr(x, causal=True, events=[-1, 10], weights_params={"var": 25})


def test_one_row_of_a_wide_kernel_is_not_a_delta_kernel():
    x = np.random.default_rng(70).standard_normal((30, 3))
    # as arrays and through the correlation functions
    for n in (1, 2, 3):
        w = tc.gaussian_weights(n, {"var": 25}, direction="backward")
        assert np.isnan(_quiet(tc.wcorr, x[:n], x[:n], w, causal=True)[:, :, :2]).all()
        assert np.isnan(_quiet(tc.autofc, x[:n], w, causal=True)[:2]).all()
        assert np.isnan(_quiet(tc.isfc, [x[:n], x[:n] + 1], w, causal=True)[0][:2]).all()
        w0 = tc.gaussian_weights(n, {"var": 25}, direction="backward",
                                 include_zero=False)
        assert np.isnan(_quiet(tc.wcorr, x[:n], x[:n], w0, causal=True)).all()
    # kernels that are delta kernels at every length still raise
    with pytest.raises(ValueError, match="delta"):
        tc.timecorr(x, causal=True, cfun=tc.autofc, weights_function=tc.boxcar_weights,
                    weights_params={"width": 1})
    with pytest.raises(ValueError, match="delta"):
        tc.timecorr(x[:1], causal=True, cfun=tc.autofc,
                    weights_function=tc.boxcar_weights, weights_params={"width": 1})
    with pytest.raises(ValueError, match="delta"):
        tc.wcorr(x[:2], x[:2], np.eye(2), causal=True)


# ----------------------------------------------------------------------------
# the reach of a restricted kernel, and of kernels without a limit
# ----------------------------------------------------------------------------


def _restricted_reach(function, params, exclude, include_zero=True, n=4000):
    """The largest effective sample size a row of the restricted kernel has
    with up to ``n - 1`` earlier timepoints."""
    kernel = helpers._lag_kernel(n, function, params, "backward", include_zero, exclude)
    # the last row, by lag: over the lags up to each one
    w = np.abs(kernel.block(n - 1, n, 0, n))[0][::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ess = np.cumsum(w) ** 2 / np.cumsum(w * w)
    return float(np.nanmax(ess)) if not np.isnan(ess).all() else float("nan")


def _kernel_reach(function, params, include_zero=True, exclude=None):
    return helpers._kernel_reach(
        helpers._lag_kernel(5, function, params, "backward", include_zero, exclude))


def test_a_kernel_of_infinite_width_has_no_limit():
    for include_zero in (True, False):
        assert _kernel_reach(tc.boxcar_weights, {"width": np.inf}, include_zero) is None
        assert _kernel_reach(tc.uniform_weights, None, include_zero) is None
        # the lags from 4 on: still without a limit
        assert _kernel_reach(tc.uniform_weights, None, include_zero, -3) is None
        assert _kernel_reach(tc.boxcar_weights, {"width": np.inf}, include_zero,
                             -3) is None
    x = np.random.default_rng(30).standard_normal((50, 3))
    kw = dict(causal=True, cfun=tc.autofc, weights_function=tc.boxcar_weights,
              weights_params={"width": np.inf})
    # more than these 50 rows hold, not more than a row can have: NaN rows
    assert np.isnan(_quiet(tc.timecorr, x, min_samples=2000, **kw)).all()
    out = _quiet(tc.timecorr, x, min_samples=30, **kw)
    assert np.array_equal(np.flatnonzero(np.isnan(out).all(axis=1)), np.arange(29))
    feats = _quiet(tc.features, x, order=1, rfun=None, delta_rfun=None,
                   delta_weights_function=tc.boxcar_weights,
                   delta_weights_params={"width": np.inf}, min_samples=2000, **kw)
    assert np.isnan(feats[1]).all()


@pytest.mark.parametrize("function,params", [
    (tc.uniform_weights, None), (tc.boxcar_weights, {"width": np.inf}),
    (tc.boxcar_weights, {"width": 30}), (tc.gaussian_weights, {"var": 25}),
    (tc.laplace_weights, {"scale": 4.0}), (tc.t_weights, {"df": 2}),
    (tc.mexican_hat_weights, {"sigma": 6.0}),
], ids=lambda v: getattr(v, "__name__", str(v)))
@pytest.mark.parametrize("exclude", [3, 0, -3, 40, -40])
@pytest.mark.parametrize("include_zero", [True, False])
def test_reach_of_a_kernel_restricted_by_exclude_timepoints(function, params, exclude,
                                                            include_zero):
    reach = _kernel_reach(function, params, include_zero, exclude)
    flat = function is tc.uniform_weights or params == {"width": np.inf}
    if flat and exclude < 0:
        assert reach is None  # every lag beyond: no limit
        return
    heavy = function is tc.t_weights and exclude < 0
    most = _restricted_reach(function, params, exclude, include_zero,
                             n=200_000 if heavy else 4000)
    if np.isnan(most):
        most = 0.0  # no lag with weight (the lag 0 alone, without the diagonal)
    assert reach == pytest.approx(most, rel=1e-3 if heavy else 1e-9, abs=1e-12)
    if flat and exclude >= 0:
        assert reach == exclude + 1 - (not include_zero)
    if exclude == 0:
        return  # the lag 0 alone: a delta kernel, which causal mode refuses
    # the decision of timecorr: a number above it raises, one below does not
    x = np.random.default_rng(31).standard_normal((60, 3))
    kw = dict(causal=True, cfun=tc.autofc, weights_function=function,
              weights_params={} if params is None else params,
              include_zero=include_zero, exclude_timepoints=exclude)
    with pytest.raises(ValueError, match="min_samples.*more than a row"):
        tc.timecorr(x, min_samples=max(1.0, reach) * (1 + 1e-6) + 1e-9, **kw)
    with pytest.raises(ValueError, match="min_samples.*more than a row"):
        tc.timecorr(x, cfun=None, causal=True, weights_function=function,
                    weights_params={} if params is None else params,
                    include_zero=include_zero, exclude_timepoints=exclude,
                    min_samples=max(1.0, reach) * (1 + 1e-6) + 1e-9)
    if reach >= 1:
        _quiet(tc.timecorr, x, min_samples=reach * (1 - 1e-6), **kw)


def test_the_uniform_kernel_restricted_to_four_lags_cannot_reach_five():
    x = np.random.default_rng(32).standard_normal((40, 3))
    kw = dict(causal=True, cfun=tc.autofc, weights_function=tc.uniform_weights,
              weights_params={}, exclude_timepoints=3)
    with pytest.raises(ValueError, match=r"min_samples=5 .*is 4\b"):
        tc.timecorr(x, min_samples=5, **kw)
    out = _quiet(tc.timecorr, x, min_samples=4, **kw)
    assert np.array_equal(np.flatnonzero(np.isnan(out).all(axis=1)), np.arange(3))


def test_reach_of_wide_kernels_with_exclude_timepoints():
    # wider than the lags that are added up, and restricted
    assert _kernel_reach(tc.uniform_weights, None, exclude=1e8) == 1e8 + 1
    assert _kernel_reach(tc.uniform_weights, None, False, 1e8) == 1e8
    assert _kernel_reach(tc.boxcar_weights, {"width": np.inf}, exclude=1e8) == 1e8 + 1
    # the lags from 6 on of the 5e8 of a boxcar
    assert _kernel_reach(tc.boxcar_weights, {"width": 1e9}, exclude=-5) == 5e8 - 6
    # the first 11 lags of a kernel that is flat over them
    assert _kernel_reach(tc.gaussian_weights, {"var": 1e14}, exclude=10) == pytest.approx(
        11, rel=1e-9)
    assert _kernel_reach(tc.laplace_weights, {"scale": 1e9}, exclude=10) == pytest.approx(
        11, rel=1e-6)
    # without its first lags a wide kernel reaches nearly what it reaches whole
    whole = _kernel_reach(tc.gaussian_weights, {"var": 1e14})
    assert _kernel_reach(tc.gaussian_weights, {"var": 1e14}, exclude=-10) == pytest.approx(
        whole, rel=1e-5)
    assert _kernel_reach(tc.gaussian_weights, {"var": 1e14}, exclude=-10) < whole
    # restricted beyond the lags added up: at most the kernel's own reach,
    # at least the reach of the lags added up
    for function, params in ((tc.boxcar_weights, {"width": 1e9}),
                             (tc.gaussian_weights, {"var": 1e18}),
                             (tc.t_weights, {"df": 1})):
        bound = _kernel_reach(function, params, exclude=1e8)
        assert bound is not None and bound <= _kernel_reach(function, params)
        assert bound >= _kernel_reach(function, params, exclude=1e5) * (1 - 1e-9)
    assert _kernel_reach(tc.boxcar_weights, {"width": 1e9}, exclude=1e8) == pytest.approx(
        1e8 + 1, rel=1e-13)
    # only lags beyond those added up
    assert _kernel_reach(tc.boxcar_weights, {"width": 1e9}, exclude=-1e8) == 4e8 - 1
    assert _kernel_reach(tc.boxcar_weights, {"width": 100}, exclude=-1e8) == 0
    far = _kernel_reach(tc.gaussian_weights, {"var": 1e18}, exclude=-1e8)
    assert far == pytest.approx(_kernel_reach(tc.gaussian_weights, {"var": 1e18}),
                                rel=0.2)
    assert _kernel_reach(tc.gaussian_weights, {"var": 25}, exclude=-1e8) == 0
    assert _kernel_reach(tc.t_weights, {"df": 1}, exclude=-1e8) > 1e8


def test_reach_of_a_very_wide_mexican_hat_is_known():
    # the profile depends on lag / sigma: the reach grows in proportion
    narrow = _kernel_reach(tc.mexican_hat_weights, {"sigma": 5000.0})
    for sigma in (5e6, 5e9, 5e15):
        wide = _kernel_reach(tc.mexican_hat_weights, {"sigma": sigma})
        assert wide is not None
        assert narrow * sigma / 5000 * 0.98 <= wide <= narrow * sigma / 5000 * 1.03


@pytest.mark.parametrize("function,params", [
    (tc.gaussian_weights, {"var": np.inf}), (tc.laplace_weights, {"scale": np.inf}),
    (tc.t_weights, {"df": np.inf}), (tc.gaussian_weights, {"var": np.nan}),
    (tc.mexican_hat_weights, {"sigma": np.inf}),
], ids=str)
def test_reach_with_parameters_that_are_not_finite_never_fails(function, params):
    with np.errstate(all="ignore"):
        w = np.asarray(helpers._lag_kernel(50, function, params, "backward")
                       .block(0, 50, 0, 50))
        reach = _kernel_reach(function, params)
    if not np.any(np.nan_to_num(w, nan=0.0, posinf=0.0)):
        assert reach == 0  # a kernel without weight reaches nothing
    else:
        assert reach is None or reach > 0
