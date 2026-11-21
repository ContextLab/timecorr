"""
Automatic kernels of ``features`` and ``two_path_step`` (``weights_params=None``
and ``delta_weights_params=None``, the defaults, with Gaussian kernels on both
paths) and the lower rate at which ``features`` passes feeding features on.

The rule: the feeding kernel of a step is a Gaussian whose standard deviation
is 5 autocorrelation widths of the step's input, at most a hundredth of its
rows and at least 5; the analysis kernel's is that or 10, whichever is larger.
When no kernel parameters and no events are given, ``features`` passes each
order's feeding features on at every ``floor(sd / 2)``-th row while at least
1000 rows remain, and returns every output at all T timepoints by linear
interpolation.

Everything is computed for real on simulated data; nothing is mocked.
"""

import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

import timecorr as tc
from timecorr.higher_order import (
    _at_every_timepoint,
    _auto_sd,
    _autocorrelation_width,
    estimate_memory_gb,
)

LEGACY = dict(rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)
# the fixed kernels that were the defaults before the automatic ones
FIXED = dict(weights_params={"var": 100}, delta_weights_params={"var": 25})


def quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return f(*args, **kwargs)


def hierarchy(seed=0, T=1500, S=2):
    """72 nodes: 3 top groups x 4 modules x 6 nodes; timescales 5 and 60
    (5 and T // 8 for T < 480)."""
    data, truth = tc.simulate_hierarchy(
        S=S, T=T, nodes_per_module=6, branching=(4,), n_top=3,
        taus=(5, min(60, T // 8)),
        set_random_seed=seed,
    )
    return data, truth


def smooth_noise(seed, T, K, sd):
    rng = np.random.default_rng(seed)
    return gaussian_filter1d(rng.standard_normal((T + 8 * sd, K)), sd, axis=0)[4 * sd:-4 * sd]


def assert_same(a, b):
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.shape == y.shape
        np.testing.assert_array_equal(x, y)


def gaussian(sd):
    return {"var": float(sd) ** 2}


# the autocorrelation width and the kernel rule --------------------------------


def test_width_of_white_noise_is_one():
    rng = np.random.default_rng(0)
    x = [rng.standard_normal((500, 8)) for _ in range(2)]
    assert _autocorrelation_width(x) == 1.0
    assert _auto_sd(x) == 5.0


@pytest.mark.parametrize("sd", [3, 8, 20])
def test_width_of_smoothed_noise(sd):
    # white noise smoothed with a Gaussian of standard deviation sd has a
    # Gaussian autocorrelation of standard deviation sqrt(2) sd
    x = [smooth_noise(s, 20000, 6, sd) for s in range(2)]
    assert _autocorrelation_width(x) == pytest.approx(np.sqrt(2) * sd, rel=0.1)
    assert _auto_sd(x) == pytest.approx(5 * np.sqrt(2) * sd, rel=0.1)


def test_width_ignores_constant_columns_and_nan_samples():
    x = smooth_noise(1, 6000, 4, 6)
    w = _autocorrelation_width([x])
    with_constant = np.column_stack([x, np.full(len(x), 3.0)])
    assert _autocorrelation_width([with_constant]) == w
    # constants whose mean is not exact in binary (see also
    # test_constant_columns_of_any_value_are_left_out)
    with_constants = np.column_stack([x, np.full((len(x), 3), 0.1), np.full(len(x), -7.7)])
    assert _autocorrelation_width([with_constants]) == w
    holes = x.copy()
    holes[::97, 0] = np.nan
    assert _autocorrelation_width([holes]) == pytest.approx(w, rel=0.05)
    # NaN samples in every column: the columns still count
    every = x.copy()
    for column in range(4):
        every[column::97, column] = np.nan
    assert _autocorrelation_width([every]) == pytest.approx(w, rel=0.05)
    assert w > 5
    assert _autocorrelation_width([np.ones((50, 3))]) == 1.0


@pytest.mark.parametrize("T, expected", [(400, 5.0), (800, 8.0), (3000, 30.0)])
def test_kernel_is_at_most_a_hundredth_of_the_recording_and_at_least_5(T, expected):
    # smooth input (5 widths would be about 210 rows): the kernel is T / 100
    # rows, and 5 rows in recordings of fewer than 500 rows
    x = [smooth_noise(2, T, 5, 30)]
    assert _auto_sd(x) == expected


@pytest.mark.parametrize("T", [20, 39, 40, 300])
def test_white_data_get_the_fixed_kernels_at_every_length(T):
    # also below 40 timepoints, where only a given feeding reducer runs
    rng = np.random.default_rng(T)
    x = [rng.standard_normal((T, 6)) for _ in range(2)]
    common = dict(order=1, cfun=tc.autofc, rfun="weighted_strength",
                  delta_rfun="weighted_strength", return_delta=True)
    X, D = quiet(tc.features, x, **common)
    Xe, De = quiet(tc.features, x, **common, **FIXED)
    assert_same(X[1], Xe[1])
    assert_same(D[1], De[1])


def test_step_on_white_data_uses_the_fixed_kernels():
    data, _ = hierarchy()
    X, D = quiet(tc.two_path_step, data, cfun=tc.autofc)
    Xn, Dn = quiet(tc.two_path_step, data, cfun=tc.autofc, weights_params=None,
                   delta_weights_params=None)
    Xe, De = quiet(tc.two_path_step, data, cfun=tc.autofc, **FIXED)
    assert_same(X, Xn)
    assert_same(D, Dn)
    assert_same(X, Xe)
    assert_same(D, De)


def test_step_on_smooth_input_follows_the_rule():
    data, _ = hierarchy(T=6000)
    _, D1 = quiet(tc.two_path_step, data, cfun=tc.autofc)
    sd = _auto_sd(D1)
    # D1 was smoothed by a kernel of sd 5 (width above 5 sqrt(2)); 5 widths
    # are below the cap of 6000 / 100 rows
    assert 35 < sd < 60
    X, D = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2)
    Xe, De = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2,
                   weights_params=gaussian(sd), delta_weights_params=gaussian(sd))
    assert_same(X, Xe)
    assert_same(D, De)
    # and the fixed kernels (sd 10 and 5) give something else
    Xo, Do = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2, **FIXED)
    assert not np.allclose(X[0], Xo[0])
    assert not np.allclose(D[0], Do[0])


def test_one_path_automatic_the_other_given():
    data, _ = hierarchy(T=6000)
    _, D1 = quiet(tc.two_path_step, data, cfun=tc.autofc)
    sd = _auto_sd(D1)
    X, D = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2,
                 weights_params={"var": 64})
    Xe, De = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2,
                   weights_params={"var": 64}, delta_weights_params=gaussian(sd))
    assert_same(X, Xe)
    assert_same(D, De)
    X, D = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2,
                 delta_weights_params={"var": 64})
    Xe, De = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2,
                   weights_params=gaussian(sd), delta_weights_params={"var": 64})
    assert_same(X, Xe)
    assert_same(D, De)


def test_autocorrelated_raw_data_get_wider_order_one_kernels():
    x = [smooth_noise(s, 3000, 8, 3) for s in range(2)]
    sd = _auto_sd(x)
    assert sd == pytest.approx(5 * np.sqrt(2) * 3, rel=0.1)  # about 21 samples
    common = dict(order=1, cfun=tc.autofc, rfun="weighted_strength",
                  delta_rfun="weighted_strength", return_delta=True)
    X, D = quiet(tc.features, x, **common)
    Xe, De = quiet(tc.features, x, **common, weights_params=gaussian(sd),
                   delta_weights_params=gaussian(sd))
    assert_same(X[1], Xe[1])
    assert_same(D[1], De[1])
    Xo, Do = quiet(tc.features, x, **common, **FIXED)
    assert not np.allclose(D[1][0], Do[1][0])


def test_nan_samples_under_the_defaults():
    data, _ = hierarchy(T=600)
    holes = [x.copy() for x in data]
    holes[0][50:60, 3] = np.nan
    holes[1][::41, 10] = np.nan
    assert _auto_sd(holes) == 5.0
    X, D = quiet(tc.features, holes, order=2, cfun=tc.autofc, return_delta=True)
    Xe, De = quiet(tc.features, holes, order=2, cfun=tc.autofc, return_delta=True,
                   **FIXED)
    for k in (1, 2):
        for x in X[k] + D[k]:
            assert x.shape[0] == 600 and np.isfinite(x).all()
    # T 600: both orders use the fixed kernels' analysis width (see
    # test_short_recordings_keep_the_fixed_analysis_kernel)
    assert_same(X[1], Xe[1])
    assert_same(D[1], De[1])
    assert_same(X[2], Xe[2])


def smooth_data():
    # not white: 5 autocorrelation widths are about 28 samples, so automatic
    # kernels would be 15 samples wide (T / 100)
    x = [smooth_noise(s, 1500, 6, 4) for s in range(2)]
    assert _auto_sd(x) == 15.0
    return x


def test_another_analysis_kernel_keeps_its_own_defaults():
    x = smooth_data()
    X = quiet(tc.features, x, order=1, cfun=tc.autofc, weights_function=tc.laplace_weights)
    Xe = quiet(tc.features, x, order=1, cfun=tc.autofc, weights_function=tc.laplace_weights,
               weights_params=dict(tc.helpers.laplace_params))
    assert_same(X[1], Xe[1])


def test_another_feeding_kernel_keeps_its_own_defaults():
    x = smooth_data()
    common = dict(order=1, cfun=tc.autofc, weights_params={"var": 100},
                  delta_rfun="weighted_strength", return_delta=True)
    _, D = quiet(tc.features, x, delta_weights_function=tc.laplace_weights, **common)
    _, De = quiet(tc.features, x, delta_weights_function=tc.laplace_weights,
                  delta_weights_params=dict(tc.helpers.laplace_params), **common)
    assert_same(D[1], De[1])


def test_original_pipeline_keeps_the_kernel_defaults():
    x = smooth_data()
    X, D = tc.features(x, order=2, return_delta=True, **LEGACY)
    Xe, De = tc.features(x, order=2, return_delta=True, weights_params={"var": 100},
                         **LEGACY)
    for k in range(3):
        assert_same(X[k], Xe[k])
        assert_same(D[k], De[k])


def test_orders_missing_from_a_map_are_automatic():
    data, _ = hierarchy(T=6000)
    common = dict(order=2, cfun=tc.autofc, return_delta=True)
    X, D = quiet(tc.features, data, weights_params={1: {"var": 100}},
                 delta_weights_params={1: {"var": 25}}, **common)
    sd = _auto_sd(D[1])
    assert 35 < sd < 60
    Xe, De = quiet(tc.features, data,
                   weights_params={1: {"var": 100}, 2: gaussian(sd)},
                   delta_weights_params={1: {"var": 25}, 2: gaussian(sd)}, **common)
    Xn, Dn = quiet(tc.features, data, weights_params={1: {"var": 100}, 2: None},
                   delta_weights_params={1: {"var": 25}, 2: None}, **common)
    for k in (1, 2):
        assert_same(X[k], Xe[k])
        assert_same(D[k], De[k])
        assert_same(X[k], Xn[k])
        assert_same(D[k], Dn[k])


# short recordings ------------------------------------------------------------


@pytest.mark.parametrize("cfun", [tc.autofc, tc.isfc])
def test_short_recordings_keep_the_fixed_analysis_kernel(cfun):
    # up to 1000 timepoints the automatic kernels are at most 10 samples wide
    # and every order is computed at every timepoint, so X at every order is
    # what the fixed analysis kernel (variance 100) gives on white data
    data, _ = hierarchy(T=1000, S=3)
    (X, D), info = quiet(tc.features, data, order=2, cfun=cfun, return_delta=True,
                         return_info=True)
    assert all(e["sampling"] == 1 for e in info)
    sd = {(e["order"], e["path"]): e["kernel_sd"] for e in info}
    assert sd[(1, "analysis")] == 10.0 and sd[(1, "feeding")] == 5.0
    assert sd[(2, "analysis")] == 10.0 and sd[(2, "feeding")] == 10.0
    Xe, De = quiet(tc.features, data, order=2, cfun=cfun, return_delta=True,
                   weights_params={"var": 100},
                   delta_weights_params={1: {"var": 25}, 2: {"var": 100}})
    for k in (1, 2):
        assert_same(X[k], Xe[k])
        assert_same(D[k], De[k])


# features: lower rate between orders ---------------------------------------


def test_interpolation_to_every_timepoint():
    rows = np.arange(0, 20, 4)  # 0, 4, 8, 12, 16
    out = np.column_stack([rows.astype(float), rows ** 2.0])
    full = _at_every_timepoint(out, rows, 20)
    assert full.shape == (20, 2)
    np.testing.assert_allclose(full[rows], out)
    np.testing.assert_allclose(full[:17, 0], np.arange(17))  # linear in between
    np.testing.assert_allclose(full[17:, 0], 16)  # held after the last row
    assert _at_every_timepoint(out, np.arange(5), 5) is out
    assert [a.shape for a in _at_every_timepoint([out, out], rows, 20)] == [(20, 2)] * 2


def test_features_equal_a_chain_of_steps_at_lower_rates():
    T = 6000
    data, _ = hierarchy(T=T)
    (X, D), info = quiet(tc.features, data, order=3, cfun=tc.autofc,
                         return_delta=True, return_info=True)
    rows = np.arange(T)
    fed = data
    sampling = []
    for k in (1, 2, 3):
        sd = _auto_sd(fed)
        x, d = quiet(tc.two_path_step, fed, cfun=tc.autofc, order=k)
        for got, step in ((X[k], x), (D[k], d)):
            assert len(got) == len(step)
            for g, s in zip(got, step):
                assert g.shape == (T, s.shape[1])
                # expected: numpy's linear interpolation (ends held), not
                # the package's helper
                assert np.isfinite(s).all()
                expected = np.column_stack(
                    [np.interp(np.arange(T), rows, s[:, j]) for j in range(s.shape[1])])
                np.testing.assert_allclose(g, expected, rtol=1e-9, atol=1e-12)
        entries = [e for e in info if e["order"] == k]
        assert [e["path"] for e in entries] == ["analysis", "feeding"]
        assert entries[1]["kernel_sd"] == pytest.approx(sd)
        assert entries[0]["kernel_sd"] == pytest.approx(max(sd, 10.0))
        assert entries[0]["sampling"] == entries[1]["sampling"] == rows[1] - rows[0]
        sampling.append(int(rows[1] - rows[0]))
        stride = max(1, min(int(sd // 2), len(rows) // 1000))
        fed = [u[::stride] for u in d]
        rows = rows[::stride]
    # order 1 on white data: sd 5, passed on at every 2nd timepoint (3000
    # rows); order 2: at every 3rd of those (1000 rows)
    assert sampling == [1, 2, 6]
    assert len(rows) == 1000


def test_recordings_are_not_thinned_below_1000_rows():
    data, _ = hierarchy(T=1999)
    (X, D), info = quiet(tc.features, data, order=3, cfun=tc.autofc,
                         return_delta=True, return_info=True)
    assert all(x.shape[0] == 1999 for k in (1, 2, 3) for x in X[k] + D[k])
    assert all(e["sampling"] == 1 for e in info)
    # the chain of steps at the full rate
    fed = data
    for k in (1, 2, 3):
        x, fed = quiet(tc.two_path_step, fed, cfun=tc.autofc, order=k)
        assert_same(X[k], x)
        assert_same(D[k], fed)


@pytest.mark.parametrize("kwargs", [
    dict(weights_params={"var": 100}),
    dict(delta_weights_params={"var": 25}),
    dict(weights_params={1: {"var": 100}}),
    dict(events=[1500]),
    dict(events={2: [1500]}),
])
def test_given_kernels_or_events_keep_every_timepoint(kwargs):
    data, _ = hierarchy(T=3000)
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc,
                         return_delta=True, return_info=True, **kwargs)
    assert all(e.get("sampling", 1) == 1 for e in info)
    # the chain of steps at the full rate
    def at(k):
        out = {}
        for name, value in kwargs.items():
            is_map = isinstance(value, dict) and all(isinstance(key, int) for key in value)
            out[name] = value.get(k) if is_map else value
        return out
    x1, d1 = quiet(tc.two_path_step, data, cfun=tc.autofc, **at(1))
    x2, d2 = quiet(tc.two_path_step, d1, cfun=tc.autofc, order=2, **at(2))
    assert_same(X[1], x1)
    assert_same(D[1], d1)
    assert_same(X[2], x2)
    assert_same(D[2], d2)


def test_an_empty_events_map_restricts_nothing():
    # events={} restricts no order: the call is the default call, lower rate
    # between orders included
    data, _ = hierarchy(T=3000)
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc,
                         return_delta=True, return_info=True)
    (Xe, De), info_e = quiet(tc.features, data, order=2, cfun=tc.autofc,
                             return_delta=True, return_info=True, events={})
    assert [e["sampling"] for e in info] == [1, 1, 2, 2]
    assert info_e == info
    for k in range(3):
        assert_same(X[k], Xe[k])
        assert_same(D[k], De[k])


def test_info_has_kernel_keys_only_for_automatic_kernels():
    data, _ = hierarchy(T=600)
    _, info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_info=True,
                    weights_params={"var": 100})
    for entry in info:
        assert ("kernel_sd" in entry) == (entry["path"] == "feeding")
        assert ("sampling" in entry) == (entry["path"] == "feeding")
    _, info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_info=True,
                    weights_params={1: {"var": 100}}, delta_weights_params={2: {"var": 25}})
    keyed = {(e["order"], e["path"]) for e in info if "kernel_sd" in e}
    assert keyed == {(1, "feeding"), (2, "analysis")}
    _, info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_info=True,
                    **FIXED)
    assert not any("kernel_sd" in e or "sampling" in e for e in info)
    _, info = tc.features(data, order=2, return_info=True, **LEGACY)
    assert not any("kernel_sd" in e or "sampling" in e for e in info)


# the feeding reducer ---------------------------------------------------------


def test_automatic_feeding_reducer_is_communities_at_every_order():
    # 4 top groups: 4 community columns at order 2, so no fallback
    data, _ = tc.simulate_hierarchy(S=2, T=3000, nodes_per_module=6, branching=(4,),
                                    n_top=4, taus=(5, 100), set_random_seed=1)
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc,
                         return_delta=True, return_info=True)
    for k in (1, 2):
        entry = [e for e in info if e["order"] == k and e["path"] == "feeding"][0]
        assert entry["effective"] == repr(tc.Communities())
        assert entry["fallback"] is None
        assert D[k][0].shape == (3000, entry["n_communities"])
        assert (D[k][0] >= 0).all()  # mean squared correlations
    assert D[1][0].shape[1] == 16 and D[2][0].shape[1] == 4


# memory -------------------------------------------------------------------------


def test_orders_at_a_lower_rate_are_checked_with_their_rows():
    # 2 subjects of (4000, 12) noise, signed weighted strength on the feeding
    # path: order 2 has 24 input features.  With the fixed kernels it runs
    # on 4000 rows; with the automatic kernels on 2000 rows, where one row
    # of correlations per block needs less.  A budget between the two
    # completes, and is refused only with the fixed kernels (before
    # anything is computed)
    rng = np.random.default_rng(11)
    x = [rng.standard_normal((4000, 12)) for _ in range(2)]
    common = dict(order=2, cfun=tc.autofc, rfun="weighted_strength",
                  delta_rfun=tc.WeightedStrength(signed=True))
    one_row = {T: estimate_memory_gb(24, 1, T, 2, streamed=True)[1] for T in (2000, 4000)}
    assert one_row[4000] > 1.2 * one_row[2000]
    budget = (one_row[2000] + one_row[4000]) / 2
    X, info = quiet(tc.features, x, max_memory_gb=budget, return_info=True, **common)
    assert [e["sampling"] for e in info if e["path"] == "analysis"] == [1, 2]
    assert X[2][0].shape == (4000, 24)
    with pytest.raises(MemoryError, match=r"order 2 .*24 input features, T=4000"):
        quiet(tc.features, x, max_memory_gb=budget, **common, **FIXED)


# recovery -------------------------------------------------------------------------


def test_default_kernels_follow_the_order2_signal_of_a_long_recording():
    # 72 nodes in 12 modules in 4 groups, T 6000, order-2 signal at timescale
    # 100: each column of the top signal is matched with the D[2] column it
    # correlates with most.  Development (seeds 0-3, same call): .75 to .79
    # with the default kernels (order-2 kernel about 47 samples), .24 to .27
    # with the fixed kernels (5 samples)
    data, truth = tc.simulate_hierarchy(S=2, T=6000, nodes_per_module=6, branching=(3,),
                                        n_top=4, taus=(5, 100), set_random_seed=10)
    top = truth["top"][200:-200]

    def followed(**kwargs):
        (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_delta=True,
                             return_info=True, rfun={1: tc.WeightedStrength(signed=True)},
                             **kwargs)
        feeding = [e for e in info if e["order"] == 2 and e["path"] == "feeding"][0]
        assert feeding["n_communities"] == 4 and feeding["fallback"] is None
        scores = []
        for d in D[2]:
            c = np.corrcoef(d[200:-200].T, top.T)[:4, 4:]
            scores.append(np.mean(np.max(np.abs(c), axis=0)))
        return float(np.mean(scores)), feeding

    default, entry = followed()
    assert 35 < entry["kernel_sd"] * entry["sampling"] < 60
    fixed, _ = followed(**FIXED)
    assert default > 0.6
    assert default > fixed + 0.3


# one-sided kernels -----------------------------------------------------------


@pytest.mark.parametrize("side", ["backward", "pre", "forward", "post"])
def test_one_sided_kernels_keep_every_timepoint(side):
    # with a one-sided kernel every order is computed at every timepoint:
    # interpolating between computed rows would read rows on the excluded
    # side
    data, _ = hierarchy(T=3000)
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_delta=True,
                         return_info=True, include_timepoints=side)
    assert all(e["sampling"] == 1 for e in info)
    x1, d1 = quiet(tc.two_path_step, data, cfun=tc.autofc, include_timepoints=side)
    x2, d2 = quiet(tc.two_path_step, d1, cfun=tc.autofc, order=2, include_timepoints=side)
    assert_same(X[2], x2)
    assert_same(D[2], d2)
    # the two-sided default at this length does use the lower rate
    _, info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_info=True)
    assert [e["sampling"] for e in info if e["order"] == 2] == [2]


def test_backward_default_kernels_do_not_read_later_samples():
    # unreduced features (nothing is fitted on the whole recording) of white
    # data with the default kernels and include_timepoints="backward".  The
    # automatic widths are read from the whole recording, so the property is
    # stated given the widths: (1) the default call equals the call with the
    # widths it chose given explicitly, at every timepoint (no interpolation
    # between computed rows); (2) with those widths, replacing every sample
    # from t0 on leaves the features before t0 unchanged (to 1e-8: the
    # estimator centers columns on their overall means,
    # tests/test_estimator_causality.py)
    T, t0 = 3000, 2000
    rng = np.random.default_rng(3)
    x = rng.standard_normal((T, 3))
    y = x.copy()
    y[t0:] = np.random.default_rng(4).standard_normal((T - t0, 3))
    common = dict(order=2, cfun=tc.autofc, rfun=None, delta_rfun=None, return_delta=True,
                  include_timepoints="backward")
    (X, D), info = quiet(tc.features, x, return_info=True, **common)
    assert all(e["sampling"] == 1 for e in info)
    sd = {(e["order"], e["path"]): e["kernel_sd"] for e in info}
    chosen = dict(
        weights_params={k: gaussian(sd[(k, "analysis")]) for k in (1, 2)},
        delta_weights_params={k: gaussian(sd[(k, "feeding")]) for k in (1, 2)},
    )
    Xe, De = quiet(tc.features, x, **common, **chosen)
    Xp, Dp = quiet(tc.features, y, **common, **chosen)
    for k in (1, 2):
        assert_same(X[k], Xe[k])
        assert_same(D[k], De[k])
        for a, b in ((Xe[k], Xp[k]), (De[k], Dp[k])):
            np.testing.assert_allclose(a[:t0], b[:t0], rtol=0, atol=1e-8)
            assert np.abs(a[t0 + 50:] - b[t0 + 50:]).max() > 1e-3


# second outside review --------------------------------------------------------


@pytest.mark.parametrize("value", [0.1, 1 / 3, -7.7, 1e-300])
def test_constant_columns_of_any_value_are_left_out(value):
    # a rounded mean of repeated 0.1 leaves a residual that is the same at
    # every sample; such a column is constant (the estimator's round-off
    # rule), not perfectly autocorrelated: 40 of them next to noise changed
    # the kernel from 5 to 30 rows
    x = np.random.default_rng(2).normal(size=(3000, 4))
    y = np.column_stack([x, np.full((3000, 40), value)])
    assert _auto_sd([x]) == 5.0
    assert _autocorrelation_width([y]) == _autocorrelation_width([x])
    assert _auto_sd([y]) == 5.0


@pytest.mark.parametrize("scale", [1e-180, 1e-100, 1e100, 1e180])
def test_width_does_not_depend_on_the_units_of_the_data(scale):
    x = smooth_noise(4, 3000, 4, 3)
    w = _autocorrelation_width([x])
    assert w > 3
    assert _autocorrelation_width([x * scale]) == pytest.approx(w, rel=1e-9)
    assert _auto_sd([x * scale]) == pytest.approx(_auto_sd([x]), rel=1e-9)


def test_interpolation_keeps_valid_samples_beside_nan():
    # expected values written by hand: a computed row keeps its value, rows
    # after the last computed row hold it, and a row between two computed
    # rows is NaN only when a NaN row has non-zero weight there
    out = np.array([[1.0], [np.nan], [3.0]])
    got = _at_every_timepoint(out, np.array([0, 2, 4]), 6).ravel()
    np.testing.assert_array_equal(got, [1.0, np.nan, np.nan, np.nan, 3.0, 3.0])
    out = np.array([[np.nan, 2.0], [4.0, 6.0]])
    got = _at_every_timepoint(out, np.array([0, 4]), 6)
    np.testing.assert_array_equal(got[:, 0], [np.nan, np.nan, np.nan, np.nan, 4.0, 4.0])
    np.testing.assert_array_equal(got[:, 1], [2.0, 3.0, 4.0, 5.0, 6.0, 6.0])
    # one computed row: held everywhere
    np.testing.assert_array_equal(
        _at_every_timepoint(np.array([[np.nan, 5.0]]), np.array([0]), 3),
        [[np.nan, 5.0]] * 3)


def test_unreduced_runs_are_checked_with_the_rows_of_each_order():
    # order 2 of 4000 rows of white noise runs on 2000 rows: an unreduced
    # call needs about 0.22 GB there, not the 0.41 GB of a step on all 4000
    x = np.random.default_rng(2).normal(size=(4000, 8))
    assert estimate_memory_gb(8, 2, 4000)[2] > 0.4
    assert estimate_memory_gb(36, 1, 2000)[1] < 0.25
    X = quiet(tc.features, x, order=2, cfun=tc.autofc, rfun=None, delta_rfun=None,
              max_memory_gb=0.4)
    assert X[2].shape == (4000, 666)
    # with the fixed kernels every order runs on all rows: refused up front
    with pytest.raises(MemoryError, match="Unreduced features at order 2"):
        quiet(tc.features, x, order=2, cfun=tc.autofc, rfun=None, delta_rfun=None,
              max_memory_gb=0.4, **FIXED)


def test_default_return_info_has_the_automatic_kernel_keys():
    # the default call (automatic kernels on both paths): every entry has the
    # kernel's standard deviation and the sampling, with the values the rule
    # gives for each order's input
    data, _ = hierarchy(T=3000)
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc,
                         return_delta=True, return_info=True)
    base = {"order", "path", "requested", "effective", "input_width", "output_width",
            "kernel_sd", "sampling"}
    feeding = base | {"n_communities", "n_exceed", "null_threshold", "n_null", "fallback"}
    assert [(e["order"], e["path"]) for e in info] == [
        (1, "analysis"), (1, "feeding"), (2, "analysis"), (2, "feeding")]
    for e in info:
        assert set(e) == (feeding if e["path"] == "feeding" else base)
    sd1 = _auto_sd(data)
    assert info[1]["kernel_sd"] == pytest.approx(sd1)
    assert info[0]["kernel_sd"] == pytest.approx(max(sd1, 10.0))
    assert info[0]["sampling"] == info[1]["sampling"] == 1
    stride = max(1, min(int(sd1 // 2), 3000 // 1000))
    fed = [d[::stride] for d in quiet(tc.two_path_step, data, cfun=tc.autofc)[1]]
    sd2 = _auto_sd(fed)
    assert info[3]["kernel_sd"] == pytest.approx(sd2)
    assert info[2]["kernel_sd"] == pytest.approx(max(sd2, 10.0))
    assert info[2]["sampling"] == info[3]["sampling"] == stride == 2

