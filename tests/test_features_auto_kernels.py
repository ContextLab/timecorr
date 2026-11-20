"""
Automatic kernels of ``features`` and ``two_path_step`` (``weights_params="auto"``,
``delta_weights_params="auto"``, the defaults), the lower rate at which
``features`` passes feeding features on, and the signed feeding reducer of
orders 2 and above.

The rule: the feeding kernel of a step is a Gaussian whose standard deviation
is 5 autocorrelation widths of the step's input (at most T / 8); the analysis
kernel's is that or 10 samples, whichever is larger.  With both automatic and
no events, ``features`` passes each order's feeding features on at every
``floor(sd / 2)``-th timepoint (keeping at least 100 timepoints) and returns
every output at all T timepoints by linear interpolation.

Everything is computed for real on simulated data; nothing is mocked.
"""

import inspect
import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

import timecorr as tc
from timecorr.higher_order import (
    _at_every_timepoint,
    _auto_sd,
    _autocorrelation_width,
)

LEGACY = dict(rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)
OLD_KERNELS = dict(weights_params=None, delta_weights_params=None)


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


# the autocorrelation width and the kernel rule --------------------------------


@pytest.mark.parametrize("function", [tc.features, tc.two_path_step])
def test_signature_defaults_are_automatic(function):
    params = inspect.signature(function).parameters
    assert params["weights_params"].default == "auto"
    assert params["delta_weights_params"].default == "auto"


def test_width_of_white_noise_is_one():
    rng = np.random.default_rng(0)
    x = [rng.standard_normal((500, 8)) for _ in range(2)]
    assert _autocorrelation_width(x) == 1.0
    assert _auto_sd(x) == 5.0


@pytest.mark.parametrize("sd", [3, 8, 20])
def test_width_of_smoothed_noise(sd):
    # white noise smoothed with a Gaussian of standard deviation sd has a
    # Gaussian autocorrelation of standard deviation sqrt(2) sd
    x = [smooth_noise(s, 6000, 6, sd) for s in range(2)]
    assert _autocorrelation_width(x) == pytest.approx(np.sqrt(2) * sd, rel=0.1)
    assert _auto_sd(x) == pytest.approx(5 * np.sqrt(2) * sd, rel=0.1)


def test_width_ignores_constant_columns_and_nan_samples():
    x = smooth_noise(1, 3000, 4, 6)
    w = _autocorrelation_width([x])
    with_constant = np.column_stack([x, np.full(len(x), 3.0)])
    assert _autocorrelation_width([with_constant]) == w
    holes = x.copy()
    holes[::97, 0] = np.nan
    assert _autocorrelation_width([holes]) == pytest.approx(w, rel=0.05)
    assert _autocorrelation_width([np.ones((50, 3))]) == 1.0


def test_kernel_is_at_most_an_eighth_of_the_recording():
    x = [smooth_noise(2, 400, 5, 30)]
    assert _auto_sd(x) == 400 / 8


def test_step_on_white_data_uses_the_old_default_kernels():
    data, _ = hierarchy()
    X, D = quiet(tc.two_path_step, data, cfun=tc.autofc)
    Xo, Do = quiet(tc.two_path_step, data, cfun=tc.autofc, **OLD_KERNELS)
    Xe, De = quiet(tc.two_path_step, data, cfun=tc.autofc,
                   weights_params={"var": 100}, delta_weights_params={"var": 25})
    assert_same(X, Xo)
    assert_same(D, Do)
    assert_same(X, Xe)
    assert_same(D, De)


def test_step_on_smooth_input_follows_the_rule():
    data, _ = hierarchy()
    _, D1 = quiet(tc.two_path_step, data, cfun=tc.autofc)
    sd = _auto_sd(D1)
    assert sd > 25  # D1 was smoothed by a kernel of sd 5: width above 5 sqrt(2)
    X, D = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2)
    Xe, De = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2,
                   weights_params={"var": sd ** 2},
                   delta_weights_params={"var": sd ** 2})
    assert_same(X, Xe)
    assert_same(D, De)
    # and the old kernels (sd 10 and 5) give something else
    Xo, _ = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2, **OLD_KERNELS)
    assert not np.allclose(X[0], Xo[0])


def test_one_path_automatic_the_other_given():
    data, _ = hierarchy()
    _, D1 = quiet(tc.two_path_step, data, cfun=tc.autofc)
    sd = _auto_sd(D1)
    X, D = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2,
                 weights_params={"var": 64})
    Xe, De = quiet(tc.two_path_step, D1, cfun=tc.autofc, order=2,
                   weights_params={"var": 64}, delta_weights_params={"var": sd ** 2})
    assert_same(X, Xe)
    assert_same(D, De)


def test_auto_with_another_analysis_kernel_is_that_kernels_defaults():
    data, _ = hierarchy(T=300)
    X = quiet(tc.features, data, order=1, cfun=tc.autofc,
              weights_function=tc.laplace_weights)
    Xn = quiet(tc.features, data, order=1, cfun=tc.autofc,
               weights_function=tc.laplace_weights, weights_params=None)
    assert_same(X[1], Xn[1])


def test_original_pipeline_keeps_the_kernel_defaults():
    x = [smooth_noise(s, 300, 6, 4) for s in range(2)]  # not white: width > 1
    assert _autocorrelation_width(x) > 1
    X, D = tc.features(x, order=2, return_delta=True, **LEGACY)
    Xn, Dn = tc.features(x, order=2, return_delta=True, weights_params=None, **LEGACY)
    for k in range(3):
        assert_same(X[k], Xn[k])
        assert_same(D[k], Dn[k])


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
    data, _ = hierarchy(T=3000)
    T = 3000
    (X, D), info = quiet(tc.features, data, order=3, cfun=tc.autofc,
                         return_delta=True, return_info=True)
    rows = np.arange(T)
    fed = data
    for k in (1, 2, 3):
        sd = _auto_sd(fed)
        x, d = quiet(tc.two_path_step, fed, cfun=tc.autofc, order=k)
        for got, step in ((X[k], x), (D[k], d)):
            assert len(got) == len(step)
            for g, s in zip(got, step):
                assert g.shape == (T, s.shape[1])
                np.testing.assert_allclose(g, _at_every_timepoint(s, rows, T),
                                           rtol=1e-9, atol=1e-12)
        entries = [e for e in info if e["order"] == k]
        assert [e["path"] for e in entries] == ["analysis", "feeding"]
        assert entries[1]["kernel_sd"] == pytest.approx(sd)
        assert entries[0]["kernel_sd"] == pytest.approx(max(sd, 10.0))
        assert entries[0]["sampling"] == entries[1]["sampling"] == rows[1] - rows[0]
        stride = max(1, min(int(sd // 2), len(rows) // 100))
        fed = [u[::stride] for u in d]
        rows = rows[::stride]
    # order 1 on white data: sd 5, passed on at every 2nd timepoint
    assert [e["sampling"] for e in info if e["path"] == "analysis"][:2] == [1, 2]
    assert len(rows) >= 100


def test_short_recordings_are_not_thinned_below_100_timepoints():
    data, _ = hierarchy(T=300)
    (X, D), info = quiet(tc.features, data, order=3, cfun=tc.autofc,
                         return_delta=True, return_info=True)
    assert all(x.shape[0] == 300 for k in (1, 2, 3) for x in X[k] + D[k])
    sampling = {e["order"]: e["sampling"] for e in info}
    # 300 -> 150 timepoints after order 1; never below 100
    assert sampling == {1: 1, 2: 2, 3: 2}


def test_given_kernels_or_events_keep_every_timepoint():
    data, _ = hierarchy(T=600)
    for kwargs in (dict(weights_params={"var": 100}),
                   dict(delta_weights_params={"var": 25}),
                   dict(events=[300])):
        (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc,
                             return_delta=True, return_info=True, **kwargs)
        assert all(e.get("sampling", 1) == 1 for e in info), kwargs
        # the chain of steps at the full rate
        x1, d1 = quiet(tc.two_path_step, data, cfun=tc.autofc, **kwargs)
        x2, d2 = quiet(tc.two_path_step, d1, cfun=tc.autofc, order=2, **kwargs)
        assert_same(X[1], x1)
        assert_same(D[1], d1)
        assert_same(X[2], x2)
        assert_same(D[2], d2)


def test_info_has_kernel_keys_only_for_automatic_kernels():
    data, _ = hierarchy(T=600)
    _, info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_info=True,
                    weights_params={"var": 100})
    for entry in info:
        assert ("kernel_sd" in entry) == (entry["path"] == "feeding")
        assert ("sampling" in entry) == (entry["path"] == "feeding")
    _, info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_info=True,
                    **OLD_KERNELS)
    assert not any("kernel_sd" in e or "sampling" in e for e in info)


def test_old_kernels_requested_explicitly():
    # weights_params=None and delta_weights_params=None are the kernels'
    # defaults at every order (variance 100 and 25), at every timepoint
    data, _ = hierarchy(T=600)
    X, D = quiet(tc.features, data, order=2, cfun=tc.autofc, return_delta=True,
                 **OLD_KERNELS)
    Xe, De = quiet(tc.features, data, order=2, cfun=tc.autofc, return_delta=True,
                   weights_params={"var": 100},
                   delta_weights_function=tc.gaussian_weights,
                   delta_weights_params={"var": 25})
    for k in range(3):
        assert_same(X[k], Xe[k])
        assert_same(D[k], De[k])


# the signed feeding reducer -------------------------------------------------


def test_signed_only_communities_are_the_signed_columns():
    data, _ = hierarchy(T=400)
    corrs = tc.timecorr(data, cfun=tc.autofc, rfun=None, weights_params={"var": 25})
    both = tc.Communities(n=12, signed=True)(corrs)
    only = tc.Communities(n=12, signed="only")(corrs)
    for b, o in zip(both, only):
        assert o.shape == (400, 12)
        np.testing.assert_array_equal(o, b[:, 12:])
    assert repr(tc.Communities(signed="only")) == "Communities(n=None, signed='only', n_null=5)"
    assert repr(tc.Communities()) == "Communities(n=None, signed=False, n_null=5)"
    with pytest.raises(ValueError, match='signed must be True, False or "only"'):
        tc.Communities(signed="yes")


def test_automatic_feeding_reducer_is_signed_from_order_two():
    data, _ = hierarchy(T=1500)
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc,
                         return_delta=True, return_info=True, **OLD_KERNELS)
    feeding = {e["order"]: e for e in info if e["path"] == "feeding"}
    assert feeding[1]["effective"] == repr(tc.Communities())
    assert feeding[1]["fallback"] is None and feeding[1]["n_communities"] == 12
    assert (D[1][0] >= 0).all()  # mean squared correlations
    # order 2: 12 module columns in 3 top groups give 3 community columns,
    # fewer than 4, so the step falls back to weighted strength
    assert feeding[2]["fallback"] == "too_few_community_columns"
    # with the reducer given, the columns are signed means (some negative)
    signed = tc.Communities(n=3, signed="only")
    _, d2 = quiet(tc.two_path_step, D[1], cfun=tc.autofc, order=2,
                  delta_rfun=signed, **OLD_KERNELS)
    assert d2[0].shape == (1500, 3)
    assert (d2[0] < 0).any()
    expected = tc.Communities(n=3, signed=True)(
        tc.timecorr(D[1], cfun=tc.autofc, rfun=None, weights_params={"var": 25}))
    np.testing.assert_allclose(d2[0], expected[0][:, 3:], rtol=1e-9, atol=1e-12)


def test_automatic_feeding_reducer_at_order_two_emits_signed_columns():
    # 4 top groups: 4 community columns at order 2, so no fallback
    data, _ = tc.simulate_hierarchy(S=2, T=3000, nodes_per_module=6, branching=(4,),
                                    n_top=4, taus=(5, 100), set_random_seed=1)
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc,
                         return_delta=True, return_info=True)
    entry = [e for e in info if e["order"] == 2 and e["path"] == "feeding"][0]
    assert entry["effective"] == repr(tc.Communities(signed="only"))
    assert entry["fallback"] is None
    assert D[2][0].shape == (3000, entry["n_communities"])
