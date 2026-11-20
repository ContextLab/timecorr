"""
The streaming engine of the squared-correlation summaries
(``timecorr.summaries._streamed_summary``) and the shift null that sets the
number of communities (``_shift_null_threshold``).

Streamed summaries are compared with the materialized route (the correlation
function's full output, then the summary) on ``simulate_nested`` data, with
scattered NaN samples and with a constant column, for a Gaussian kernel and
for per-subject event-masked kernels, at several block sizes.  Everything is
computed for real; nothing is mocked.
"""

import copy
import tracemalloc
import warnings

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

import timecorr as tc
from timecorr import summaries as sm
from timecorr.events import mask_kernel

T_SMALL = 90
CFUNS = {"autofc": tc.autofc, "isfc": tc.isfc, "wisfc": tc.wisfc}
SUMMARIES = {
    "ws": tc.WeightedStrength(),
    "ws-signed": tc.WeightedStrength(signed=True),
    "ws-split": tc.WeightedStrength(split=True, n=4),
    "comm": tc.Communities(n=4),
    "comm-signed": tc.Communities(n=4, signed=True),
}


def _base_data():
    data, _ = tc.simulate_nested(
        S=3,
        T=T_SMALL,
        n_supergroups=2,
        modules_per_supergroup=2,
        nodes_per_module=3,
        set_random_seed=1,
    )
    return data  # K = 12


def _variant(kind):
    data = [d.copy() for d in _base_data()]
    if kind == "nan":
        rng = np.random.default_rng(4)
        for d in data:
            d[rng.random(d.shape) < 0.03] = np.nan
    elif kind == "constant":
        for d in data:
            d[:, 2] = 1.5
    return data


def _kernels(kind):
    w = tc.gaussian_weights(T_SMALL, {"var": 25})
    if kind == "gaussian":
        return w
    return [mask_kernel(w, b) for b in ([30], [45], [30, 60])]


@pytest.fixture(scope="module")
def materialized():
    """Cache of the materialized correlations and their estimator warnings."""
    cache = {}

    def get(data_kind, kernel_kind, cfun_name):
        key = (data_kind, kernel_kind, cfun_name)
        if key not in cache:
            data = _variant(data_kind)
            kernels = _kernels(kernel_kind)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                corrs = CFUNS[cfun_name](data, kernels)
            cache[key] = (data, kernels, corrs, [str(r.message) for r in rec])
        return cache[key]

    return get


@pytest.mark.parametrize("b", [1, 2, 5, 12])
@pytest.mark.parametrize("summary_name", list(SUMMARIES))
@pytest.mark.parametrize("kernel_kind", ["gaussian", "events"])
@pytest.mark.parametrize("data_kind", ["plain", "nan", "constant"])
@pytest.mark.parametrize("cfun_name", list(CFUNS))
def test_streamed_equals_materialized(
    materialized, cfun_name, data_kind, kernel_kind, summary_name, b
):
    data, kernels, corrs, est_messages = materialized(data_kind, kernel_kind, cfun_name)
    assert len(est_messages) in (0, 1)
    K = data[0].shape[1]

    mat = copy.deepcopy(SUMMARIES[summary_name])
    with warnings.catch_warnings(record=True) as rec_m:
        warnings.simplefilter("always")
        expected = mat(corrs)

    streamed = copy.deepcopy(SUMMARIES[summary_name])
    with warnings.catch_warnings(record=True) as rec_s:
        warnings.simplefilter("always")
        outs, message = sm._streamed_summary(
            data,
            kernels,
            streamed,
            CFUNS[cfun_name],
            max_block_elements=b * K * T_SMALL,
        )

    assert len(outs) == len(expected)
    for o, e in zip(outs, expected):
        assert o.shape == e.shape
        np.testing.assert_allclose(o, e, rtol=1e-9, atol=1e-12)
    fm, fs = mat.fits_[-1], streamed.fits_[-1]
    if fm["labels"] is None:
        assert fs["labels"] is None
    else:
        np.testing.assert_array_equal(fs["labels"], fm["labels"])
    assert fs["n"] == fm["n"]
    for wm, ws in zip(fm["weights"], fs["weights"]):
        np.testing.assert_allclose(ws, wm, rtol=1e-9, atol=1e-12)

    # the estimator's warning is returned, not issued
    assert (message is None) == (len(est_messages) == 0)
    if message is not None:
        assert message == est_messages[0]
    # the summary's own warnings are the same in both routes
    assert [(r.category, str(r.message)) for r in rec_s] == [
        (r.category, str(r.message)) for r in rec_m
    ]


def test_streamed_single_subject_and_delta():
    data = [_base_data()[0]]
    for kernels in (tc.gaussian_weights(T_SMALL, {"var": 25}), tc.eye_weights(T_SMALL)):
        for cfun in (tc.autofc, tc.isfc):
            corrs = cfun(data, kernels)
            expected = tc.WeightedStrength(signed=True)(corrs)
            outs, message = sm._streamed_summary(
                data, kernels, tc.WeightedStrength(signed=True), cfun,
                max_block_elements=3 * 12 * T_SMALL,
            )
            assert message is None
            np.testing.assert_allclose(outs[0], expected[0], rtol=1e-9, atol=1e-12)


def test_streamed_delta_kernel_cross():
    data = _base_data()
    kernels = tc.eye_weights(T_SMALL)
    corrs = tc.isfc(data, kernels)
    expected = tc.Communities(n=4, signed=True)(corrs)
    outs, message = sm._streamed_summary(
        data, kernels, tc.Communities(n=4, signed=True), tc.isfc,
        max_block_elements=5 * 12 * T_SMALL,
    )
    assert message is None
    for o, e in zip(outs, expected):
        np.testing.assert_allclose(o, e, rtol=1e-9, atol=1e-12)


def test_streamed_errors():
    data = _base_data()
    w = tc.gaussian_weights(T_SMALL, {"var": 25})
    K = data[0].shape[1]
    with pytest.raises(MemoryError, match="streamed"):
        sm._streamed_summary(
            data, w, tc.WeightedStrength(), tc.autofc, max_block_elements=K * T_SMALL - 1
        )
    with pytest.raises(ValueError, match="cfun"):
        sm._streamed_summary(
            data, w, tc.WeightedStrength(), tc.helpers.mean_combine,
            max_block_elements=K * K * T_SMALL,
        )
    # n=None without the automatic count: the summary's own error
    with pytest.raises(ValueError, match="number of communities n"):
        sm._streamed_summary(
            data, w, tc.Communities(), tc.autofc, max_block_elements=K * K * T_SMALL
        )
    # the automatic count needs autofc
    with pytest.raises(ValueError, match="autofc"):
        sm._streamed_summary(
            data, w, tc.Communities(), tc.isfc,
            max_block_elements=K * K * T_SMALL, auto_n=True,
        )


# ----------------------------------------------------------------------------
# shift null and the automatic number of communities
# ----------------------------------------------------------------------------


def _nested_auto(n_sg, per_sg, per_mod, seed):
    data, truth = tc.simulate_nested(
        S=2,
        T=600,
        n_supergroups=n_sg,
        modules_per_supergroup=per_sg,
        nodes_per_module=per_mod,
        set_random_seed=seed,
    )
    return data, truth


@pytest.mark.parametrize(
    "n_sg, per_sg, per_mod, seed",
    [(3, 2, 4, 0), (3, 2, 4, 1), (6, 3, 3, 0)],
    ids=["6-modules-seed0", "6-modules-seed1", "18-modules-seed0"],
)
def test_auto_n_counts_modules(n_sg, per_sg, per_mod, seed):
    data, truth = _nested_auto(n_sg, per_sg, per_mod, seed)
    K = data[0].shape[1]
    w = tc.gaussian_weights(600, {"var": 25})
    comm = tc.Communities()
    outs, _ = sm._streamed_summary(
        data, w, comm, tc.autofc, max_block_elements=K * K * 600, auto_n=True,
        bandwidth=5.0,
    )
    n_true = n_sg * per_sg
    f = comm.fits_[-1]
    assert f["null_threshold"] is not None
    assert f["n"] == n_true
    assert outs[0].shape == (600, n_true)


def test_auto_n_on_independent_noise():
    T, K = 400, 24
    w = tc.gaussian_weights(T, {"var": 25})
    none_found = 0
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        data = [
            gaussian_filter1d(rng.standard_normal((T, K)), 2, axis=0) for _ in range(2)
        ]
        comm = tc.Communities()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            outs, _ = sm._streamed_summary(
                data, w, comm, tc.autofc, max_block_elements=K * K * T, auto_n=True,
                bandwidth=5.0,
            )
        warned = any(
            "no community structure above the shift null" in str(r.message) for r in rec
        )
        if warned and all(o.shape == (T, 1) for o in outs):
            none_found += 1
    assert none_found >= 6


def test_shift_null_deterministic_and_global_state():
    rng = np.random.default_rng(3)
    data = [gaussian_filter1d(rng.standard_normal((120, 10)), 2, axis=0) for _ in range(2)]
    w = tc.gaussian_weights(120, {"var": 25})
    np.random.seed(2024)
    before = np.random.get_state()
    a = sm._shift_null_threshold(data, w, max_block_elements=3 * 10 * 120)
    b = sm._shift_null_threshold(data, w, max_block_elements=10 * 10 * 120)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert isinstance(a, float) and np.isfinite(a)
    np.testing.assert_allclose(a, b, rtol=1e-12)
    assert a == sm._shift_null_threshold(data, w, max_block_elements=3 * 10 * 120)
    # per-subject kernels are accepted
    sm._shift_null_threshold(data, [w, mask_kernel(w, [60])], max_block_elements=10 * 10 * 120)


def test_shift_null_needs_T_40():
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((39, 6)) for _ in range(2)]
    with pytest.raises(ValueError, match="40"):
        sm._shift_null_threshold(data, tc.gaussian_weights(39, {"var": 9}))
    data = [rng.standard_normal((40, 6)) for _ in range(2)]
    sm._shift_null_threshold(data, tc.gaussian_weights(40, {"var": 9}))


# ----------------------------------------------------------------------------
# memory
# ----------------------------------------------------------------------------


def test_block_rows_for():
    T, K, S = 300, 60, 2
    for b in (1, 4, 17, K):
        need = 8 * sm._streamed_peak_elements(T, K, S, b)
        assert sm._block_rows_for(T, K, S, need) == b
        assert sm._block_rows_for(T, K, S, need - 1) == b - 1
    assert sm._block_rows_for(T, K, S, 10**15) == K
    assert sm._block_rows_for(T, K, S, 8) == 0


@pytest.mark.parametrize("cfun", [tc.autofc, tc.isfc], ids=["autofc", "isfc"])
def test_streamed_peak_memory(cfun):
    # measured at 7.7 MB (autofc) and 12.4 MB (isfc) against 68.6 MB and
    # 69.0 MB for the materialized route; one (K, K, T) array is 8.6 MB and
    # the formula gives 27.9 MB
    T, K, S, b = 300, 60, 2, 4
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((T, K)) for _ in range(S)]
    w = tc.gaussian_weights(T, {"var": 25})

    def peak_of(f):
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            f()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    streamed = peak_of(
        lambda: sm._streamed_summary(
            data, w, tc.WeightedStrength(split=True, signed=True, n=4), cfun,
            max_block_elements=b * K * T,
        )
    )
    materialized = peak_of(
        lambda: tc.WeightedStrength(split=True, signed=True, n=4)(cfun(data, w))
    )
    assert streamed < 1.25 * 8 * sm._streamed_peak_elements(T, K, S, b)
    assert streamed < 0.25 * materialized


def _traced_peak(f):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        f()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("cfun", [tc.autofc, tc.isfc], ids=["autofc", "isfc"])
def test_streamed_peak_below_one_correlation_array_at_larger_k(cfun):
    # what streaming exists for: at K 150 with one row per block (and NaN
    # samples, the estimator's most expensive path) the step holds less than
    # one (K, K, T) array
    # (measured: 10.0 MB autofc, 13.2 MB isfc; one array 36.0 MB; formula 18.8 MB)
    T, K, S, b = 200, 150, 2, 1
    rng = np.random.default_rng(5)
    data = [rng.standard_normal((T, K)) for _ in range(S)]
    for d in data:
        d[rng.random(d.shape) < 0.03] = np.nan
    w = tc.gaussian_weights(T, {"var": 25})
    streamed = _traced_peak(
        lambda: sm._streamed_summary(
            data, w, tc.WeightedStrength(signed=True), cfun,
            max_block_elements=b * K * T,
        )
    )
    assert streamed < K * K * T * 8
    assert streamed < 1.25 * 8 * sm._streamed_peak_elements(T, K, S, b)


# ----------------------------------------------------------------------------
# shift-null lags and the number of surrogates (review items 3 and 4)
# ----------------------------------------------------------------------------


def test_shift_lags_are_distinct_within_a_subject():
    rng = np.random.default_rng(0)
    for T, K, L in [(40, 21, 10), (40, 5, 10), (600, 108, 50), (300, 24, 50)]:
        for _ in range(20):
            lags = sm._shift_lags(rng, T, K, L)
            assert lags.shape == (K,)
            assert len(np.unique(lags)) == K
            assert lags.min() >= L and lags.max() <= T - L


def test_shift_null_with_more_columns_than_lags_runs():
    # T 40: L 10, so 21 distinct lags in [10, 30] for 22 columns: the lags
    # are reused (balanced), and the threshold is deterministic
    rng = np.random.default_rng(1)
    w = tc.gaussian_weights(40, {"var": 9})
    data = [rng.standard_normal((40, 22)) for _ in range(2)]
    with pytest.warns(UserWarning, match="22 columns") as record:
        first = sm._shift_null_threshold(data, w)
    # one warning per fit (not per subject or surrogate)
    assert len([r for r in record if "share a lag" in str(r.message)]) == 1
    message = str(record[0].message)
    assert "21 distinct lags" in message
    assert "keep their dependence" in message
    assert "give the number of communities n" in message
    assert "1/231" in message  # [n q (q - 1) + 2 r q] / [K (K - 1)], q 1, r 1
    assert np.isfinite(first)
    with pytest.warns(UserWarning, match="share a lag"):
        assert sm._shift_null_threshold(data, w) == first
    # K <= n: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sm._shift_null_threshold([d[:, :21] for d in data], w)


@pytest.mark.parametrize("T, K, L", [(40, 22, 10), (40, 42, 10), (40, 64, 10),
                                     (120, 150, 30), (100, 51, 25), (100, 60, 25)])
def test_fraction_of_pairs_sharing_a_lag_is_exact(T, K, L):
    # every draw of _shift_lags has [n q (q - 1) + 2 r q] / [K (K - 1)] of
    # its column pairs sharing a lag, K = q n + r
    n = T - 2 * L + 1
    q, r = divmod(K, n)
    expected = (n * q * (q - 1) + 2 * r * q) / (K * (K - 1))
    assert sm._shared_lag_fraction(K, n) == pytest.approx(expected, rel=1e-15)
    rng = np.random.default_rng(0)
    for _ in range(5):
        lags = sm._shift_lags(rng, T, K, L)
        counts = np.bincount(lags - L)
        shared = np.sum(counts * (counts - 1))  # ordered pairs
        assert shared / (K * (K - 1)) == pytest.approx(expected, rel=1e-15)


def test_shift_lags_up_to_the_number_of_lags_are_unchanged():
    # K <= number of distinct lags: the lags drawn without replacement, as
    # before (bit for bit, same generator state afterwards)
    for T, K, L in [(40, 21, 10), (40, 5, 10), (600, 108, 50), (300, 201, 50)]:
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        for _ in range(5):
            np.testing.assert_array_equal(
                sm._shift_lags(a, T, K, L),
                L + b.choice(T - 2 * L + 1, size=K, replace=False))
        assert a.integers(1 << 30) == b.integers(1 << 30)


@pytest.mark.parametrize("T, K, L", [(40, 22, 10), (40, 63, 10), (40, 64, 10),
                                     (120, 150, 30), (300, 700, 75)])
def test_shift_lags_beyond_the_number_of_lags_are_balanced(T, K, L):
    n = T - 2 * L + 1
    rng = np.random.default_rng(0)
    for _ in range(10):
        lags = sm._shift_lags(rng, T, K, L)
        assert lags.shape == (K,)
        assert lags.min() >= L and lags.max() <= T - L
        counts = np.bincount(lags - L, minlength=n)
        assert counts.sum() == K
        assert counts.min() == K // n and counts.max() == -(-K // n)
        assert np.sum(counts == -(-K // n)) == (K % n or n)
    # deterministic for a given generator state
    np.testing.assert_array_equal(
        sm._shift_lags(np.random.default_rng(3), T, K, L),
        sm._shift_lags(np.random.default_rng(3), T, K, L))
    # the order of the columns is shuffled: not the repeated permutation
    # itself (columns n apart do not always share a lag)
    lags = sm._shift_lags(np.random.default_rng(3), T, K, L)
    assert not np.array_equal(lags[:K - n], lags[n:])


def test_shift_lags_need_one_lag():
    with pytest.raises(ValueError, match="no lag"):
        sm._shift_lags(np.random.default_rng(0), 40, 5, 21)


@pytest.mark.parametrize("make", [
    lambda m: tc.Communities(n_null=m),
    lambda m: tc.WeightedStrength(split=True, n_null=m),
], ids=["communities", "ws-split"])
def test_n_null_is_used_and_recorded(make):
    data, _ = _nested_auto(3, 2, 4, 0)
    K = data[0].shape[1]
    w = tc.gaussian_weights(600, {"var": 25})
    thresholds = {}
    for m in (1, 5, 9):
        summary = make(m)
        assert summary.n_null == m
        sm._streamed_summary(
            data, w, summary, tc.autofc, max_block_elements=K * K * 600, auto_n=True,
            bandwidth=5.0,
        )
        fit = summary.fits_[-1]
        assert fit["n_null"] == m
        thresholds[m] = fit["null_threshold"]
        assert fit["null_threshold"] == sm._shift_null_threshold(
            data, w, n_null=m, bandwidth=5.0
        )
    # the first surrogates are the same draws, so the maximum can only grow
    assert thresholds[1] <= thresholds[5] <= thresholds[9]
    assert make(5).n_null == 5 and tc.Communities().n_null == 5


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "5", None])
def test_bad_n_null_raises(bad):
    with pytest.raises(ValueError, match="n_null must be a positive integer"):
        tc.Communities(n_null=bad)
    with pytest.raises(ValueError, match="n_null must be a positive integer"):
        tc.WeightedStrength(split=True, n_null=bad)


def _reference_null_tops(data, kernel, n_null, bandwidth):
    """Top eigenvalues of the shift-null surrogates, recomputed here from the
    documented procedure: numpy.random.default_rng(0); per repetition, per
    subject, K lags without replacement from [L, T - L]; circular shifts;
    autofc pair weights mean_t r_ij**2 (zero diagonal), mean over subjects,
    minus their median over i < j (zero diagonal); largest eigenvalue."""
    T, K = data[0].shape
    L = int(min(T // 4, max(10, 10 * bandwidth)))
    rng = np.random.default_rng(0)
    rows, cols = np.arange(T)[:, None], np.arange(K)
    tops = []
    for _ in range(n_null):
        shifted = []
        for x in data:
            lags = L + rng.choice(T - 2 * L + 1, size=K, replace=False)
            shifted.append(x[(rows - lags[None, :]) % T, cols])
        weights = []
        for v in tc.autofc(shifted, kernel):
            M = tc.vec2mat(v)
            w = (M**2).mean(axis=2)
            np.fill_diagonal(w, 0)
            weights.append(w)
        wbar = np.mean(weights, axis=0)
        C = wbar - np.median(wbar[np.triu_indices(K, 1)])
        np.fill_diagonal(C, 0)
        tops.append(np.linalg.eigvalsh(C)[-1])
    return np.array(tops)


def test_n_null_matches_an_independent_surrogate_loop():
    data, _ = _nested_auto(3, 2, 4, 0)
    K = data[0].shape[1]
    w = tc.gaussian_weights(600, {"var": 25})
    tops = _reference_null_tops(data, w, 9, 5.0)
    # these data give a larger maximum over 9 surrogates than over 1
    assert tops.max() > tops[0]
    for m in (1, 9):
        expected = tops[:m].max()
        assert sm._shift_null_threshold(data, w, n_null=m, bandwidth=5.0) == pytest.approx(
            expected, rel=1e-9)
        comm = tc.Communities(n_null=m)
        sm._streamed_summary(data, w, comm, tc.autofc, max_block_elements=K * K * 600,
                             auto_n=True, bandwidth=5.0)
        assert comm.fits_[-1]["null_threshold"] == pytest.approx(expected, rel=1e-9)
    assert sm._shift_null_threshold(data, w, n_null=1, bandwidth=5.0) != pytest.approx(
        sm._shift_null_threshold(data, w, n_null=9, bandwidth=5.0), rel=1e-9)
