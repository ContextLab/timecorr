"""
End-to-end recovery of structure simulated at one order.

``simulate_data(order=n)`` embeds a known dynamic correlation matrix at exactly
order ``n`` (the sign-parity cascade).  These tests run the real pipeline
(``tc.features``) on those data and check that

- the order-``n`` correlations among the block products match the truth
  (recovery r: Pearson correlation between estimated and true off-diagonal
  entries, pooled over entries and timepoints), and
- every order below ``n`` looks like independent noise: its mean absolute
  off-diagonal correlation is within 0.03 of the same quantity computed the same
  way on independent Gaussian noise with the same T, the same number of columns
  and the same kernel.

All runs: one subject, block template with 4 events, T=1000, Gaussian kernel with
var=625 (sd 25 timepoints), ``cfun=autofc``, fixed seeds.
"""

import tracemalloc
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr.higher_order import estimate_memory_gb

T = 1000
B = 4
SEED = 0  # the slow unreduced order-3 case and the multi-order case
SEEDS = [0, 1, 2, 3, 4]
RECOVERY = 0.8
MULTI_ORDER_RECOVERY = 0.75
CONTROL_TOLERANCE = 0.03
KERNEL = {
    "weights_function": tc.gaussian_weights,
    "weights_params": {"var": 625},
    "cfun": tc.autofc,
}


def _simulate(order, K, seed=SEED):
    """simulate_data with the shared settings; checks exactly the expected warnings."""
    p = K // 2 ** (order - 1)
    expected = (["single entry"] if p == 2 else []) + (
        ["carrier_reducer"] if order >= 3 else []
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        data, truth = tc.simulate_data(
            datagen="block",
            return_corrs=True,
            set_random_seed=seed,
            T=T,
            K=K,
            B=B,
            order=order,
        )
    messages = [str(w.message) for w in record]
    assert all(issubclass(w.category, UserWarning) for w in record)
    assert len(messages) == len(expected)
    for text in expected:
        assert sum(text in m for m in messages) == 1
    return data, truth


def _width(x):
    """Number of features d whose correlation matrices give x's mat2vec columns."""
    v = x.shape[1]
    d = int(round((np.sqrt(8 * v + 1) - 1) / 2))
    assert d * (d + 1) // 2 == v
    return d


def _vec_column(d, a, b):
    """Column of entry (a, b), a != b, in mat2vec layout of d x d matrices."""
    i, j = np.triu_indices(d, 1)
    a, b = min(a, b), max(a, b)
    return d + int(np.flatnonzero((i == a) & (j == b))[0])


def _recovery(X, cols, corrs):
    """
    Pooled Pearson r between estimated and true off-diagonal entries.

    ``X``: (T, d(d+1)/2) estimates; ``cols``: the p feature indices (of d) whose
    p x p correlation matrix is scored; ``corrs``: (T, p(p+1)/2) truth.
    """
    d, p = _width(X), len(cols)
    i, j = np.triu_indices(p, 1)
    est = X[:, [_vec_column(d, cols[a], cols[b]) for a, b in zip(i, j)]]
    true = corrs[:, p:]  # mat2vec: p diagonal entries first, then the upper triangle
    assert est.shape == true.shape == (T, p * (p - 1) // 2)
    return np.corrcoef(est.ravel(), true.ravel())[0, 1]


def _mean_abs_offdiag(X):
    return float(np.abs(X[:, _width(X) :]).mean())


def _noise_control(d, seed):
    """Mean |off-diagonal correlation| of independent Gaussian noise, (T, d)."""
    noise = np.random.default_rng(seed).standard_normal((T, d))
    return _mean_abs_offdiag(tc.features(noise, order=1, rfun=None, **KERNEL)[1])


def _assert_lower_orders_at_control(X, n, seed):
    """Orders 1 .. n-1 of X carry no more structure than independent noise."""
    for k in range(1, n):
        observed = _mean_abs_offdiag(X[k])
        control = _noise_control(_width(X[k]), seed=1000 * (seed + 1) + k)
        print(f"  order {k}: mean |r| {observed:.4f}, control {control:.4f}")
        assert abs(observed - control) <= CONTROL_TOLERANCE, (k, observed, control)


# ------------------------------------------------------------- unreduced runs


@pytest.mark.parametrize("seed", SEEDS)
def test_unreduced_order1_recovery(seed):
    data, corrs = tc.simulate_data(
        datagen="block", return_corrs=True, set_random_seed=seed, T=T, K=4, B=B
    )
    X = tc.features(data, order=1, rfun=None, **KERNEL)
    r = _recovery(X[1], np.arange(4), corrs)
    print(f"unreduced order 1, p=4, seed {seed}: recovery r {r:.4f}")
    assert r >= RECOVERY


@pytest.mark.parametrize("seed", SEEDS)
def test_unreduced_order2_recovery(seed):
    data, truth = _simulate(order=2, K=8, seed=seed)
    X = tc.features(data, order=2, rfun=None, **KERNEL)
    cols = tc.carrier_columns(truth["blocks"], 1, 8).ravel()
    r = _recovery(X[2], cols, truth["corrs"])
    print(f"unreduced order 2, p=4, seed {seed}: recovery r {r:.4f}")
    assert r >= RECOVERY
    _assert_lower_orders_at_control(X, 2, seed)


@pytest.mark.slow
def test_unreduced_order3_recovery():
    data, truth = _simulate(order=3, K=8)
    # The order-3 step correlates 666 delta features: about 5.7 GB, above the
    # default max_memory_gb of 4.
    step_gb = estimate_memory_gb(8, 3, T)[3]
    assert 4.0 < step_gb < 6.0
    with pytest.raises(MemoryError):
        tc.features(data, order=3, rfun=None, **KERNEL)

    tracemalloc.start()
    try:
        X = tc.features(data, order=3, rfun=None, max_memory_gb=6.0, **KERNEL)
        peak_gb = tracemalloc.get_traced_memory()[1] / 1e9
    finally:
        tracemalloc.stop()
    # the run's live memory stays close to the guard's estimate for its largest step
    print(f"unreduced order 3: traced peak {peak_gb:.2f} GB, estimate {step_gb:.2f} GB")
    assert peak_gb <= 1.1 * step_gb

    cols = tc.carrier_columns(truth["blocks"], 2, 8).ravel()
    r = _recovery(X[3], cols, truth["corrs"])
    print(f"unreduced order 3, p=2, seed {SEED}: recovery r {r:.4f}")
    assert r >= RECOVERY
    _assert_lower_orders_at_control(X, 3, SEED)


# ---------------------------------------------------------- carrier reducer


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_carrier_reducer_recovery(order, seed):
    p = 4
    data, truth = _simulate(order=order, K=tc.min_features(order, p=p), seed=seed)
    X = tc.features(
        data,
        order=order,
        rfun=None,
        delta_rfun=tc.carrier_reducer(truth["blocks"]),
        **KERNEL,
    )
    assert X[order].shape == (T, p * (p + 1) // 2)
    r = _recovery(X[order], np.arange(p), truth["corrs"])
    print(f"carrier reducer, order {order}, p={p}, seed {seed}: recovery r {r:.4f}")
    assert r >= RECOVERY
    _assert_lower_orders_at_control(X, order, seed)


def test_order3_data_run_with_pca():
    # PCA did not retain structure that exists only at order 3 in the tutorial's
    # measurement (the generator warns), but the pipeline must still run.
    data, _ = _simulate(order=3, K=16)
    X = tc.features(data, order=3, rfun="PCA", **KERNEL)
    assert len(X) == 4
    for x in X:
        assert x.shape == (T, 16)
        assert np.all(np.isfinite(x))


def test_simulate_multi_order_recovery():
    p = 4
    with pytest.warns(UserWarning, match="carrier_reducer") as record:
        data, truths = tc.simulate_multi_order(
            [1, 2, 3], p=p, datagen="block", set_random_seed=SEED, T=T, B=B
        )
    assert len(record) == 1
    K = data.shape[1]
    assert K == 4 + 8 + 16
    for truth in truths:
        n = truth["order"]
        X = tc.features(
            data,
            order=n,
            rfun=None,
            delta_rfun=tc.carrier_reducer(truth["blocks"], K=K),
            **KERNEL,
        )
        # order 1 is scored among the group's own columns of the data; at higher
        # orders the delta features entering order n are this group's p carriers
        cols = truth["blocks"][:, 0] if n == 1 else np.arange(p)
        r = _recovery(X[n], cols, truth["corrs"])
        print(f"multi-order [1, 2, 3], order-{n} group: recovery r {r:.4f}")
        assert r >= MULTI_ORDER_RECOVERY
