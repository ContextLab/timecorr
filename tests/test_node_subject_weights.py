"""
Subject weights per node: ``wisfc(subject_weights=W)`` with ``W`` of shape
(S, S, K), and ``tc.distinct_subject_weights``, which leaves out of each
subject's reference the subjects that hold one of its own series.

Real computations on simulated data, no mocks.
"""

import functools
import warnings

import numpy as np
import pytest

import timecorr as tc
import timecorr.summaries as sm
from timecorr.helpers import _fisher_symmetrize, vec2mat

S, T, K = 4, 120, 5


def make_data(seed=0, S=S, T=T, K=K):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(T, K))
    return [common + rng.normal(size=(T, K)) for _ in range(S)]


def kernel(T=T, var=100):
    return tc.gaussian_weights(T, {"var": var})


def reference(data, w):
    """Subject weights (S, K) applied node by node, with plain loops."""
    T, K = data[0].shape
    ref = np.zeros((T, K))
    for k in range(K):
        total = sum(w[r, k] for r in range(len(data)))
        if total != 0:
            ref[:, k] = sum(w[r, k] * data[r][:, k] for r in range(len(data))) / total
    return ref


def direct(data, W, kern, s):
    """Subject s's matrices from wcorr and the documented rules."""
    a = data[s]
    ref = reference(data, W[s])
    empty = ~np.any(W[s] != 0, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = tc.wcorr(a, ref, kern)
    c[:, empty, :] = c[empty, :, :].transpose(1, 0, 2)
    c[np.ix_(empty, empty)] = 0
    return _fisher_symmetrize(c)


# --- the helper ---------------------------------------------------------------

def test_distinct_subject_weights_rule():
    labels = np.array([["a", "b", "c"],
                       ["a", "d", "e"],
                       ["f", "b", "a"],
                       ["g", "h", "i"]])
    W = tc.distinct_subject_weights(labels)
    assert W.shape == (4, 4, 3) and W.dtype == np.float64
    for s in range(4):
        for r in range(4):
            for k in range(3):
                expected = float(r != s and labels[r, k] not in labels[s])
                assert W[s, r, k] == expected, (s, r, k)
    # subject 2 holds "a" at node 2: left out of subject 0's reference there,
    # although subject 0 holds "a" at node 0
    assert W[0, 2, 2] == 0 and W[0, 1, 0] == 0 and W[0, 3, 0] == 1


def test_distinct_subject_weights_without_repeats_is_isfc():
    labels = np.arange(S * K).reshape(S, K)
    W = tc.distinct_subject_weights(labels)
    np.testing.assert_array_equal(
        W, np.repeat((1 - np.eye(S))[:, :, None], K, axis=2))
    data = make_data()
    kern = kernel()
    for x, y in zip(tc.wisfc(data, kern, subject_weights=W), tc.isfc(data, kern)):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)


@pytest.mark.parametrize("labels", [
    np.arange(6),
    np.arange(24).reshape(2, 3, 4),
    np.empty((0, 3)),
    np.empty((3, 0)),
])
def test_distinct_subject_weights_validation(labels):
    with pytest.raises(ValueError, match="labels must be"):
        tc.distinct_subject_weights(labels)


def test_distinct_subject_weights_accepts_lists_and_numbers():
    W = tc.distinct_subject_weights([[1, 2], [1, 3], [4, 2]])
    assert W.shape == (3, 3, 2)
    assert W[0, 1, 0] == 0 and W[0, 2, 0] == 1 and W[0, 2, 1] == 0


def test_distinct_subject_weights_compares_labels_as_given():
    # converting to one dtype would make these labels equal
    W = tc.distinct_subject_weights([[9007199254740992, 0.0],
                                     [9007199254740993, 1.0]])
    assert W[0, 1, 0] == 1
    W = tc.distinct_subject_weights([[1, "a"], ["1", "b"]])
    assert W[0, 1, 0] == 1
    # numpy arrays of numbers or strings
    W = tc.distinct_subject_weights(np.array([[1.5, 2.0], [1.5, 3.0]]))
    assert W[0, 1, 0] == 0 and W[0, 1, 1] == 1
    W = tc.distinct_subject_weights(np.array([["x", "y"], ["z", "y"]]))
    assert W[0, 1, 0] == 1 and W[0, 1, 1] == 0


def test_a_single_subject_has_no_reference():
    data = make_data(7, S=1)
    W = tc.distinct_subject_weights(np.arange(K)[None, :])
    assert W.shape == (1, 1, K) and not W.any()
    with pytest.warns(UserWarning, match="5 of 5"):
        out = tc.wisfc(data, kernel(), subject_weights=W)
    np.testing.assert_array_equal(out[0], 0)


# --- wisfc with weights per node ---------------------------------------------------

def test_equal_weights_at_every_node_are_the_matrix_weights():
    data = make_data(1)
    kern = kernel()
    rng = np.random.default_rng(2)
    W2 = rng.uniform(0.2, 1.0, size=(S, S))
    np.fill_diagonal(W2, 0)
    W3 = np.repeat(W2[:, :, None], K, axis=2)
    for x, y in zip(tc.wisfc(data, kern, subject_weights=W3),
                    tc.wisfc(data, kern, subject_weights=W2)):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)


def test_weights_per_node_give_the_reference_of_each_node():
    data = make_data(3)
    kern = kernel()
    rng = np.random.default_rng(4)
    W = rng.uniform(0.2, 1.0, size=(S, S, K))
    W[np.arange(S), np.arange(S)] = 0
    W[0, 1, 2] = 0
    W[2, 3, 0] = 0
    out = tc.wisfc(data, kern, subject_weights=W)
    for s in range(S):
        np.testing.assert_allclose(vec2mat(out[s]), direct(data, W, kern, s),
                                   rtol=0, atol=1e-12)


def repeated():
    """Subjects 0 and 1 hold the same series at node 2."""
    data = make_data(5, S=3)
    data[1][:, 2] = data[0][:, 2]
    labels = np.arange(3 * K).reshape(3, K)
    labels[1, 2] = labels[0, 2]
    return data, labels


def test_a_repeated_series_is_left_out_of_the_reference():
    data, labels = repeated()
    kern = kernel()
    W = tc.distinct_subject_weights(labels)
    plain = vec2mat(tc.isfc(data, kern)[0])
    out = tc.wisfc(data, kern, subject_weights=W)
    got = vec2mat(out[0])
    # isfc correlates the series with a mean that holds it
    independent = data[0][:, 2], data[2][:, 2]
    expected = tc.wcorr(independent[0][:, None], independent[1][:, None], kern)[0, 0]
    np.testing.assert_allclose(got[2, 2], expected, rtol=0, atol=1e-12)
    assert np.mean(plain[2, 2] - got[2, 2]) > 0.1
    for s in range(3):
        np.testing.assert_allclose(vec2mat(out[s]), direct(data, W, kern, s),
                                   rtol=0, atol=1e-12)
    # subject 2 holds neither copy: its reference is isfc's
    np.testing.assert_allclose(out[2], tc.isfc(data, kern)[2], rtol=0, atol=1e-12)


def no_reference():
    """Two subjects with the same series at node 1: neither has a reference
    there."""
    data = make_data(6, S=2)
    data[1][:, 1] = data[0][:, 1]
    labels = np.arange(2 * K).reshape(2, K)
    labels[1, 1] = labels[0, 1]
    return data, tc.distinct_subject_weights(labels)


def test_a_node_without_reference():
    data, W = no_reference()
    kern = kernel()
    with pytest.warns(UserWarning, match="no reference") as record:
        out = tc.wisfc(data, kern, subject_weights=W)
    text = [str(w.message) for w in record if "no reference" in str(w.message)]
    assert len(text) == 1
    assert "2 of 10" in text[0]
    for s in range(2):
        got = vec2mat(out[s])
        assert np.isfinite(got).all()
        # the diagonal entry is undefined: 0
        np.testing.assert_array_equal(got[1, 1], 0)
        # the other entries of the node: the subject's series against the
        # other nodes' references, the one orientation that is defined
        other = data[1 - s]
        for i in (0, 2, 3, 4):
            expected = tc.wcorr(data[s][:, [1]], other[:, [i]], kern)[0, 0]
            np.testing.assert_allclose(got[1, i], expected, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(got[1, i], got[i, 1])
        # entries of the other nodes: as isfc
        keep = [0, 2, 3, 4]
        plain = vec2mat(tc.isfc(data, kern)[s])
        np.testing.assert_allclose(got[np.ix_(keep, keep)],
                                   plain[np.ix_(keep, keep)], rtol=0, atol=1e-12)


def test_no_warning_when_every_node_has_a_reference():
    data, labels = repeated()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tc.wisfc(data, kernel(), subject_weights=tc.distinct_subject_weights(labels))


@pytest.mark.parametrize("shape", [(S, S, K + 1), (S, S + 1, K), (S + 1, S, K),
                                   (S, S, K, 1)])
def test_weights_of_another_shape_raise(shape):
    with pytest.raises(ValueError, match="subject_weights"):
        tc.wisfc(make_data(), kernel(), subject_weights=np.ones(shape))


# --- through timecorr and features -------------------------------------------------

def test_timecorr_with_weights_per_node():
    data, labels = repeated()
    W = tc.distinct_subject_weights(labels)
    cfun = functools.partial(tc.wisfc, subject_weights=W)
    kern = kernel()
    out = tc.timecorr(data, weights_function=tc.gaussian_weights,
                      weights_params={"var": 100}, cfun=cfun,
                      combine=tc.null_combine)
    for x, y in zip(out, tc.wisfc(data, kern, subject_weights=W)):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)


def causal_kernel(T):
    return tc.gaussian_weights(T, {"var": 100}, direction="backward")


@pytest.mark.parametrize("make", [repeated, no_reference])
def test_causal_rows_do_not_change_when_later_rows_are_removed(make):
    made = make()
    data = made[0]
    W = made[1] if make is no_reference else tc.distinct_subject_weights(made[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = tc.wisfc(data, causal_kernel(T), subject_weights=W, causal=True)
        cut = 77
        part = tc.wisfc([d[:cut] for d in data], causal_kernel(cut),
                        subject_weights=W, causal=True)
    for x, y in zip(full, part):
        assert np.array_equal(x[:cut], y, equal_nan=True)
        defined = ~np.isnan(x).any(axis=1)
        assert defined.sum() > T - 10
        # a row is NaN entirely or not at all
        assert np.array_equal(np.isnan(x).all(axis=1), np.isnan(x).any(axis=1))


def weights_of(make):
    made = make()
    return made[0], (made[1] if make is no_reference
                     else tc.distinct_subject_weights(made[1]))


@pytest.mark.parametrize("b", [1, 2, 5])
@pytest.mark.parametrize("make", [repeated, no_reference])
@pytest.mark.parametrize("causal", [False, True])
def test_streamed_summaries_agree_with_the_summary_of_wisfc(make, causal, b):
    # the comparison of tests/test_summaries_streaming.py, for weights per node
    data, W = weights_of(make)
    cfun = functools.partial(tc.wisfc, subject_weights=W)
    kern = causal_kernel(T) if causal else kernel()
    estimator = {"causal": True, "min_samples": None} if causal else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrs = tc.wisfc(data, kern, subject_weights=W, causal=causal)
        rows = [~np.isnan(c).any(axis=1) for c in corrs]
        dense = tc.WeightedStrength(signed=True)
        dense.fit([c[r] for c, r in zip(corrs, rows)])
        expected = dense.transform([c[r] for c, r in zip(corrs, rows)])
        outs, _ = sm._streamed_summary(
            data, kern, tc.WeightedStrength(signed=True), cfun,
            max_block_elements=b * K * T, estimator=estimator)
    for o, e, r in zip(outs, expected, rows):
        assert np.isnan(o[~r]).all()
        np.testing.assert_allclose(o[r], e, rtol=1e-9, atol=1e-12)


def test_features_warns_about_nodes_without_reference():
    data, W = weights_of(no_reference)
    cfun = functools.partial(tc.wisfc, subject_weights=W)
    with pytest.warns(UserWarning, match="no reference"):
        out = tc.features(
            data, order=1, cfun=cfun, combine=tc.null_combine, causal=True,
            weights_function=tc.gaussian_weights, weights_params={"var": 100},
            delta_weights_params={"var": 100},
            rfun=tc.WeightedStrength(signed=True), fit_timepoints=80)
    # the data and the order-1 features of each subject
    assert len(out) == 2 and len(out[1]) == 2
    assert out[1][0].shape[0] == T


def test_features_at_an_order_with_another_width_raises():
    # the weights are for K nodes; order 2 has the width of the feeding
    # features, here 3 components
    data, W = weights_of(repeated)
    cfun = functools.partial(tc.wisfc, subject_weights=W)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="subject_weights"):
            tc.features(data, order=2, cfun=cfun, combine=tc.null_combine,
                        rfun="PCA", delta_rfun="PCA", ndims=3,
                        weights_params={"var": 100},
                        delta_weights_params={"var": 100})
