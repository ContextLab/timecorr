"""Tests for timecorr.helpers.reduce (real hypertools / scikit-learn models)."""

import os
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA

from timecorr.helpers import mat2vec, pca_decoder, reduce, vec2mat

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "graph_reference.npz")


def seeded_vectorized_corrs(T=30, K=5, seed=0):
    rng = np.random.RandomState(seed)
    mats = np.stack([np.corrcoef(rng.randn(K, 8)) for _ in range(T)])
    return np.stack([mat2vec(m) for m in mats])


def test_none_returns_input_unchanged():
    V = seeded_vectorized_corrs()
    assert reduce(V, rfun=None) is V


def test_callable_is_applied_and_output_returned_for_list():
    a = seeded_vectorized_corrs(seed=1)
    b = seeded_vectorized_corrs(seed=2)
    calls = []

    def f(arrays):
        calls.append(len(arrays))
        return [x[:, :2] * 2.0 for x in arrays]

    out = reduce([a, b], rfun=f)
    assert calls == [2]
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_array_equal(out[0], a[:, :2] * 2.0)
    np.testing.assert_array_equal(out[1], b[:, :2] * 2.0)


def test_callable_with_single_array_returns_array():
    a = seeded_vectorized_corrs(seed=1)
    out = reduce(a, rfun=lambda arrays: [x[:, :4] + 1.0 for x in arrays])
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, a[:, :4] + 1.0)


def test_list_input_is_fit_jointly_with_one_model():
    a = seeded_vectorized_corrs(T=40, seed=1)
    b = seeded_vectorized_corrs(T=40, seed=2) * 0.5 + 0.1
    out = reduce([a, b], rfun="PCA", ndims=3)
    model = PCA(n_components=3).fit(np.vstack([a, b]))
    for got, src in zip(out, [a, b]):
        np.testing.assert_allclose(
            np.abs(got), np.abs(model.transform(src)), atol=1e-8
        )


def test_shared_matrix_supplied_twice_gives_equal_outputs():
    a = seeded_vectorized_corrs(T=40, seed=3)
    out = reduce([a, a.copy()], rfun="PCA", ndims=3)
    np.testing.assert_allclose(out[0], out[1], rtol=1e-12, atol=1e-12)


def test_ndims_sets_number_of_columns():
    a = seeded_vectorized_corrs(T=40, K=6)
    assert reduce(a, rfun="PCA", ndims=3).shape == (40, 3)
    out = reduce([a, a * 0.9], rfun="PCA", ndims=3)
    assert [x.shape for x in out] == [(40, 3), (40, 3)]


def test_ndims_none_gives_K_columns():
    a = seeded_vectorized_corrs(T=40, K=6)
    assert a.shape[1] == 21
    assert reduce(a, rfun="PCA").shape == (40, 6)
    assert [x.shape for x in reduce([a, a], rfun="PCA")] == [(40, 6), (40, 6)]


def test_random_state_makes_stochastic_model_reproducible():
    a = seeded_vectorized_corrs(T=40, K=6)
    first = reduce(a, rfun="FastICA", ndims=3)
    second = reduce(a, rfun="FastICA", ndims=3)
    np.testing.assert_array_equal(first, second)


def test_dict_rfun_with_params_is_supported_without_deprecation_warning():
    a = seeded_vectorized_corrs(T=40, K=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = reduce(a, rfun={"model": "PCA", "params": {"whiten": True}}, ndims=3)
    np.testing.assert_allclose(out.std(axis=0, ddof=1), 1.0, atol=1e-8)


@pytest.mark.parametrize(
    "name", ["eigenvector_centrality", "pagerank_centrality", "strength"]
)
def test_graph_measures_unchanged_on_seeded_input(name):
    ref = np.load(FIXTURE)
    V = seeded_vectorized_corrs()
    np.testing.assert_array_equal(V, ref["V"])
    np.testing.assert_array_equal(reduce(V, rfun=name), ref[name])
    out = reduce([V, V * 0.5], rfun=name)
    np.testing.assert_array_equal(out[0], ref[name + "_l0"])
    np.testing.assert_array_equal(out[1], ref[name + "_l1"])


def test_strength_matches_row_sums_of_matrix():
    V = seeded_vectorized_corrs()
    expected = np.stack([vec2mat(V)[:, :, t].sum(axis=1) for t in range(V.shape[0])])
    np.testing.assert_allclose(reduce(V, rfun="strength"), expected)


def test_pca_decoder_with_more_dims_than_features_emits_no_warning():
    rng = np.random.RandomState(0)
    data = np.array([rng.randn(8, 5) for _ in range(4)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = pca_decoder(data, nfolds=2, dims=10)
    assert len(result) > 0


def test_reduce_requests_no_more_dimensions_than_exist():
    # K=6 features but only 4 timepoints: must not trigger the hypertools warning
    a = seeded_vectorized_corrs(T=4, K=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = reduce(a, rfun="PCA")
    assert out.shape == (4, 6)


def test_smooth_shorter_than_window_keeps_series_length():
    from timecorr.helpers import smooth

    rng = np.random.RandomState(0)
    x = rng.randn(8, 3)
    out = smooth(x, windowsize=10)
    assert out.shape == x.shape
    # for a series at least as long as the window, out[t] = sum_j kernel[j] *
    # y[t + j - 4]: row 4 of the 10 x 10 kernel matrix weights sample j at offset
    # j - 4 (row convention; Task 6 fixed smooth, which used to apply the row
    # mirrored, as np.convolve(kernel, y, mode="same") does)
    y = rng.randn(30, 3)
    out = smooth(y, windowsize=10)
    assert out.shape == y.shape
    from timecorr.helpers import laplace_params, laplace_weights

    k = laplace_weights(10, params=laplace_params)
    kernel = k[4, :]  # windowsize 10: floor(10/2)-1 == ceil(10/2)-1 == 4
    kernel = kernel / kernel.sum()
    expected = np.zeros_like(y)
    for t in range(y.shape[0]):
        for j in range(10):
            if 0 <= t + j - 4 < y.shape[0]:
                expected[t] += kernel[j] * y[t + j - 4]
    for c in range(y.shape[1]):
        np.testing.assert_allclose(out[:, c], expected[:, c])


@pytest.mark.parametrize("rfun", ["PCA", {"model": "PCA", "params": {}}])
def test_one_element_list_returns_one_element_list(rfun):
    V = seeded_vectorized_corrs(seed=3)
    out = reduce([V], rfun=rfun)
    assert isinstance(out, list) and len(out) == 1
    assert np.array_equal(out[0], reduce(V, rfun=rfun))


def test_one_element_list_padded_output_stays_list():
    V = seeded_vectorized_corrs(T=4, K=5, seed=4)
    out = reduce([V], rfun="PCA", ndims=6)
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (4, 6)
