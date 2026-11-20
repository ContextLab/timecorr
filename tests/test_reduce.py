"""Tests for timecorr.helpers.reduce (real hypertools / scikit-learn models)."""

import os
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA

from timecorr.helpers import mat2vec, pca_decoder, reduce, smooth, vec2mat

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


# Graph-measure tests.  The references below are computed in the test, on the
# same machine as the code under test, by straightforward numpy routines, and
# the comparison is by tolerance: two correct implementations of the same
# linear algebra round differently under different BLAS/LAPACK builds
# (Accelerate on macOS, OpenBLAS on Linux), so exact equality is not portable.
#
# Tolerance derivation.  The seeded inputs are 5 x 5 correlation matrices.
# Strength: a sum of 5 terms of size <= 1, so the error is <= 4 * eps ~ 1e-15.
# Eigenvector centrality: the error is about eps * ||W|| / gap, where gap is
# the distance between the two largest eigenvalues (>= 0.03 here), so about
# 1e-13 or less.  PageRank: the error is about n * eps * cond(B) for the
# linear system B r = b (cond(B) <= 2e3 here), so about 1e-12 or less.
# ATOL = 1e-9 leaves a margin of at least 1e3 over every bound, while the
# measures change by ~1e-2 or more if their definition changes (for example
# the damping factor).
ATOL = 1e-9
D = 0.85


def _reference_strength(W):
    return W.sum(axis=1)


def _reference_eigenvector_centrality(W):
    # absolute value of the leading eigenvector of the symmetrized matrix
    # (eigh returns eigenvalues in ascending order, so the last column)
    S = (W + W.T) / 2
    _, vecs = np.linalg.eigh(S)
    return np.abs(vecs[:, -1])


def _reference_pagerank_centrality(W):
    # r solves (I - D * W diag(1/colsum)) r = (1 - D)/N * 1, then r / sum(r);
    # zero column sums are replaced by 1.  Computed here through an explicit
    # inverse (the package solves the system with a LU solve).
    N = W.shape[0]
    deg = W.sum(axis=0)
    deg = np.where(deg == 0, 1.0, deg)
    B = np.eye(N) - D * (W / deg[None, :])
    r = np.linalg.inv(B) @ np.full(N, (1 - D) / N)
    return r / r.sum()


REFERENCES = {
    "eigenvector_centrality": _reference_eigenvector_centrality,
    "pagerank_centrality": _reference_pagerank_centrality,
    "strength": _reference_strength,
}


def _reference_series(V, name):
    M = vec2mat(V)
    return np.stack([REFERENCES[name](M[:, :, t]) for t in range(M.shape[2])])


@pytest.mark.parametrize(
    "name", ["eigenvector_centrality", "pagerank_centrality", "strength"]
)
def test_graph_measures_match_independent_reference_on_seeded_input(name):
    V = seeded_vectorized_corrs()
    np.testing.assert_allclose(
        reduce(V, rfun=name), _reference_series(V, name), atol=ATOL, rtol=0
    )
    # a list input is reduced one element at a time
    out = reduce([V, V * 0.5], rfun=name)
    np.testing.assert_allclose(out[0], _reference_series(V, name), atol=ATOL, rtol=0)
    np.testing.assert_allclose(
        out[1], _reference_series(V * 0.5, name), atol=ATOL, rtol=0
    )


@pytest.mark.parametrize(
    "name", ["eigenvector_centrality", "pagerank_centrality", "strength"]
)
def test_graph_measures_match_saved_outputs_within_tolerance(name):
    # Saved input and saved outputs (written before the reduction refactor, on
    # another machine): the saved input is fed to the code, so both sides of
    # the comparison come from the file except for the code under test.
    ref = np.load(FIXTURE)
    V = ref["V"]
    np.testing.assert_allclose(reduce(V, rfun=name), ref[name], atol=ATOL, rtol=0)
    out = reduce([V, V * 0.5], rfun=name)
    np.testing.assert_allclose(out[0], ref[name + "_l0"], atol=ATOL, rtol=0)
    np.testing.assert_allclose(out[1], ref[name + "_l1"], atol=ATOL, rtol=0)


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
    # K=6 features but only 4 timepoints: must not trigger the hypertools
    # warning; the only warning is timecorr's own padding notice (Ruling R-U)
    a = seeded_vectorized_corrs(T=4, K=6)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = reduce(a, rfun="PCA")
    assert [(r.category, str(r.message)) for r in rec] == [(
        UserWarning,
        "reduce: the data supply at most 4 components (21 features, 4 samples) "
        "but 6 were requested; the output is padded with 2 zero columns.",
    )]
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
    with pytest.warns(UserWarning, match="2 zero columns"):  # R-U
        out = reduce([V], rfun="PCA", ndims=6)
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (4, 6)


def test_dict_rfun_with_params_and_kwargs_keeps_both():
    a = seeded_vectorized_corrs(T=40, K=6)
    rfun = {"model": "PCA", "params": {"svd_solver": "full"}, "kwargs": {"whiten": True}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = reduce(a, rfun=rfun, ndims=3)
    # whiten (from 'kwargs') reached the model: unit-variance components
    np.testing.assert_allclose(out.std(axis=0, ddof=1), 1.0, atol=1e-8)
    ref = reduce(a, rfun={"model": "PCA", "kwargs": {"whiten": True, "svd_solver": "full"}},
                 ndims=3)
    np.testing.assert_array_equal(out, ref)
    # the caller's dict is not modified
    assert rfun == {"model": "PCA", "params": {"svd_solver": "full"},
                    "kwargs": {"whiten": True}}


def test_dict_rfun_with_conflicting_params_and_kwargs_raises():
    a = seeded_vectorized_corrs(T=40, K=6)
    rfun = {"model": "PCA", "params": {"whiten": True}, "kwargs": {"whiten": False}}
    with pytest.raises(ValueError, match="whiten"):
        reduce(a, rfun=rfun, ndims=3)


def test_smooth_integer_input_matches_float_input():
    # integer data must not be truncated by an integer output array
    x = np.arange(20).reshape(10, 2) * 3
    out = smooth(x, windowsize=3)
    assert np.issubdtype(out.dtype, np.floating)
    assert np.allclose(out, smooth(x.astype(float), windowsize=3), rtol=0, atol=1e-12)
    assert not np.allclose(out, np.trunc(out))


# --- all-issues wave: ndims (R-I), zero-padding warning (R-U), UMAP (R-T)


@pytest.mark.parametrize("rfun", ["strength", "eigenvector_centrality", lambda xs: xs])
def test_ndims_with_a_graph_measure_or_callable_raises(rfun):
    # graph measures and callables define their own output width
    a = seeded_vectorized_corrs(T=10, K=4)
    with pytest.raises(ValueError, match="ndims"):
        reduce(a, rfun=rfun, ndims=2)
    reduce(a, rfun=rfun)  # without ndims they run


@pytest.mark.parametrize("ndims", [0, -1, 2.5, True, "3"])
def test_ndims_must_be_a_positive_integer(ndims):
    a = seeded_vectorized_corrs(T=10, K=4)
    with pytest.raises(ValueError, match="ndims must be a positive integer"):
        reduce(a, rfun="PCA", ndims=ndims)


def test_ndims_accepts_numpy_integers():
    a = seeded_vectorized_corrs(T=10, K=4)
    assert reduce(a, rfun="PCA", ndims=np.int64(3)).shape == (10, 3)


def test_zero_padding_warns_with_the_number_of_padded_columns():
    # 4 samples cannot give 6 components: 4 are fit, 2 zero columns are added
    a = seeded_vectorized_corrs(T=4, K=5)
    with pytest.warns(UserWarning, match="2 zero columns") as rec:
        out = reduce(a, rfun="PCA", ndims=6)
    assert len([r for r in rec if issubclass(r.category, UserWarning)]) == 1
    assert out.shape == (4, 6)
    assert np.all(out[:, 4:] == 0) and np.all(np.any(out[:, :4] != 0, axis=0))


def test_umap_runs_without_the_compatibility_layout_future_warning():
    a = seeded_vectorized_corrs(T=30, K=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = reduce(a, rfun="UMAP", ndims=2)
        again = reduce(a, rfun={"model": "UMAP", "kwargs": {"n_neighbors": 5}}, ndims=2)
    assert out.shape == (30, 2) and again.shape == (30, 2)
    assert np.all(np.isfinite(out)) and np.all(np.isfinite(again))


@pytest.mark.parametrize(
    "rfun", ["PCA", "IncrementalPCA", "TruncatedSVD", "FactorAnalysis", "KernelPCA",
             "SparsePCA", "UMAP"]
)
def test_reduce_is_deterministic_for_each_model(rfun):
    # random_state=0 is passed to every model: the same input gives the same output
    a = seeded_vectorized_corrs(T=40, K=6)
    first = reduce(a, rfun=rfun, ndims=3)
    second = reduce(a, rfun=rfun, ndims=3)
    np.testing.assert_array_equal(first, second)


# --- PageRank on signed correlations: ill-posed systems


def _previous_pagerank(W, d=0.85):
    # the package's algorithm at commit 863a412 (scipy LU solve, no checks)
    from scipy import linalg

    N = len(W)
    deg = np.sum(W, axis=0)
    deg[deg == 0] = 1
    B = np.eye(N) - d * np.dot(W, np.diag(1 / deg))
    r = linalg.solve(B, (1 - d) * (np.ones((N,)) / N))
    return r / np.sum(r)


def _agy_singular(a):
    # valid correlation matrix (positive definite for a near 0.054)
    return np.array([[1, -a, -a], [-a, 1, a], [-a, a, 1]], dtype=float)


_AGY_A = (57 - np.sqrt(2841)) / 68


def _pagerank_warnings(record):
    return [
        str(w.message) for w in record
        if issubclass(w.category, UserWarning) and "pagerank_centrality" in str(w.message)
    ]


@pytest.mark.parametrize(
    "M, expected, n_lstsq, n_uniform",
    [
        # 2 x 2, off-diagonal -3/37: B is singular, b in its range; the
        # minimum-norm solution is the symmetric one
        ((1 + 3 / 37) * np.eye(2) - 3 / 37, [0.5, 0.5], 1, 0),
        # 3 x 3, off-diagonal -1/19: two-dimensional null space
        ((1 + 1 / 19) * np.eye(3) - 1 / 19, [1 / 3] * 3, 1, 0),
        # agy's exactly singular 3 x 3 (|r| about 0.054): inconsistent system,
        # so the uniform vector
        (_agy_singular(_AGY_A), [1 / 3] * 3, 0, 1),
    ],
)
def test_pagerank_ill_posed_timepoint_falls_back_and_warns_once(
    M, expected, n_lstsq, n_uniform
):
    K = len(M)
    assert np.linalg.eigvalsh(M).min() > 0  # a valid correlation matrix
    V = np.stack([mat2vec(np.eye(K)), mat2vec(M), mat2vec(np.eye(K))])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = reduce(V, rfun="pagerank_centrality")
    assert out.shape == (3, K) and np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1], expected, rtol=0, atol=1e-12)
    # the well-posed timepoints are what the previous algorithm gave
    for t in (0, 2):
        assert np.array_equal(out[t], _previous_pagerank(vec2mat(V)[:, :, t]))
    messages = _pagerank_warnings(record)
    assert len(messages) == 1
    message = messages[0]
    assert "1 of 3 timepoints" in message
    assert f"{n_lstsq} used the minimum-norm least-squares solution" in message
    assert f"{n_uniform} the uniform vector" in message
    assert "not always defined" in message


def test_pagerank_one_warning_per_reduce_call_for_lists():
    bad = _agy_singular(_AGY_A)
    V1 = np.stack([mat2vec(np.eye(3)), mat2vec(bad)])
    V2 = np.stack([mat2vec(bad), mat2vec(bad), mat2vec(np.eye(3))])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = reduce([V1, V2], rfun="pagerank_centrality")
    assert isinstance(out, list) and len(out) == 2
    messages = _pagerank_warnings(record)
    assert len(messages) == 1
    assert "3 of 5 timepoints" in messages[0]
    assert "0 used the minimum-norm least-squares solution" in messages[0]
    assert "3 the uniform vector" in messages[0]
    # a second call warns again (one warning per call, not per session)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        reduce(V1, rfun="pagerank_centrality")
    assert len(_pagerank_warnings(record)) == 1


def test_pagerank_well_posed_input_does_not_warn():
    V = seeded_vectorized_corrs(T=20, K=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reduce(V, rfun="pagerank_centrality")


def test_pagerank_well_posed_outputs_unchanged_bit_for_bit():
    rng = np.random.RandomState(7)
    mats = [np.corrcoef(rng.randn(6, 10)) for _ in range(25)]
    mats.append((1 - (-3 / 37)) * np.eye(4)[:4, :4] + (-3 / 37))  # K 4: well posed
    # badly scaled (a column sum of -2e-12) but well posed: the solve is kept
    mats.append(_agy_singular(0.5 + 1e-12))
    for M in mats:
        V = mat2vec(M)[None]
        got = reduce(V, rfun="pagerank_centrality")[0]
        # the matrix as reduce sees it (after vec2mat)
        assert np.array_equal(got, _previous_pagerank(vec2mat(V)[:, :, 0]))


# --- PageRank on non-negative weights |r| ('pagerank_centrality_abs')


def _power_iteration_pagerank(A, d=0.85, tol=1e-15, max_iter=10000):
    # Textbook PageRank by power iteration on the non-negative weights A
    # (column j gives the out-links of node j, as in the package), uniform
    # teleportation, and the mass of nodes without links (zero columns)
    # spread uniformly.  Independent of the package's linear solve.
    N = A.shape[0]
    out = A.sum(axis=0)
    dangling = out == 0
    P = np.divide(A, out[None, :], out=np.zeros_like(A), where=~dangling[None, :])
    r = np.full(N, 1.0 / N)
    for _ in range(max_iter):
        new = d * (P @ r) + d * r[dangling].sum() / N + (1 - d) / N
        if np.abs(new - r).sum() < tol:
            return new
        r = new
    raise AssertionError("power iteration did not converge")


def _abs_pagerank_reference_series(V):
    M = vec2mat(V)
    return np.stack(
        [_power_iteration_pagerank(np.abs(M[:, :, t])) for t in range(M.shape[2])]
    )


def _signed_test_matrices():
    rng = np.random.RandomState(11)
    mats = [
        (1 + 3 / 37) * np.eye(2) - 3 / 37,       # singular signed system
        (1 + 1 / 19) * np.eye(3) - 1 / 19,       # two-dimensional null space
        _agy_singular(_AGY_A),                   # inconsistent signed system
        _agy_singular(0.5 + 1e-12),              # column sum near zero
        np.zeros((4, 4)),                        # no links at all
        np.diag([1.0, 0.0, 2.0, 0.0]),           # isolated nodes
        np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float),
        1e300 * np.corrcoef(rng.randn(5, 9)),    # large weights
        1e-300 * np.corrcoef(rng.randn(5, 9)),   # tiny weights
    ]
    for K in (3, 6, 12):
        for _ in range(5):
            S = rng.uniform(-1, 1, (K, K))
            mats.append((S + S.T) / 2)           # symmetric, signed, any diagonal
        mats.append(np.corrcoef(rng.randn(K, K + 2)))
    return mats


def test_abs_pagerank_matches_power_iteration_without_warning():
    for M in _signed_test_matrices():
        V = mat2vec(M)[None]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = reduce(V, rfun="pagerank_centrality_abs")
        assert got.shape == (1, len(M)) and np.all(np.isfinite(got))
        assert np.all(got > 0)
        np.testing.assert_allclose(got[0].sum(), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            got, _abs_pagerank_reference_series(V), rtol=0, atol=1e-12
        )


def test_abs_pagerank_on_seeded_series_and_lists():
    V = seeded_vectorized_corrs(T=40, K=7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = reduce([V, -V], rfun="pagerank_centrality_abs")
    expected = _abs_pagerank_reference_series(V)
    np.testing.assert_allclose(out[0], expected, rtol=0, atol=1e-12)
    # the sign of the weights does not matter
    np.testing.assert_allclose(out[1], expected, rtol=0, atol=1e-12)


def test_abs_pagerank_equals_signed_pagerank_on_non_negative_weights():
    rng = np.random.RandomState(5)
    for _ in range(10):
        M = np.abs(np.corrcoef(rng.randn(6, 9)))
        V = mat2vec(M)[None]
        np.testing.assert_allclose(
            reduce(V, rfun="pagerank_centrality_abs"),
            reduce(V, rfun="pagerank_centrality"),
            rtol=0, atol=1e-12,
        )


def test_abs_pagerank_differs_from_signed_pagerank_on_signed_weights():
    V = seeded_vectorized_corrs(T=10, K=5)
    a = reduce(V, rfun="pagerank_centrality_abs")
    s = reduce(V, rfun="pagerank_centrality")
    assert np.abs(a - s).max() > 1e-3


def test_abs_pagerank_non_finite_timepoint_gives_nan_row():
    V = seeded_vectorized_corrs(T=3, K=4)
    V[1, 2] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = reduce(V, rfun="pagerank_centrality_abs")
    assert np.all(np.isnan(out[1]))
    np.testing.assert_allclose(
        out[[0, 2]], _abs_pagerank_reference_series(V[[0, 2]]), rtol=0, atol=1e-12
    )


def test_abs_pagerank_is_a_graph_measure_not_a_model():
    from timecorr.helpers import _check_rfun

    _check_rfun("pagerank_centrality_abs")
    with pytest.raises(ValueError, match="graph measures are given"):
        _check_rfun({"model": "pagerank_centrality_abs"})
    with pytest.raises(ValueError, match="ndims"):
        reduce(seeded_vectorized_corrs(), rfun="pagerank_centrality_abs", ndims=2)
