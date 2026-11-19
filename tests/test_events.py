"""Tests for timecorr.events (boundaries, specification grammar, kernel mask) and for
event-restricted kernels in ``timecorr`` and the correlation functions."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

import timecorr as tc
from timecorr.events import (
    block_labels,
    check_boundaries,
    distinct_event_sets,
    mask_kernel,
    resolve_events,
    resolve_events_by_order,
)
from timecorr.helpers import eye_weights, gaussian_weights, mat2vec, one_sided, r2z, z2r

T = 12
FEATURES = np.random.RandomState(0).randn(T, 3)


# ---------------------------------------------------------------- check_boundaries
def test_check_boundaries_sorted_unique_with_zero():
    out = check_boundaries([7, 3, 3, 5], T)
    assert out.dtype == np.int64
    np.testing.assert_array_equal(out, [0, 3, 5, 7])


def test_check_boundaries_zero_and_duplicates_equal_without():
    # Review focus 1
    np.testing.assert_array_equal(check_boundaries([0, 4, 4, 8, 8], T), check_boundaries([4, 8], T))


def test_check_boundaries_empty_is_one_block():
    np.testing.assert_array_equal(check_boundaries([], T), [0])
    np.testing.assert_array_equal(check_boundaries(np.array([]), T), [0])


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.array([2, 5], dtype=np.int32),
        lambda: np.array([2, 5], dtype=np.uint8),
        lambda: pd.Series([2, 5]),
        lambda: pl.Series([2, 5]),
        lambda: (2, 5),
        lambda: [np.int64(2), np.int64(5)],
    ],
)
def test_check_boundaries_containers(make):
    # Review focus 4
    np.testing.assert_array_equal(check_boundaries(make(), T), [0, 2, 5])


@pytest.mark.parametrize(
    "bad",
    [
        [[1, 2], [3, 4]],  # 2-D
        np.array([1.0, 2.0]),  # integral floats still rejected
        [1.5, 2],
        np.array([True, False]),
        [True, 3],
        [-1],
        [T],
        [T + 5],
        ["a"],
        3,  # scalar (0-D)
    ],
)
def test_check_boundaries_errors(bad):
    with pytest.raises(ValueError):
        check_boundaries(bad, T)


def test_check_boundaries_bad_T():
    with pytest.raises(ValueError):
        check_boundaries([1], 0)


# ---------------------------------------------------------------- block_labels
def test_block_labels():
    labels = block_labels(np.array([0, 3, 7]), 10)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    assert labels.dtype.kind == "i"
    assert len(labels) == 10


def test_block_labels_validates():
    with pytest.raises(ValueError):
        block_labels(np.array([0, 10]), 10)


# ---------------------------------------------------------------- resolve_events grammar
def test_none_is_no_restriction():
    assert resolve_events(None, T, 3) is None


def test_boundary_array_shared():
    out = resolve_events(np.array([4, 8]), T, 3)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [0, 4, 8])


def test_boundary_list_shared():
    np.testing.assert_array_equal(resolve_events([4, 8], T, 3), [0, 4, 8])
    np.testing.assert_array_equal(resolve_events((4, 8), T, 3), [0, 4, 8])
    np.testing.assert_array_equal(resolve_events([np.int64(4), 8], T, 3), [0, 4, 8])


def test_empty_list_is_one_block():
    np.testing.assert_array_equal(resolve_events([], T, 3), [0])


def test_boundary_series_shared():
    np.testing.assert_array_equal(resolve_events(pd.Series([4, 8]), T, 2), [0, 4, 8])
    np.testing.assert_array_equal(resolve_events(pl.Series([4, 8]), T, 2), [0, 4, 8])


def test_single_subject_list_and_bare_agree():
    # Review focus 2
    bare = resolve_events(np.array([4, 8]), T, 1)
    wrapped = resolve_events([np.array([4, 8])], T, 1)
    assert isinstance(wrapped, list) and len(wrapped) == 1
    np.testing.assert_array_equal(wrapped[0], bare)
    wrapped_list = resolve_events([[4, 8]], T, 1)
    np.testing.assert_array_equal(wrapped_list[0], bare)


def test_per_subject_list():
    out = resolve_events([[4], None, np.array([2, 9]), []], T, 4)
    assert isinstance(out, list) and len(out) == 4
    np.testing.assert_array_equal(out[0], [0, 4])
    assert out[1] is None
    np.testing.assert_array_equal(out[2], [0, 2, 9])
    np.testing.assert_array_equal(out[3], [0])


def test_per_subject_tuple_and_series():
    out = resolve_events((pd.Series([3]), (5,)), T, 2)
    np.testing.assert_array_equal(out[0], [0, 3])
    np.testing.assert_array_equal(out[1], [0, 5])


def test_per_subject_wrong_length():
    with pytest.raises(ValueError, match="subjects"):
        resolve_events([[4], [5]], T, 3)


def test_per_subject_all_none():
    out = resolve_events([None, None], T, 2)
    assert out == [None, None]


def test_per_subject_nested_list_is_error():
    with pytest.raises(ValueError, match="array or DataFrame"):
        resolve_events([[[1.0, 2.0], [3.0, 4.0]], [3]], T, 2)


def test_per_subject_scalar_entries_error():
    with pytest.raises(ValueError):
        resolve_events([1.5, 2.5], T, 2)
    with pytest.raises(ValueError):
        resolve_events([True, False], T, 2)


def test_boundary_errors_through_resolve():
    with pytest.raises(ValueError):
        resolve_events([T], T, 1)
    with pytest.raises(ValueError):
        resolve_events(np.array([1.0, 2.0]), T, 1)
    with pytest.raises(ValueError):
        resolve_events(np.array([True, False]), T, 1)
    with pytest.raises(ValueError):
        resolve_events([[4], [-1]], T, 2)
    with pytest.raises(ValueError):
        resolve_events(np.zeros((2, 2, 2)), T, 1)


def test_dict_not_allowed_in_single_order():
    with pytest.raises(ValueError):
        resolve_events({1: [3]}, T, 1)


def test_numpy_bool_not_integer_in_list():
    with pytest.raises(ValueError):
        resolve_events([np.bool_(True), np.bool_(False)], T, 2)


# ---------------------------------------------------------------- feature entries
FEATURE_ENTRIES = {
    "ndarray": FEATURES,
    "pandas": pd.DataFrame(FEATURES),
    "polars": pl.DataFrame({"a": FEATURES[:, 0], "b": FEATURES[:, 1]}),
}


@pytest.mark.parametrize("name", list(FEATURE_ENTRIES))
def test_features_entry_reaches_segmentation(name):
    # Review focus 4: a features entry is segmented with tc.segment.
    out = resolve_events(FEATURE_ENTRIES[name], T, 2)
    assert isinstance(out, np.ndarray) and out.dtype == np.int64 and out[0] == 0
    np.testing.assert_array_equal(out, tc.segment(FEATURE_ENTRIES[name]))


@pytest.mark.parametrize("name", list(FEATURE_ENTRIES))
def test_features_entry_wrong_rows(name):
    entry = FEATURE_ENTRIES[name]
    with pytest.raises(ValueError, match=rf"{T}.*boundaries|boundaries.*{T}"):
        resolve_events(entry, T + 3, 2)


def test_features_in_per_subject_list_reach_segmentation():
    out = resolve_events([FEATURES, [3]], T, 2)
    np.testing.assert_array_equal(out[0], tc.segment(FEATURES))
    np.testing.assert_array_equal(out[1], [0, 3])


def test_features_wrong_rows_message_names_both_lengths():
    with pytest.raises(ValueError) as exc:
        resolve_events(FEATURES, 20, 2)
    assert "12" in str(exc.value) and "20" in str(exc.value)


def test_events_params_unknown_key():
    with pytest.raises(ValueError, match="bogus"):
        resolve_events(FEATURES, T, 2, events_params={"n": 3, "bogus": 1})


def test_events_params_validated_before_segmentation():
    # Unknown key is a ValueError raised before any segmentation.
    with pytest.raises(ValueError):
        resolve_events(FEATURES, T, 2, events_params={"nn": 3})


def test_events_params_allowed_keys_reach_segmentation():
    params = {"n": 3, "kind": "ordered", "max_events": 5, "random_state": 0}
    out = resolve_events(FEATURES, T, 2, events_params=params)
    np.testing.assert_array_equal(out, tc.segment(FEATURES, **params))
    assert len(out) == 3


def test_events_params_without_features():
    with pytest.raises(ValueError, match="features"):
        resolve_events([3, 6], T, 2, events_params={"n": 3})
    with pytest.raises(ValueError, match="features"):
        resolve_events(None, T, 2, events_params={"n": 3})
    with pytest.raises(ValueError, match="features"):
        resolve_events([[3], None], T, 2, events_params={"n": 3})


def test_events_params_must_be_dict():
    with pytest.raises(ValueError):
        resolve_events(FEATURES, T, 2, events_params=[("n", 3)])


# ---------------------------------------------------------------- resolve_events_by_order
def test_by_order_single_spec_applies_to_every_order():
    out = resolve_events_by_order([4, 8], T, 2, 3)
    assert sorted(out) == [1, 2, 3]
    for k in (1, 2, 3):
        np.testing.assert_array_equal(out[k], [0, 4, 8])


def test_by_order_none():
    out = resolve_events_by_order(None, T, 2, 2)
    assert out == {1: None, 2: None}


def test_by_order_dict():
    out = resolve_events_by_order({1: [4], 3: [[2], None]}, T, 2, 3)
    assert sorted(out) == [1, 2, 3]
    np.testing.assert_array_equal(out[1], [0, 4])
    assert out[2] is None
    assert isinstance(out[3], list)
    np.testing.assert_array_equal(out[3][0], [0, 2])
    assert out[3][1] is None


def test_by_order_numpy_integer_key():
    out = resolve_events_by_order({np.int64(2): [4]}, T, 1, 2)
    np.testing.assert_array_equal(out[2], [0, 4])
    assert out[1] is None


@pytest.mark.parametrize("key", [0, 4, -1, True, 1.0, "1", None, 2.5])
def test_by_order_bad_keys(key):
    # Review focus 5
    with pytest.raises(ValueError):
        resolve_events_by_order({key: [3]}, T, 1, 3)


def test_by_order_dict_value_dict_is_error():
    with pytest.raises(ValueError):
        resolve_events_by_order({1: {1: [3]}}, T, 1, 2)


def test_by_order_validates_order():
    with pytest.raises(ValueError):
        resolve_events_by_order([3], T, 1, 0)


def test_by_order_features_reach_segmentation():
    expected = tc.segment(FEATURES)
    out = resolve_events_by_order({2: FEATURES}, T, 2, 2)
    assert out[1] is None
    np.testing.assert_array_equal(out[2], expected)
    out = resolve_events_by_order(FEATURES, T, 2, 2)
    np.testing.assert_array_equal(out[1], expected)
    np.testing.assert_array_equal(out[2], expected)


def test_by_order_flat_params_need_features_somewhere():
    with pytest.raises(ValueError, match="features"):
        resolve_events_by_order({1: [3]}, T, 1, 2, events_params={"n": 2})
    # features at one order, boundaries at another: flat params are valid
    out = resolve_events_by_order({1: [3], 2: FEATURES}, T, 1, 2, events_params={"n": 2})
    np.testing.assert_array_equal(out[1], [0, 3])
    np.testing.assert_array_equal(out[2], tc.segment(FEATURES, n=2))


def test_by_order_params_keyed_by_order():
    out = resolve_events_by_order({1: [3], 2: FEATURES}, T, 1, 2, events_params={2: {"n": 2}})
    np.testing.assert_array_equal(out[1], [0, 3])
    np.testing.assert_array_equal(out[2], tc.segment(FEATURES, n=2))
    # keyed by an order that holds no features
    with pytest.raises(ValueError, match="features"):
        resolve_events_by_order({1: [3], 2: FEATURES}, T, 1, 2, events_params={1: {"n": 2}})
    # keyed by order but events is not a dict
    with pytest.raises(ValueError):
        resolve_events_by_order(FEATURES, T, 1, 2, events_params={1: {"n": 2}})
    # bad key inside an order's params
    with pytest.raises(ValueError):
        resolve_events_by_order({2: FEATURES}, T, 1, 2, events_params={2: {"bogus": 2}})
    # mixed keys
    with pytest.raises(ValueError):
        resolve_events_by_order({2: FEATURES}, T, 1, 2, events_params={2: {"n": 2}, "n": 3})


def test_by_order_params_unknown_flat_key():
    with pytest.raises(ValueError, match="bogus"):
        resolve_events_by_order({2: FEATURES}, T, 1, 2, events_params={"bogus": 2})


# ---------------------------------------------------------------- mask_kernel
def test_mask_kernel_none_returns_same_object():
    w = gaussian_weights(T, {"var": 4})
    assert mask_kernel(w, None) is w


def test_mask_kernel_zero_exactly_across_blocks_untouched_within():
    w = gaussian_weights(T, {"var": 4})
    original = w.copy()
    b = np.array([0, 5, 9])
    out = mask_kernel(w, b)
    np.testing.assert_array_equal(w, original)  # input never mutated
    assert out is not w
    labels = block_labels(b, T)
    same = labels[:, None] == labels[None, :]
    assert np.all(out[~same] == 0)
    np.testing.assert_array_equal(out[same], original[same])
    # the kernel really had weight across blocks
    assert np.any(original[~same] != 0)


def test_mask_kernel_single_block_equal_but_new():
    w = gaussian_weights(T, {"var": 4})
    for b in (np.array([0]), check_boundaries([], T)):
        out = mask_kernel(w, b)
        np.testing.assert_array_equal(out, w)
        assert out is not w


def test_mask_kernel_delta_passes_through():
    w = eye_weights(T, None)
    out = mask_kernel(w, np.array([0, 3, 6]))
    np.testing.assert_array_equal(out, w)


def test_mask_kernel_all_short_blocks_raises():
    w = gaussian_weights(T, {"var": 4})
    b = np.arange(0, T, 2)  # blocks of length 2: no row has 3 weights
    with pytest.raises(ValueError, match="no block is long enough"):
        mask_kernel(w, b)
    b = np.arange(T)  # singleton blocks
    with pytest.raises(ValueError, match="no block is long enough"):
        mask_kernel(w, b)


def test_mask_kernel_mixed_short_blocks_ok():
    w = gaussian_weights(T, {"var": 4})
    b = np.array([0, 2, 8])  # blocks of length 2, 6, 4
    out = mask_kernel(w, b)
    assert np.sum(out != 0, axis=1).max() >= 3


def test_mask_kernel_one_sided_kernel():
    w = np.tril(gaussian_weights(T, {"var": 4}))
    out = mask_kernel(w, np.array([0, 6]))
    labels = block_labels(np.array([0, 6]), T)
    assert np.all(out[labels[:, None] != labels[None, :]] == 0)
    assert np.all(np.triu(out, 1) == 0)


def test_mask_kernel_nan_not_propagated_across_blocks():
    w = gaussian_weights(T, {"var": 4})
    w[0, T - 1] = np.nan
    out = mask_kernel(w, np.array([0, 6]))
    assert out[0, T - 1] == 0


def test_mask_kernel_validates():
    w = gaussian_weights(T, {"var": 4})
    with pytest.raises(ValueError):
        mask_kernel(w, np.array([0, T]))
    with pytest.raises(ValueError):
        mask_kernel(w[:, :5], np.array([0, 3]))


# ---------------------------------------------------------------- distinct_event_sets
def test_distinct_none_and_shared():
    out = distinct_event_sets(None, 3)
    assert len(out) == 1 and out[0][0] is None and out[0][1] == [0, 1, 2]
    b = np.array([0, 4])
    out = distinct_event_sets(b, 3)
    assert len(out) == 1 and out[0][1] == [0, 1, 2]
    np.testing.assert_array_equal(out[0][0], b)


def test_distinct_groups_identical_boundaries():
    resolved = resolve_events([[4], [2, 6], np.array([4]), None, [2, 6], None], T, 6)
    groups = distinct_event_sets(resolved, 6)
    assert [g[1] for g in groups] == [[0, 2], [1, 4], [3, 5]]
    np.testing.assert_array_equal(groups[0][0], [0, 4])
    np.testing.assert_array_equal(groups[1][0], [0, 2, 6])
    assert groups[2][0] is None


def test_distinct_every_subject_once():
    resolved = resolve_events([[1], [2], [3]], T, 3)
    groups = distinct_event_sets(resolved, 3)
    assert sorted(i for _, idx in groups for i in idx) == [0, 1, 2]


def test_distinct_wrong_length():
    with pytest.raises(ValueError):
        distinct_event_sets([np.array([0])], 3)


# ---------------------------------------------------------------- Task 1 review cleanups
@pytest.mark.parametrize("bad", [[], 0, "", ()])
def test_by_order_falsy_non_dict_params_raise(bad):
    with pytest.raises(ValueError):
        resolve_events_by_order([4], T, 2, 2, events_params=bad)


def test_by_order_empty_params_dict_is_no_params():
    out = resolve_events_by_order([4], T, 2, 2, events_params={})
    np.testing.assert_array_equal(out[1], [0, 4])
    np.testing.assert_array_equal(out[2], [0, 4])


@pytest.mark.parametrize("bad", [[], 0, ""])
def test_resolve_falsy_non_dict_params_raise(bad):
    with pytest.raises(ValueError):
        resolve_events([4], T, 2, events_params=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, np.True_, "2", None])
def test_n_subjects_must_be_positive_integer(bad):
    with pytest.raises(ValueError):
        resolve_events([4], T, bad)
    with pytest.raises(ValueError):
        resolve_events_by_order([4], T, bad, 2)
    with pytest.raises(ValueError):
        distinct_event_sets(None, bad)


def test_n_subjects_numpy_integer_ok():
    np.testing.assert_array_equal(resolve_events([4], T, np.int64(2)), [0, 4])


def test_distinct_none_and_single_block_are_separate_groups():
    groups = distinct_event_sets([None, np.array([0])], 2)
    assert [g[1] for g in groups] == [[0], [1]]


# ================================================================ timecorr(events=...)
NT = 60
K = 4
S = 3
B = np.array([20, 38])  # blocks [0, 20), [20, 38), [38, 60)
EDGES = [0, 20, 38, 60]
PARAMS = {"var": 25}
SW = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 0.5], [3.0, 1.0, 0.0]])
PER_SUBJECT = [np.array([20, 38]), np.array([15, 40]), np.array([30])]


def _subjects(seed=0, n=S):
    rng = np.random.RandomState(seed)
    return [rng.randn(NT, K) for _ in range(n)]


def _kernel(include):
    w = tc.gaussian_weights(NT, PARAMS)
    if include == "backward":
        w = one_sided(w, "backward", True)
    return w


def _fixed_wisfc(data, weights):
    n = len(data) if isinstance(data, list) else 1
    return tc.wisfc(data, weights, subject_weights=SW[:n, :n])


CFUNS = {"autofc": tc.autofc, "isfc": tc.isfc, "smooth": None, "wisfc": _fixed_wisfc}


def _count_user_warnings(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn()
    return out, [w for w in rec if issubclass(w.category, UserWarning)]


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


# ---------------------------------------------------------------- (a) split-and-stitch oracle
@pytest.mark.parametrize("include", ["all", "backward"])
@pytest.mark.parametrize("name", list(CFUNS))
def test_split_and_stitch_oracle(name, include):
    cfun = CFUNS[name]
    data = _subjects()
    full = _kernel(include)
    got = _quiet(
        lambda: tc.timecorr(
            data, weights_params=PARAMS, include_timepoints=include, cfun=cfun, events=B
        )
    )
    pieces = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sub = full[a:b, a:b]
        pieces.append(
            _quiet(
                lambda sub=sub, a=a, b=b: tc.timecorr(
                    [d[a:b] for d in data],
                    weights_function=lambda n, p, sub=sub: sub,
                    cfun=cfun,
                )
            )
        )
    for s in range(S):
        expected = np.vstack([p[s] for p in pieces])
        assert got[s].shape == expected.shape
        assert np.allclose(got[s], expected, atol=1e-10)


# ---------------------------------------------------------------- (b) no restriction
@pytest.mark.parametrize("name", list(CFUNS))
def test_events_none_and_empty_reproduce_today(name):
    cfun = CFUNS[name]
    data = _subjects()
    base = tc.timecorr(data, weights_params=PARAMS, cfun=cfun)
    for ev in (None, []):
        out = tc.timecorr(data, weights_params=PARAMS, cfun=cfun, events=ev)
        for s in range(S):
            assert np.array_equal(out[s], base[s])


def test_events_empty_single_subject_reproduces_today():
    x = _subjects()[0]
    base = tc.timecorr(x, weights_params=PARAMS)
    assert np.array_equal(tc.timecorr(x, weights_params=PARAMS, events=[]), base)
    assert np.array_equal(tc.timecorr(x, weights_params=PARAMS, events=[0]), base)


# ---------------------------------------------------------------- (c) per-subject events
def test_per_subject_autofc_equals_single_subject_calls():
    data = _subjects()
    out = tc.timecorr(data, weights_params=PARAMS, cfun=tc.autofc, events=PER_SUBJECT)
    for s in range(S):
        single = tc.timecorr(data[s], weights_params=PARAMS, cfun=tc.autofc, events=PER_SUBJECT[s])
        np.testing.assert_allclose(out[s], single, rtol=0, atol=1e-12)


def test_per_subject_smoothing_equals_single_subject_calls():
    data = _subjects()
    out = tc.timecorr(data, weights_params=PARAMS, cfun=None, events=PER_SUBJECT)
    for s in range(S):
        single = tc.timecorr(data[s], weights_params=PARAMS, cfun=None, events=PER_SUBJECT[s])
        np.testing.assert_allclose(out[s], single, rtol=0, atol=1e-12)


def test_per_subject_isfc_matches_oracle():
    data = _subjects()
    out = tc.timecorr(data, weights_params=PARAMS, cfun=tc.isfc, events=PER_SUBJECT)
    full = tc.gaussian_weights(NT, PARAMS)
    for s in range(S):
        w_s = mask_kernel(full, PER_SUBJECT[s])
        others = np.mean([data[o] for o in range(S) if o != s], axis=0)
        c = tc.wcorr(data[s], others, w_s)
        sym = z2r((r2z(c) + r2z(c.transpose(1, 0, 2))) / 2)
        assert np.allclose(out[s], mat2vec(sym), atol=1e-10)


@pytest.mark.parametrize("name", list(CFUNS))
def test_per_subject_identical_equals_shared(name):
    cfun = CFUNS[name]
    data = _subjects()
    shared = tc.timecorr(data, weights_params=PARAMS, cfun=cfun, events=B)
    for ev in ([B, B, B], [B.copy(), list(B), tuple(int(b) for b in B)]):
        per = tc.timecorr(data, weights_params=PARAMS, cfun=cfun, events=list(ev))
        for s in range(S):
            assert np.array_equal(per[s], shared[s])


def test_per_subject_none_entry_is_unrestricted():
    data = _subjects()
    out = tc.timecorr(data, weights_params=PARAMS, cfun=tc.autofc, events=[B, None, [0]])
    base = tc.timecorr(data, weights_params=PARAMS, cfun=tc.autofc)
    np.testing.assert_allclose(out[1], base[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out[2], base[2], rtol=0, atol=1e-12)
    assert not np.allclose(out[0], base[0])


# ---------------------------------------------------------------- correlation functions, list form
@pytest.mark.parametrize("fn", [tc.wisfc, tc.isfc, tc.autofc])
def test_list_of_kernels_takes_subject_kernel(fn):
    data = _subjects()
    full = tc.gaussian_weights(NT, PARAMS)
    kernels = [mask_kernel(full, b) for b in PER_SUBJECT]
    out = fn(data, kernels)
    for s in range(S):
        np.testing.assert_allclose(out[s], fn(data, kernels[s])[s], rtol=0, atol=1e-12)


def test_list_of_kernels_wisfc_fixed_subject_weights():
    data = _subjects()
    full = tc.gaussian_weights(NT, PARAMS)
    kernels = [mask_kernel(full, b) for b in PER_SUBJECT]
    out = tc.wisfc(data, kernels, subject_weights=SW)
    for s in range(S):
        ref = tc.wisfc(data, kernels[s], subject_weights=SW)[s]
        np.testing.assert_allclose(out[s], ref, rtol=0, atol=1e-12)


def test_list_of_kernels_single_array_data():
    x = _subjects()[0]
    w = mask_kernel(tc.gaussian_weights(NT, PARAMS), B)
    np.testing.assert_array_equal(tc.autofc(x, [w]), tc.autofc(x, w))


@pytest.mark.parametrize("fn", [tc.wisfc, tc.isfc, tc.autofc])
def test_list_of_kernels_wrong_length(fn):
    data = _subjects()
    w = tc.gaussian_weights(NT, PARAMS)
    with pytest.raises(ValueError):
        fn(data, [w, w])


def test_list_of_equal_kernels_warns_as_single_kernel():
    data = _subjects()
    w = mask_kernel(tc.gaussian_weights(NT, PARAMS), [2, 4, 30])
    _, single = _count_user_warnings(lambda: tc.isfc(data, w))
    _, listed = _count_user_warnings(lambda: tc.isfc(data, [w.copy() for _ in range(S)]))
    assert len(single) == 1 and len(listed) == 1
    assert str(single[0].message) == str(listed[0].message)


def test_list_of_different_kernels_one_warning():
    data = _subjects()
    full = tc.gaussian_weights(NT, PARAMS)
    kernels = [mask_kernel(full, b) for b in ([2, 4, 30], [10], [50, 52])]
    _, rec = _count_user_warnings(lambda: tc.isfc(data, kernels))
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert "fewer than 3" in msg
    # each kernel with a problem is prefixed by the subject that uses it; subject 1's
    # kernel has no short rows, so the message names only subjects 0 and 2
    assert msg.startswith("Subject 0: ")
    assert " Subject 2: " in msg
    assert "Subject 1" not in msg and "Subjects" not in msg


@pytest.mark.parametrize("fn", [tc.wisfc, tc.isfc, tc.autofc])
@pytest.mark.parametrize("n_subjects", [S, 6])
def test_nested_list_kernel_is_one_kernel(fn, n_subjects):
    # a T x T kernel given as a Python list of lists is one shared kernel, also when
    # T equals the number of subjects
    rng = np.random.RandomState(5)
    n_time = 6
    data = [rng.randn(n_time, 4) for _ in range(n_subjects)]
    w = tc.gaussian_weights(n_time, {"var": 4})
    ref = fn(data, w)
    for given in (w.tolist(), tuple(map(tuple, w.tolist()))):
        out = fn(data, given)
        for s in range(n_subjects):
            assert np.array_equal(out[s], ref[s])


@pytest.mark.parametrize("fn", [tc.wisfc, tc.isfc, tc.autofc])
def test_list_mixing_kernels_and_rows_raises(fn):
    data = _subjects()
    w = tc.gaussian_weights(NT, PARAMS)
    with pytest.raises(ValueError, match="timepoint_weights"):
        fn(data, [w, w[0], w])


# ---------------------------------------------------------------- (d) custom cfun
def test_custom_cfun_per_subject_matches_builtin():
    data = _subjects()
    calls = []

    def my_isfc(d, w):
        calls.append(w)
        return tc.isfc(d, w)

    got = tc.timecorr(data, weights_params=PARAMS, cfun=my_isfc, events=PER_SUBJECT)
    ref = tc.timecorr(data, weights_params=PARAMS, cfun=tc.isfc, events=PER_SUBJECT)
    for s in range(S):
        np.testing.assert_allclose(got[s], ref[s], rtol=0, atol=1e-12)
    assert len(calls) == 3


def test_custom_cfun_called_once_per_distinct_kernel():
    data = _subjects()
    calls = []

    def my_autofc(d, w):
        calls.append(w)
        return tc.autofc(d, w)

    got = tc.timecorr(data, weights_params=PARAMS, cfun=my_autofc, events=[B, [30], list(B)])
    ref = tc.timecorr(data, weights_params=PARAMS, cfun=tc.autofc, events=[B, [30], list(B)])
    assert len(calls) == 2
    for s in range(S):
        np.testing.assert_allclose(got[s], ref[s], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        lambda d, w: tc.isfc(d, w)[0],  # one array
        lambda d, w: tc.isfc(d, w)[:2],  # wrong length
        lambda d, w: tuple(tc.isfc(d, w)),  # not a list
    ],
)
def test_custom_cfun_bad_return(bad):
    with pytest.raises(ValueError):
        tc.timecorr(_subjects(), weights_params=PARAMS, cfun=bad, events=PER_SUBJECT)


# ---------------------------------------------------------------- (e) isolation
def test_isolation_block_two_does_not_affect_block_one():
    x = _subjects()[0]
    y = x.copy()
    y[20:38] = np.random.RandomState(9).randn(18, K) * 5 + 3
    a = tc.timecorr(x, weights_params=PARAMS, cfun=tc.autofc, events=B)
    b = tc.timecorr(y, weights_params=PARAMS, cfun=tc.autofc, events=B)
    np.testing.assert_allclose(a[:20], b[:20], rtol=0, atol=1e-12)
    np.testing.assert_allclose(a[38:], b[38:], rtol=0, atol=1e-12)
    assert not np.allclose(a[20:38], b[20:38])
    # without events, block 1 does change
    c = tc.timecorr(x, weights_params=PARAMS, cfun=tc.autofc)
    d = tc.timecorr(y, weights_params=PARAMS, cfun=tc.autofc)
    assert not np.allclose(c[:20], d[:20], atol=1e-6)


# ---------------------------------------------------------------- (f) short blocks, exclude
def test_backward_without_zero_block_starts_are_zero_one_warning():
    data = _subjects()
    out, rec = _count_user_warnings(
        lambda: tc.timecorr(
            data,
            weights_params=PARAMS,
            include_timepoints="backward",
            include_zero=False,
            cfun=tc.autofc,
            events=B,
        )
    )
    assert len(rec) == 1
    for s in range(S):
        for start in EDGES[:-1]:
            assert np.all(out[s][start] == 0)


@pytest.mark.parametrize("exclude", [3, -2])
def test_events_with_exclude_timepoints(exclude):
    data = _subjects()
    out, rec = _count_user_warnings(
        lambda: tc.timecorr(
            data,
            weights_params=PARAMS,
            include_timepoints="backward",
            exclude_timepoints=exclude,
            cfun=tc.isfc,
            events=B,
        )
    )
    assert len(rec) == 1
    for s in range(S):
        assert out[s].shape == (NT, K * (K + 1) // 2)
        for start in EDGES[:-1]:
            assert np.all(out[s][start] == 0)


@pytest.mark.parametrize("name", ["autofc", "isfc", "wisfc"])
def test_short_blocks_one_warning_shared(name):
    _, rec = _count_user_warnings(
        lambda: tc.timecorr(_subjects(), weights_params=PARAMS, cfun=CFUNS[name], events=[2, 4, 30])
    )
    assert len(rec) == 1
    assert "fewer than 3" in str(rec[0].message)


@pytest.mark.parametrize("name", ["autofc", "isfc"])
def test_short_blocks_one_warning_per_subject(name):
    _, rec = _count_user_warnings(
        lambda: tc.timecorr(
            _subjects(),
            weights_params=PARAMS,
            cfun=CFUNS[name],
            events=[[2, 4, 30], [10], [50, 52]],
        )
    )
    assert len(rec) == 1
    assert "fewer than 3" in str(rec[0].message)


def test_no_block_long_enough_raises():
    with pytest.raises(ValueError):
        tc.timecorr(_subjects()[0], weights_params=PARAMS, events=list(range(1, NT)))


def test_events_params_without_features_raises():
    with pytest.raises(ValueError):
        tc.timecorr(_subjects()[0], weights_params=PARAMS, events=B, events_params={"n": 3})


# ---------------------------------------------------------------- (g) one subject, list or bare
@pytest.mark.parametrize("name", list(CFUNS))
def test_timecorr_single_subject_list_and_bare_agree(name):
    x = _subjects()[0]
    bare = tc.timecorr(x, weights_params=PARAMS, cfun=CFUNS[name], events=B)
    for ev in ([B], [list(B)], list(B)):
        out = tc.timecorr(x, weights_params=PARAMS, cfun=CFUNS[name], events=ev)
        assert np.array_equal(out, bare)
    assert not np.array_equal(bare, tc.timecorr(x, weights_params=PARAMS, cfun=CFUNS[name]))


# ---------------------------------------------------------------- (h) rfun and combine
@pytest.mark.parametrize("events", [B, PER_SUBJECT])
def test_rfun_and_combine_after_masking(events):
    data = _subjects()
    reduced = tc.timecorr(data, weights_params=PARAMS, rfun="PCA", events=events)
    assert len(reduced) == S and all(r.shape == (NT, K) for r in reduced)
    combined = tc.timecorr(
        data, weights_params=PARAMS, combine=tc.corrmean_combine, events=events
    )
    assert combined.shape == (NT, K * (K + 1) // 2)
    both = tc.timecorr(
        data, weights_params=PARAMS, combine=tc.corrmean_combine, rfun="PCA", events=events
    )
    assert both.shape == (NT, K)

