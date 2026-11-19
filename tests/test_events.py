"""Tests for timecorr.events: boundaries, specification grammar and kernel mask."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from timecorr.events import (
    block_labels,
    check_boundaries,
    distinct_event_sets,
    mask_kernel,
    resolve_events,
    resolve_events_by_order,
)
from timecorr.helpers import eye_weights, gaussian_weights

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
    # Review focus 4; the NotImplementedError branch is replaced when segmentation exists.
    with pytest.raises(NotImplementedError, match="segmentation is added in a later task"):
        resolve_events(FEATURE_ENTRIES[name], T, 2)


@pytest.mark.parametrize("name", list(FEATURE_ENTRIES))
def test_features_entry_wrong_rows(name):
    entry = FEATURE_ENTRIES[name]
    with pytest.raises(ValueError, match=rf"{T}.*boundaries|boundaries.*{T}"):
        resolve_events(entry, T + 3, 2)


def test_features_in_per_subject_list_reach_segmentation():
    with pytest.raises(NotImplementedError):
        resolve_events([FEATURES, [3]], T, 2)


def test_features_wrong_rows_message_names_both_lengths():
    with pytest.raises(ValueError) as exc:
        resolve_events(FEATURES, 20, 2)
    assert "12" in str(exc.value) and "20" in str(exc.value)


def test_events_params_unknown_key():
    with pytest.raises(ValueError, match="bogus"):
        resolve_events(FEATURES, T, 2, events_params={"n": 3, "bogus": 1})


def test_events_params_validated_before_segmentation():
    # Unknown key is a ValueError, not the later-task NotImplementedError.
    with pytest.raises(ValueError):
        resolve_events(FEATURES, T, 2, events_params={"nn": 3})


def test_events_params_allowed_keys_reach_segmentation():
    params = {"n": 3, "kind": "ordered", "max_events": 5, "random_state": 0}
    with pytest.raises(NotImplementedError):
        resolve_events(FEATURES, T, 2, events_params=params)


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
    with pytest.raises(NotImplementedError):
        resolve_events_by_order({2: FEATURES}, T, 2, 2)
    with pytest.raises(NotImplementedError):
        resolve_events_by_order(FEATURES, T, 2, 2)


def test_by_order_flat_params_need_features_somewhere():
    with pytest.raises(ValueError, match="features"):
        resolve_events_by_order({1: [3]}, T, 1, 2, events_params={"n": 2})
    # features at one order, boundaries at another: flat params are valid
    with pytest.raises(NotImplementedError):
        resolve_events_by_order({1: [3], 2: FEATURES}, T, 1, 2, events_params={"n": 2})


def test_by_order_params_keyed_by_order():
    with pytest.raises(NotImplementedError):
        resolve_events_by_order({1: [3], 2: FEATURES}, T, 1, 2, events_params={2: {"n": 2}})
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
