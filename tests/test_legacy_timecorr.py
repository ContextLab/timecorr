"""Frozen outputs of ``tc.timecorr``.

``tests/fixtures/legacy_timecorr.npz`` holds the outputs of ``tc.timecorr``
computed at commit d4bb5ba for the calls in ``CASES`` (written by
``tests/fixtures/make_legacy_timecorr.py``).  The test recomputes each call
and compares every output array at 1e-12.
"""

import os
import warnings

import numpy as np
import pytest

import timecorr as tc

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "legacy_timecorr.npz")

T, K = 60, 4


def _array(seed, k=K):
    return np.random.default_rng(seed).standard_normal((T, k))


def _list(seed, n, k=K):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((T, k)) for _ in range(n)]


def _nan_list(seed, n=3):
    rng = np.random.default_rng(seed)
    data = [rng.standard_normal((T, K)) for _ in range(n)]
    for d in data:
        d[rng.integers(0, T, 8), rng.integers(0, K, 8)] = np.nan
    return data


def own_autofc(data, weights):
    """A correlation function that is not one of the built-in ones (it is
    called once per distinct event set with per-subject events)."""
    return tc.autofc(data, weights)


WIDE = {"var": 10}
PER_SUBJECT = [[20], [15, 40], None]

# (name, make_data, kwargs)
CASES = [
    ("array", lambda: _array(201), {"weights_params": WIDE}),
    ("list_isfc", lambda: _list(202, 3), {"weights_params": WIDE, "cfun": tc.isfc}),
    ("list_wisfc", lambda: _list(203, 3), {"weights_params": WIDE, "cfun": tc.wisfc}),
    (
        "list_autofc_shared_events",
        lambda: _list(204, 3),
        {"weights_params": WIDE, "cfun": tc.autofc, "events": [20, 40]},
    ),
    (
        "list_isfc_per_subject_events",
        lambda: _list(205, 3),
        {"weights_params": WIDE, "cfun": tc.isfc, "events": PER_SUBJECT},
    ),
    (
        "list_autofc_per_subject_events",
        lambda: _list(206, 3),
        {"weights_params": WIDE, "cfun": tc.autofc, "events": PER_SUBJECT},
    ),
    (
        "list_custom_cfun_per_subject_events",
        lambda: _list(207, 3),
        {"weights_params": WIDE, "cfun": own_autofc, "events": PER_SUBJECT},
    ),
    ("nan_list_isfc", lambda: _nan_list(208), {"weights_params": WIDE, "cfun": tc.isfc}),
    (
        "nan_list_autofc_backward",
        lambda: _nan_list(209),
        {"weights_params": WIDE, "cfun": tc.autofc, "include_timepoints": "backward"},
    ),
    ("smoothing", lambda: _nan_list(210), {"weights_params": WIDE, "cfun": None}),
    (
        "smoothing_per_subject_events",
        lambda: _list(211, 3),
        {"weights_params": WIDE, "cfun": None, "events": PER_SUBJECT},
    ),
    (
        "corrmean_pca",
        lambda: _list(212, 3),
        {"weights_params": WIDE, "combine": tc.corrmean_combine, "rfun": "PCA"},
    ),
    (
        "laplace_exclude",
        lambda: _list(213, 2),
        {"weights_function": tc.laplace_weights, "weights_params": {"scale": 5},
         "exclude_timepoints": 3, "cfun": tc.autofc},
    ),
]


def run_case(make_data, kwargs):
    """Recompute one case (warnings are not recorded here)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tc.timecorr(make_data(), **kwargs)


def collect(index, out):
    """The output of case ``index`` as ``case<N>`` or ``case<N>_s<j>`` arrays."""
    if isinstance(out, np.ndarray):
        return {f"case{index}": out}
    return {f"case{index}_s{j}": x for j, x in enumerate(out)}


def compute_all():
    """Every case's outputs in one dict (used by the fixture script)."""
    store = {}
    for index, (_, make_data, kwargs) in enumerate(CASES, start=1):
        store.update(collect(index, run_case(make_data, kwargs)))
    return store


@pytest.mark.parametrize("index", range(1, len(CASES) + 1), ids=[c[0] for c in CASES])
def test_matches_frozen_outputs(index):
    _, make_data, kwargs = CASES[index - 1]
    got = collect(index, run_case(make_data, kwargs))
    frozen = np.load(FIXTURE)
    expected_keys = sorted(
        k for k in frozen.files if k == f"case{index}" or k.startswith(f"case{index}_")
    )
    assert expected_keys, f"no frozen outputs for case {index}"
    assert sorted(got) == expected_keys
    for key in expected_keys:
        assert got[key].shape == frozen[key].shape, key
        np.testing.assert_allclose(
            got[key], frozen[key], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=key
        )
