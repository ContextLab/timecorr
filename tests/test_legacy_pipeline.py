"""Frozen outputs of the two-path pipeline.

``tests/fixtures/legacy_features.npz`` holds the outputs of ``tc.features`` and
``tc.two_path_step`` computed at commit d4bb5ba for the calls in ``CASES``.  The
test recomputes each call and compares every output array at 1e-12.
``WARNINGS`` holds the warnings (category and start of the message) that some
of these calls, and three more calls with default settings, issued at commit
d4bb5ba.
"""

import os
import warnings

import numpy as np
import pytest

import timecorr as tc
from timecorr.higher_order import two_path_step

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "legacy_features.npz")

# Arguments merged into every call: the settings of the original pipeline,
# the package defaults at commit d4bb5ba.
LEGACY = dict(rfun="PCA", delta_rfun=None, delta_weights_function=tc.eye_weights)

T, K = 60, 5


def _array(seed, k=K):
    return np.random.default_rng(seed).standard_normal((T, k))


def _list(seed, n, k=K):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((T, k)) for _ in range(n)]


def _nan_list(seed):
    rng = np.random.default_rng(seed)
    data = [rng.standard_normal((T, 4)) for _ in range(2)]
    for d in data:
        d[rng.integers(0, T, 6), rng.integers(0, 4, 6)] = np.nan
    return data


def _call_features(data, **kwargs):
    return tc.features(data, **{**LEGACY, **kwargs})


def _call_step(data, **kwargs):
    return two_path_step(data, **{**LEGACY, **kwargs})


# (name, make_data, kwargs); kwargs hold no legacy arguments.  ``call`` is
# looked up by name in _CALLERS.
CASES = [
    ("array_order3", lambda: _array(101), {"order": 3, "return_delta": True}),
    (
        "list_isfc_order2",
        lambda: _list(102, 3),
        {"order": 2, "cfun": tc.isfc, "return_delta": True},
    ),
    (
        "list_autofc_ndims2",
        lambda: _list(103, 3),
        {"order": 2, "cfun": tc.autofc, "ndims": 2, "return_delta": True},
    ),
    (
        "nan_list_no_reduction",
        lambda: _nan_list(104),
        {"order": 2, "rfun": None, "delta_rfun": None, "return_delta": True},
    ),
    (
        "array_graph_measure",
        lambda: _array(105),
        {"order": 2, "rfun": "eigenvector_centrality", "delta_rfun": "PCA"},
    ),
    ("list_events", lambda: _list(106, 3), {"order": 2, "events": [20, 40]}),
    ("two_path_step_list", lambda: _list(107, 3), {"_step": True}),
]


def run_case(make_data, kwargs):
    """Recompute one case."""
    kwargs = dict(kwargs)
    if kwargs.pop("_step", False):
        return _call_step(make_data(), **kwargs)
    return _call_features(make_data(), **kwargs)


def collect(index, kwargs, out):
    """Flatten the output of case ``index`` into a dict of arrays.

    Keys are ``case<N>_X<k>[_s<j>]`` / ``case<N>_D<k>[_s<j>]``.  ``features``
    returns the list ``X_0..X_order`` (or ``(X, D)`` with ``return_delta``);
    ``two_path_step`` returns ``(X_next, D_next)``.
    """
    store = {}

    def add(label, item):
        if isinstance(item, np.ndarray):
            store[f"case{index}_{label}"] = item
        else:
            for j, sub in enumerate(item):
                store[f"case{index}_{label}_s{j}"] = sub

    if kwargs.get("_step", False):
        add("X1", out[0])
        add("D1", out[1])
    elif kwargs.get("return_delta", False):
        xs, ds = out
        for k, item in enumerate(xs):
            add(f"X{k}", item)
        for k, item in enumerate(ds):
            add(f"D{k}", item)
    else:
        for k, item in enumerate(out):
            add(f"X{k}", item)
    return store


def compute_all():
    """Every case's outputs in one dict (used by the fixture script)."""
    store = {}
    for index, (_, make_data, kwargs) in enumerate(CASES, start=1):
        store.update(collect(index, kwargs, run_case(make_data, kwargs)))
    return store


@pytest.mark.parametrize(
    "index", range(1, len(CASES) + 1), ids=[c[0] for c in CASES]
)
def test_matches_frozen_outputs(index):
    _, make_data, kwargs = CASES[index - 1]
    got = collect(index, kwargs, run_case(make_data, kwargs))
    frozen = np.load(FIXTURE)
    prefix = f"case{index}_"
    expected_keys = sorted(k for k in frozen.files if k.startswith(prefix))
    assert expected_keys, f"no frozen outputs for case {index}"
    assert sorted(got) == expected_keys
    for key in expected_keys:
        assert got[key].shape == frozen[key].shape, key
        np.testing.assert_allclose(
            got[key], frozen[key], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=key
        )


# ------------------------------------------------------------------ warnings

_NAN_FILL = "NaN samples: their co-fluctuations are set to 0 before the delta path's"
_SPARSE = "2 of 60 timepoints have fewer than 3 non-zero kernel weights"

# Warnings recorded at commit d4bb5ba (warnings.simplefilter("always")):
# name -> (call, [(category, start of the message), ...])
WARNINGS = {
    "array_order3": (lambda: run_case(*CASES[0][1:]), []),
    "nan_list_no_reduction": (lambda: run_case(*CASES[3][1:]), []),
    "list_events": (lambda: run_case(*CASES[5][1:]), []),
    "nan_list_defaults": (
        lambda: _call_features(_nan_list(104), order=2, return_delta=True),
        [(UserWarning, _NAN_FILL)],
    ),
    "backward_list": (
        lambda: _call_features(
            _list(108, 3), order=2, include_timepoints="backward",
            weights_params={"var": 10}, return_delta=True,
        ),
        [(UserWarning, _SPARSE), (UserWarning, _SPARSE)],
    ),
    "nan_step_defaults": (
        lambda: _call_step(_nan_list(109)), [(UserWarning, _NAN_FILL)]
    ),
}


@pytest.mark.parametrize("name", list(WARNINGS))
def test_warnings_match_the_legacy_pipeline(name):
    call, expected = WARNINGS[name]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    got = [(r.category, str(r.message)) for r in record]
    assert len(got) == len(expected), got
    for (category, message), (want_category, start) in zip(got, expected):
        assert category is want_category, got
        assert message.startswith(start), got
