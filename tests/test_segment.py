"""Tests for ``timecorr.segment`` (HMM segmentation into events), for features entries
in ``events``, and for the on-demand installation of hmmlearn (``timecorr._lazy``).

Every test runs real fits on synthetic data; the installation tests build real virtual
environments. Nothing is mocked.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import timecorr as tc
from timecorr.events import (
    _Features,
    _finish,
    block_labels,
    resolve_events,
    resolve_events_by_order,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- synthetic data
def _event_lengths(rng, n_events=6, T=300, low=30, high=70):
    """Event lengths drawn from low..high (inclusive) that sum to T."""
    while True:
        head = rng.randint(low, high + 1, n_events - 1)
        last = T - head.sum()
        if low <= last <= high:
            return np.concatenate([head, [last]])


def ordered_data(seed=0, T=300, K=20, n_events=6, noise=0.5):
    """Events with random mean patterns (sd 1) plus noise; returns data and true starts."""
    rng = np.random.RandomState(seed)
    lengths = _event_lengths(rng, n_events, T)
    labels = np.repeat(np.arange(n_events), lengths)
    means = rng.randn(n_events, K)
    data = means[labels] + noise * rng.randn(T, K)
    return data, np.concatenate([[0], np.cumsum(lengths)[:-1]])


def recurring_data(seed=0, K=20, noise=0.5):
    """Three states visited in the order 0, 1, 2, 0, 1 (five runs of 60 timepoints)."""
    rng = np.random.RandomState(seed)
    states = np.repeat([0, 1, 2, 0, 1], 60)
    means = rng.randn(3, K)
    return means[states] + noise * rng.randn(300, K), np.arange(0, 300, 60)


# ---------------------------------------------------------------- ordered kind
def test_ordered_known_events_n_given():
    data, truth = ordered_data(seed=0)
    b = tc.segment(data, n=6)
    assert b.dtype == np.int64 and b.ndim == 1
    assert len(b) == 6
    assert b[0] == 0
    assert np.max(np.abs(b - truth)) <= 2, (b, truth)


def test_ordered_known_events_n_chosen():
    data, truth = ordered_data(seed=0)
    b = tc.segment(data)
    assert len(b) == 6, b
    assert np.max(np.abs(b - truth)) <= 2, (b, truth)


def test_ordered_return_states():
    data, truth = ordered_data(seed=0)
    b, states = tc.segment(data, n=6, return_states=True)
    assert states.shape == (300,) and np.issubdtype(states.dtype, np.integer)
    np.testing.assert_array_equal(states, block_labels(b, 300))


def test_ordered_pure_noise_exactly_n_blocks():
    noise = np.random.RandomState(3).randn(200, 10)
    b, states = tc.segment(noise, n=7, return_states=True)
    assert len(b) == 7
    assert b[0] == 0
    assert np.all(np.diff(np.concatenate([b, [200]])) >= 1)
    np.testing.assert_array_equal(np.unique(states), np.arange(7))
    assert np.all(np.diff(states) >= 0)  # visited once each, in order


@pytest.mark.parametrize("n", [1, 2, 25, 25 - 1])
def test_ordered_extreme_n(n):
    x = np.random.RandomState(1).randn(25, 4)
    b = tc.segment(x, n=n)
    assert len(b) == n and b[0] == 0


def test_ordered_n_equals_T():
    x = np.random.RandomState(1).randn(12, 3)
    np.testing.assert_array_equal(tc.segment(x, n=12), np.arange(12))


# ---------------------------------------------------------------- recurring kind
def test_recurring_known_states():
    data, truth = recurring_data(seed=0)
    b, labels = tc.segment(data, n=3, kind="recurring", random_state=0, return_states=True)
    assert len(b) == 5, b
    assert np.max(np.abs(b - truth)) <= 2, (b, truth)
    assert labels.shape == (300,)
    assert labels[0] == labels[200]  # the same state, revisited
    blocks = block_labels(b, 300)
    assert blocks[0] != blocks[200]  # but in different blocks


# ---------------------------------------------------------------- validation
def test_recurring_requires_n():
    with pytest.raises(ValueError, match="recurring"):
        tc.segment(np.random.randn(50, 3), kind="recurring")


def test_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        tc.segment(np.random.randn(50, 3), n=2, kind="bogus")


@pytest.mark.parametrize("n", [0, -1, 51, 2.0, True, "3"])
def test_n_out_of_range_or_not_integer(n):
    with pytest.raises(ValueError):
        tc.segment(np.random.randn(50, 3), n=n)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_features(bad):
    x = np.random.RandomState(0).randn(40, 3)
    x[5, 1] = bad
    with pytest.raises(ValueError, match="finite|NaN"):
        tc.segment(x, n=3)
    with pytest.raises(ValueError, match="finite|NaN"):
        tc.segment(pd.DataFrame(x), n=3)


def test_automatic_n_needs_four_timepoints_and_two_features():
    with pytest.raises(ValueError, match="4"):
        tc.segment(np.random.randn(3, 5))
    with pytest.raises(ValueError, match="2 features|two features"):
        tc.segment(np.random.randn(40, 1))
    # with n given, one feature is fine
    assert len(tc.segment(np.random.RandomState(0).randn(40, 1), n=2)) == 2


@pytest.mark.parametrize("max_events", [1, 0, 2.5, True])
def test_bad_max_events(max_events):
    with pytest.raises(ValueError, match="max_events"):
        tc.segment(np.random.randn(40, 3), max_events=max_events)


def test_max_events_limits_candidates():
    data, _ = ordered_data(seed=0)
    assert len(tc.segment(data, max_events=4)) <= 4


# ---------------------------------------------------------------- reproducibility, inputs
def test_reproducible_with_random_state():
    data, _ = recurring_data(seed=1)
    a = tc.segment(data, n=3, kind="recurring", random_state=0, return_states=True)
    b = tc.segment(data, n=3, kind="recurring", random_state=0, return_states=True)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    data, _ = ordered_data(seed=1)
    np.testing.assert_array_equal(
        tc.segment(data, random_state=0), tc.segment(data, random_state=0)
    )


def test_list_input_returns_list():
    d0, _ = ordered_data(seed=0)
    d1, _ = ordered_data(seed=1)
    out = tc.segment([d0, d1], n=6)
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_array_equal(out[0], tc.segment(d0, n=6))
    np.testing.assert_array_equal(out[1], tc.segment(d1, n=6))
    # subjects may differ in length; with return_states each item is a pair
    out = tc.segment([d0, d1[:200]], n=4, return_states=True)
    assert len(out[1][1]) == 200 and len(out[1][0]) == 4


def test_dataframe_input():
    data, _ = ordered_data(seed=0)
    np.testing.assert_array_equal(tc.segment(pd.DataFrame(data), n=6), tc.segment(data, n=6))


def test_input_not_modified():
    data, _ = ordered_data(seed=0)
    copy = data.copy()
    tc.segment(data, n=6)
    tc.segment(data, n=3, kind="recurring", random_state=0)
    np.testing.assert_array_equal(data, copy)


# ---------------------------------------------------------------- features entries in events
def test_timecorr_features_entry_equals_explicit_segment():
    features_array, _ = ordered_data(seed=0)
    rng = np.random.RandomState(5)
    brain = [rng.randn(300, 4), rng.randn(300, 4)]
    params = {"n": 6, "random_state": 0}
    a = tc.timecorr(brain, events=features_array, events_params=params)
    b = tc.timecorr(brain, events=tc.segment(features_array, **params))
    np.testing.assert_array_equal(a, b)


def test_resolve_events_features_entry_gives_boundaries():
    data, truth = ordered_data(seed=0)
    out = resolve_events(data, 300, 2, events_params={"n": 6})
    np.testing.assert_array_equal(out, tc.segment(data, n=6))


def test_shared_features_object_segmented_once_across_subjects_and_orders():
    # Single segmentation is observed through object identity: the resolver returns the
    # cached result itself, so every place the same features object appears gets the SAME
    # array object; two segmentations would give two distinct (if equal) arrays.
    data, _ = ordered_data(seed=0)
    out = resolve_events([data, data], 300, 2, events_params={"n": 6})
    assert out[0] is out[1]
    out = resolve_events_by_order(data, 300, 2, 3, events_params={"n": 6})
    assert out[1] is out[2] is out[3]
    out = resolve_events_by_order(
        {1: [data, data], 2: data}, 300, 2, 2, events_params={"n": 6}
    )
    assert out[1][0] is out[1][1] is out[2]
    # an equal copy is a different object and is segmented separately (same result)
    other = data.copy()
    out = resolve_events([data, other], 300, 2, events_params={"n": 6})
    assert out[0] is not out[1]
    np.testing.assert_array_equal(out[0], out[1])


def test_finish_cache_holds_one_entry_per_object_and_params():
    data, _ = ordered_data(seed=0)
    feat = _Features(data)
    cache = {}
    res = _finish([feat, feat, _Features(data)], {"n": 6}, cache)
    assert len(cache) == 1  # two _Features wrappers of one object: one segmentation
    assert res[0] is res[1] is res[2]
    _finish(feat, {"n": 4}, cache)
    assert len(cache) == 2  # different parameters: segmented again


def test_same_features_different_params_per_order():
    data, _ = ordered_data(seed=0)
    out = resolve_events_by_order(
        {1: data, 2: data}, 300, 1, 2, events_params={1: {"n": 3}, 2: {"n": 6}}
    )
    assert len(out[1]) == 3 and len(out[2]) == 6


# ---------------------------------------------------------------- lazy import
def _run(code, env_extra=None, python=sys.executable, timeout=900):
    env = dict(os.environ)
    env.pop("TIMECORR_AUTO_INSTALL", None)
    env.update(env_extra or {})
    return subprocess.run(
        [python, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        cwd=os.path.dirname(python) if python != sys.executable else None,
    )


def test_import_timecorr_does_not_import_hmmlearn():
    r = _run("import sys, timecorr; print('hmmlearn' in sys.modules)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_lazy_import_returns_installed_module():
    from timecorr._lazy import lazy_import

    import json

    assert lazy_import("json", "json", "segment") is json


@pytest.mark.parametrize("value", ["0", "false", "No", "FALSE"])
def test_lazy_import_auto_install_off(value):
    r = _run(
        """
        from timecorr._lazy import lazy_import
        try:
            lazy_import("timecorr_no_such_module_xyz", "timecorr-no-such-pkg", "segment")
        except ImportError as e:
            print("IMPORTERROR", e)
        """,
        {"TIMECORR_AUTO_INSTALL": value},
    )
    assert r.returncode == 0, r.stderr
    assert "IMPORTERROR" in r.stdout
    assert 'pip install "timecorr[segment]"' in r.stdout
    assert "installing" not in r.stderr


def test_lazy_import_reraises_missing_inner_dependency(tmp_path):
    # A module that exists but fails to import one of ITS dependencies is not installed
    # by name: the original ModuleNotFoundError propagates.
    (tmp_path / "tc_lazy_probe_pkg.py").write_text("import tc_lazy_probe_missing_dep\n")
    r = _run(
        """
        from timecorr._lazy import lazy_import
        try:
            lazy_import("tc_lazy_probe_pkg", "tc-lazy-probe-pkg", "segment")
        except ModuleNotFoundError as e:
            print("MNF", e.name)
        """,
        {"PYTHONPATH": str(tmp_path), "TIMECORR_AUTO_INSTALL": "1"},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "MNF tc_lazy_probe_missing_dep"
    assert "installing" not in r.stderr


def _source_copy(dest):
    """Copy of the working tree's package sources (building in it leaves the repository clean)."""
    if not dest.exists():
        shutil.copytree(
            os.path.join(REPO_ROOT, "timecorr"),
            dest / "timecorr",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        for name in ("setup.py", "README.md"):
            shutil.copy(os.path.join(REPO_ROOT, name), dest / name)
    return str(dest)


def _fresh_venv(path, seed_pip):
    """A new virtual environment with timecorr installed from (a copy of) the working tree."""
    source = _source_copy(path.parent / "timecorr_source")
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "venv", "--python", sys.executable, str(path)]
        if seed_pip:
            cmd.insert(2, "--seed")
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    else:
        cmd = [sys.executable, "-m", "venv", str(path)]
        if not seed_pip:
            cmd.append("--without-pip")
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    python = str(path / ("Scripts" if os.name == "nt" else "bin") / "python")
    if uv:
        install = [uv, "pip", "install", "--python", python, source]
    elif seed_pip:
        install = [python, "-m", "pip", "install", source]
    else:  # no installer inside the environment: use this interpreter's pip
        install = [sys.executable, "-m", "pip", "--python", python, "install", source]
    r = subprocess.run(install, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
    has = _run("import importlib.util as u; print(u.find_spec('hmmlearn') is None)", python=python)
    assert has.stdout.strip() == "True", "the fresh environment must not have hmmlearn"
    return python


SEGMENT_CALL = """
import numpy as np, timecorr as tc
x = np.random.RandomState(0).randn(60, 4)
try:
    print("RESULT", tc.segment(x, n=3).tolist())
except ImportError as e:
    print("IMPORTERROR", e)
"""


@pytest.mark.slow
def test_lazy_install_in_fresh_environments(tmp_path):
    with_pip = _fresh_venv(tmp_path / "with_pip", seed_pip=True)
    without_pip = _fresh_venv(tmp_path / "without_pip", seed_pip=False)

    # (1) installation turned off: ImportError with the manual command
    for python in (with_pip, without_pip):
        r = _run(SEGMENT_CALL, {"TIMECORR_AUTO_INSTALL": "0"}, python=python)
        assert r.returncode == 0, r.stderr
        assert "IMPORTERROR" in r.stdout and 'pip install "timecorr[segment]"' in r.stdout

    # (2a) no pip in the environment: ImportError with the manual command
    r = _run(SEGMENT_CALL, python=without_pip)
    assert r.returncode == 0, r.stderr
    assert "IMPORTERROR" in r.stdout and 'pip install "timecorr[segment]"' in r.stdout

    # pip present but the requirement cannot be installed: ImportError with pip's output
    r = _run(
        """
        from timecorr._lazy import lazy_import
        try:
            lazy_import("tc_no_such_module", "/nonexistent/tc-no-such-dir", "segment")
        except ImportError as e:
            print("IMPORTERROR", e)
        """,
        python=with_pip,
    )
    assert r.returncode == 0, r.stderr
    assert "IMPORTERROR" in r.stdout and 'pip install "timecorr[segment]"' in r.stdout
    assert "nonexistent" in r.stdout  # pip's own error text is included

    # (2b) pip present: hmmlearn is installed on demand and the call succeeds
    r = _run(SEGMENT_CALL, python=with_pip)
    assert r.returncode == 0, r.stderr
    assert "RESULT [0," in r.stdout, r.stdout + r.stderr
    assert (
        "timecorr: installing hmmlearn (needed for segment; "
        "set TIMECORR_AUTO_INSTALL=0 to disable)" in r.stderr
    )
    r = _run("import hmmlearn; print('OK')", python=with_pip)
    assert r.stdout.strip() == "OK", r.stderr
