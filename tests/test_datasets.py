"""Tests for ``timecorr.datasets``: the cached download (``fetch``) and the loaders of
the example data of the real-data tutorials.

Nothing is mocked. ``fetch`` is tested on ``file://`` URLs of real temporary files;
``industry_returns`` downloads the real archive of the Kenneth R. French Data Library
(4 MB) into a temporary cache; ``story_listening`` (slow) reads the archive cached under
``$TIMECORR_DATA`` (default ``~/timecorr_data``), downloading it (460 MB) when it is not
there. A download that fails makes its test fail: these tests do not skip.

The reference functions below are the loading code of the tutorial notebooks as it was
before the loaders existed (``htfa_network_animations.ipynb``, ``stocks.ipynb``,
``trajectories.ipynb`` at commit f9db4af); the loaders must return exactly the same
arrays.
"""

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io

import timecorr as tc
from timecorr import datasets

FRENCH_ZIP = "49_Industry_Portfolios_daily_CSV.zip"


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def source(tmp_path):
    """A real file to download: ``(file:// URL, its bytes)``."""
    path = tmp_path / "source" / "payload.bin"
    path.parent.mkdir()
    data = np.random.default_rng(0).bytes(3_000_000)
    path.write_bytes(data)
    return path.as_uri(), data


# ---------------------------------------------------------------- fetch
def test_datasets_is_a_submodule_only():
    assert tc.datasets is datasets
    for name in ("fetch", "cache_dir", "story_listening", "industry_returns"):
        assert callable(getattr(tc.datasets, name))
        assert not hasattr(tc, name)


def test_fetch_downloads_once_and_keeps_no_part_file(tmp_path, source):
    url, data = source
    target = tmp_path / "cache" / "sub" / "payload.bin"
    path = datasets.fetch(url, target)
    assert isinstance(path, Path) and path == target
    assert path.read_bytes() == data
    assert sorted(p.name for p in target.parent.iterdir()) == ["payload.bin"]
    # cached: the source is not read again
    Path(url[len("file://"):]).unlink()
    assert datasets.fetch(url, target).read_bytes() == data


def test_fetch_checks_sha256(tmp_path, source):
    url, data = source
    target = tmp_path / "cache" / "payload.bin"
    wrong = digest(b"something else")
    with pytest.raises(RuntimeError, match="checksum"):
        datasets.fetch(url, target, sha256=wrong)
    # a download that fails its check is not kept
    assert list(target.parent.iterdir()) == []
    assert datasets.fetch(url, target, sha256=digest(data)).read_bytes() == data
    # a cached file is checked too, and kept for the user to delete
    with pytest.raises(RuntimeError, match="delete"):
        datasets.fetch(url, target, sha256=wrong)
    assert target.read_bytes() == data


def test_fetch_extracts_one_member_and_removes_the_archive(tmp_path):
    members = {"data/a.txt": b"first member\n" * 1000, "data/b.bin": bytes(range(256)) * 500}
    archive = tmp_path / "source" / "archive.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in members.items():
            z.writestr(name, content)
    target = tmp_path / "cache" / "b.bin"
    path = datasets.fetch(archive.as_uri(), target, member="data/b.bin",
                          sha256=digest(members["data/b.bin"]))
    assert path.read_bytes() == members["data/b.bin"]
    assert sorted(p.name for p in target.parent.iterdir()) == ["b.bin"]
    with pytest.raises(RuntimeError, match="no_such_member"):
        datasets.fetch(archive.as_uri(), tmp_path / "cache" / "c.bin", member="no_such_member")
    assert sorted(p.name for p in target.parent.iterdir()) == ["b.bin"]


def test_fetch_failure_names_the_url_and_leaves_nothing(tmp_path):
    url = (tmp_path / "missing.bin").as_uri()
    target = tmp_path / "cache" / "missing.bin"
    with pytest.raises(RuntimeError) as err:
        datasets.fetch(url, target)
    assert url in str(err.value)
    assert list(target.parent.iterdir()) == []


def test_relative_names_are_under_the_cache_root(tmp_path, source, monkeypatch):
    url, data = source
    monkeypatch.setenv("TIMECORR_DATA", str(tmp_path / "root"))
    assert datasets.cache_dir() == tmp_path / "root"
    path = datasets.fetch(url, "folder/payload.bin")
    assert path == tmp_path / "root" / "folder" / "payload.bin"
    assert path.read_bytes() == data
    monkeypatch.delenv("TIMECORR_DATA")
    assert datasets.cache_dir() == Path.home() / "timecorr_data"


# ---------------------------------------------------------------- industry returns
def reference_stocks(zip_path):
    """stocks.ipynb, cells 5 and 7: ``(first line, missing count, returns)``."""
    with zipfile.ZipFile(zip_path) as z:
        returns_text = z.read(z.namelist()[0]).decode("latin-1")
    lines = returns_text.splitlines()
    first = next(i for i, line in enumerate(lines)
                 if "Average Value Weighted Returns -- Daily" in line)
    last = next(i for i in range(first + 1, len(lines)) if not lines[i].strip())
    table = pd.read_csv(io.StringIO("\n".join(lines[first + 1:last])), index_col=0)
    table.index = pd.to_datetime(table.index.astype(str).str.strip(), format="%Y%m%d")
    table.columns = table.columns.str.strip()
    table = table.loc["2005-01-01":"2024-12-31"]
    missing = (table <= -99.99).sum()
    table = table.loc[:, missing == 0]
    returns = np.log1p(table / 100)
    return returns_text.splitlines()[0], missing.sum(), returns


def reference_trajectories_stocks(zip_path):
    """trajectories.ipynb, cell 7: ``(stock_dates, stocks)``."""
    with zipfile.ZipFile(zip_path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    first = next(i for i, line in enumerate(lines)
                 if "Average Value Weighted Returns -- Daily" in line)
    last = next(i for i in range(first + 1, len(lines)) if not lines[i].strip())
    table = pd.read_csv(io.StringIO("\n".join(lines[first + 1:last])), index_col=0)
    table.index = pd.to_datetime(table.index.astype(str).str.strip(), format="%Y%m%d")
    table = table.loc["2005-01-01":"2024-12-31"]
    table = table.loc[:, (table > -99.99).all()]
    return table.index, np.log1p(table.values / 100)


def check_industry_returns(returns, zip_path):
    header, missing, expected = reference_stocks(zip_path)
    pd.testing.assert_frame_equal(returns, expected, check_exact=True, check_freq=True)
    assert np.array_equal(returns.values, expected.values)
    assert returns.values.dtype == expected.values.dtype == np.float64
    assert list(returns.columns) == list(expected.columns)
    assert returns.index.equals(expected.index) and returns.index.dtype == expected.index.dtype
    assert returns.attrs == {"header": header, "missing": int(missing)}
    assert type(returns.attrs["missing"]) is int
    dates, values = reference_trajectories_stocks(zip_path)
    assert returns.index.equals(dates)
    assert np.array_equal(returns.values, values) and returns.values.dtype == values.dtype


def test_industry_returns_equal_the_notebooks_loading_code(tmp_path, monkeypatch):
    # a real download into an empty cache
    monkeypatch.setenv("TIMECORR_DATA", str(tmp_path))
    returns = datasets.industry_returns()
    zip_path = tmp_path / "french" / FRENCH_ZIP
    assert zip_path.exists()
    check_industry_returns(returns, zip_path)
    assert returns.shape[1] == 49 and returns.shape[0] > 5000
    assert returns.index[0] >= pd.Timestamp("2005-01-01")
    assert returns.index[-1] <= pd.Timestamp("2024-12-31")
    assert np.isfinite(returns.values).all()
    # another range, read from the cache: within the range, without missing values
    early = datasets.industry_returns(start="1950-01-01", end="1950-12-31")
    assert early.index[0].year == 1950 and early.index[-1].year == 1950
    assert 0 < early.shape[1] < 49 and early.attrs["missing"] > 0
    assert np.isfinite(early.values).all()


# ---------------------------------------------------------------- story listening
def farthest_points(points, k):
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(k - 1):
        chosen.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return np.array(chosen)


def reference_htfa_story(fit_path, K=100):
    """htfa_network_animations.ipynb, cell 7."""
    post = scipy.io.loadmat(fit_path)["posterior"][0, 0]
    names = [str(n[0]) for n in post["subjects_names"][0]]
    centers = np.asarray(post["centers"][0, 0]["mean"], dtype=float)
    widths = np.asarray(post["widths"][0, 0]["mean"], dtype=float).ravel()
    intact = sorted((i for i, n in enumerate(names) if n.startswith("intact_")),
                    key=lambda i: int(names[i].split("_s")[1].split(".")[0]))
    weights = np.stack([post["subjects"][0, i]["image_weights"][0, 0]["mean"][0, 0]
                        for i in intact])
    del post
    n_nodes = weights.shape[2]
    varying = np.flatnonzero((weights.std(axis=1) > 1e-6).all(axis=0))
    nodes = varying[farthest_points(centers[varying], K)]
    data = [w[:, nodes] for w in weights]
    return data, centers, widths, nodes, varying, n_nodes


def reference_trajectories_story(fit_path):
    """trajectories.ipynb, cell 5."""
    post = scipy.io.loadmat(fit_path)["posterior"][0, 0]
    names = [str(n[0]) for n in post["subjects_names"][0]]
    centers = np.asarray(post["centers"][0, 0]["mean"], dtype=float)
    intact = sorted((i for i, n in enumerate(names) if n.startswith("intact_")),
                    key=lambda i: int(names[i].split("_s")[1].split(".")[0]))
    weights = np.stack([post["subjects"][0, i]["image_weights"][0, 0]["mean"][0, 0]
                        for i in intact])
    del post
    usable = np.flatnonzero(~(weights.std(axis=1) < 1e-6).any(axis=0))
    nodes = usable[farthest_points(centers[usable], 100)]
    return [w[:, nodes] for w in weights]


def same_arrays(a, b):
    return (len(a) == len(b) and all(
        x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)
        for x, y in zip(a, b)))


@pytest.mark.slow
def test_story_listening_equals_the_notebooks_loading_code():
    data, nodes = datasets.story_listening(return_nodes=True)
    fit_path = datasets.cache_dir() / "pieman" / "pieman_posterior_K700.mat"
    assert fit_path.exists()
    ref, centers, widths, index, varying, total = reference_htfa_story(fit_path)
    assert same_arrays(data, ref)
    assert same_arrays(data, reference_trajectories_story(fit_path))
    assert set(nodes) == {"centers", "widths", "index", "varying", "total"}
    for found, expected in ((nodes["centers"], centers[index]), (nodes["widths"], widths[index]),
                            (nodes["index"], index), (nodes["varying"], varying)):
        assert found.dtype == expected.dtype and np.array_equal(found, expected)
    assert nodes["total"] == total == 700
    # without the nodes: the same list; another number of nodes
    assert same_arrays(datasets.story_listening(), ref)
    few, few_nodes = datasets.story_listening(n_nodes=12, return_nodes=True)
    assert same_arrays(few, reference_htfa_story(fit_path, K=12)[0])
    assert np.array_equal(few_nodes["index"], index[:12])   # farthest points are nested
    assert len(data) == 36 and data[0].shape == (300, 100)


# ---------------------------------------------------------------- review fixes
def test_concurrent_fetches_of_one_target_all_succeed(tmp_path):
    # eight threads download the same 40 MB file to the same target at once
    from concurrent.futures import ThreadPoolExecutor
    import threading

    src = tmp_path / "source" / "big.bin"
    src.parent.mkdir()
    data = np.random.default_rng(1).bytes(40_000_000)
    src.write_bytes(data)
    target = tmp_path / "cache" / "big.bin"
    n = 8
    barrier = threading.Barrier(n)

    def call(_):
        barrier.wait()
        return datasets.fetch(src.as_uri(), target, sha256=digest(data))

    with ThreadPoolExecutor(n) as pool:
        paths = list(pool.map(call, range(n)))      # raises if any call raised
    assert paths == [target] * n
    assert target.read_bytes() == data
    assert sorted(p.name for p in target.parent.iterdir()) == ["big.bin"]


def test_concurrent_fetches_of_one_archive_member(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    content = np.random.default_rng(2).bytes(20_000_000)
    archive = tmp_path / "source" / "archive.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        z.writestr("data/member.bin", content)
    target = tmp_path / "cache" / "member.bin"
    barrier = threading.Barrier(6)

    def call(_):
        barrier.wait()
        return datasets.fetch(archive.as_uri(), target, member="data/member.bin")

    with ThreadPoolExecutor(6) as pool:
        paths = list(pool.map(call, range(6)))
    assert paths == [target] * 6 and target.read_bytes() == content
    assert sorted(p.name for p in target.parent.iterdir()) == ["member.bin"]


@pytest.mark.parametrize("k", [0, -1, 6, 2.0, 2.5, "2", None, True])
def test_farthest_points_rejects_counts_outside_the_points(k):
    points = np.random.default_rng(0).standard_normal((5, 3))
    with pytest.raises(ValueError, match=r"between 1 and 5\b"):
        datasets._farthest_points(points, k)


def test_farthest_points_returns_distinct_points():
    points = np.random.default_rng(0).standard_normal((5, 3))
    for k in (1, 3, 5, np.int64(4)):
        chosen = datasets._farthest_points(points, k)
        assert len(chosen) == k == len(set(chosen.tolist()))


@pytest.mark.slow
def test_story_listening_validates_n_nodes():
    _, nodes = datasets.story_listening(n_nodes=1, return_nodes=True)
    eligible = len(nodes["varying"])
    assert eligible == 679
    for bad in (0, -1, eligible + 1, 700, 2.5, "100", True):
        with pytest.raises(ValueError, match=rf"n_nodes.*between 1 and {eligible}\b"):
            datasets.story_listening(n_nodes=bad)
    every, all_nodes = datasets.story_listening(n_nodes=eligible, return_nodes=True)
    assert every[0].shape == (300, eligible)
    assert sorted(all_nodes["index"].tolist()) == nodes["varying"].tolist()
