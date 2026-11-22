# coding: utf-8
"""
Example datasets of the tutorials, downloaded once and cached.

Files are kept under the cache root: the directory named by the environment variable
``TIMECORR_DATA``, or ``~/timecorr_data`` (see :func:`cache_dir`). :func:`fetch` downloads
any file into it; :func:`story_listening` and :func:`industry_returns` load the data of the
real-data tutorials. The module is used as ``tc.datasets`` after ``import timecorr as tc``.
"""

import hashlib
import io
import operator
import os
import urllib.request
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

__all__ = ["cache_dir", "fetch", "story_listening", "industry_returns"]

# environment variable that names the cache root
_CACHE_VARIABLE = "TIMECORR_DATA"

_CHUNK = 1 << 20  # bytes read at a time

# HTFA fit (700 nodes) of the story-listening data of Simony et al. (2016), as analyzed
# by Owen, Chang & Manning (2021)
_STORY_URL = ("https://drive.usercontent.google.com/download"
              "?id=1CZYe8eyAkZFuLqfwwlKoeijgkjdW6vFs&export=download&confirm=t")
_STORY_MEMBER = "data/pieman_posterior_K700.mat"
_STORY_SHA256 = "870179316f127776ff78096c361df6b1c5c7957d6456deb26b5489bcee6c0793"

_FRENCH_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
               "49_Industry_Portfolios_daily_CSV.zip")
_FRENCH_SECTION = "Average Value Weighted Returns -- Daily"
_FRENCH_MISSING = -99.99  # the library writes missing returns as -99.99 or -999


def cache_dir():
    """
    The directory under which downloaded files are cached.

    Returns
    -------
    path : pathlib.Path
        The value of the environment variable ``TIMECORR_DATA`` if it is set and not
        empty, otherwise ``~/timecorr_data``. The directory is not created.

    Examples
    --------
    >>> import os, pathlib
    >>> import timecorr as tc
    >>> default = pathlib.Path.home() / "timecorr_data"
    >>> tc.datasets.cache_dir() == pathlib.Path(os.environ.get("TIMECORR_DATA") or default)
    True
    """
    return Path(os.environ.get(_CACHE_VARIABLE) or Path.home() / "timecorr_data")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _extract(archive, member, target):
    """Write one member of a zip archive to ``target``."""
    try:
        with zipfile.ZipFile(archive) as z, z.open(member) as src, open(target, "wb") as dst:
            while chunk := src.read(_CHUNK):
                dst.write(chunk)
    except (zipfile.BadZipFile, KeyError) as err:
        raise RuntimeError(f"could not read {member!r} from the downloaded archive ({err})") from err


def fetch(url, path, sha256=None, member=None):
    """
    Download a file unless it is already cached, and return its path.

    The download is written to temporary files next to the target (uniquely named,
    ending in ``.part``) and moved into place in one step only when it is complete and,
    with ``sha256``, has the expected checksum, so an interrupted or failed download
    leaves no file behind, and calls that run at the same time for one target (in
    threads or processes) do not disturb each other: each downloads its own copy, and
    one that finds the target complete when its own copy is ready uses the target.

    Parameters
    ----------
    url : str
        Where the file (or, with ``member``, the zip archive holding it) is downloaded
        from.
    path : str or pathlib.Path
        Where the file is kept. A relative path is relative to the cache root
        (:func:`cache_dir`), for example ``"french/Siccodes49.zip"``; missing
        directories are created.
    sha256 : str, optional
        The SHA-256 checksum of the file, as a hexadecimal string. It is checked after
        a download and every time the cached file is used. Default: None (no check).
    member : str, optional
        The name of one member of the zip archive at ``url``. The member is extracted
        to ``path`` and the archive is removed. Default: None (``url`` is the file).

    Returns
    -------
    path : pathlib.Path
        The path of the file.

    Raises
    ------
    RuntimeError
        If the download fails (the message names ``url``), the archive has no such
        member, or the file does not match ``sha256``. A cached file that does not
        match is kept, and the message asks to delete it.

    Examples
    --------
    A file is downloaded once; here from a ``file://`` URL:

    >>> import pathlib, tempfile
    >>> import timecorr as tc
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (folder / "source.txt").write_text("1,2,3")
    >>> path = tc.datasets.fetch((folder / "source.txt").as_uri(), folder / "cache" / "copy.txt")
    >>> path.name, path.read_text()
    ('copy.txt', '1,2,3')
    """
    path = Path(path)
    if not path.is_absolute():
        path = cache_dir() / path
    if path.exists():
        if sha256 is not None and _sha256(path) != sha256:
            raise RuntimeError(
                f"{path} does not match its checksum; delete it and run this again"
            )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    # temporary files of this call only
    unique = uuid.uuid4().hex
    downloaded = path.with_name(f"{path.name}.{unique}.download.part")
    complete = path.with_name(f"{path.name}.{unique}.part")
    try:
        try:
            urllib.request.urlretrieve(url, downloaded)
        except (OSError, ValueError) as err:
            raise RuntimeError(f"could not download {url} ({err})") from err
        if member is None:
            downloaded.replace(complete)
        else:
            _extract(downloaded, member, complete)
        if sha256 is not None and _sha256(complete) != sha256:
            raise RuntimeError(f"the file downloaded from {url} does not match its checksum")
        # another call may have finished first: its file is used if it is valid
        if not (path.exists() and (sha256 is None or _sha256(path) == sha256)):
            complete.replace(path)  # atomic
    finally:
        downloaded.unlink(missing_ok=True)
        complete.unlink(missing_ok=True)
    return path


def _farthest_points(points, k):
    """
    Indices of ``k`` rows of ``points``, spread out: the row nearest to the mean, then,
    one at a time, the row farthest from those chosen so far.
    """
    k = _count(k, len(points), "k", "points")
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(k - 1):
        chosen.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return np.array(chosen)


def _count(value, largest, name, what):
    """``value`` as an int; ``ValueError`` unless it is an integer in 1..``largest``."""
    try:
        if isinstance(value, bool):
            raise TypeError
        value = operator.index(value)
    except TypeError:
        value = None
    if value is None or not 1 <= value <= largest:
        raise ValueError(
            f"{name} must be an integer between 1 and {largest} (the number of {what})."
        )
    return value


def story_listening(n_nodes=100, return_nodes=False):
    """
    Brain activity of people listening to a story: the node weights of the
    hierarchical topographic factor analysis (HTFA) of Owen, Chang & Manning (2021).

    The data are the functional MRI recordings of Simony et al. (2016) of participants
    who heard the intact story "Pie Man", described by 700 nodes (radial basis
    functions) shared by all participants; the weight of a node is its activity at one
    timepoint. The first call downloads a 460 MB archive and keeps the fit (233 MB) as
    ``pieman/pieman_posterior_K700.mat`` under the cache root (:func:`cache_dir`); its
    checksum is verified on every call.

    Parameters
    ----------
    n_nodes : int
        Number of nodes returned. They are chosen among the nodes whose weights vary
        over time in every participant (standard deviation above 1e-6), spread over
        the brain: the node nearest to the mean position, then, one at a time, the
        node farthest from those chosen so far. An integer between 1 and the number
        of those nodes (679). Default: 100.
    return_nodes : bool
        If True, also return ``nodes``. Default: False.

    Returns
    -------
    data : list of numpy.ndarray
        One (300 timepoints, ``n_nodes``) array of node weights per participant (36
        participants, in the order of their numbers).
    nodes : dict
        Only with ``return_nodes=True``: ``"centers"`` ((``n_nodes``, 3) positions of
        the returned nodes in MNI space, in mm), ``"widths"`` ((``n_nodes``,) widths of
        their radial basis functions), ``"index"`` ((``n_nodes``,) the returned nodes
        as indices into the 700 nodes of the fit, in the order of the columns of
        ``data``), ``"varying"`` (indices of the nodes that vary in every
        participant, increasing) and ``"total"`` (700, the number of nodes of the
        fit).

    Raises
    ------
    RuntimeError
        If the download fails or the file does not match its checksum (see
        :func:`fetch`).
    ValueError
        If ``n_nodes`` is not an integer between 1 and the number of nodes that vary
        in every participant.

    References
    ----------
    Owen LLW, Chang TH, Manning JR (2021) High-level cognition during story listening
    is reflected in high-order dynamic correlations in neural activity patterns.
    Nature Communications 12: 5728.

    Simony E, Honey CJ, Chen J, Lositsky O, Yeshurun Y, Wiesel A, Hasson U (2016)
    Dynamic reconfiguration of the default mode network during narrative
    comprehension. Nature Communications 7: 12141.

    Examples
    --------
    >>> import timecorr as tc
    >>> data, nodes = tc.datasets.story_listening(return_nodes=True)   # doctest: +SKIP
    >>> len(data), data[0].shape, nodes["centers"].shape              # doctest: +SKIP
    (36, (300, 100), (100, 3))
    """
    fit_path = fetch(_STORY_URL, "pieman/pieman_posterior_K700.mat",
                     sha256=_STORY_SHA256, member=_STORY_MEMBER)
    post = scipy.io.loadmat(fit_path)["posterior"][0, 0]
    names = [str(n[0]) for n in post["subjects_names"][0]]
    centers = np.asarray(post["centers"][0, 0]["mean"], dtype=float)
    widths = np.asarray(post["widths"][0, 0]["mean"], dtype=float).ravel()
    intact = sorted((i for i, n in enumerate(names) if n.startswith("intact_")),
                    key=lambda i: int(names[i].split("_s")[1].split(".")[0]))
    weights = np.stack([post["subjects"][0, i]["image_weights"][0, 0]["mean"][0, 0]
                        for i in intact])
    del post
    varying = np.flatnonzero((weights.std(axis=1) > 1e-6).all(axis=0))
    n_nodes = _count(n_nodes, len(varying), "n_nodes", "nodes that vary in every participant")
    index = varying[_farthest_points(centers[varying], n_nodes)]
    data = [w[:, index] for w in weights]
    if not return_nodes:
        return data
    return data, {"centers": centers[index], "widths": widths[index], "index": index,
                  "varying": varying, "total": int(weights.shape[2])}


def industry_returns(start="2005-01-01", end="2024-12-31"):
    """
    Daily returns of the 49 Industry Portfolios of the Kenneth R. French Data Library.

    Each portfolio holds the US stocks of one industry, weighted by market value. The
    first call downloads the library's archive (4 MB) and keeps it as
    ``french/49_Industry_Portfolios_daily_CSV.zip`` under the cache root
    (:func:`cache_dir`); the library updates the archive every month, and a cached copy
    is used as it is.

    Parameters
    ----------
    start, end : str or pandas.Timestamp
        The first and last day of the period returned (both included). Default:
        ``"2005-01-01"`` and ``"2024-12-31"``.

    Returns
    -------
    returns : pandas.DataFrame
        Log returns, ``log(1 + r)`` for a daily return ``r`` (the file gives returns in
        percent): one row per trading day (a ``DatetimeIndex``) and one column per
        industry (the library's abbreviations, for example ``"Banks"``), in the order
        of the file. Industries with a missing return within the period are left out.
        ``returns.attrs`` holds ``"header"`` (the first line of the file, which says
        which release of the stock database it was built from) and ``"missing"`` (the
        number of missing returns within the period, before those industries were
        left out).

    Raises
    ------
    RuntimeError
        If the download fails (see :func:`fetch`).

    References
    ----------
    Kenneth R. French Data Library,
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

    Examples
    --------
    >>> import timecorr as tc
    >>> returns = tc.datasets.industry_returns()        # doctest: +SKIP
    >>> returns.shape                                   # doctest: +SKIP
    (5033, 49)
    """
    zip_path = fetch(_FRENCH_URL, "french/49_Industry_Portfolios_daily_CSV.zip")
    with zipfile.ZipFile(zip_path) as z:
        lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    # the file holds several tables; this one ends at the next empty line
    first = next(i for i, line in enumerate(lines) if _FRENCH_SECTION in line)
    last = next(i for i in range(first + 1, len(lines)) if not lines[i].strip())
    table = pd.read_csv(io.StringIO("\n".join(lines[first + 1:last])), index_col=0)
    table.index = pd.to_datetime(table.index.astype(str).str.strip(), format="%Y%m%d")
    table.columns = table.columns.str.strip()
    table = table.loc[start:end]
    missing = (table <= _FRENCH_MISSING).sum()
    returns = np.log1p(table.loc[:, missing == 0] / 100)
    returns.attrs = {"header": lines[0], "missing": int(missing.sum())}
    return returns
