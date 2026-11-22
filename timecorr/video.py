# coding: utf-8
"""
Videos from frames drawn in parallel: :func:`render_video` draws the frames in worker
processes and encodes them (H.264), :func:`show_video` shows a video in a notebook, and
:func:`media_dir` gives the directory the tutorials write their videos to.

joblib, imageio-ffmpeg (which brings its own ffmpeg) and IPython are imported on first use
(``timecorr._lazy``); they are listed in the ``tutorials`` extra.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ._lazy import lazy_import

__all__ = ["render_video", "show_video", "media_dir"]

_EXTRA = "tutorials"
_WORKERS = 3  # default number of worker processes (fewer on fewer cores)


def _usable_cores():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _in_docs():
    """True when the working directory is a folder of Sphinx sources (docs/tutorial)."""
    return (Path("..") / "conf.py").exists() and (Path("..") / "_static").is_dir()


def media_dir(name="tutorial_media"):
    """
    The directory for the videos a notebook writes; it is created if missing.

    Parameters
    ----------
    name : str
        The directory used outside a documentation tree, relative to the working
        directory. Default: ``"tutorial_media"``.

    Returns
    -------
    path : pathlib.Path
        ``../_static/tutorial_media`` when the working directory is a folder of Sphinx
        documentation sources (its parent holds ``conf.py`` and a ``_static`` directory,
        as ``docs/tutorial`` of timecorr does), where :func:`show_video` links the
        videos; otherwise ``name`` (for example on Google Colab), where
        :func:`show_video` embeds them. The path is relative to the working directory.

    Examples
    --------
    >>> import os, tempfile
    >>> import timecorr as tc
    >>> os.chdir(tempfile.mkdtemp())
    >>> tc.media_dir().as_posix(), tc.media_dir("videos").is_dir()
    ('tutorial_media', True)
    """
    path = Path("..") / "_static" / "tutorial_media" if _in_docs() else Path(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_video(draw, n_frames, path, fps=30, crf=23, hold=0, n_jobs=None,
                 chunk_size=25, retries=0):
    """
    Draw the frames of a video in parallel worker processes and encode them as H.264.

    The frames are divided into runs of consecutive frames, and ``draw`` is called once
    per run in a worker process, so it can build its figure or scene once and then loop
    over its frames. The frames are written as PNG files in a temporary folder, encoded
    with the ffmpeg of imageio-ffmpeg (``-c:v libx264 -preset slow -crf <crf> -pix_fmt
    yuv420p -movflags +faststart``) and removed.

    Parameters
    ----------
    draw : callable
        ``draw(frames, paths)`` draws each frame ``i`` of ``frames`` (a list of
        consecutive frame numbers, counted from 0) and saves it as the PNG file
        ``paths[j]`` for ``frames[j]``. It is sent to the worker processes with what it
        refers to, so a function defined in a notebook works. All frames need the same
        size, with an even width and height (required by ``yuv420p``).
    n_frames : int
        Number of frames drawn.
    path : str or pathlib.Path
        The video file written (``.mp4``); an existing file is replaced.
    fps : float
        Frames per second. Default: 30.
    crf : int
        Constant rate factor of the encoder: 0 is lossless, higher values give smaller
        files of lower quality. Default: 23.
    hold : float
        Seconds for which the last frame stays at the end of the video
        (``round(hold * fps)`` copies of it are appended). Default: 0.
    n_jobs : int, optional
        Number of worker processes. Default: None, 3 or the number of usable CPU cores,
        whichever is smaller. With 1, ``draw`` runs in the calling process.
    chunk_size : int
        Largest number of frames per call of ``draw`` (there are at least ``n_jobs``
        calls when there are that many frames). Default: 25.
    retries : int
        How many times drawing is started again after a call of ``draw`` raised or a
        worker process died: the workers are stopped, and new ones draw the frames
        whose files do not exist yet. A frame file therefore has to be complete as soon
        as it exists: with ``retries``, save each frame under another name and rename
        it. Default: 0 (the first error is raised).

    Returns
    -------
    path : pathlib.Path
        The video file.

    Raises
    ------
    RuntimeError
        If ``draw`` returned without writing one of its frames.
    subprocess.CalledProcessError
        If ffmpeg fails.
    ImportError
        If joblib or imageio-ffmpeg is missing and cannot be installed (see
        ``pip install "timecorr[tutorials]"``).

    Examples
    --------
    Ten frames of a moving dot, drawn with one figure:

    >>> import pathlib, tempfile
    >>> import timecorr as tc
    >>> def draw(frames, paths):
    ...     import matplotlib
    ...     matplotlib.use("Agg")
    ...     import matplotlib.pyplot as plt
    ...     fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)
    ...     dot, = ax.plot([], [], "o")
    ...     ax.set(xlim=(0, 10), ylim=(-1, 1))
    ...     for i, path in zip(frames, paths):
    ...         dot.set_data([i], [0])
    ...         fig.savefig(path)
    ...     plt.close(fig)
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> video = tc.render_video(draw, 10, folder / "dot.mp4", fps=10, n_jobs=1)
    >>> video.name, video.stat().st_size > 0
    ('dot.mp4', True)
    """
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1 (got {n_frames}).")
    joblib = lazy_import("joblib", "joblib", _EXTRA)
    ffmpeg = lazy_import("imageio_ffmpeg", "imageio-ffmpeg", _EXTRA).get_ffmpeg_exe()
    path = Path(path)
    if n_jobs is None:
        n_jobs = min(_WORKERS, _usable_cores())
    with tempfile.TemporaryDirectory(prefix="timecorr_frames_") as folder:
        frame = [Path(folder) / f"{i:05d}.png" for i in range(n_frames + round(hold * fps))]
        failures = 0
        while True:
            todo = [i for i in range(n_frames) if not frame[i].exists()]
            n_chunks = max(n_jobs, -(-len(todo) // chunk_size))
            chunks = [c.tolist() for c in np.array_split(todo, n_chunks) if len(c)]
            try:
                joblib.Parallel(n_jobs=n_jobs)(
                    joblib.delayed(draw)(chunk, [frame[i] for i in chunk]) for chunk in chunks
                )
                break
            except Exception:
                failures += 1
                if failures > retries:
                    raise
                # stop the workers; new ones draw the frames not yet written
                from joblib.externals.loky import get_reusable_executor
                get_reusable_executor().shutdown(wait=True, kill_workers=True)
        missing = [i for i in range(n_frames) if not frame[i].exists()]
        if missing:
            raise RuntimeError(
                f"draw did not write {len(missing)} of {n_frames} frames (the first: "
                f"frame {missing[0]}, {frame[missing[0]].name})."
            )
        for held in frame[n_frames:]:
            shutil.copyfile(frame[n_frames - 1], held)
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
             "-i", str(Path(folder) / "%05d.png"), "-c:v", "libx264", "-preset", "slow",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            check=True,
        )
    return path


def show_video(path, width=960):
    """
    Show a video in a notebook.

    Parameters
    ----------
    path : str or pathlib.Path
        A video file, or the URL (``http://`` or ``https://``) of one.
    width : int
        Largest width of the player, in pixels. Default: 960.

    Returns
    -------
    None
        The video is displayed with IPython: when the working directory is a folder of
        Sphinx documentation sources (see :func:`media_dir`), and for a URL, as an HTML
        ``<video>`` tag that links the file (by its path relative to the working
        directory), so the notebook stays small; otherwise (for example on Google
        Colab) embedded in the notebook's output. The player has controls, loops and is
        muted.

    Raises
    ------
    ImportError
        If IPython is missing and cannot be installed (see
        ``pip install "timecorr[tutorials]"``).

    Examples
    --------
    Outside a notebook the displayed object is printed:

    >>> import timecorr as tc
    >>> tc.show_video("https://timecorr.readthedocs.io/en/latest/_static/"
    ...               "tutorial_media/stocks_network.mp4")
    <IPython.core.display.HTML object>
    """
    display = lazy_import("IPython.display", "ipython", _EXTRA)
    is_url = str(path).startswith(("http://", "https://"))
    if is_url or _in_docs():
        src = str(path) if is_url else Path(os.path.relpath(path)).as_posix()
        shown = display.HTML(
            f'<video src="{src}" controls loop muted playsinline preload="metadata" '
            f'style="width: 100%; max-width: {width}px;"></video>'
        )
    else:
        shown = display.Video(str(path), embed=True, width=width,
                              html_attributes="controls loop muted playsinline")
    display.display(shown)
