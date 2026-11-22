"""Tests for ``timecorr.video``: ``render_video``, ``show_video`` and ``media_dir``.

Nothing is mocked: frames are drawn with matplotlib in real worker processes, encoded
with the ffmpeg of imageio-ffmpeg and decoded again with the same ffmpeg; the display is
captured from a real IPython shell.
"""

import base64
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

import timecorr as tc

SIZE = (160, 120)   # width, height of the test frames


def make_draw(log, n_frames, noise=False, fail_once=False, skip=()):
    """
    A ``draw(frames, paths)`` as the tutorials write them: one figure per call, then a
    loop over the frames. Frame ``i`` is a gray level rising with ``i`` (or noise).
    Every call appends ``frames`` to a file in ``log``; with ``fail_once``, the first call
    of each chunk that holds frame 0 raises after drawing its first frame.
    """
    log = Path(log)

    def draw(frames, paths):
        import os
        import uuid

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        (log / f"call_{uuid.uuid4().hex}_{os.getpid()}.txt").write_text(
            " ".join(map(str, frames)) + "\n" + str(paths[0].parent))
        fig = plt.figure(figsize=(1.6, 1.2), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        try:
            for i, path in zip(frames, paths):
                if i in skip:
                    continue
                ax.clear()
                ax.axis("off")
                if noise:
                    image = np.random.default_rng(i).random((120, 160))
                else:
                    image = np.full((120, 160), (i + 1) / (n_frames + 1))
                ax.imshow(image, cmap="gray", vmin=0, vmax=1, aspect="auto")
                tmp = path.with_suffix(".tmp.png")
                fig.savefig(tmp, dpi=100)
                tmp.replace(path)             # a frame file exists only once it is complete
                if fail_once and 0 in frames and not (log / "failed").exists():
                    (log / "failed").write_text("")
                    raise RuntimeError("a worker failed")
        finally:
            plt.close(fig)

    return draw


def calls(log):
    """``(frames of each call of draw, the folders the frames were written to)``."""
    found, folders = [], set()
    for path in Path(log).glob("call_*.txt"):
        frames, folder = path.read_text().split("\n")
        found.append([int(i) for i in frames.split()])
        folders.add(Path(folder))
    return found, folders


def decode(path):
    """``(meta, frames)`` of a video, decoded with ffmpeg: frames as (h, w, 3) arrays."""
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    w, h = meta["size"]
    frames = [np.frombuffer(f, dtype=np.uint8).reshape(h, w, 3) for f in reader]
    return meta, frames


def test_render_video_draws_in_chunks_and_encodes_h264(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    out = tmp_path / "videos" / "gray.mp4"
    out.parent.mkdir()
    path = tc.render_video(make_draw(log, 23), 23, out, n_jobs=2, chunk_size=5)
    assert isinstance(path, Path) and path == out and out.stat().st_size > 0
    meta, frames = decode(out)
    assert meta["codec"].startswith("h264") and meta["pix_fmt"].startswith("yuv420p")
    assert meta["size"] == SIZE and meta["fps"] == 30
    assert len(frames) == 23
    # the frames are in order: gray levels rise from frame to frame
    levels = [f.mean() for f in frames]
    assert all(b > a + 2 for a, b in zip(levels, levels[1:]))
    # +faststart: the index (moov) comes before the samples (mdat)
    data = out.read_bytes()
    assert 0 < data.index(b"moov") < data.index(b"mdat")
    # every frame was drawn once, in calls of at most chunk_size consecutive frames
    drawn, folders = calls(log)
    assert sorted(i for c in drawn for i in c) == list(range(23))
    assert len(drawn) == 5 and all(1 <= len(c) <= 5 for c in drawn)
    assert all(c == list(range(c[0], c[0] + len(c))) for c in drawn)
    # the frame files and their folder are removed
    assert len(folders) == 1 and not folders.pop().exists()
    assert [p.name for p in out.parent.iterdir()] == ["gray.mp4"]


def test_render_video_fps_crf_and_hold(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    draw = make_draw(log, 8, noise=True)
    fine = tc.render_video(draw, 8, tmp_path / "fine.mp4", fps=10, crf=10, n_jobs=1)
    coarse = tc.render_video(draw, 8, tmp_path / "coarse.mp4", fps=10, crf=40, n_jobs=1)
    assert fine.stat().st_size > 2 * coarse.stat().st_size
    meta, frames = decode(fine)
    assert meta["fps"] == 10 and len(frames) == 8
    # the last frame held for half a second: 5 more frames, equal to the last one
    held = tc.render_video(draw, 8, tmp_path / "held.mp4", fps=10, crf=10, hold=0.5, n_jobs=1)
    meta, held_frames = decode(held)
    assert len(held_frames) == 13
    assert abs(meta["duration"] - 1.3) < 0.05
    for a, b in zip(frames, held_frames):
        assert np.abs(a.astype(int) - b).mean() < 4
    for f in held_frames[8:]:
        assert np.abs(f.astype(int) - held_frames[7]).mean() < 4
    assert np.abs(held_frames[6].astype(int) - held_frames[7]).mean() > 20
    # a frame rate that is not an integer (the preview quality of the HTFA tutorial)
    slow = tc.render_video(draw, 8, tmp_path / "slow.mp4", fps=3.75, n_jobs=1)
    assert decode(slow)[0]["fps"] == 3.75


def test_render_video_equals_the_encoding_of_the_tutorials(tmp_path):
    # the same frames encoded with the command the tutorial notebooks ran before the
    # helper existed (stocks.ipynb cell 29, CRF 27; trajectories.ipynb cell 15 draws the
    # frames of its one-second hold itself): the files are identical
    log = tmp_path / "log"
    log.mkdir()
    draw = make_draw(log, 9, noise=True)
    FPS, CRF = 30, 27
    for hold, name in ((0, "plain"), (1, "held")):
        frames_dir = tmp_path / f"frames_{name}"
        frames_dir.mkdir()
        ids = list(range(9))
        draw(ids, [frames_dir / f"{i:05d}.png" for i in ids])
        for i in range(9, 9 + hold * FPS):
            draw([8], [frames_dir / f"{i:05d}.png"])
        video = tmp_path / f"notebook_{name}.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", str(CRF),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                       check=True)
        ours = tc.render_video(draw, 9, tmp_path / f"ours_{name}.mp4", fps=FPS, crf=CRF,
                               hold=hold, n_jobs=2, chunk_size=4)
        assert ours.read_bytes() == video.read_bytes()
        assert len(decode(ours)[1]) == 9 + hold * FPS


def test_render_video_retries_only_the_missing_frames(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    with pytest.raises(Exception, match="a worker failed"):
        tc.render_video(make_draw(log, 12, fail_once=True), 12, tmp_path / "a.mp4",
                        n_jobs=2, chunk_size=4)
    assert not (tmp_path / "a.mp4").exists()
    _, folders = calls(log)
    assert folders and not any(f.exists() for f in folders)    # removed after a failure too

    log = tmp_path / "log2"
    log.mkdir()
    out = tc.render_video(make_draw(log, 12, fail_once=True), 12, tmp_path / "b.mp4",
                          n_jobs=2, chunk_size=4, retries=1)
    assert len(decode(out)[1]) == 12
    drawn, _ = calls(log)
    counts = np.bincount([i for c in drawn for i in c], minlength=12)
    assert counts[0] == 1                 # frame 0 was written before its chunk failed
    assert counts.max() <= 2 and (counts == 2).any()
    assert (counts[4:] == 1).sum() >= 4   # chunks that finished were not drawn again


def test_render_video_names_frames_that_were_not_written(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    with pytest.raises(RuntimeError, match=r"did not write.*\b3\b"):
        tc.render_video(make_draw(log, 6, skip=(3,)), 6, tmp_path / "gap.mp4", n_jobs=1)
    assert not (tmp_path / "gap.mp4").exists()
    with pytest.raises(ValueError, match="n_frames"):
        tc.render_video(make_draw(log, 6), 0, tmp_path / "none.mp4")


# ---------------------------------------------------------------- display
LINK = ('<video src="{src}" controls loop muted playsinline preload="metadata" '
        'style="width: 100%; max-width: {width}px;"></video>')


def shown(*args, **kwargs):
    """The HTML that ``tc.show_video`` displays in an IPython shell."""
    InteractiveShell.instance()
    with capture_output() as captured:
        assert tc.show_video(*args, **kwargs) is None
    assert len(captured.outputs) == 1
    return captured.outputs[0].data["text/html"]


@pytest.fixture
def video(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    return tc.render_video(make_draw(log, 3), 3, tmp_path / "three.mp4", n_jobs=1)


def test_inside_a_docs_tree_videos_are_linked(tmp_path, video, monkeypatch):
    docs = tmp_path / "docs"
    (docs / "tutorial").mkdir(parents=True)
    (docs / "_static").mkdir()
    (docs / "conf.py").write_text("project = 'example'\n")
    monkeypatch.chdir(docs / "tutorial")
    media = tc.media_dir()
    assert media == Path("..") / "_static" / "tutorial_media" and media.is_dir()
    assert tc.media_dir("htfa_videos") == media
    target = media / "three.mp4"
    target.write_bytes(video.read_bytes())
    html = shown(target)
    assert html == LINK.format(src="../_static/tutorial_media/three.mp4", width=960)
    assert shown(target.resolve(), width=640) == LINK.format(
        src="../_static/tutorial_media/three.mp4", width=640)


def test_elsewhere_videos_are_embedded(tmp_path, video, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert tc.media_dir() == Path("tutorial_media") and (work / "tutorial_media").is_dir()
    assert tc.media_dir("htfa_videos") == Path("htfa_videos") and (work / "htfa_videos").is_dir()
    html = shown(video)
    assert base64.b64encode(video.read_bytes()).decode() in html
    assert "controls loop muted playsinline" in html and 'width="960"' in html
    assert 'width="480"' in shown(str(video), width=480)
    # conf.py without _static is not a docs tree
    (tmp_path / "conf.py").write_text("")
    assert tc.media_dir() == Path("tutorial_media")


def test_a_url_is_linked_everywhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    url = "https://timecorr.readthedocs.io/en/latest/_static/tutorial_media/htfa_order1.mp4"
    assert shown(url) == LINK.format(src=url, width=960)


def test_optional_packages_are_imported_lazily():
    # importing timecorr does not import imageio-ffmpeg or IPython (joblib is imported by
    # scikit-learn, a runtime dependency)
    code = ("import sys, timecorr; "
            "print([m for m in ('imageio_ffmpeg', 'IPython') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
