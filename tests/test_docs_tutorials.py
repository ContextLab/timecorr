"""
The tutorial notebooks are the gallery of the documentation.

These tests read the real notebooks in docs/tutorial/ and the real
docs/tutorials.rst (nothing is mocked) and check that:

- every notebook is in the gallery (the ``nbgallery`` directive of
  docs/tutorials.rst) and every gallery entry is a notebook;
- each notebook has exactly one code cell tagged ``nbsphinx-thumbnail``, and
  that cell has an image output (the gallery card's figure);
- each notebook has an "Open in Colab" badge whose URL names its own file;
- no notebook has an error output;
- every public name of the package (listed as in tests/test_api_docs.py)
  appears in a code cell of at least one tutorial.
"""

import json
import os
import re

import pytest

from .test_api_docs import USER_FACING_MODULE_NAMES, _exported

DOCS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
TUTORIAL_DIR = os.path.join(DOCS, "tutorial")
TUTORIALS_RST = os.path.join(DOCS, "tutorials.rst")
COLAB = (
    "https://colab.research.google.com/github/ContextLab/timecorr/blob/master/"
    "docs/tutorial/{name}.ipynb"
)


def _notebook_names():
    return sorted(
        f[: -len(".ipynb")] for f in os.listdir(TUTORIAL_DIR) if f.endswith(".ipynb")
    )


def _load(name):
    with open(os.path.join(TUTORIAL_DIR, name + ".ipynb"), encoding="utf-8") as f:
        return json.load(f)


def _source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _gallery_entries():
    """Entries of the nbgallery directive in docs/tutorials.rst, in order."""
    with open(TUTORIALS_RST, encoding="utf-8") as f:
        lines = f.read().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == ".. nbgallery::"]
    assert len(starts) == 1, "docs/tutorials.rst needs exactly one nbgallery directive"
    entries = []
    for line in lines[starts[0] + 1 :]:
        if line.strip() == "" or line.strip().startswith(":"):
            continue
        if not line.startswith("   "):
            break
        entries.append(line.strip())
    return entries


def test_gallery_and_notebooks_match():
    entries = _gallery_entries()
    assert len(entries) == len(set(entries)), f"duplicate gallery entries: {entries}"
    assert all(e.startswith("tutorial/") for e in entries), entries
    in_gallery = sorted(e[len("tutorial/") :] for e in entries)
    assert in_gallery == _notebook_names()


def test_quickstart_is_first():
    assert _gallery_entries()[0] == "tutorial/quickstart"


@pytest.mark.parametrize("name", _notebook_names())
def test_one_thumbnail_cell_with_an_image(name):
    cells = _load(name)["cells"]
    tagged = [
        c for c in cells if "nbsphinx-thumbnail" in c.get("metadata", {}).get("tags", [])
    ]
    assert len(tagged) == 1, f"{name}: {len(tagged)} cells tagged nbsphinx-thumbnail"
    cell = tagged[0]
    assert cell["cell_type"] == "code"
    images = [
        o
        for o in cell.get("outputs", [])
        if any(k.startswith("image/") for k in o.get("data", {}))
    ]
    assert images, f"{name}: the thumbnail cell has no image output"


@pytest.mark.parametrize("name", _notebook_names())
def test_colab_badge_names_its_own_file(name):
    markdown = "\n".join(
        _source(c) for c in _load(name)["cells"] if c["cell_type"] == "markdown"
    )
    urls = re.findall(r"https://colab\.research\.google\.com/github/[^\s)\"']+", markdown)
    assert urls, f"{name}: no Open in Colab badge"
    assert COLAB.format(name=name) in urls, f"{name}: badge URLs {urls}"


@pytest.mark.parametrize("name", _notebook_names())
def test_no_error_outputs(name):
    errors = [
        i
        for i, c in enumerate(_load(name)["cells"])
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"{name}: error outputs in cells {errors}"


def _public_names():
    names = set(_exported())
    for names_in_module in USER_FACING_MODULE_NAMES.values():
        names.update(names_in_module)
    return sorted(names)


def test_every_public_name_is_in_a_tutorial():
    code = "\n".join(
        _source(c)
        for name in _notebook_names()
        for c in _load(name)["cells"]
        if c["cell_type"] == "code"
    )
    missing = [n for n in _public_names() if not re.search(r"\b%s\b" % re.escape(n), code)]
    assert not missing, f"public names in no tutorial code cell: {missing}"
