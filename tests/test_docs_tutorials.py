"""
The tutorial notebooks are the gallery of the documentation.

These tests read the real notebooks in docs/tutorial/ and the real
docs/tutorials.rst (nothing is mocked) and check that:

- every notebook is in the gallery (the ``nbgallery`` directive of
  docs/tutorials.rst) and every gallery entry is a notebook;
- each notebook has exactly one code cell tagged ``nbsphinx-thumbnail``, and
  that cell has an image output (the gallery card's figure);
- each notebook has an "Open in Colab" badge whose URL names its own file,
  and its first code cell installs timecorr when it runs on Google Colab;
- no notebook has an error output;
- every relative link in a notebook's text names another tutorial notebook
  (links to other pages are absolute URLs, which also work on Colab);
- every public name of the package (listed as in tests/test_api_docs.py)
  is used in a code cell of at least one tutorial: as ``tc.<name>``, imported
  or called (comments and string literals do not count).
"""

import io
import json
import os
import re
import tokenize

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
def test_first_code_cell_is_the_colab_install_cell(name):
    code_cells = [c for c in _load(name)["cells"] if c["cell_type"] == "code"]
    assert code_cells, f"{name}: no code cells"
    first = _source(code_cells[0])
    assert re.search(r"[\"']google\.colab[\"']\s+in\s+sys\.modules", first), (
        f"{name}: the first code cell does not check for Google Colab: {first!r}"
    )
    # a pip install whose requirement is timecorr (with or without extras,
    # a version or a git URL)
    installs = re.findall(r"pip install[^\n|]*", first)
    assert any(
        re.search(r"[\s\"']timecorr(\[[^\]]*\])?\s*(>=|==|@|[\"'\s]|$)", line)
        for line in installs
    ), f"{name}: the first code cell does not pip install timecorr: {first!r}"


@pytest.mark.parametrize("name", _notebook_names())
def test_no_error_outputs(name):
    errors = [
        i
        for i, c in enumerate(_load(name)["cells"])
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"{name}: error outputs in cells {errors}"


@pytest.mark.parametrize("name", _notebook_names())
def test_relative_links_name_tutorial_notebooks(name):
    # On Colab a relative link resolves next to the notebook on GitHub, so it
    # works only for another tutorial notebook; other pages need absolute URLs.
    markdown = "\n".join(
        _source(c) for c in _load(name)["cells"] if c["cell_type"] == "markdown"
    )
    targets = re.findall(r"\]\(([^)\s]+)\)", markdown)
    relative = [t for t in targets if not re.match(r"[a-z]+:", t) and not t.startswith("#")]
    notebooks = set(_notebook_names())
    bad = [t for t in relative if t.split("#")[0][: -len(".ipynb")] not in notebooks
           or not t.split("#")[0].endswith(".ipynb")]
    assert not bad, f"{name}: relative links that are not tutorial notebooks: {bad}"


def _public_names():
    names = set(_exported())
    for names_in_module in USER_FACING_MODULE_NAMES.values():
        names.update(names_in_module)
    return sorted(names)


def _used_names(cell_source):
    """
    Names a code cell uses: ``tc.<name>``, imported, or called.

    Comments and string literals are skipped (tokenize drops them), and
    IPython magics (lines starting with ``%`` or ``!``) are replaced by
    ``pass``, so a name only printed or mentioned in a comment does not count.
    """
    lines = []
    for line in cell_source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            line = line[: len(line) - len(stripped)] + "pass"
        lines.append(line)
    # Python >= 3.12 tokenizes the inside of f-strings; skip it like other strings
    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    tokens = []
    depth = 0
    for t in tokenize.generate_tokens(io.StringIO("\n".join(lines) + "\n").readline):
        if fstring_start is not None and t.type == fstring_start:
            depth += 1
        elif fstring_end is not None and t.type == fstring_end:
            depth -= 1
        elif depth == 0 and t.type in (tokenize.NAME, tokenize.OP, tokenize.NEWLINE):
            tokens.append(t)
    used = set()
    statement = []
    for i, t in enumerate(tokens):
        if t.type == tokenize.NEWLINE:
            if statement and statement[0] in ("import", "from"):
                used.update(statement)
            statement = []
            continue
        statement.append(t.string)
        if t.type != tokenize.NAME:
            continue
        follows_tc = i >= 2 and tokens[i - 1].string == "." and tokens[i - 2].string == "tc"
        called = i + 1 < len(tokens) and tokens[i + 1].string == "("
        if follows_tc or called:
            used.add(t.string)
    return used


def test_used_names_ignores_strings_and_comments():
    src = (
        "import sys\n"
        "from timecorr.decoding import decoder\n"
        "if 'x' in sys.modules:\n"
        "    !pip install timecorr\n"
        "print('null_combine:')  # mat2vec\n"
        "y = tc.vec2mat(x)\n"
        "z = smooth(y)\n"
        "w = f'{tc.isfc}'\n"
    )
    used = _used_names(src)
    assert {"decoder", "vec2mat", "smooth"} <= used
    assert not {"null_combine", "mat2vec", "isfc"} & used


def test_every_public_name_is_in_a_tutorial():
    used = set()
    for name in _notebook_names():
        for c in _load(name)["cells"]:
            if c["cell_type"] == "code":
                used |= _used_names(_source(c))
    missing = [n for n in _public_names() if n not in used]
    assert not missing, (
        "public names not used (as tc.<name>, imported or called) in any tutorial "
        f"code cell: {missing}"
    )
