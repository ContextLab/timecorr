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

import ast
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
    assert _installs_timecorr(first), (
        f"{name}: the first code cell does not pip install timecorr: {first!r}"
    )


# a requirement naming timecorr: with or without extras, a version or a URL
_TIMECORR_REQUIREMENT = re.compile(r"timecorr(\[[^\]]*\])?\s*(>=|==|~=|<|@|;|$)")


def _static_text(node, env):
    """The text a string expression evaluates to, from literals, f-strings,
    + and names assigned a string in the cell (None when not static; an
    f-string field that is not static becomes "\x00")."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            text = _static_text(v.value if isinstance(v, ast.FormattedValue) else v, env)
            parts.append("\x00" if text is None else text)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_text(node.left, env), _static_text(node.right, env)
        return None if left is None or right is None else left + right
    return None


def _pip_requirements(call):
    """The argument nodes that a call passes to ``pip install`` as
    requirements: the elements after "pip", "install" of a list argument
    (options, starting with "-", left out), or the words after "pip install"
    of a string argument (a shell command)."""
    out = []
    for arg in call.args:
        if isinstance(arg, (ast.List, ast.Tuple)):
            texts = [_static_text(e, {}) for e in arg.elts]
            for i in range(len(texts) - 1):
                if texts[i] == "pip" and texts[i + 1] == "install":
                    out += [e for e, t in zip(arg.elts[i + 2:], texts[i + 2:])
                            if not (t or "").startswith("-") and not isinstance(e, ast.Starred)]
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            m = re.search(r"pip install((?:\s+\S+)+)", arg.value)
            if m:
                out += [ast.Constant(w) for w in m.group(1).split() if not w.startswith("-")]
    return out


def _installs_timecorr(source):
    """
    True when the cell passes a requirement naming timecorr to pip: in an
    IPython shell line (``!pip install ...``, ``%pip install ...``), in a
    call whose arguments include pip's ("pip", "install", requirement) or a
    "pip install requirement" command, directly or through a string
    variable, or through a parameter of a function of the cell that does
    so, called with such a requirement.  A string naming timecorr that is
    not what pip is asked to install does not count.
    """
    shell = [line for line in source.splitlines() if line.lstrip().startswith(("!", "%"))]
    for line in shell:
        m = re.search(r"pip install((?:\s+\S+)+)", line)
        if m and any(_TIMECORR_REQUIREMENT.match(w.strip("\"'"))
                     for w in m.group(1).split() if not w.startswith("-")):
            return True
    tree = ast.parse("\n".join("" if line in shell else line for line in source.splitlines()))
    env = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            text = _static_text(node.value, env)
            if text is not None:
                env[node.targets[0].id] = text
    requirements = []
    # functions that pass one of their parameters to pip, by position
    passes = {}
    for f in ast.walk(tree):
        if not isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = [a.arg for a in f.args.args]
        for call in ast.walk(f):
            if isinstance(call, ast.Call):
                for r in _pip_requirements(call):
                    if isinstance(r, ast.Name) and r.id in params:
                        passes.setdefault(f.name, set()).add(params.index(r.id))
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        requirements += _pip_requirements(call)
        if isinstance(call.func, ast.Name) and call.func.id in passes:
            requirements += [call.args[i] for i in passes[call.func.id] if i < len(call.args)]
    texts = [_static_text(r, env) for r in requirements]
    return any(t is not None and _TIMECORR_REQUIREMENT.match(t.strip()) for t in texts)


def test_the_install_checker_needs_timecorr_to_be_what_pip_installs():
    # the checker itself: a string naming timecorr that pip is not asked to
    # install does not count
    unused = (
        'import subprocess, sys\n'
        'if "google.colab" in sys.modules:\n'
        '    PACKAGE = "timecorr"\n'
        '    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy"])\n'
    )
    assert not _installs_timecorr(unused)
    assert not _installs_timecorr('X = "timecorr"\n!pip install numpy\n')
    assert not _installs_timecorr('subprocess.run(["pip", "install", "timecorrx"])\n')
    # what counts: a list, a variable, an f-string through a function, a
    # shell line, extras, a version and a URL
    assert _installs_timecorr('subprocess.run(["pip", "install", "timecorr"])\n')
    assert _installs_timecorr('P = "timecorr[segment]"\n'
                              'subprocess.run([sys.executable, "-m", "pip", "install", "-q", P])\n')
    assert _installs_timecorr(
        'PACKAGE = "timecorr"\n'
        'def pip_install(requirement):\n'
        '    return subprocess.run([sys.executable, "-m", "pip", "install", requirement])\n'
        'pip_install(f"{PACKAGE}>=0.3.0")\n')
    assert _installs_timecorr("!pip install -q timecorr>=0.3\n")
    assert _installs_timecorr('os.system("pip install timecorr @ git+https://x")\n')
    # a function that installs its parameter, called with something else
    assert not _installs_timecorr(
        'PACKAGE = "timecorr"\n'
        'def pip_install(requirement):\n'
        '    return subprocess.run(["pip", "install", requirement])\n'
        'pip_install("numpy")\n')


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
