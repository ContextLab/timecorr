"""
Every user-facing name of timecorr is documented: a numpy-style docstring with
Parameters, Returns and Examples, and an entry on the API reference page
(docs/api.rst) that resolves to the real object.  The default kernel parameter
table on that page must match the dictionaries in the code.

These tests read the real package and the real docs/api.rst; nothing is
mocked.
"""

import ast
import importlib
import inspect
import os
import re
import types

import pytest

import timecorr as tc

API_RST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "api.rst")

MODULES = [
    "timecorr.timecorr",
    "timecorr.helpers",
    "timecorr.higher_order",
    "timecorr.decoding",
    "timecorr.simulate",
    "timecorr.events",
    "timecorr.summaries",
]

# Public names that are not exported from ``timecorr`` but are documented on the
# API page, under the module they live in.
USER_FACING_MODULE_NAMES = {
    "timecorr.higher_order": ["two_path_step", "estimate_memory_gb"],
    "timecorr.events": [
        "check_boundaries",
        "block_labels",
        "resolve_events",
        "resolve_events_by_order",
        "mask_kernel",
        "distinct_event_sets",
    ],
    "timecorr.decoding": [
        "weighted_timepoint_decoder_ec",
        "optimize_weights",
        "folding_levels",
        "folding_levels_ec",
        "pca_decoder",
        "decoder",
        "weight_corrs",
        "get_xval_assignments",
    ],
    "timecorr.helpers": ["r2z", "z2r", "plot_weights"],
}

# Public names that are internal helpers: they need a docstring, not an API page.
INTERNAL_NAMES = {
    "timecorr.helpers": ["apply_by_row", "weighted_mean", "rmdiag", "symmetric"],
    "timecorr.decoding": ["reduce_wrapper", "sum_to_x", "calculate_error"],
    "timecorr.simulate": [
        "random_corrmat",
        "ramping_dataset",
        "random_dataset",
        "constant_dataset",
        "block_dataset",
    ],
}

KERNEL_PARAMS = {
    "gaussian_params": "gaussian_weights",
    "laplace_params": "laplace_weights",
    "eye_params": "eye_weights",
    "t_params": "t_weights",
    "mexican_hat_params": "mexican_hat_weights",
    "uniform_params": "uniform_weights",
    "boxcar_params": "boxcar_weights",
}


def _exported():
    """Public, non-module names importable from ``timecorr``."""
    if hasattr(tc, "__all__"):
        names = list(tc.__all__)
    else:
        names = [n for n in dir(tc) if not n.startswith("_")]
    return sorted(
        n for n in names if not isinstance(getattr(tc, n), types.ModuleType)
    )


def _defined_public(modname):
    """Public functions, classes and data defined in (not imported into) a module."""
    mod = importlib.import_module(modname)
    out = []
    for name, obj in vars(mod).items():
        if name.startswith("_") or isinstance(obj, types.ModuleType):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if obj.__module__ == modname:
                out.append(name)
        elif not callable(obj):
            # module-level data: count it only where it is assigned
            source = inspect.getsource(mod)
            if re.search(r"^%s\s*=" % re.escape(name), source, re.M):
                out.append(name)
    return sorted(out)


def _user_facing_functions():
    """(label, function) for every user-facing function."""
    out = [(f"timecorr.{n}", getattr(tc, n)) for n in _exported()]
    for modname, names in USER_FACING_MODULE_NAMES.items():
        mod = importlib.import_module(modname)
        out += [(f"{modname}.{n}", getattr(mod, n)) for n in names]
    return [(label, f) for label, f in out if inspect.isfunction(f)]


def _has_section(doc, title):
    return re.search(r"^\s*%s\s*\n\s*-{3,}\s*$" % title, doc, re.M) is not None


def _autosummary_entries():
    """(currentmodule, entry, has_toctree) for every autosummary entry in api.rst."""
    with open(API_RST, encoding="utf-8") as f:
        lines = f.read().splitlines()
    current = None
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^\.\. currentmodule::\s*(\S+)\s*$", line)
        if m:
            current = m.group(1)
        if line.strip() == ".. autosummary::":
            i += 1
            toctree = False
            # options, then a blank line, then indented entries
            while i < len(lines) and (
                lines[i].strip() == "" or lines[i].startswith("   ")
            ):
                text = lines[i].strip()
                if text.startswith(":toctree:"):
                    toctree = True
                elif text and not text.startswith(":"):
                    entries.append((current, text, toctree))
                i += 1
            continue
        i += 1
    return entries


def _kernel_param_table():
    """Parse the list-table in the "Default kernel parameters" section."""
    with open(API_RST, encoding="utf-8") as f:
        text = f.read()
    m = re.search(
        r"^Default kernel parameters\n-+\n(.*?)(?=^\S[^\n]*\n-{3,}\n)",
        text,
        re.M | re.S,
    )
    assert m, "docs/api.rst has no 'Default kernel parameters' section"
    section = m.group(1)
    table = section[section.index(".. list-table::") :]
    rows = []
    for line in table.splitlines():
        if line.startswith("   * - "):
            rows.append([line[len("   * - ") :].strip()])
        elif line.startswith("     - ") and rows:
            rows[-1].append(line[len("     - ") :].strip())
        elif line.startswith("       ") and line.strip() and rows:
            # continuation of the last cell
            rows[-1][-1] += " " + line.strip()
        elif line.strip() and not line.startswith("   ") and rows:
            break
    header, body = rows[0], rows[1:]
    assert header == ["Name", "Value", "Kernel and meaning of each key"], header
    return body


def _literal(cell):
    m = re.fullmatch(r"``(.*)``", cell)
    assert m, f"expected a literal in double backquotes, got {cell!r}"
    return m.group(1)


def test_every_public_name_is_classified():
    exported = set(_exported())
    facing = {(m, n) for m, ns in USER_FACING_MODULE_NAMES.items() for n in ns}
    internal = {(m, n) for m, ns in INTERNAL_NAMES.items() for n in ns}
    assert not facing & internal
    unclassified = []
    for modname in MODULES:
        for name in _defined_public(modname):
            key = (modname, name)
            if name in exported or key in facing or key in internal:
                continue
            unclassified.append(f"{modname}.{name}")
    assert not unclassified, (
        "public names that are neither exported, user-facing nor internal "
        f"(classify them in tests/test_api_docs.py): {unclassified}"
    )
    for modname, name in facing | internal:
        assert hasattr(importlib.import_module(modname), name), (modname, name)


@pytest.mark.parametrize(
    "label,func", _user_facing_functions(), ids=lambda x: x if isinstance(x, str) else ""
)
def test_user_facing_docstring_sections(label, func):
    doc = inspect.getdoc(func)
    assert doc, f"{label} has no docstring"
    if inspect.signature(func).parameters:
        assert _has_section(doc, "Parameters"), f"{label}: no Parameters section"
    assert _has_section(doc, "Returns"), f"{label}: no Returns section"
    assert _has_section(doc, "Examples"), f"{label}: no Examples section"
    assert ">>> " in doc, f"{label}: Examples has no doctest"


@pytest.mark.parametrize(
    "modname,name",
    [(m, n) for m, ns in INTERNAL_NAMES.items() for n in ns],
)
def test_internal_helpers_have_docstrings(modname, name):
    doc = inspect.getdoc(getattr(importlib.import_module(modname), name))
    assert doc and len(doc.split()) >= 5, f"{modname}.{name} has no docstring"


def test_every_user_facing_name_is_on_the_api_page():
    listed = {(mod, entry) for mod, entry, _ in _autosummary_entries()}
    missing = [n for n in _exported() if ("timecorr", n) not in listed]
    for modname, names in USER_FACING_MODULE_NAMES.items():
        missing += [f"{modname}.{n}" for n in names if (modname, n) not in listed]
    assert not missing, f"not listed in docs/api.rst: {missing}"


def test_every_api_page_entry_resolves_and_gets_a_page():
    entries = _autosummary_entries()
    assert len(entries) > 40
    for mod, entry, toctree in entries:
        assert mod is not None, f"{entry} is listed before any currentmodule"
        obj = importlib.import_module(mod)
        for part in entry.split("."):
            assert hasattr(obj, part), f"{mod}.{entry} does not exist"
            obj = getattr(obj, part)
        assert toctree, f"{mod}.{entry} is in an autosummary without :toctree:"


def test_kernel_parameter_table_matches_code():
    rows = _kernel_param_table()
    table = {}
    for row in rows:
        name_cell, value_cell = row[0], row[1]
        meaning = " ".join(row[2:])
        name = re.fullmatch(r":data:`(\w+)`", name_cell).group(1)
        assert name not in table, f"{name} is listed twice"
        table[name] = ast.literal_eval(_literal(value_cell))
        kernel = KERNEL_PARAMS[name]
        assert meaning.startswith(f":func:`{kernel}`"), (name, meaning)
        for key in table[name]:
            assert f"``{key}``:" in meaning, f"{name}: no meaning for key {key!r}"
        if not table[name]:
            assert "No keys" in meaning, name
    assert set(table) == set(KERNEL_PARAMS)
    for name, value in table.items():
        assert getattr(tc, name) == value, (name, getattr(tc, name), value)
        assert getattr(tc.helpers, name) is getattr(tc, name)


@pytest.mark.parametrize("params_name,kernel_name", sorted(KERNEL_PARAMS.items()))
def test_kernel_docstring_names_its_defaults(params_name, kernel_name):
    kernel = getattr(tc, kernel_name)
    doc = inspect.getdoc(kernel)
    assert f"``{params_name}``" in doc, f"{kernel_name} does not name {params_name}"
    for key, value in getattr(tc, params_name).items():
        assert f"``{key!r}``" in doc or f"``{key}``" in doc, (kernel_name, key)
        assert repr(value) in doc, (kernel_name, key, value)
    # the default is the dictionary itself
    assert inspect.signature(kernel).parameters["params"].default is getattr(
        tc, params_name
    )
