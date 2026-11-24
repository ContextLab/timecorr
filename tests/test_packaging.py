"""
The declared dependencies allow only versions the package runs with.

``timecorr.helpers.reduce`` calls ``hypertools.reduce(..., random_state=0)``,
and the causal mode fits models with ``return_model=True`` and applies them
with ``model=``.  hypertools 0.8 has none of these arguments (the default
``tc.features`` call raised ``TypeError: reduce() got an unexpected keyword
argument 'random_state'`` there), so the declared floor must exclude it.

These tests read the real setup.py and requirements.txt and the installed
hypertools; nothing is mocked.
"""

import ast
import inspect
import os

import hypertools
import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = os.path.dirname(os.path.dirname(__file__))
FLOOR = Version("1.1")


def _setup_keyword(name):
    """The literal value of a keyword of the ``setup(...)`` call in setup.py."""
    with open(os.path.join(ROOT, "setup.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    ]
    assert len(calls) == 1, "setup.py needs exactly one setup(...) call"
    values = [k.value for k in calls[0].keywords if k.arg == name]
    assert len(values) == 1, f"setup.py: no {name}= keyword"
    return ast.literal_eval(values[0])


def _requirements_txt():
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        lines = [line.split("#")[0].strip() for line in f]
    return [line for line in lines if line]


def _hypertools_floor(requirements, where):
    """The lowest hypertools version that a list of requirement strings allows."""
    found = [r for r in map(Requirement, requirements) if r.name == "hypertools"]
    assert len(found) == 1, f"{where}: {len(found)} hypertools requirements"
    floors = [
        Version(s.version) for s in found[0].specifier if s.operator in (">=", "==", "~=")
    ]
    assert floors, f"{where}: hypertools has no lower bound ({found[0]})"
    return max(floors)


@pytest.mark.parametrize(
    "where,requirements",
    [
        ("setup.py install_requires", _setup_keyword("install_requires")),
        ("requirements.txt", _requirements_txt()),
        ("setup.py extras_require['tutorials']", _setup_keyword("extras_require")["tutorials"]),
    ],
    ids=["install_requires", "requirements.txt", "tutorials-extra"],
)
def test_hypertools_floor_is_at_least_1_1(where, requirements):
    assert _hypertools_floor(requirements, where) >= FLOOR, where


def test_the_floor_excludes_the_versions_without_the_arguments_used():
    # the specifier of install_requires rejects 0.8.x and 1.0.x and accepts 1.1.0
    requirement = [
        r
        for r in map(Requirement, _setup_keyword("install_requires"))
        if r.name == "hypertools"
    ][0]
    for version in ("0.8.0", "0.8.2", "1.0.0"):
        assert not requirement.specifier.contains(version), version
    assert requirement.specifier.contains("1.1.0")


def test_installed_hypertools_meets_the_floor_and_has_the_arguments():
    assert Version(hypertools.__version__) >= FLOOR, hypertools.__version__
    parameters = inspect.signature(hypertools.reduce).parameters
    for name in ("random_state", "return_model", "model"):
        assert name in parameters, f"hypertools.reduce has no {name}="


def test_install_requires_and_requirements_txt_agree():
    def norm(requirements):
        return sorted(str(Requirement(r)) for r in requirements)

    assert norm(_setup_keyword("install_requires")) == norm(_requirements_txt())
