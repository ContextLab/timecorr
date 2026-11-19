# coding: utf-8
"""
Import optional dependencies, installing them on demand.

A missing optional module is installed into the running interpreter with
``python -m pip install <requirement>`` and then imported; a one-line notice is printed to
standard error first. Setting the environment variable ``TIMECORR_AUTO_INSTALL`` to ``0``,
``false`` or ``no`` (in any case) turns installation off; the import then raises
``ImportError`` with the manual command. Without pip in the running interpreter (some uv or
conda environments) no other installer is tried: ``ImportError`` names the manual command.
"""

import importlib
import importlib.util
import os
import subprocess
import sys

__all__ = ["lazy_import"]

#: environment variable that turns automatic installation off
AUTO_INSTALL_VARIABLE = "TIMECORR_AUTO_INSTALL"

#: values of :data:`AUTO_INSTALL_VARIABLE` (compared in lower case) that turn installation off
_OFF_VALUES = ("0", "false", "no")

#: seconds before a pip installation is abandoned
INSTALL_TIMEOUT_SECONDS = 600

#: lines of pip's output quoted in the error when an installation fails
_PIP_OUTPUT_LINES = 15


def _auto_install_enabled():
    value = os.environ.get(AUTO_INSTALL_VARIABLE, "")
    return value.strip().lower() not in _OFF_VALUES


def _manual_command(extra):
    return f'pip install "timecorr[{extra}]"'


def lazy_import(module, requirement, extra):
    """
    Import a module, installing it with pip first if it is missing.

    Parameters
    ----------
    module : str
        Module to import (for example ``"hmmlearn.hmm"``). Only a missing top-level
        package triggers an installation; any other import error propagates unchanged.
    requirement : str
        What ``pip install`` is given to provide the module (for example ``"hmmlearn"``).
    extra : str
        The timecorr extra that lists the requirement (for example ``"segment"``), used in
        the notice and in the manual command ``pip install "timecorr[<extra>]"``.

    Returns
    -------
    module : types.ModuleType
        The imported module.

    Raises
    ------
    ImportError
        If the module is missing and automatic installation is off
        (``TIMECORR_AUTO_INSTALL`` is ``0``, ``false`` or ``no``), pip is not available in
        the running interpreter, pip fails or times out (after 10 minutes; the message
        then includes pip's last output lines), or the module still cannot be imported
        after installation. The message gives the manual command.

    Examples
    --------
    >>> from timecorr._lazy import lazy_import
    >>> lazy_import("json", "json", "segment").__name__
    'json'
    """
    top = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != top:
            raise
        missing = exc

    manual = _manual_command(extra)
    if not _auto_install_enabled():
        raise ImportError(
            f"{top} is needed for {extra} but is not installed, and automatic installation "
            f"is off ({AUTO_INSTALL_VARIABLE}). Install it with: {manual}"
        ) from missing
    if importlib.util.find_spec("pip") is None:
        raise ImportError(
            f"{top} is needed for {extra} but is not installed, and pip is not available "
            f"in this Python environment ({sys.executable}) to install it. "
            f"Install it with: {manual}"
        ) from missing

    print(
        f"timecorr: installing {requirement} (needed for {extra}; "
        f"set {AUTO_INSTALL_VARIABLE}=0 to disable)",
        file=sys.stderr,
        flush=True,
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", requirement],
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ImportError(
            f"installing {requirement} (needed for {extra}) did not finish within "
            f"{INSTALL_TIMEOUT_SECONDS} seconds. Install it with: {manual}"
        ) from exc
    if result.returncode != 0:
        output = (result.stdout or "") + (result.stderr or "")
        tail = "\n".join(output.strip().splitlines()[-_PIP_OUTPUT_LINES:])
        raise ImportError(
            f"installing {requirement} (needed for {extra}) failed with exit code "
            f"{result.returncode}. Install it with: {manual}\npip's output ended with:\n{tail}"
        ) from missing

    importlib.invalidate_caches()
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{requirement} was installed but {module} still cannot be imported. "
            f"Install it with: {manual}"
        ) from exc
