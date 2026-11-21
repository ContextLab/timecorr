"""Write ``private_signs_reference.npz``: the reference values (see
``reference_values`` in ``tests/test_simulate_hierarchy.py``) of
``small(S=3, shared=..., set_random_seed=seed)`` for each key of
``PRIVATE_SIGNS_KEYS``.

Run from the repository root, with the timecorr package of the commit whose
outputs are to be frozen first on the import path (for example
``PYTHONPATH=<checkout of that commit> python
tests/fixtures/make_private_signs_reference.py``).  The file was written with
the package at commit 012f5e0^, before the ``signs`` option was added.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE))

import timecorr  # noqa: E402
from test_simulate_hierarchy import (  # noqa: E402
    PRIVATE_SIGNS_KEYS,
    private_signs_reference_name,
    reference_values,
    small,
)

if __name__ == "__main__":
    print(f"timecorr from {os.path.dirname(timecorr.__file__)}")
    values = {}
    for shared, seed in PRIVATE_SIGNS_KEYS:
        data, truth = small(S=3, shared=shared, set_random_seed=seed)
        values[private_signs_reference_name((shared, seed))] = reference_values(
            data, truth
        )
    out = os.path.join(HERE, "private_signs_reference.npz")
    np.savez_compressed(out, **values)
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
