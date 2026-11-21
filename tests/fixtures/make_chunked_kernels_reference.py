"""Write ``chunked_kernels_reference.npz``: the reference values (see
``reference_values`` in ``tests/test_kernels_reference.py``) of each case of
``CASES``.

Run from the repository root, with the timecorr package of the commit whose
outputs are to be frozen first on the import path (for example
``PYTHONPATH=<checkout of that commit> python
tests/fixtures/make_chunked_kernels_reference.py``).  The file was written
with the package at commit 6b3d23a, before the built-in kernels were read in
chunks of rows.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE))

import timecorr  # noqa: E402
from test_kernels_reference import CASES, reference_values  # noqa: E402

if __name__ == "__main__":
    print(f"timecorr from {os.path.dirname(timecorr.__file__)}")
    values = {name: reference_values(name) for name in sorted(CASES)}
    out = os.path.join(HERE, "chunked_kernels_reference.npz")
    np.savez_compressed(out, **values)
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
