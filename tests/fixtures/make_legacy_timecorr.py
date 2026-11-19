"""Write ``legacy_timecorr.npz``: the outputs of the calls in ``CASES`` of
``tests/test_legacy_timecorr.py``.

Run from the repository root, with the timecorr package of the commit whose
outputs are to be frozen first on the import path (for example
``PYTHONPATH=<checkout of that commit> python
tests/fixtures/make_legacy_timecorr.py``).  The file was written with the
package at commit d4bb5ba.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE))

import timecorr  # noqa: E402
from test_legacy_timecorr import compute_all  # noqa: E402

if __name__ == "__main__":
    print(f"timecorr from {os.path.dirname(timecorr.__file__)}")
    out = os.path.join(HERE, "legacy_timecorr.npz")
    np.savez_compressed(out, **compute_all())
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
