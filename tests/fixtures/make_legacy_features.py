"""Write ``legacy_features.npz``: the outputs of the calls in ``CASES``.

Run from the repository root at the commit whose outputs are to be frozen:
``python tests/fixtures/make_legacy_features.py``.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from test_legacy_pipeline import compute_all  # noqa: E402

if __name__ == "__main__":
    out = os.path.join(HERE, "legacy_features.npz")
    np.savez_compressed(out, **compute_all())
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
