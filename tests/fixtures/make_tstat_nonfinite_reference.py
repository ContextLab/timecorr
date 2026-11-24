"""Write ``tstat_nonfinite_reference.npz``: what ``tstat_combine`` returns
(t and p) for every list of ``nonfinite_inputs`` in
``tests/test_round_off_variance.py``, the lists of 2 and 3 values with at
least one infinite or NaN value.

Run from the repository root, with the timecorr package of the commit whose
outputs are to be frozen first on the import path (for example
``PYTHONPATH=<checkout of that commit> python
tests/fixtures/make_tstat_nonfinite_reference.py``).  The file was written
with the package at commit 589c2a1 (release 0.3.1's ``tstat_combine``).
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE))

import timecorr  # noqa: E402
from test_round_off_variance import nonfinite_inputs  # noqa: E402

if __name__ == "__main__":
    print(f"timecorr from {os.path.dirname(timecorr.__file__)}")
    out = {}
    for size, values in nonfinite_inputs().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t, p = timecorr.tstat_combine([column[np.newaxis] for column in values.T],
                                          return_pvals=True)
        out[f"t{size}"], out[f"p{size}"] = t[0], p[0]
    np.savez(os.path.join(HERE, "tstat_nonfinite_reference.npz"), **out)
