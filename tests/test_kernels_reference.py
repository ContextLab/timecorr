"""
Outputs of ``timecorr`` compared with those of the package at commit 6b3d23a,
before the built-in kernels were read in chunks of rows (``_kernels``).

The other tests of the chunked kernels compare a built-in kernel with the
same kernel given as a T x T array; both are read by the same chunk code.
These compare with stored outputs of the earlier code, which built the T x T
kernel and multiplied it with the data in full.  The reference file
``tests/fixtures/chunked_kernels_reference.npz`` was written by
``tests/fixtures/make_chunked_kernels_reference.py`` with the package at
commit 6b3d23a.  The values are compared up to round-off (rtol 1e-9, atol
1e-12): the products are summed in a different order, and BLAS libraries
round differently in their last bits.

Only the public interface of ``timecorr`` is used here, so the fixture
script can import this module with the earlier package.
"""

import os
import warnings

import numpy as np
import pytest

import timecorr as tc

T, K, S = 160, 3, 3  # 160 timepoints: two chunks of 128 rows, the second short
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "chunked_kernels_reference.npz")


def as_array_function(f):
    """The kernel function ``f`` as a function of the user's (T x T arrays)."""
    return lambda T, p=None: f(T) if p is None else f(T, p)


def subjects(kind):
    rng = np.random.RandomState(0)
    data = [rng.randn(T, K) for _ in range(S)]
    if kind == "nan":
        for x in data:
            x[rng.rand(T, K) < 0.08] = np.nan
    elif kind == "inf":
        data[0][20, 1] = np.inf  # outside the columns of rows 128:160
        data[1][150, 2] = -np.inf  # outside the columns of rows 0:128
    return data


# name: (data, weights_function, weights_params, cfun, options)
CASES = {
    "gaussian_isfc": ("clean", tc.gaussian_weights, None, tc.isfc, {}),
    "gaussian_array_isfc": ("clean", as_array_function(tc.gaussian_weights), None,
                            tc.isfc, {}),
    "gaussian100_autofc_nan": ("nan", tc.gaussian_weights, {"var": 100}, tc.autofc, {}),
    "gaussian_wisfc": ("clean", tc.gaussian_weights, None, tc.wisfc, {}),
    "gaussian25_exclude3_isfc": ("clean", tc.gaussian_weights, {"var": 25}, tc.isfc,
                                 {"exclude_timepoints": 3}),
    "gaussian_subject_events_isfc": ("clean", tc.gaussian_weights, None, tc.isfc,
                                     {"events": [[40], None, [70, 129]]}),
    "laplace20_backward_isfc": ("clean", tc.laplace_weights, {"scale": 20}, tc.isfc,
                                {"include_timepoints": "backward"}),
    "laplace_autofc": ("clean", tc.laplace_weights, None, tc.autofc, {}),
    "t_isfc": ("clean", tc.t_weights, None, tc.isfc, {}),
    "t3_forward_nozero_autofc": ("clean", tc.t_weights, {"df": 3}, tc.autofc,
                                 {"include_timepoints": "forward",
                                  "include_zero": False}),
    "mexican_hat10_isfc": ("clean", tc.mexican_hat_weights, {"sigma": 10}, tc.isfc, {}),
    "boxcar12_events_autofc": ("clean", tc.boxcar_weights, {"width": 12}, tc.autofc,
                               {"events": [50, 129]}),
    "boxcar12_exclude_isfc": ("clean", tc.boxcar_weights, {"width": 12}, tc.isfc,
                              {"exclude_timepoints": -2}),
    "uniform_autofc": ("clean", tc.uniform_weights, None, tc.autofc, {}),
    "uniform_backward_isfc_nan": ("nan", tc.uniform_weights, None, tc.isfc,
                                  {"include_timepoints": "backward"}),
    "uniform_array_backward_isfc_nan": ("nan", as_array_function(tc.uniform_weights),
                                        None, tc.isfc,
                                        {"include_timepoints": "backward"}),
    "eye_autofc": ("clean", tc.eye_weights, None, tc.autofc, {}),
    "smooth_gaussian": ("clean", tc.gaussian_weights, None, None, {}),
    "smooth_boxcar_nan": ("nan", tc.boxcar_weights, {"width": 12}, None, {}),
    "smooth_laplace_backward": ("clean", tc.laplace_weights, {"scale": 20}, None,
                                {"include_timepoints": "backward"}),
    "smooth_boxcar_inf": ("inf", tc.boxcar_weights, {"width": 10}, None, {}),
    "smooth_gaussian_inf": ("inf", tc.gaussian_weights, {"var": 25}, None, {}),
}


def reference_values(name):
    """The case's output: subject 0 in full, then each subject's column sums
    (NaN and inf entries kept)."""
    kind, f, params, cfun, options = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = tc.timecorr(subjects(kind), weights_function=f, weights_params=params,
                          cfun=cfun, combine=tc.null_combine, **options)
    out = [np.asarray(x, dtype=np.float64) for x in out]
    with np.errstate(invalid="ignore"):
        sums = np.stack([x.sum(axis=0) for x in out])
    return np.concatenate([out[0].ravel(), sums.ravel()])


@pytest.mark.parametrize("name", sorted(CASES))
def test_outputs_equal_those_of_the_full_kernel_products(name):
    with np.load(FIXTURE) as reference:
        expected = reference[name]
    np.testing.assert_allclose(reference_values(name), expected, rtol=1e-9,
                               atol=1e-12, equal_nan=True)
