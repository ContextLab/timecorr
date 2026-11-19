import numpy as np
import pandas as pd
import polars as pl
import pytest

import timecorr as tc
from timecorr.helpers import format_data

RNG = np.random.default_rng(0)
VALUES = RNG.standard_normal((20, 4))
VALUES2 = RNG.standard_normal((20, 4))


def _as(kind, v):
    if kind == "numpy":
        return v.copy()
    if kind == "pandas":
        return pd.DataFrame(v.copy(), columns=[f"f{i}" for i in range(v.shape[1])])
    return pl.DataFrame({f"f{i}": v[:, i].copy() for i in range(v.shape[1])})


@pytest.mark.parametrize("kind", ["numpy", "pandas", "polars"])
def test_single_input(kind):
    src = _as(kind, VALUES)
    out = format_data(src)
    assert isinstance(out, list) and len(out) == 1
    assert isinstance(out[0], np.ndarray) and out[0].dtype == np.float64
    assert out[0].flags.writeable
    np.testing.assert_array_equal(out[0], VALUES)
    np.testing.assert_array_equal(
        np.asarray(src if kind == "numpy" else src.to_numpy()), VALUES
    )


@pytest.mark.parametrize("kind", ["numpy", "pandas", "polars"])
def test_list_input(kind):
    out = format_data([_as(kind, VALUES), _as(kind, VALUES2)])
    assert len(out) == 2
    np.testing.assert_array_equal(out[0], VALUES)
    np.testing.assert_array_equal(out[1], VALUES2)
    assert all(o.dtype == np.float64 and o.flags.writeable for o in out)


def test_mixed_list():
    out = format_data(
        [_as("numpy", VALUES), _as("pandas", VALUES2), _as("polars", VALUES)]
    )
    np.testing.assert_array_equal(out[0], VALUES)
    np.testing.assert_array_equal(out[1], VALUES2)
    np.testing.assert_array_equal(out[2], VALUES)


def test_nan_zeroed_without_mutating_input():
    v = VALUES.copy()
    v[3, 1] = np.nan
    df = pd.DataFrame(v)
    arr = v.copy()
    out = format_data([df])[0]
    assert out[3, 1] == 0
    assert np.isnan(df.iloc[3, 1])
    out_arr = format_data(arr)[0]
    assert out_arr[3, 1] == 0
    assert np.isnan(arr[3, 1])
    out_pl = format_data(pl.DataFrame({"a": [1.0, float("nan")]}))[0]
    assert out_pl[1, 0] == 0


def test_does_not_alias_input():
    arr = VALUES.copy()
    out = format_data(arr)[0]
    out[0, 0] = 123.0
    assert arr[0, 0] == VALUES[0, 0]


def test_integer_input_becomes_float64():
    out = format_data(np.arange(12).reshape(6, 2))[0]
    assert out.dtype == np.float64


def test_mismatched_timepoints_raise():
    with pytest.raises(ValueError, match=r"\(20, 4\).*\(15, 4\)"):
        format_data([VALUES, VALUES[:15]])


def test_mismatched_features_raise():
    with pytest.raises(ValueError, match=r"\(20, 4\).*\(20, 3\)"):
        format_data([VALUES, VALUES[:, :3]])


@pytest.mark.parametrize("lib", ["pandas", "polars"])
def test_non_numeric_column_raises(lib):
    d = {"a": VALUES[:, 0], "bad_col": ["x"] * 20}
    df = pd.DataFrame(d) if lib == "pandas" else pl.DataFrame(d)
    with pytest.raises(ValueError, match="bad_col"):
        format_data(df)


@pytest.mark.parametrize("kind", ["pandas", "polars"])
@pytest.mark.parametrize("as_list", [False, True])
def test_timecorr_matches_numpy(kind, as_list):
    wrap = (lambda x: [x]) if as_list else (lambda x: x)
    ref = tc.timecorr(wrap(_as("numpy", VALUES)), cfun=tc.isfc)
    got = tc.timecorr(wrap(_as(kind, VALUES)), cfun=tc.isfc)
    if as_list:
        for r, g in zip(ref, got):
            np.testing.assert_array_equal(r, g)
    else:
        np.testing.assert_array_equal(ref, got)


def test_timecorr_multisubject_matches_numpy():
    ref = tc.timecorr([VALUES, VALUES2], cfun=tc.isfc)
    got = tc.timecorr([_as("polars", VALUES), _as("pandas", VALUES2)], cfun=tc.isfc)
    for r, g in zip(ref, got):
        np.testing.assert_array_equal(r, g)


def _layout_inputs():
    wide = RNG.standard_normal((20, 8))
    return {
        "numpy": VALUES.copy(),
        "numpy-fortran": np.asfortranarray(VALUES),
        "numpy-sliced": wide[:, ::2],
        "pandas": _as("pandas", VALUES),
        "polars": _as("polars", VALUES),
    }


@pytest.mark.parametrize(
    "kind", ["numpy", "numpy-fortran", "numpy-sliced", "pandas", "polars"]
)
def test_output_is_c_contiguous_float64(kind):
    # The memory layout changes floating-point rounding in matrix products,
    # so every input type must reach the estimator in the same layout.
    src = _layout_inputs()[kind]
    out = format_data(src)[0]
    assert out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"]
    assert out.flags["OWNDATA"] and out.flags["WRITEABLE"]
    np.testing.assert_array_equal(out, np.asarray(src))


def test_layouts_give_identical_timecorr():
    ins = _layout_inputs()
    ref = tc.timecorr(np.ascontiguousarray(ins["numpy-sliced"]), cfun=tc.isfc)
    np.testing.assert_array_equal(tc.timecorr(ins["numpy-sliced"], cfun=tc.isfc), ref)
    ref2 = tc.timecorr(VALUES.copy(), cfun=tc.isfc)
    for kind in ["numpy-fortran", "pandas", "polars"]:
        np.testing.assert_array_equal(tc.timecorr(ins[kind], cfun=tc.isfc), ref2)


@pytest.mark.parametrize("kind", ["numpy", "pandas", "polars"])
def test_one_dimensional_input_is_one_feature(kind):
    v = VALUES[:, 0].copy()
    if kind == "numpy":
        src = v
    elif kind == "pandas":
        src = pd.Series(v, name="x")
    else:
        src = pl.Series("x", v)
    out = format_data(src)
    assert len(out) == 1
    assert out[0].shape == (20, 1)
    np.testing.assert_array_equal(out[0][:, 0], v)
    # also inside a list (one subject per item)
    out2 = format_data([src, src])
    assert [o.shape for o in out2] == [(20, 1), (20, 1)]


def test_series_with_nan_is_filled():
    v = VALUES[:, 0].copy()
    v[2] = np.nan
    out = format_data(pd.Series(v))[0]
    assert out.shape == (20, 1) and out[2, 0] == 0
    out = format_data(pl.Series("x", v), fill_nan=False)[0]
    assert out.shape == (20, 1) and np.isnan(out[2, 0])


@pytest.mark.parametrize("shape", [(2, 20, 3), (2, 2, 20, 3)])
def test_three_or_more_dimensions_raise(shape):
    with pytest.raises(ValueError, match=r"dimension.*" + r"\(" + ", ".join(map(str, shape)) + r"\)"):
        format_data(np.zeros(shape))
    with pytest.raises(ValueError, match=r"\(" + ", ".join(map(str, shape)) + r"\)"):
        format_data([np.zeros(shape)])


@pytest.mark.parametrize("value", [np.array(3.0), 3.0])
def test_zero_dimensional_input_raises(value):
    with pytest.raises(ValueError, match="0-dimensional"):
        format_data(value)


def test_timecorr_on_one_dimensional_input():
    out = tc.timecorr(VALUES[:, 0].copy())
    assert out.shape == (20, 1)
    np.testing.assert_array_equal(out, np.ones((20, 1)))
