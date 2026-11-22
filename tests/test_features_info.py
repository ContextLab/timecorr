"""
``return_info`` of ``features``: the community labels (``"labels"``) and the constant
output columns (``"constant_columns"``) of each step.

The labels must be those a second run with an explicit ``tc.Communities()`` at the same
orders records in ``fits_`` (how the tutorials obtained them before the key existed), and
the constant columns those of the returned arrays (``np.ptp(x, axis=0) == 0``).
Everything is computed for real on simulated data; nothing is mocked.
"""

import warnings

import numpy as np
import pytest

import timecorr as tc


def quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return f(*args, **kwargs)


def hierarchy(T, S=2, seed=0):
    """72 nodes: 3 top groups x 4 modules x 6 nodes."""
    data, _ = tc.simulate_hierarchy(S=S, T=T, nodes_per_module=6, branching=(4,), n_top=3,
                                    taus=(5, 60), set_random_seed=seed)
    return data


def constant(x):
    """Constant columns of a returned output, as the tutorials computed them."""
    if isinstance(x, list):
        return [constant(a) for a in x]
    return np.flatnonzero(np.ptp(x, axis=0) == 0).tolist()


def is_int_list(value):
    return type(value) is list and all(type(v) is int for v in value)


def entry(info, k, path):
    return next(e for e in info if e["order"] == k and e["path"] == path)


@pytest.mark.parametrize("T", [600, 3000])   # 3000: orders >= 2 are computed at a lower rate
def test_labels_equal_those_of_an_explicit_communities_run(T):
    data = hierarchy(T)
    (X, D), info = quiet(tc.features, data, order=3, cfun=tc.autofc, return_delta=True,
                         return_info=True)
    feeding = [e for e in info if e["path"] == "feeding"]
    community_orders = [e["order"] for e in feeding if e["effective"].startswith("Communities")]
    assert 1 in community_orders
    assert len(community_orders) < 3    # a step fell back: its labels are None
    explicit = {k: tc.Communities() for k in community_orders}
    X_check, D_check = quiet(tc.features, data, order=3, cfun=tc.autofc, return_delta=True,
                             delta_rfun=explicit)
    for out, check in ((X, X_check), (D, D_check)):
        for k in range(4):
            assert all(np.array_equal(a, b) for a, b in zip(out[k], check[k]))
    for e in feeding:
        if e["order"] in explicit:
            labels = explicit[e["order"]].fits_[-1]["labels"]
            assert is_int_list(e["labels"])
            assert e["labels"] == labels.tolist()
            assert len(e["labels"]) == e["input_width"]
            # one output column per community of two or more columns
            assert int(np.sum(np.bincount(e["labels"]) >= 2)) == e["output_width"]
        else:
            assert e["fallback"] is not None and e["labels"] is None
    if T == 3000:
        assert any(e.get("sampling", 1) > 1 for e in info)
    assert all("labels" not in e for e in info if e["path"] == "analysis")


def test_labels_of_explicit_reducers():
    data = hierarchy(600)
    communities = tc.Communities(n=12)
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_delta=True,
                         return_info=True, delta_rfun={1: communities})
    found = entry(info, 1, "feeding")["labels"]
    assert is_int_list(found) and found == communities.fits_[-1]["labels"].tolist()
    assert len(set(found)) == 12 == D[1][0].shape[1]
    # only feeding entries have labels
    assert all(("labels" in e) == (e["path"] == "feeding") for e in info)
    # one object on both paths: the labels of the feeding fit
    both = tc.Communities(n=12)
    _, info = quiet(tc.features, data, order=1, cfun=tc.autofc, return_delta=True,
                    return_info=True, rfun=both, delta_rfun=None,
                    weights_params={"var": 400}, delta_weights_params={"var": 9})
    fits = {f["path"]: f["labels"].tolist() for f in both.fits_}
    assert set(fits) == {"analysis", "feeding"}
    assert entry(info, 1, "feeding")["labels"] == fits["feeding"]
    # a summary that is not Communities, reducers given by name, no reduction
    for delta_rfun in (tc.WeightedStrength(), "weighted_strength", "communities", "PCA", None):
        _, info = quiet(tc.features, data, order=1, cfun=tc.autofc, return_delta=True,
                        return_info=True, rfun="PCA", delta_rfun=delta_rfun)
        assert entry(info, 1, "feeding")["labels"] is None


def test_constant_columns_per_subject_and_for_one_array():
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((120, 6)) for _ in range(2)]
    data[0][:, 2] = 1.5                       # one constant feature in the first subject
    kwargs = dict(order=2, cfun=tc.autofc, rfun=None, delta_rfun="weighted_strength",
                  return_info=True, weights_params={"var": 25}, delta_weights_params={"var": 25})
    (X, D), info = quiet(tc.features, data, return_delta=True, **kwargs)
    assert len(info) == 4
    for e in info:
        out = (X if e["path"] == "analysis" else D)[e["order"]]
        assert e["constant_columns"] == constant(out)
        assert len(e["constant_columns"]) == 2 and all(map(is_int_list, e["constant_columns"]))
    # unreduced correlations (the 6 diagonal entries first, then the pairs): the diagonal
    # and, in the first subject, the pairs of the constant feature (correlation 0)
    diagonal, pairs = [0, 1, 2, 3, 4, 5], [7, 11, 15, 16, 17]
    assert entry(info, 1, "analysis")["constant_columns"] == [diagonal + pairs, diagonal]
    assert entry(info, 1, "feeding")["constant_columns"] == [[2], []]
    assert entry(info, 2, "analysis")["constant_columns"][0] != []
    # one array: one list of columns; the feeding features of order 2 are not computed
    X1, info1 = quiet(tc.features, data[0], **kwargs)
    assert [(e["order"], e["path"]) for e in info1] == [
        (1, "analysis"), (1, "feeding"), (2, "analysis")]
    assert entry(info1, 1, "analysis")["constant_columns"] == constant(X1[1]) == diagonal + pairs
    assert entry(info1, 2, "analysis")["constant_columns"] == constant(X1[2])
    # feeding features that are not returned: the same columns as when they are
    assert entry(info1, 1, "feeding")["constant_columns"] == [2]
    # combined subjects: one array, one list
    Xc, infoc = quiet(tc.features, data, combine=tc.mean_combine, **kwargs)
    assert isinstance(Xc[1], np.ndarray)
    assert entry(infoc, 1, "analysis")["constant_columns"] == constant(Xc[1]) == diagonal
    assert is_int_list(entry(infoc, 1, "analysis")["constant_columns"])


def test_constant_columns_are_those_of_the_returned_rows():
    # T 3000 and automatic kernels: order 2 is computed at every other row or fewer and
    # returned at all timepoints; the constant columns are those of the returned arrays
    data = hierarchy(3000)
    for x in data:
        x[:, 5] = 0.25
    (X, D), info = quiet(tc.features, data, order=2, cfun=tc.autofc, return_delta=True,
                         return_info=True, delta_rfun="weighted_strength")
    assert entry(info, 2, "analysis")["sampling"] > 1
    assert all(len(x) == 3000 for out in (X, D) for o in out for x in o)
    for e in info:
        out = (X if e["path"] == "analysis" else D)[e["order"]]
        assert e["constant_columns"] == constant(out)
    assert entry(info, 1, "feeding")["constant_columns"] == [[5], [5]]
    assert all(cols != [] for cols in entry(info, 2, "analysis")["constant_columns"])
    assert all(cols != [] for cols in entry(info, 2, "feeding")["constant_columns"])


def test_info_without_orders_and_default_keys():
    data = hierarchy(600)
    _, info = quiet(tc.features, data, order=1, cfun=tc.autofc, return_info=True)
    assert all("constant_columns" in e for e in info)
    assert tc.features(data, order=0, return_info=True)[1] == []
