# -*- coding: utf-8 -*-
"""
=====================================
Recover structure at a chosen order
=====================================

``simulate_data(order=n)`` embeds a known 4 x 4 dynamic correlation matrix (the
template) at exactly order ``n``: the features form 4 blocks of ``2**(n-1)``
columns, and only the product of all of a block's columns carries the template.
Products of fewer columns have mean zero, so orders below ``n`` hold no
structure.

For each embedded order ``n`` from 1 to 4, this example computes features up to
order ``n`` with ``tc.features`` and, at every analyzed order ``k <= n``, scores
the estimated 4 x 4 correlation matrix among the blocks' (first) carriers
against the template: the Pearson correlation between estimated and true
off-diagonal entries, pooled over entries and timepoints. ``carrier_reducer``
keeps, at each order, only the products that carry the simulated structure,
so the cost stays small at every order.

The recovery is high at the embedded order and near zero below it.

"""
# Code source: Contextual Dynamics Laboratory
# License: MIT

import warnings

import numpy as np
from matplotlib import pyplot as plt

import timecorr as tc

T = 1000  # timepoints
p = 4  # template size (number of blocks)
seeds = [0, 1, 2, 3, 4]
orders = [1, 2, 3, 4]
kernel = {
    "weights_function": tc.gaussian_weights,
    "weights_params": {"var": 625},  # standard deviation of 25 timepoints
    "cfun": tc.autofc,
}


def first_carriers(blocks, k, n):
    """Columns of D_{k-1} (carrier_reducer layout) holding each block's first carrier."""
    if k == 1:
        return np.asarray(blocks)[:, 0]
    per_block = 2 ** (n - 1) // 2 ** (k - 1)
    return np.arange(p) * per_block


def recovery(X, cols, corrs):
    """Pooled r between estimated and true off-diagonal entries of the p x p matrix."""
    est = tc.vec2mat(X)[np.ix_(cols, cols, np.arange(T))]
    est = tc.mat2vec(est)[:, p:]  # mat2vec: p diagonal entries first
    return np.corrcoef(est.ravel(), corrs[:, p:].ravel())[0, 1]


# r[n][k - 1]: one value per seed
r = {n: np.full((n, len(seeds)), np.nan) for n in orders}
for s, seed in enumerate(seeds):
    for n in orders:
        with warnings.catch_warnings():
            # simulate_data warns that variance-based reduction (PCA) may not
            # retain structure that exists only at order >= 3; this example uses
            # carrier_reducer instead
            warnings.filterwarnings("ignore", message="The simulated structure")
            data, truth = tc.simulate_data(
                datagen="block", K=p * 2 ** (n - 1), T=T, B=4, order=n,
                return_corrs=True, set_random_seed=seed,
            )
        if n == 1:
            corrs, blocks = truth, np.arange(p)[:, np.newaxis]
            X = tc.features(data, order=1, rfun=None, **kernel)
        else:
            corrs, blocks = truth["corrs"], truth["blocks"]
            X = tc.features(
                data, order=n, rfun=None,
                delta_rfun=tc.carrier_reducer(blocks), **kernel,
            )
        for k in range(1, n + 1):
            r[n][k - 1, s] = recovery(X[k], first_carriers(blocks, k, n), corrs)

for n in orders:
    print(f"embedded order {n}: mean r by analyzed order", np.round(r[n].mean(axis=1), 2))

# grouped bars: x = analyzed order, one bar per embedded order
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
width = 0.2
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, n in enumerate(orders):
    ks = np.arange(1, n + 1)
    x = ks + (i - 1.5) * width
    means = r[n].mean(axis=1)
    ax.bar(x, means, width * 0.9, color=colors[i], label=f"embedded at order {n}")
    for xi, row in zip(x, r[n]):
        ax.plot(np.full(len(seeds), xi), row, "o", ms=3, color="0.25", zorder=3)
    for xi, m, row in zip(x, means, r[n]):
        ax.annotate(
            f"{m:.2f}", (xi, max(row.max(), 0) + 0.04), ha="center", va="bottom",
            rotation=90, fontsize=8, bbox=dict(fc="white", ec="none", pad=0.5),
        )
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(orders)
ax.set_xlabel("Analyzed order k")
ax.set_ylabel("Recovery r (estimate vs. template)")
ax.set_ylim(-0.4, 1.25)
ax.set_title("Simulated structure is recovered only at its embedded order")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=9)
ax.set_yticks(np.arange(-0.4, 1.01, 0.2))
fig.tight_layout()
plt.show()
