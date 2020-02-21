# -*- coding: utf-8 -*-
"""
=============================
Simulate subject data
=============================

In this example, we simulate data

"""
# Code source: Lucy Owen
# License: MIT

# load timecorr
import timecorr as tc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist



def recovery_corrs(truth, guess):

    return np.diag(1 - cdist(truth[:, K:], guess[:, K:], 'correlation'))


# simulate some data

T = 200
K = 20

data0, zeroth = tc.simulate_data(datagen='random', return_corrs=True, set_random_seed=False, S=1, T=T, K=K)

data1, first = tc.simulate_data(datagen='random', return_corrs=True, set_random_seed=False, S=1, T=T, K=K)

data2, second = tc.simulate_data(datagen='random', return_corrs=True, set_random_seed=False, S=1, T=T, K=K)

data3, third = tc.simulate_data(datagen='random', return_corrs=True, set_random_seed=False, S=1, T=T, K=K)

# convert from vector to matrix format
first_order = tc.vec2mat(first)

# convert from vector to matrix format
second_order = tc.vec2mat(second)


Y = np.zeros([T, K])

for t in np.arange(T):

    k = np.kron(second_order[:, :, t], first_order[:, :, t])

    k2 = np.random.multivariate_normal(mean=np.zeros([k.shape[0]]), cov=k).reshape((K, K))
    ks = (k2 + k2.T)/2

    Y[t, :] = np.random.multivariate_normal(mean=np.zeros([ks.shape[0]]), cov=ks)


second_order_vec = tc.timecorr(Y, weights_function=tc.gaussian_weights, weights_params={'var': 5},
                          combine=tc.helpers.corrmean_combine)


second_order_approx = tc.vec2mat(second_order_vec)

corr_corr2 = np.diag(1 - cdist(np.atleast_2d(second[:, K:]), np.atleast_2d(second_order_vec[:, K:]), 'correlation'))

print(corr_corr2.mean())

corr_corr1 = np.diag(1 - cdist(np.atleast_2d(first[:, K:]), np.atleast_2d(second_order_vec[:, K:]), 'correlation'))

print(corr_corr1.mean())

corr_corr0 = np.diag(1 - cdist(np.atleast_2d(zeroth[:, K:]), np.atleast_2d(second_order_vec[:, K:]), 'correlation'))

print(corr_corr0.mean())


conditions = ['zeroth', 'first', 'second']
colors = sns.color_palette("cubehelix", 3)

for e, c in enumerate(conditions):

    plt.plot(np.diag(1 - cdist(np.atleast_2d(eval(c)[:, K:]), np.atleast_2d(second_order_vec[:, K:]), 'correlation'))
, color = colors[e])
    plt.xlabel('time')
    plt.ylabel('correlation of matrices')

f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
legend = plt.legend([f("s", colors[i]) for i in range(3)], conditions, loc=4, framealpha=1, frameon=False, fontsize = 'small')
plt.show()