
import supereeg as se
import timecorr as tc
import numpy as np
from timecorr.helpers import wcorr, gaussian_weights


locs = se.simulate_locations(n_elecs=10)
bo_s = se.simulate_bo(n_samples=1000, locs=locs, noise=.5)

try_rolling =bo_s.get_data().rolling(window=120).corr(other=bo_s.get_data()[0])

try_rolling_12 =bo_s.get_data().iloc[:,0].rolling(window=120).corr(other=bo_s.get_data().iloc[:,1])

gaussian_params = {'var': 100}

data = bo_s.get_data().values
T = data.shape[0]

weights = gaussian_weights(T, gaussian_params)

col_1 = np.atleast_2d(data[:,0]).T

col_2 = np.atleast_2d(data[:,1]).T

corrs_col_arrays = np.squeeze(wcorr(data, data, weights))

corrs_multidim = wcorr(col_1, data, weights)

corrs_col = np.squeeze(wcorr(col_1, col_1, weights))

corrs_col_neg = np.squeeze(wcorr(-col_1, col_1, weights))

corrs_col_12 = np.squeeze(wcorr(col_2, col_1, weights))

assert(np.allclose(corrs_col,corrs_multidim[0][0]))
assert(np.allclose(corrs_col_12,corrs_multidim[0][1]))

