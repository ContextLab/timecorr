
import supereeg as se
import timecorr as tc
import numpy as np
from timecorr.helpers import wcorr, gaussian_weights


locs = se.simulate_locations(n_elecs=10)
bo_s = se.simulate_bo(n_samples=1000, locs=locs)

try_rolling =bo_s.get_data().rolling(window=120).corr(other=bo_s.get_data()[0])

gaussian_params = {'var': 1000}

data = bo_s.get_data().values
T = data.shape[0]

weights = gaussian_weights(T, gaussian_params)

## negative times itself, neg 1s
## random data

corrs = np.squeeze(wcorr(np.atleast_2d(data[:,0]).T, np.atleast_2d(data[:,0]).T, weights))


corrs = wcorr(np.atleast_2d(data[:,0]).T, data, weights)

# create list of simulated brain objects
model_bos = [se.simulate_model_bos(n_samples=1000, sample_rate=1000,
                                   locs=locs, sample_locs=10) for x in range(3)]


data = np.dstack((model_bos[0].get_data().as_matrix().T, model_bos[1].get_data().as_matrix().T,
                  model_bos[2].get_data().as_matrix().T)).T





mo_corrs = wcorr(data, np.atleast_3d(model_bos[0].get_data().as_matrix().T).T, weights)