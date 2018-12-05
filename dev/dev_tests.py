import numpy as np
from scipy.linalg import toeplitz
from scipy.io import loadmat
import os
import timecorr as tc

#using method from supereeg
from timecorr.helpers import gaussian_weights, gaussian_params, wcorr, wisfc, mat2vec, vec2mat, isfc, mean_combine, \
    corrmean_combine

T = 10
D = 4
S = 5

n_elecs = 20

n_samples = 100

R = toeplitz(np.linspace(0, 1, n_elecs)[::-1])

data_sim = np.random.multivariate_normal(np.zeros(n_elecs), R, size=n_samples)

width = 10
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}
try_data = []
repdata = 4
for i in range(repdata):
    try_data.append(data_sim)

try_data = np.array(try_data)
## check if level is integer

# results_level_int = tc.timepoint_decoder(try_data, level=np.array([0,1]), combine= corrmean_combine, cfun=isfc,
#                                    rfun='eigenvector_centrality', weights_params=laplace['params'])
#
# results_level_array = tc.timepoint_decoder(try_data, level=np.array([0,1]), combine= corrmean_combine, cfun=isfc,
#                                    rfun='eigenvector_centrality', weights_params=laplace['params'])
#
# results_level_list = tc.timepoint_decoder(try_data, level=[0,1], combine= corrmean_combine, cfun=isfc,
#                                    rfun='eigenvector_centrality', weights_params=laplace['params'])
#
#
# results_level_wrong_list = tc.timepoint_decoder(try_data, level=[1,2], combine= corrmean_combine, cfun=isfc,
#                                    rfun='eigenvector_centrality', weights_params=laplace['params'])
#
# results_level_wrong_array = tc.timepoint_decoder(try_data, level=np.array([1,2]), combine= corrmean_combine, cfun=isfc,
#                                    rfun='eigenvector_centrality', weights_params=laplace['params'])

results_combine_int = tc.timepoint_decoder(try_data, level=1, combine=corrmean_combine, cfun=isfc,
                                   rfun='eigenvector_centrality', weights_params=laplace['params'])

results_combine_list= tc.timepoint_decoder(try_data, level=np.array([0,1]), combine=[mean_combine, corrmean_combine], cfun=isfc,
                                   rfun='eigenvector_centrality', weights_params=laplace['params'])

results_combine_array = tc.timepoint_decoder(try_data, level=[0,1], combine= np.array([mean_combine, corrmean_combine]), cfun=isfc,
                                   rfun='eigenvector_centrality', weights_params=laplace['params'])


results_level_wrong_list = tc.timepoint_decoder(try_data, level=[1,2], combine= corrmean_combine, cfun=isfc,
                                   rfun='eigenvector_centrality', weights_params=laplace['params'])

results_level_wrong_array = tc.timepoint_decoder(try_data, level=np.array([1,2]), combine= corrmean_combine, cfun=isfc,
                                   rfun='eigenvector_centrality', weights_params=laplace['params'])
