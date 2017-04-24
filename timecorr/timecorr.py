import numpy as np
# cimport numpy as np
from scipy.stats.stats import pearsonr
from scipy.io import loadmat
import sys
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from corr_helper import ISFC, correlation_calculation_single

def timecorr(activations, gaussian_variance, estimation_range=3, mode = "within"):
    if len(activations.shape)==2:
        return correlation_calculation_single(activations.T, gaussian_variance, estimation_range)
    else:
        # the data file is expected to be of dimensions [subject number, time length, activations length]
        # and converted to dimensions [subject number, activations length, time length]
        activations = np.array(activations)
        activations = np.swapaxes(activations,1,2)
        subject_num, activations_len, time_len = activations.shape

        if mode=="within":
            subject_num = len(activations)
            activations_len, time_len= activations[0].shape
            result = np.zeros([subject_num, time_len,(activations_len * (activations_len-1) / 2)])
            for subject in range(subject_num):
                result[subject] = correlation_calculation_single(activations[subject], gaussian_variance, estimation_range)
            return result
        else:
            return ISFC(activations, gaussian_variance, estimation_range)




if __name__ == "__main__":
    filename, gaussian_variance, estimation_range, mode = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
    data = np.load(filename)
    activations = data['arr_2']
    true_covariances = data['arr_0']
    true_covariances1 = data['arr_1']
    activations_len, time_len = activations.shape
    covariances_vector = timecorr(activations.T, int(gaussian_variance), int(estimation_range), mode)
    true_covariances_vector = squareform(true_covariances,checks=False)
    true_covariances_vector1 = squareform(true_covariances1,checks=False)
    Y = np.array([pearsonr(covariances_vector[i,],true_covariances_vector)[0] for i in range(time_len)])
    Y1 = np.array([pearsonr(covariances_vector[i,],true_covariances_vector1)[0] for i in range(time_len)])
    plt.plot(range(time_len),Y,'b-')
    plt.plot(range(time_len),Y1,'r-')
    plt.show()
