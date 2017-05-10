##Packages##
import sys
import numpy as np
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import squareform
from corr_helper import ISFC, correlation_calculation_single

# the data file is expected to be of dimensions [subject number, time length, activations length]
# and converted to dimensions [subject number, activations length, time length]
def timecorr(activations, gaussian_variance, estimation_range=3, mode = "within", coefficients = None):
    # if activations for only one subject is input, then calculate activations correlation for single subject
    if len(activations)==1:
        return correlation_calculation_single(activations[0].T, gaussian_variance, estimation_range)

    # if activations for multiple subjects is input, then two options are available
    else:
        activations = np.swapaxes(np.array(activations),1,2)
        subject_num, activations_len, time_len = activations.shape

        # Calculate correlation for activations within each subject
        if mode=="within":
            subject_num = len(activations)
            activations_len, time_len= activations[0].shape
            result = np.zeros([subject_num, time_len,(activations_len * (activations_len-1) / 2)])
            for subject in range(subject_num):
                result[subject] = correlation_calculation_single(activations[subject], gaussian_variance, estimation_range,coefficients)
            return result

        # Calculate ISFC, average correlation between activations across subjects
        else:
            return ISFC(activations, gaussian_variance, estimation_range,coefficients)


if __name__ == "__main__":
    filename, gaussian_variance, estimation_range, mode = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
    data = np.load(filename)
    activations = data['arr_2']
    true_covariances = data['arr_0']
    true_covariances1 = data['arr_1']
    activations_len, time_len = activations.shape
    covariances_vector = timecorr(np.tile(activations.T,[10,1,1]), int(gaussian_variance), int(estimation_range), mode)
    true_covariances_vector = squareform(true_covariances,checks=False)
    true_covariances_vector1 = squareform(true_covariances1,checks=False)
    Y = np.array([pearsonr(covariances_vector[i,],true_covariances_vector)[0] for i in range(time_len)])
    Y1 = np.array([pearsonr(covariances_vector[i,],true_covariances_vector1)[0] for i in range(time_len)])
    plt.plot(range(time_len),Y,'b-')
    plt.plot(range(time_len),Y1,'r-')
    plt.show()
