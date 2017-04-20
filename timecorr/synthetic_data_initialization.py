import numpy as np
from math import sqrt, pi, exp

#parameter initialization
filename = "../../data/BDM_DATA/synthetic_data_validation.npz"
time_len = 1000
activations_len = 5
feature_std = 0.1
gaussian_variance = 20

#data matrices
feature_maps = np.zeros([time_len,activations_len,activations_len])
# covariances = np.zeros([time_len,activations_len,activations_len])
# activations = np.zeros([time_len,time_len,activations_len])
final_activation = np.zeros([activations_len,time_len])
gaussian_array = np.array([exp(-timepoint**2/2/gaussian_variance)/sqrt(2*pi*gaussian_variance) for timepoint in range(-time_len+1,time_len)])
covariance, covariance1 = np.zeros([activations_len,activations_len]), np.zeros([activations_len,activations_len])

def initial_activations_generation():
    for timepoint in range(time_len):
        #generate feature map
        if timepoint == 0:
            feature_maps[timepoint] = np.random.normal(0,feature_std,[activations_len,activations_len])
        else:
            feature_maps[timepoint] = feature_maps[timepoint-1]
            # +np.random.normal(0,0.01,[activations_len,activations_len])

        #calculate covariance
        covariances[timepoint] = np.dot(feature_maps[timepoint],feature_maps[timepoint].T)

        #generate activation for timepoint 0
        if timepoint == 0:
            activations[timepoint, 0] = initial_activations  = np.random.multivariate_normal(np.zeros(activations_len), covariances[0])
        else:
            activations[timepoint, 0] = initial_activations

        #generate activation for following timepoints
        for i in range(1, time_len):
            activations[timepoint, i] = np.random.multivariate_normal(activations[timepoint,i-1],covariances[timepoint])

def final_activations_generation():
    for timepoint in range(time_len):
        coefficients = gaussian_array[(time_len-1-timepoint):(2*time_len-1-timepoint)]
        coefficients = coefficients/np.sum(coefficients)
        final_activation[timepoint] = np.dot(coefficients, activations[:,timepoint])

def simplified_activations_generation():
    global covariance, covariance1,final_activation
    feature_map = np.random.normal(0,feature_std,[activations_len,activations_len])
    covariance = np.dot(feature_map,feature_map.T)
    feature_map1 = np.random.normal(0,feature_std,[activations_len,activations_len])
    covariance1 = np.dot(feature_map1,feature_map1.T)
    final_activation[:,0] = initial_activation  = np.random.multivariate_normal(np.zeros(activations_len), covariance)
    for i in range(1, time_len):
        # cov_temp = (1000-i)*covariance/1000.0+i*covariance1/1000.0
        final_activation[:, i] = np.random.multivariate_normal(final_activation[:,(i-1)],covariance)

    # for i in range(1, time_len/2):
    #     final_activation[:, i] = np.random.multivariate_normal(final_activation[:,(i-1)],covariance)
    # for i in range(time_len/2, time_len):
    #     final_activation[:, i] = np.random.multivariate_normal(final_activation[:,(i-1)],covariance1)

if __name__ == "__main__":
    # initial_activations_generation()
    # final_activations_generation()
    # print(covariances)
    # print(final_activation)
    simplified_activations_generation()
    # print(final_activation)
    # print(covariance)
    np.savez(filename, covariance, covariance1, final_activation)
