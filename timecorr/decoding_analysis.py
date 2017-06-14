import numpy as np
import sys
from random import shuffle
from os import listdir,getcwd
from os.path import isfile, join
from timecorr import levelup, decode, decode_raw_data, timecorr, decode_pair
from sklearn import decomposition
from scipy.io import loadmat
from scipy import optimize
np.seterr(all='ignore')

def load_and_levelup(directory, nvoxels, nlevels):
    '''
    Read in fRMI data from nii files in directory, reduce voxel dimensions and then level up fRMI activations to the specified number of levels using levelup function from timecorr and store all level activations in a temporary file

    Input:
        directory: the path of the directory in which the nii files are stored

        nvoxels: the number of voxels to reduce the fRMI data to

    Return:
        A list of numpy matrices containing the reduced voxels activations
    '''
    directory = directory+"/"
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    data = []
    ipca = decomposition.IncrementalPCA(n_components=nvoxels, batch_size=nvoxels)
    for fname in onlyfiles:
        if '.npy' in fname:
            temp = np.load(join(directory, fname)).T
        elif '.mat' in fname:
            temp = loadmat(join(directory,fname))['Y']
        else:
            continue
        data.append(temp)

    subject_num, time_len = len(data), data[0].shape[0]
    stacked_data = np.concatenate(data,0)
    pca_reduced = ipca.fit_transform(stacked_data[:,~np.isnan(stacked_data).any(axis=0)])
    all_activations = np.zeros([(nlevels+1),subject_num,time_len,nvoxels])
    for i in range(subject_num):
        all_activations[0,i] = pca_reduced[i*time_len:(i+1)*time_len]
    for l in range(nlevels):
        all_activations[(l+1)] = np.array(levelup(all_activations[(l)],mode="within"))
    np.save(directory+"/results/all_level_activations", all_activations)
    print("saved to "+directory+"/results/all_level_activations")

def decoding_analysis(directory, nlevels, repetition_index, nfolds=1, noise=0):
    '''
    Performs decoding analysis

    Input:
        directory: string
            path to the directory with the nii files

        nlevels: int
            the number of levels to levelup and perform decoding analysis

        repetition_index: int
            the index of the repetition to help record data

        nfolds: int, defaults to 3
            The number of repetitions to reshuffle in decoding analysis

    Return:
        A numpy array with the decoding accuracy at each level
    '''
    activations_file=directory+"/results/all_level_activations.npy"
    accuracy_file=directory+"/results/decoding_accuracy_"+str(repetition_index)
    all_activations = np.load(activations_file)
    decoding_accuracy = np.zeros(nlevels+1)
    decoding_accuracy[0] = decode_raw_data(all_activations[0],nfolds=nfolds)
    for l in range(int(nlevels)):
        decoding_accuracy[l+1]=decode(all_activations[l],nfolds=nfolds)
    accuracy = np.save(accuracy_file, decoding_accuracy)
    print("saved to "+accuracy_file)

def divide_and_timecorr(directory, repetition_index):
    '''
    Randomly divides the subjects into 4 groups and calculate the isfc at each level for each group

    Input:
        director: string
            path to the directory containing activations at all levels

        repetition_index: string
            the index of the repetition to help record data in the

    '''
    activations = np.load(directory+"/results/all_level_activations.npy")
    nlevels, nsubjects, ntimepoints, nvoxels = activations.shape
    group_size, subjects = int(nsubjects/4), range(nsubjects)
    shuffle(subjects)
    groups = [subjects[0:group_size],subjects[group_size:2*(group_size)],subjects[2*group_size:3*(group_size)], subjects[3*(group_size):]]
    np.save(directory+"/results/group_assignment_"+str(repetition_index), subjects)
    isfc = np.zeros([4,nlevels, ntimepoints,(nvoxels**2-nvoxels)/2])
    for level in range(nlevels):
        for group in range(4):
            isfc[group, level] = timecorr(activations[level, groups[group]],mode = "across")
    np.save(directory+"/results/isfc_"+str(repetition_index), isfc)

def optimal_level_weights(corr1,corr2):
    '''
    Find the weights array for each level that returns the highest decoding accuracy

    Input:
        corr1: numpy array
            containing the "across" timecorr correlation for group 1

        corr2: numpy array
            containing the "across" timecorr correlation for group 2

    Returns optimal weights array
    '''
    corr1 = 0.5*(np.log(1e-5+1+corr1) - np.log(1e-5+1-corr1))
    corr2 = 0.5*(np.log(1e-5+1+corr2) - np.log(1e-5+1-corr2))
    nlevels = len(corr1)
    def weighted_decoding_analysis(w):
        '''
        Find the decoding accuracy of the weighted sum of correlations at each level

        Input:
            w: numpy array
                contains the weights for each level

        Returns decoding accuracy between the two groups after summing each level by the corresponding weights
        '''
        weighted1 = np.sum(map(lambda x: corr1[x]*w[x],range(nlevels)),axis=0)
        weighted2 = np.sum(map(lambda x: corr2[x]*w[x],range(nlevels)),axis=0)
        weighted1 =  (np.exp(2*weighted1) - 1)/(np.exp(2*weighted1) + 1)
        weighted2 =  (np.exp(2*weighted2) - 1)/(np.exp(2*weighted2) + 1)
        a = -1*decode_pair(weighted1,weighted2)
        return a

    def constraint1(x):
        return 1-np.max(x)
    def constraint2(x):
        return np.min(x)

    w = np.absolute(np.random.normal(0,1,nlevels))
    w = w/np.sum(w)
    weights = optimize.minimize(weighted_decoding_analysis, w, method="COBYLA", constraints = ({'type': 'ineq', 'fun': constraint1},{'type': 'ineq', 'fun': constraint2}),tol=1e-5)["x"]
    return np.absolute(weights)/np.sum(np.absolute(weights))

def optimal_decoding_accuracy(directory, repetition_index):
    '''
    Given random division of subjects into groups A1, A2, B1 and B2. Calculate optimal level weights using A1 and A2, then find the new decoding accuracy using optimal weights on B1 and B2

    Input:
        directory: string
            Path to the directory containing the activations dataset

        repetition_index: str
            specific repetition to access

    Returns None
    '''
    activations = np.load(directory+"/results/isfc_"+str(repetition_index)+".npy")
    nlevels = activations.shape[1]

    activations = 0.5*(np.log(1e-5+1+activations) - np.log(1e-5+1-activations))

    weights = optimal_level_weights(activations[0],activations[1])

    weighted1 = np.sum(map(lambda x: activations[2,x]*weights[x],range(nlevels)),axis=0)
    weighted2 = np.sum(map(lambda x: activations[3,x]*weights[x],range(nlevels)),axis=0)
    weighted1 =  (np.exp(2*weighted1) - 1)/(np.exp(2*weighted1) + 1)
    weighted2 =  (np.exp(2*weighted2) - 1)/(np.exp(2*weighted2) + 1)

    accuracy = decode_pair(weighted1,weighted2)
    out_file=directory+"/results/optimal_weights_and_accuracy_"+repetition_index
    np.savez(out_file,weights,accuracy)
    print(weights,accuracy)
    print("saved to "+out_file)

if __name__== '__main__':
#    load_and_levelup(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
 #   decoding_analysis(sys.argv[1],int(sys.argv[2]),sys.argv[3])
#    divide_and_timecorr(sys.argv[1],sys.argv[2])
    optimal_decoding_accuracy(sys.argv[1],sys.argv[2])
