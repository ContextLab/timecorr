import hypertools as hyp
import numpy as np
from os import listdir
from os.path import isfile, join
from loadnii import loadnii as ln
from timecorr import levelup, decode, decode_raw_data
from sklearn.decomposition import IncrementalPCA
np.seterr(all='ignore')

def load_fmri_data(directory, nvoxels):
    '''
    Read in fRMI data from nii files in directory, reduce voxel dimensions and then store as a list of numpy arrays

    Input:
        directory: the path of the directory in which the nii files are stored

        nvoxels: the number of voxels to reduce the fRMI data to

    Return:
        A list of numpy matrices containing the reduced voxels activations
    '''
    directory = directory+"/"
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    data = []
    ipca = IncrementalPCA(n_components=nvoxels, batch_size=nvoxels)
    for fname in onlyfiles:
        if '.nii' in fname:
            temp = ln(directory+fname)
            temp = ipca.fit_transform(temp.Y)
        elif '.npy' in fname:
            temp = np.load(directory+fname)
            temp = ipca.fit_transform(temp)
        else:
            continue

        data.append(temp)
    # np.save(directory+"/raw_data", np.array(data))
    return np.array(data)

def leveling(directory,activations, nlevels,noise=0):
    '''
    Level up fRMI activations to the specified number of levels using levelup function from timecorr and store all level activations in a temporary file

    Input:
        activations: list of numpy arrays
            containing a list of subject fRMI data as process by load_fmri_data function

        nlevels: int
            number of levels to level up

    Return:
        None
    '''
    # activations = np.load(directory+"/raw_data.npy")
    subject_num, time_len, voxel_num = activations.shape
    all_activations = np.zeros([(nlevels+1),subject_num,time_len,voxel_num])
    all_activations[0] = activations + np.random.normal(0,noise,activations.shape)
    for l in range(nlevels):
        all_activations[(l+1)] = np.array(levelup(all_activations[(l)],mode="within"))
    np.save("./all_level_activations", all_activations)

def decoding_analysis(directory, nvoxels, nlevels, var=None, nfolds=3, noise=0):
    '''
    Performs decoding analysis

    Input:
        directory: string
            path to the directory with the nii files

        nvoxels: int
            the number of voxels to reduce the dataset to

        nlevels: int
            the number of levels to levelup and perform decoding analysis

        var: int, defaults to None
            The variance to use for timecorr

        nfolds: int, defaults to 3
            The number of repetitions to reshuffle in decoding analysis

    Return:
        A numpy array with the decoding accuracy at each level
    '''
    activations = load_fmri_data(directory, nvoxels)
    leveling(directory, activations, nlevels,noise=noise)
    all_activations = np.load("./all_level_activations.npy")
    decoding_accuracy = np.zeros(nlevels+1)
    decoding_accuracy[0] = decode_raw_data(all_activations[0],nfolds=nfolds)
    for l in range(nlevels):
        print(np.sum(np.absolute(all_activations[l])))
        decoding_accuracy[l+1]=decode(all_activations[l],nfolds=nfolds)
    print decoding_accuracy
