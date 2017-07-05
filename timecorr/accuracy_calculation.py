import numpy as np
from os import listdir
from os.path import join, isfile
import sys

def decoding_accuracy(directory,nlevels=11):
    directory = directory+"/results/"
    onlyfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and "decoding_accuracy" in f)]
    print(len(onlyfiles))
    accuracy = np.zeros([len(onlyfiles),nlevels])
    for index,fname in enumerate(onlyfiles):
        accuracy[index]=np.load(join(directory,fname))
    mean = np.mean(accuracy,axis=0)
    std = np.std(accuracy,axis=0)
    print(mean, std)
    np.savez(directory+"/intra_level_decoding_analysis",mean, std)

def mixture_level_accuracy(directory, nlevels = 11):
    directory = directory + "/results/"
    onlyfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and "optimal_weights_and_accuracy_" in f)]
    weights = np.zeros([nlevels, len(onlyfiles)])
    accuracy = np.zeros([nlevels+1,len(onlyfiles)])
    for index,fname in enumerate(onlyfiles):
        data = np.load(join(directory,fname))
        weights[:,index] = data["arr_0"]
        accuracy[0,index] = data["arr_1"]

    raw_activation = np.load(directory+"/results/all_level_activations.npy")[0]
    index=0
    for repetition_index in range(100):
        isfc = np.load(directory+"/results/isfc_"+str(repetition_index)+".npy")[:,:10]
        group_assignments = np.load(directory+"/results/group_assignment_"+str(repetition_index)+".npy")
        if not isfile(isfc):
            continue
        group_size = int(len(group_assignments)/4)
        group_assignments = [[group_assignments[0:group_size]],[group_assignments[group_size:2*(group_size)]],[group_assignments[2*group_size:3*(group_size)]], [group_assignments[3*(group_size):]]]
        nlevels, ntimepoints = isfc.shape[1]+1, isfc.shape[2]
        include_inds = np.arange(ntimepoints)
        correlations= np.zeros([nlevels, ntimepoints, ntimepoints])
        correlations[0] = 1 - sd.cdist(np.mean(raw_activation[group_assignments[2]],axis=0), np.mean(raw_activation[group_assignments[3]],axis=0), 'correlation')
        for i in range(0,nlevels-1):
            correlations[i+1] = 1 - sd.cdist(isfc[2,i], isfc[3,i], 'correlation')
        for i in range(1,nlevels+1)
            for t in range(0, ntimepoints):
                decoded_inds = include_inds[np.where(correlations[i-1,t, include_inds] == np.max(correlations[i-1, t, include_inds]))]
                accuracy[i,index] += np.mean(decoded_inds == np.array(t))
            accuracy[i,index]/=float(ntimepoints)
        print(accuracy[:,index])
        index+=1
    print(np.mean(weights,1),np.mean(accuracy,1))
    np.savez(directory+"/mixture_weights_for_plot",weights, accuracy)
    print("saved to "+directory+"/mixture_weights_for_plot")

df = pd.DataFrame(weights,columns = range(nlevels))
newplot = sns.violinplot(data = df,orient = 'v')
plt.show()
if __name__== '__main__':
    directory, nlevels = sys.argv[1], sys.argv[2]
    decoding_accuracy(directory, int(nlevels))
