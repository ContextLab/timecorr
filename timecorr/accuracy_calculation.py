import numpy as np
from os import listdir
from os.path import join, isfile
import sys

def circular_decoding_summary(directory):
    directory = directory+"/results/"
    onlyfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and "circle_data_" in f)]
    print(len(onlyfiles))
    config_num,time_len=np.load(join(directory,onlyfiles[0]))['arr_0'].shape
    tc_diag,sw_diag = np.zeros([len(onlyfiles),config_num,time_len]), np.zeros([len(onlyfiles),config_num,time_len])
    timecorr_accuracy,sliding_accuracy = np.zeros([len(onlyfiles),config_num]),np.zeros([len(onlyfiles),config_num])
    for index,fname in enumerate(onlyfiles):
        data=np.load(join(directory,fname))
        tc_diag[index],sw_diag[index],timecorr_accuracy[index],sliding_accuracy[index]=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
        print(timecorr_accuracy[index],sliding_accuracy[index])
    tc_mean, sw_mean = np.mean(tc_diag,0),np.mean(sw_diag,0)
    tc_std, sw_std = np.std(tc_diag,0)/10, np.std(sw_diag,0)/10
    timecorr_accuracy,sliding_accuracy = np.mean(timecorr_accuracy,0),np.mean(sliding_accuracy,0)
    np.savez(directory+"/circular_decoding_analysis",tc_mean, sw_mean,tc_std,sw_std,timecorr_accuracy,sliding_accuracy)
    print("Saved to "+directory+"/circular_decoding_analysis"2)

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
