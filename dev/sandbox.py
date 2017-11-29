import timecorr as tc
import hypertools as hyp
from timecorr.helpers import isfc, wisfc
import seaborn as sns
import numpy as np
from scipy.io import loadmat as load

data = load('/Users/jmanning/Desktop/fMRI/sherlock/sherlock_data.mat')['HTFA'][0]
data = [d for d in data]
x = np.mean(tc.timecorr(data, mode='across', cfun=tc.helpers.wisfc), axis=1)


x = hyp.load('weights_sample')

isfc_across = tc.timecorr(x, mode='across', cfun=isfc)
isfc_within = tc.timecorr(x, mode='within', cfun=isfc)

wisfc_across = tc.timecorr(x, mode='across', cfun=wisfc)
wisfc_within = tc.timecorr(x, mode='within', cfun=wisfc)

print('Sanity check passed: ' + str(np.array(map(lambda x, y: np.isclose(x, y).all(), isfc_within, wisfc_within)).all()))


hyp.plot([isfc_across, wisfc_across])
hyp.plot(x)

sns.heatmap(isfc_across)
sns.heatmap(wisfc_across)

levelup_isfc = tc.levelup(x)
levelup_wisfc = tc.levelup(x, cfun=wisfc)

level2_isfc_across = tc.timecorr(levelup_isfc, mode='across', cfun=isfc)
level2_wisfc_across = tc.timecorr(levelup_wisfc, mode='across', cfun=wisfc)

hyp.plot([isfc_across, wisfc_across, level2_isfc_across, level2_wisfc_across], legend=['isfc', 'wisfc', 'l2 isfc', 'l2 wisfc'])

sns.heatmap(level2_isfc_across)

