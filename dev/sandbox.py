import timecorr as tc
import hypertools as hyp
from timecorr.helpers import isfc, wisfc
import seaborn as sns
import numpy as np

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


