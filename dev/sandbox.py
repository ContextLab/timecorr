import timecorr as tc
import hypertools as hyp
import seaborn as sns

x = hyp.load('weights_sample')

isfc = tc.timecorr(x, mode='across')

sns.heatmap(isfc)