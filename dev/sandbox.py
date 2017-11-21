import timecorr as tc
import hypertools as hyp
import seaborn as sns
from timecorr.helpers import wisfc

x = hyp.load('weights_sample')

wisfc = tc.timecorr(x, mode='across', cfun=wisfc)

sns.heatmap(wisfc)