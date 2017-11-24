import timecorr as tc
import hypertools as hyp
import seaborn as sns
from timecorr.helpers import wisfc

x = hyp.load('weights')

wisfc_across = tc.timecorr(x, mode='across', cfun=wisfc)
wisfc_within = tc.timecorr(x, mode='within', cfun=wisfc)

hyp.plot(wisfc_across)
hyp.plot(wisfc_within)
hyp.plot(x)

