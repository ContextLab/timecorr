import timecorr as tc
import hypertools as hyp

x = hyp.load('weights_sample')

isfc = tc.timecorr(x, mode='across')
