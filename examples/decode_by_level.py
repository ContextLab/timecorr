import timecorr as tc
import seaborn as sns
import hypertools as hyp
import numpy as np
import pandas as pd

data = hyp.tools.load('weights')

results = []
levels = np.arange(3)
next = data
for i in levels:
    results[i] = tc.timepoint_decoder(next)
    next = tc.levelup(next)
