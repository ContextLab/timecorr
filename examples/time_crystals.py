import timecorr as tc
import numpy as np
import matplotlib
matplotlib.use("Agg")

## simulate some data
data = tc.generate_multisubject_data()

## create a timecrystal with just timedata
timecrystal = tc.TimeCrystal(data=data)

## timecrystal's data field filled
timecrystal.get_data()

## this could be done inplace if neccessary, which returns a timecrystal with the passed arguments saved in meta
timecrystal.timecorr(mode='within')
#
# ## or this can be done by passing a timecrystal and returning an array
# timecorred = tc.timecorr(timecrystal, mode='within')
#
# ## timecrystal's corrs field can then be populated with timecorred
# timecrystal = tc.TimeCrystal(data=data, corr=timecorred)

## you can also use levelup as method
timecrystal.levelup(mode='within', reduce='IncrementalPCA')

## and updates meta
timecrystal.info()

## save data (saves as npz files, but would be nice to have a different extension)
#timecrystal.save('/filepath/to/temp_folder/temp_tc')

## loading data
#time_load = tc.load('/filepath/to/temp_folder/temp_tc.npz')

## see the loaded data
#time_load.info()