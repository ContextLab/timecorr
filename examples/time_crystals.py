import timecorr as tc
import numpy as np

## simulate some data
data = tc.simulate_timecorr_data()

## create a timecrystal with just timedata
timecrystal = tc.TimeCrystal(time_data=data)

## timecrystal's covs field empty
timecrystal.get_covs()

## run timecrystal through timecorr
## this could be done inplace if neccessary
timecorred = tc.timecorr(timecrystal, mode='within')

## timecrystal's covs field populated after timecorr
timecorred.get_covs()

## level up the timecorred data
## anything else we should put in for leveled up data?
## like keep track of the number of levels up??
leveled = tc.levelup(timecorred)

## check out the data
leveled.get_time_data()

## check out the covs
leveled.get_covs()

## save data (saves as npz files, but would be nice to have a different extension)
timecorred.save('/Users/lucyowen/Desktop/temp_folder/temp_tc')

## loading data
time_load = tc.load('/Users/lucyowen/Desktop/temp_folder/temp_tc.npz')

## see the loaded data
time_load.info()