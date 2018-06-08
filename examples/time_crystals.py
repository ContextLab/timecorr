import timecorr as tc

data = tc.simulate_timecorr_data()

timecrystal = tc.TimeCrystal(time_data=data)

timecorred = tc.timecorr(timecrystal)

timecorred.save('/Users/lucyowen/Desktop/temp_folder/temp_tc')

time = tc.load('/Users/lucyowen/Desktop/temp_folder/temp_tc.npz')