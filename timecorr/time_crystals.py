import numpy as np
import pandas as pd
import time

class TimeCrystal(object):

    def __init__(self, time_data=None, covs=None, meta=None, date_created=None):

        from .timecorr import timecorr

        self.time_data = time_data

        if meta:
            self.meta = meta
        else:
            self.meta = {}

        if not date_created:
            self.date_created = time.strftime("%c")
        else:
            self.date_created = date_created

        self.n_subs = np.shape(self.time_data)[0]

        self.covs = covs


    def update_info(self):
        self.n_subs = np.shape(self.time_data)[0] # needs to be calculated by sessions

    def get_time_data(self):

        return self.time_data

    def get_covs(self):

        if not self.covs:
            return []
        else:
            return self.covs

    def info(self):
        """
        Print info about the time crystal

        Prints the number of
        """
        self.update_info()
        print('Number of subjects: ' + str(self.n_subs))
        print('Date created: ' + str(self.date_created))
        print('Meta data: ' + str(self.meta))

    def save(self, fname):

        np.savez(fname, time_data=self.time_data, covs=self.covs, meta=self.meta, date_created=self.date_created)


def load(fname):
    temp_data = np.load(fname, mmap_mode='r')
    return TimeCrystal(time_data = temp_data['time_data'], covs = temp_data['covs'], meta = temp_data['meta'], date_created = temp_data['date_created'])
