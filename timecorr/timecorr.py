import numpy as np
from _shared.helpers import isfc, wcorr


def timecorr(x, var=100, wlen=3, mode="within"):
    if (not type(x) == list) and (len(x.shape)==2):
        return wcorr(x.T, var, wlen)
    else:
        # the data file is expected to be of dimensions [subject number, time length, activations length]
        # and converted to dimensions [subject number, activations length, time length]
        x = np.array(x)
        x = np.swapaxes(x, 1, 2)

        if mode=="within":
            S = len(x)
            V, T = x[0].shape
            result = np.zeros([S, T, (V * (V - 1) / 2)])
            for i in range(S):
                result[i] = wcorr(x[i], var, wlen)
            return result
        elif mode=="across":
            return isfc(x, var, wlen)
        else:
            raise NameError('Mode unknown or not supported: ' + mode)