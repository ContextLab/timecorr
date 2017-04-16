import numpy as np
import scipy.spatial.distance as sd

def wcorr(x, y, w):
    def helper(x, y, w):
        def wmean(x, w):
            return np.sum(np.dot(x, w) / np.sum(w))

        def wstd(x, w):
            m = wmean(x, w)
            return np.sqrt(wmean((x - m) ** 2, w))

        mx = wmean(x, w)
        my = wmean(y, w)
        sx = wstd(x, w)
        sy = wstd(y, w)

        return wmean(np.dot(x - mx, y - my), w) / (sx*sy)

    r = np.zeros([x.shape[1], y.shape[1]])

    cfun = lambda (x, y): helper(x, y, w)
    return sd.cdist(x, y, cfun)

r2z = lambda r: np.dot(0.5, np.log(1 + r) - np.log(1 - r))
z2r = lambda z: (np.exp(2 * z) - 1) / (np.exp(2*z) + 1)

def wcorr_tensor(x, y, w):
    z = np.zeros([x.shape[1], y.shape[1]])
    for t in range(np.min([x.shape[0], y.shape[0]])):
        z += r2z(1 - sd.cdist(x[t, :, :], y[t, :, :], metric='correlation')) * w(t)
    return z / sum(w)

# def timecorr(x, s=1, d=20, aggregate=False, q=None):
#    def levelup(x, f, q, d, weights, aggregate):
#        v = x[0].shape[1]
#        if f == 1: # level 0 --> level 1
#            if aggregate:
#                c = np.zeros([v, v])
#                for s in range(len(x)):
#                    other_inds = np.where(np.arange(len(x) != s))[0]
#                    m = np.mean(np.stack(x[other_inds], axis=3), axis=3)
#                    c += (r2z(wcorr(x[s], m, weights))*q)[:, 1:d]
#            c =
