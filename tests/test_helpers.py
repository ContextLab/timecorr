locs = np.array([[-61., -77.,  -3.],
                 [-41., -77., -23.],
                 [-21., -97.,  17.],
                 [-21., -37.,  77.],
                 [-21., -37.,  77.],
                 [ 59., -17.,  17.]])
                 
gaussian_params = {'var': 1000}

### function:
def gaussian_weights(T, params=gaussian_params):
    c1 = np.divide(1, np.sqrt(2 * np.math.pi * params['var']))
    c2 = np.divide(-1, 2 * params['var'])
    sqdiffs = toeplitz(np.arange(T)) ** 2
    return c1 * np.exp(c2 * sqdiffs)


def sort_unique_locs(locs):
    """
    Sorts unique locations

    Parameters
    ----------
    locs : pandas DataFrame or ndarray
        Electrode locations

    Returns
    ----------
    results : ndarray
        Array of unique locations

    """
    if isinstance(locs, pd.DataFrame):
        unique_full_locs = np.vstack(set(map(tuple, locs.as_matrix())))
    elif isinstance(locs, np.ndarray):
        unique_full_locs = np.vstack(set(map(tuple, locs)))
    else:
        print('unknown location type')

    return unique_full_locs[unique_full_locs[:, 0].argsort(),]


### test:
def test_gaussian_weights()

def test_sort_unique_locs():
    sorted = sort_unique_locs(locs)
    assert isinstance(sorted, np.ndarray)
