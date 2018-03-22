from past.utils import old_div
import numpy as np
import pandas as pd
import hypertools as hyp
import scipy.spatial.distance as sd
from scipy.linalg import toeplitz

from timecorr.timecorr import timecorr, levelup
from timecorr.helpers import isfc, gaussian_weights, gaussian_params

gaussian_params = {'var': 1000}
data_list= np.random.randn(10,3)
pandas_dataframe= pd.DataFrame(np.random.randint(low=0, high=10, size=(2, 2)))
numpy_array= np.array([[5, 9], [10, 7]])
numpy_array_list= np.array([[8,2],[4,6]]).tolist()
random_numbers= (2 ,3 ,5, 10, 12, 4, 6)


def test_timecorr():
    data_df = hyp.tools.format_data(pandas_dataframe)
    data_npa = hyp.tools.format_data(numpy_array)
    data_npl = hyp.tools.format_data(numpy_array_list)
    data_rn = hyp.tools.format_data(random_numbers)
    assert isinstance(data_df, np.array)
    assert isinstance(data_npa, np.array)
    assert isinstance(data_npl, np.array)
    assert isinstance(data_rn, np.array)

    T1=  data_df[0].shape[0]
    T2=  data_npa[0].shape[0]
    T3=  data_npl[0].shape[0]
    T4=  data_rn[0].shape[0]

    first_tester = gaussian_weights(T1, params=gaussian_params)
    second_tester = gaussian_weights(T2, params=gaussian_params)
    thrid_tester = gaussian_weights(T3, params=gaussian_params)
    fourth_tester = gaussian_weights(T4, params=gaussian_params)

    if (mode == 'across') or (type(data) != list) or (len(data) == 1):
        return cfun(data, weights)

    assert isinstance (first_tester, )
    assert isinstance ()

   # test_weights_functions=
    assert isinstance(first_tester, np.array)

def test_levelup ():
    data = hyp.tools.format_data(data_list)
    if type(data) == list:
        V = data[0].shape[1]
    else:
        V = data.shape[1]

    corrs = timecorr(data, weights_function= gaussian_weights, weights_params=gaussian_params, mode="within", cfun=isfc)
   # return hyp.reduce(corrs, reduce=reduce, ndims=V)
    assert isinstance(corrs.shape == data.shape)
