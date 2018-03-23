
import numpy as np
import pandas as pd
import hypertools as hyp

from timecorr.timecorr import timecorr, levelup
from timecorr.helpers import isfc, gaussian_weights, gaussian_params

gaussian_params = {'var': 1000}
data_list= np.random.randn(10,3)
pandas_dataframe= pd.DataFrame(np.random.randint(low=0, high=10, size=(2, 2)))
numpy_array= np.array([[5, 9], [10, 7]])
numpy_array_list= np.array([[8,2],[4,6]]).tolist()
# if above is how to make a numpy list than TC isn't capible np.lists currently
random_numbers= (2 ,3 ,5, 10, 12, 4, 6)


def test_timecorr():

    data_dl = hyp.tools.format_data(data_list)
    data_pdf = hyp.tools.format_data(pandas_dataframe)
    data_npa = hyp.tools.format_data(numpy_array)
#   data_npl = hyp.tools.format_data(numpy_array_list)
#   data_rand = hyp.tools.format_data(random_numbers)
#   these are now lists
    assert isinstance (data_dl, list)

    Test_dl=  data_dl[0].shape[0]
    Test_pdf=  data_pdf[0].shape[0]
    Test_npa=  data_npa[0].shape[0]
    #Test returns the shape of the weights_function
#   Test_npl=  data_npl[0].shape[0]
#   Test_rand=  data_rn[0].shape[0]

    assert isinstance (Test_pdf, int)

    dl_tester = gaussian_weights(Test_dl, params=gaussian_params)
    pdf_tester = gaussian_weights(Test_pdf, params=gaussian_params)
    npa_tester = gaussian_weights(Test_npa, params=gaussian_params)
#   thrid_tester = gaussian_weights(T3, params=gaussian_params)
#   fourth_tester = gaussian_weights(T4, params=gaussian_params)

    assert isinstance (npa_tester, np.ndarray)
    assert npa_tester.shape == data_npa.shape
    assert dl_tester.shape > data_dl.shape

    
#unsure how to test 'across' mode


def test_levelup ():
    data = hyp.tools.format_data(numpy_array)
    if type(data) == list:
        V = data[0].shape[0]
    else:
        V = data.shape[1]

    corrs = timecorr(data, weights_function= gaussian_weights, weights_params=gaussian_params, mode="within", cfun=isfc)

    assert len(corrs) == len(numpy_array)
