
Use timecorr functions
----------------------

.. code:: ipython3

    import numpy as np
    import timecorr as tc
    import seaborn as sns

.. code:: ipython3

    sim_data = tc.simulate_data(T=1000, K=300, set_random_seed=100)

.. code:: ipython3

    tc_data = tc.timecorr(sim_data, weights_function=tc.gaussian_weights, weights_params={'var': 5})

