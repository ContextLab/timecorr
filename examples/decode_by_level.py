# -*- coding: utf-8 -*-
"""
=============================
Decode by level
=============================

In this example, we load in some example data, and decode by level of higher order correlation.

NOTE: This example currently has compatibility issues with the timepoint_decoder function.
For a working example, please see the enhanced version in docs/auto_examples/decode_by_level.py

"""
# Code source: Lucy Owen
# License: MIT

# load timecorr and other packages
import timecorr as tc
import hypertools as hyp
import numpy as np

print("Timepoint Decoding Example")
print("="*30)
print("NOTE: This example currently has compatibility issues.")
print("Please see docs/auto_examples/decode_by_level.py for a working version.")
print("="*30)

# load example data
data = hyp.load('weights').get_data()

# Convert to numpy array format required by timepoint_decoder
# timepoint_decoder expects a numpy array with shape (n_subjects, T, K)
data_array = np.array(data)
print(f"Data shape: {data_array.shape} (subjects, timepoints, features)")

# define your weights parameters
width = 10
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}

# set your number of levels
# if integer, returns decoding accuracy, error, and rank for specified level
level = 2

print(f"\nAttempting timepoint decoding at level {level}...")

try:
    # run timecorr with specified functions for calculating correlations, as well as combining and reducing
    results = tc.timepoint_decoder(data_array, level=level, combine=tc.corrmean_combine,
                                   cfun=tc.isfc, rfun='eigenvector_centrality', weights_fun=laplace['weights'],
                                   weights_params=laplace['params'])
    
    # returns only decoding results for level 2
    print("✓ SUCCESS: Level 2 decoding results:")
    print(results)
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    print("This function has compatibility issues with the current version.")

# set your number of levels
# if list or array of integers, returns decoding accuracy, error, and rank for all levels
levels = np.arange(int(level) + 1)

print(f"\nAttempting multi-level decoding for levels {levels}...")

try:
    # run timecorr with specified functions for calculating correlations, as well as combining and reducing
    results = tc.timepoint_decoder(data_array, level=levels, combine=tc.corrmean_combine,
                                   cfun=tc.isfc, rfun='eigenvector_centrality', weights_fun=laplace['weights'],
                                   weights_params=laplace['params'])
    
    # returns decoding results for all levels up to level 2
    print("✓ SUCCESS: Multi-level decoding results:")
    print(results)
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    print("This function has compatibility issues with the current version.")

print("\n" + "="*60)
print("RECOMMENDATION: Use the enhanced version in docs/auto_examples/decode_by_level.py")
print("which uses synthetic data and includes comprehensive error handling.")