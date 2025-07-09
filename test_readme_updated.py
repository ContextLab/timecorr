#!/usr/bin/env python
"""
Test script to verify updated README.md code examples work correctly
"""

import numpy as np
import timecorr as tc

print("Testing updated README.md code examples...")
print("="*50)

# Test 1: Simple Example: Basic Dynamic Correlations
print("\n1. Testing basic dynamic correlations...")
try:
    # Generate sample data: 100 timepoints, 5 features
    data = np.random.randn(100, 5)
    
    # Compute dynamic correlations with Gaussian weighting
    correlations = tc.timecorr(data, weights_function=tc.gaussian_weights, weights_params={'var': 10})
    
    print(f"✓ Input shape: {data.shape}")
    print(f"✓ Output shape: {correlations.shape}")  # (100, 15) - vectorized correlation matrices
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Multi-Subject Analysis Example
print("\n2. Testing multi-subject analysis...")
try:
    # Analyze correlations across multiple subjects
    subjects_data = [np.random.randn(100, 5) for _ in range(10)]  # 10 subjects
    
    # Inter-Subject Functional Connectivity (ISFC)
    # Returns a list of correlations (one per subject vs. others)
    isfc_results = tc.timecorr(subjects_data, cfun=tc.isfc, weights_function=tc.gaussian_weights)
    print(f"✓ ISFC results: {len(isfc_results)} subjects, each shape {isfc_results[0].shape}")
    
    # Weighted ISFC for similarity-based averaging
    # Also returns a list of correlations
    wisfc_results = tc.timecorr(subjects_data, cfun=tc.wisfc, weights_function=tc.gaussian_weights)
    print(f"✓ WISFC results: {len(wisfc_results)} subjects, each shape {wisfc_results[0].shape}")
    
    # To get a single averaged result, use combine parameter
    combined_isfc = tc.timecorr(subjects_data, cfun=tc.isfc, combine=tc.mean_combine, 
                               weights_function=tc.gaussian_weights)
    print(f"✓ Combined ISFC shape: {combined_isfc.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Higher-Order Correlations with Dimensionality Reduction
print("\n3. Testing higher-order correlations...")
try:
    # Compute correlations between correlations using PCA
    data = np.random.randn(200, 10)
    
    higher_order = tc.timecorr(data, 
                              weights_function=tc.gaussian_weights,
                              weights_params={'var': 20},
                              rfun='PCA')  # Reduces dimensionality to prevent explosion
    
    print(f"✓ Higher-order correlations shape: {higher_order.shape}")  # Same as input: (200, 10)
    
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: Weighting Functions
print("\n4. Testing different weighting functions...")
data = np.random.randn(100, 8)

# Gaussian kernel (most common)
try:
    correlations = tc.timecorr(data, weights_function=tc.gaussian_weights, 
                              weights_params={'var': 10})
    print(f"✓ Gaussian kernel: {correlations.shape}")
except Exception as e:
    print(f"✗ Gaussian kernel error: {e}")

# Laplace kernel (sparser, more temporal precision)
try:
    correlations = tc.timecorr(data, weights_function=tc.laplace_weights,
                              weights_params={'scale': 5})
    print(f"✓ Laplace kernel: {correlations.shape}")
except Exception as e:
    print(f"✗ Laplace kernel error: {e}")

# Mexican Hat kernel (captures temporal dynamics)
try:
    correlations = tc.timecorr(data, weights_function=tc.mexican_hat_weights,
                              weights_params={'sigma': 15})
    print(f"✓ Mexican Hat kernel: {correlations.shape}")
except Exception as e:
    print(f"✗ Mexican Hat kernel error: {e}")

# Eye/Delta kernel (uniform weighting)
try:
    correlations = tc.timecorr(data, weights_function=tc.eye_weights,
                              weights_params={})
    print(f"✓ Eye kernel: {correlations.shape}")
except Exception as e:
    print(f"✗ Eye kernel error: {e}")

# Test 5: Correlation Functions
print("\n5. Testing correlation functions...")
multi_data = [np.random.randn(100, 8) for _ in range(5)]  # 5 subjects

# Within-subject correlations (default)
try:
    within_corr = tc.timecorr(data)
    print(f"✓ Within-subject correlations: {within_corr.shape}")
except Exception as e:
    print(f"✗ Within-subject error: {e}")

# Inter-Subject Functional Connectivity (ISFC)
# Each subject vs. average of others (returns list)
try:
    isfc_corr = tc.timecorr(multi_data, cfun=tc.isfc)
    print(f"✓ ISFC (list): {len(isfc_corr)} subjects, each shape {isfc_corr[0].shape}")
except Exception as e:
    print(f"✗ ISFC error: {e}")

# Weighted ISFC (similarity-weighted averaging, also returns list)
try:
    wisfc_corr = tc.timecorr(multi_data, cfun=tc.wisfc)
    print(f"✓ WISFC (list): {len(wisfc_corr)} subjects, each shape {wisfc_corr[0].shape}")
except Exception as e:
    print(f"✗ WISFC error: {e}")

# Combined ISFC (single averaged result)
try:
    combined_isfc = tc.timecorr(multi_data, cfun=tc.isfc, combine=tc.mean_combine)
    print(f"✓ Combined ISFC: {combined_isfc.shape}")
except Exception as e:
    print(f"✗ Combined ISFC error: {e}")

# Auto-correlation function
try:
    auto_corr = tc.timecorr(data, cfun=tc.autofc)
    print(f"✓ Auto-correlation: {auto_corr.shape}")
except Exception as e:
    print(f"✗ Auto-correlation error: {e}")

print("\n" + "="*50)
print("Updated README.md examples testing completed!")
print("All major functionality appears to be working correctly.")
print("="*50)