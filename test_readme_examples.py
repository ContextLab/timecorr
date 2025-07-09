#!/usr/bin/env python
"""
Test script to verify README.md code examples work correctly
"""

import numpy as np
import timecorr as tc

print("Testing README.md code examples...")
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
    isfc_results = tc.timecorr(subjects_data, cfun=tc.isfc, weights_function=tc.gaussian_weights)
    print(f"✓ ISFC results shape: {isfc_results.shape}")
    
    # Weighted ISFC for similarity-based averaging
    wisfc_results = tc.timecorr(subjects_data, cfun=tc.wisfc, weights_function=tc.gaussian_weights)
    print(f"✓ WISFC results shape: {wisfc_results.shape}")
    
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
try:
    isfc_corr = tc.timecorr(multi_data, cfun=tc.isfc)
    print(f"✓ ISFC: {isfc_corr.shape}")
except Exception as e:
    print(f"✗ ISFC error: {e}")

# Weighted ISFC (similarity-weighted averaging)
try:
    wisfc_corr = tc.timecorr(multi_data, cfun=tc.wisfc)
    print(f"✓ WISFC: {wisfc_corr.shape}")
except Exception as e:
    print(f"✗ WISFC error: {e}")

# Auto-correlation function
try:
    auto_corr = tc.timecorr(data, cfun=tc.autofc)
    print(f"✓ Auto-correlation: {auto_corr.shape}")
except Exception as e:
    print(f"✗ Auto-correlation error: {e}")

# Test 6: Dimensionality Reduction
print("\n6. Testing dimensionality reduction...")

# Principal Component Analysis
try:
    pca_corr = tc.timecorr(data, rfun='PCA')
    print(f"✓ PCA: {pca_corr.shape}")
except Exception as e:
    print(f"✗ PCA error: {e}")

# Independent Component Analysis  
try:
    ica_corr = tc.timecorr(data, rfun='ICA')
    print(f"✓ ICA: {ica_corr.shape}")
except Exception as e:
    print(f"✗ ICA error: {e}")

# Factor Analysis
try:
    fa_corr = tc.timecorr(data, rfun='FactorAnalysis')
    print(f"✓ Factor Analysis: {fa_corr.shape}")
except Exception as e:
    print(f"✗ Factor Analysis error: {e}")

# Graph-theoretic measures
try:
    pagerank_corr = tc.timecorr(data, rfun='pagerank_centrality')
    print(f"✓ PageRank centrality: {pagerank_corr.shape}")
except Exception as e:
    print(f"✗ PageRank error: {e}")

try:
    eigenvector_corr = tc.timecorr(data, rfun='eigenvector_centrality')
    print(f"✓ Eigenvector centrality: {eigenvector_corr.shape}")
except Exception as e:
    print(f"✗ Eigenvector error: {e}")

print("\n" + "="*50)
print("README.md examples testing completed!")
print("="*50)