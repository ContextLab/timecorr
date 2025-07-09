# -*- coding: utf-8 -*-
"""
=============================
Decode by level (Simple Version)
=============================

This example demonstrates timepoint decoding using hierarchical correlation levels.
This simplified version uses synthetic data to ensure reliability.

"""
# Code source: Lucy Owen & Enhanced by Claude
# License: MIT

import numpy as np

# Load timecorr and other packages
import timecorr as tc

print("Timepoint Decoding by Correlation Level (Simple Version)")
print("=" * 60)

# Generate synthetic multi-subject data
print("\n1. Generating synthetic multi-subject data...")

S = 8  # Number of subjects
T = 60  # Number of timepoints (reduced for faster computation)
K = 30  # Number of features

# Generate synthetic data with temporal structure
np.random.seed(42)
structured_data = []
for s in range(S):
    # Create data with temporal autocorrelation
    base_pattern = np.cumsum(np.random.randn(T, K), axis=0) * 0.1
    noise = np.random.randn(T, K) * 0.05
    subject_data = base_pattern + noise
    structured_data.append(subject_data)

# Convert to numpy array as expected by timepoint_decoder
data = np.array(structured_data)

print(f"Generated data shape: {data.shape} (subjects, timepoints, features)")

# Define kernel parameters
width = 10
laplace = {"name": "Laplace", "weights": tc.laplace_weights, "params": {"scale": width}}

print(f"\n2. Using {laplace['name']} kernel with scale={width}")

# Set your number of levels
# if integer, returns decoding accuracy, error, and rank for specified level
level = 2

print(f"\n3. Testing timepoint decoding at level {level}...")

try:
    # Run timecorr with specified functions for calculating correlations, as well as combining and reducing
    results = tc.timepoint_decoder(
        data,
        level=level,
        combine=tc.corrmean_combine,
        cfun=tc.isfc,
        rfun="eigenvector_centrality",
        weights_fun=laplace["weights"],
        weights_params=laplace["params"],
    )

    print(f"✓ Level {level} decoding results:")
    print(f"  Accuracy: {results['accuracy']:.3f}")
    print(f"  Error: {results['error']:.3f}")
    print(f"  Rank: {results['rank']:.3f}")

except Exception as e:
    print(f"✗ Error at level {level}: {e}")
    # Try with simpler parameters
    print("\nTrying with simpler parameters...")
    try:
        results = tc.timepoint_decoder(data, level=0, nfolds=2)
        print(f"✓ Basic decoding results:")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  Error: {results['error']:.3f}")
        print(f"  Rank: {results['rank']:.3f}")
    except Exception as e2:
        print(f"✗ Error with basic parameters: {e2}")
        print("This suggests an issue with the timepoint_decoder function.")

# Test multiple levels
print(f"\n4. Testing multiple levels...")
levels = np.arange(3)  # Test levels 0, 1, 2

try:
    results = tc.timepoint_decoder(
        data,
        level=levels,
        combine=tc.corrmean_combine,
        cfun=tc.isfc,
        rfun="eigenvector_centrality",
        weights_fun=laplace["weights"],
        weights_params=laplace["params"],
    )

    print(f"✓ Multi-level decoding results:")
    if isinstance(results, dict):
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        print(f"  Results: {results}")

except Exception as e:
    print(f"✗ Error with multiple levels: {e}")
    print("Timepoint decoder may have compatibility issues with this timecorr version.")

print("\n" + "=" * 60)
print("EXAMPLE COMPLETED")
print("Note: If errors occurred, this indicates potential compatibility issues")
print("with the timepoint_decoder function in the current timecorr version.")
