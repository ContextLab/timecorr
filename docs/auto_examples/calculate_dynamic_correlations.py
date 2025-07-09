# -*- coding: utf-8 -*-
"""
=============================
Calculate dynamic correlations
=============================

This example demonstrates how to calculate dynamic correlations between two 
timeseries datasets using the `wcorr` function with different kernel functions.

The `wcorr` function computes weighted correlations between two datasets using
a kernel-based approach, allowing you to see how correlations change over time.

"""
# Code source: Lucy Owen & Enhanced by Claude
# License: MIT

# Load timecorr and other packages
import timecorr as tc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Define data parameters
S = 1      # Number of subjects
T = 1000   # Number of timepoints  
K = 10     # Number of features
B = 5      # Number of blocks (for block data generation)

print("Dynamic Correlation Analysis Example")
print("="*40)
print(f"Parameters: T={T}, K={K}, B={B}")

# Generate two different synthetic datasets for comparison
print("\n1. Generating synthetic datasets...")

# Dataset 1: Ramping data with seed 1
subs_data_1 = tc.simulate_data(datagen='ramping', return_corrs=False, 
                              set_random_seed=1, S=S, T=T, K=K, B=B)

# Dataset 2: Ramping data with seed 2 (different pattern)
subs_data_2 = tc.simulate_data(datagen='ramping', return_corrs=False, 
                              set_random_seed=2, S=S, T=T, K=K, B=B)

print(f"Dataset 1 shape: {subs_data_1.shape}")
print(f"Dataset 2 shape: {subs_data_2.shape}")

# Define kernel parameters for dynamic correlation analysis
width = 100
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}

print(f"\n2. Computing dynamic correlations with Laplace kernel (scale={width})...")

# Calculate dynamic correlations between the two datasets
# The wcorr function expects weight matrices, so we generate them with the kernel
laplace_weights = laplace['weights'](T, laplace['params'])
wcorred_data = tc.wcorr(np.array(subs_data_1), np.array(subs_data_2), weights=laplace_weights)

print(f"Dynamic correlations shape: {wcorred_data.shape}")
print(f"Interpretation: ({K}, {K}, {T}) = (features1, features2, timepoints)")

# Analyze the results
print(f"\n3. Analysis of results:")
print(f"   Min correlation: {wcorred_data.min():.3f}")
print(f"   Max correlation: {wcorred_data.max():.3f}")
print(f"   Mean correlation: {wcorred_data.mean():.3f}")
print(f"   Std correlation: {wcorred_data.std():.3f}")

# Visualize results
try:
    import matplotlib.pyplot as plt
    
    # Plot correlation time series for specific feature pairs
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Correlation matrices at different time points
    timepoints = [100, 300, 500, 700]
    for i, t in enumerate(timepoints):
        plt.subplot(3, 4, i+1)
        plt.imshow(wcorred_data[:, :, t], cmap='RdBu_r', vmin=-1, vmax=1)
        plt.title(f'Correlations at t={t}')
        plt.colorbar()
    
    # Subplot 2: Time series of specific correlations
    plt.subplot(3, 1, 2)
    plt.plot(wcorred_data[0, 1, :], label='Features (0,1)', linewidth=2)
    plt.plot(wcorred_data[2, 5, :], label='Features (2,5)', linewidth=2)
    plt.plot(wcorred_data[3, 7, :], label='Features (3,7)', linewidth=2)
    plt.title('Dynamic Correlations Over Time')
    plt.xlabel('Timepoints')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Distribution of all correlations
    plt.subplot(3, 1, 3)
    plt.hist(wcorred_data.flatten(), bins=50, alpha=0.7, density=True)
    plt.title('Distribution of All Dynamic Correlations')
    plt.xlabel('Correlation Value')
    plt.ylabel('Density')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dynamic_correlations_example.png', dpi=150, bbox_inches='tight')
    print(f"\n4. Visualization saved as 'dynamic_correlations_example.png'")
    
except ImportError:
    print("\n4. Matplotlib not available for visualization")

print("\n✓ Dynamic correlation analysis complete!")
print("\nKey insights:")
print("- wcorr computes correlations between two datasets at each timepoint")
print("- Kernel functions control temporal smoothing of correlations")
print("- Results show how inter-dataset correlations evolve over time")
print("- This is useful for comparing dynamic patterns between conditions/groups")