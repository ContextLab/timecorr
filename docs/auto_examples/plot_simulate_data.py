# -*- coding: utf-8 -*-
"""
=============================
Simulate subject data
=============================

This example demonstrates how to simulate synthetic timeseries data using timecorr
and visualize the resulting dynamic correlations. The simulate_data function provides
several data generation methods for testing and validation.

Data generation types:
- **block**: Data with block structure and temporal transitions
- **random**: Random multivariate normal data
- **ramping**: Gradual transitions between correlation patterns
- **constant**: Constant correlation structure

This is useful for:
- Testing timecorr algorithms with known ground truth
- Understanding how different data structures affect correlation estimates
- Validating analysis pipelines before applying to real data
- Creating reproducible examples and tutorials

"""
# Code source: Lucy Owen & Enhanced by Claude
# License: MIT

# Load required packages
import timecorr as tc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Configure matplotlib for all environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

print("Data Simulation and Dynamic Correlation Visualization")
print("=" * 55)

# Define simulation parameters
S = 1    # Number of subjects
T = 100  # Number of timepoints
K = 10   # Number of features
B = 5    # Number of blocks (for block data generation)

print(f"\nSimulation parameters:")
print(f"  Subjects: {S}")
print(f"  Timepoints: {T}")
print(f"  Features: {K}")
print(f"  Blocks: {B}")

# Example 1: Compare different data generation methods
print(f"\n1. Comparing different data generation methods...")

data_types = ['block', 'random', 'ramping', 'constant']
simulated_datasets = {}

for data_type in data_types:
    try:
        # Generate data with ground truth correlations
        data, corrs = tc.simulate_data(
            datagen=data_type, 
            return_corrs=True, 
            set_random_seed=42,  # For reproducibility
            S=S, T=T, K=K, B=B
        )
        simulated_datasets[data_type] = {'data': data, 'corrs': corrs}
        print(f"   ✓ {data_type}: data shape {data.shape}, corrs shape {corrs.shape}")
    except Exception as e:
        print(f"   ✗ {data_type}: Failed - {e}")

# Visualize the different data types
if simulated_datasets:
    print(f"\n2. Visualizing different data generation methods...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (data_type, dataset) in enumerate(simulated_datasets.items()):
        ax = axes[i]
        
        # Plot the raw data as a heatmap
        data = dataset['data']
        im = ax.imshow(data.T, aspect='auto', cmap='RdBu_r')
        ax.set_title(f'{data_type.capitalize()} Data Generation')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Features')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('data_generation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: data_generation_comparison.png")

# Example 2: Detailed analysis of block data
print(f"\n3. Detailed analysis of block-structured data...")

# Generate block data for detailed analysis
block_data, block_corrs = tc.simulate_data(
    datagen='block', 
    return_corrs=True, 
    set_random_seed=42,
    S=1, T=T, K=K, B=B
)

print(f"   Block data shape: {block_data.shape}")
print(f"   Ground truth correlations shape: {block_corrs.shape}")

# Compute dynamic correlations using timecorr
print(f"\n4. Computing dynamic correlations...")

# Calculate correlations with Gaussian kernel
tc_vec_data = tc.timecorr(
    block_data, 
    weights_function=tc.gaussian_weights, 
    weights_params={'var': 5}
)

# Convert from vector to matrix format
tc_mat_data = tc.vec2mat(tc_vec_data)

print(f"   Vectorized correlations shape: {tc_vec_data.shape}")
print(f"   Matrix correlations shape: {tc_mat_data.shape}")

# Example 3: Visualization of dynamic correlations
print(f"\n5. Visualizing dynamic correlations at different timepoints...")

# Select timepoints to visualize
timepoints = [20, 50, 80]  # Early, middle, late
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, t in enumerate(timepoints):
    ax = axes[i]
    
    # Plot correlation matrix at this timepoint
    sns.heatmap(tc_mat_data[:, :, t], 
                cmap='RdBu_r', 
                center=0, 
                vmin=-1, vmax=1,
                square=True,
                ax=ax)
    ax.set_title(f'Dynamic Correlations at t={t}')
    ax.set_xlabel('Features')
    if i == 0:
        ax.set_ylabel('Features')

plt.tight_layout()
plt.savefig('dynamic_correlations_timepoints.png', dpi=150, bbox_inches='tight')
print(f"   Saved: dynamic_correlations_timepoints.png")

# Example 4: Compare ground truth vs estimated correlations
print(f"\n6. Comparing ground truth vs estimated correlations...")

# Ground truth correlations at middle timepoint
mid_timepoint = T // 2

# Convert ground truth correlations to matrix format
ground_truth_mat = tc.vec2mat(block_corrs)
ground_truth = ground_truth_mat[:, :, mid_timepoint]
estimated = tc_mat_data[:, :, mid_timepoint]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ground truth
sns.heatmap(ground_truth, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, ax=axes[0])
axes[0].set_title('Ground Truth Correlations')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Features')

# Estimated
sns.heatmap(estimated, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, ax=axes[1])
axes[1].set_title('Estimated Correlations (Gaussian kernel)')
axes[1].set_xlabel('Features')

# Difference
difference = estimated - ground_truth
sns.heatmap(difference, cmap='RdBu_r', center=0,
            square=True, ax=axes[2])
axes[2].set_title('Difference (Estimated - Ground Truth)')
axes[2].set_xlabel('Features')

plt.tight_layout()
plt.savefig('ground_truth_vs_estimated.png', dpi=150, bbox_inches='tight')
print(f"   Saved: ground_truth_vs_estimated.png")

# Calculate correlation between ground truth and estimated
correlation_quality = np.corrcoef(ground_truth.flatten(), estimated.flatten())[0, 1]
print(f"   Correlation between ground truth and estimated: {correlation_quality:.3f}")

# Example 5: Temporal dynamics analysis
print(f"\n7. Analyzing temporal dynamics...")

# Track how specific correlations change over time
feature_pairs = [(0, 1), (2, 5), (4, 8)]

plt.figure(figsize=(12, 8))

for i, (f1, f2) in enumerate(feature_pairs):
    plt.subplot(3, 1, i+1)
    
    # Ground truth temporal evolution (use matrix format)
    plt.plot(ground_truth_mat[f1, f2, :], 'b-', linewidth=2, label='Ground Truth')
    
    # Estimated temporal evolution
    plt.plot(tc_mat_data[f1, f2, :], 'r--', linewidth=2, label='Estimated')
    
    plt.title(f'Temporal Evolution: Features {f1} and {f2}')
    plt.xlabel('Timepoints')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temporal_correlation_evolution.png', dpi=150, bbox_inches='tight')
print(f"   Saved: temporal_correlation_evolution.png")

print(f"\n8. Summary statistics:")
print(f"   Mean absolute difference: {np.mean(np.abs(difference)):.3f}")
print(f"   Root mean square error: {np.sqrt(np.mean(difference**2)):.3f}")
print(f"   Correlation quality: {correlation_quality:.3f}")

print(f"\n✓ Data simulation and visualization complete!")
print(f"\nCreated visualizations:")
print("   - data_generation_comparison.png: Different data generation methods")
print("   - dynamic_correlations_timepoints.png: Correlations at different times")
print("   - ground_truth_vs_estimated.png: Ground truth vs estimated comparison")
print("   - temporal_correlation_evolution.png: How correlations change over time")

print(f"\nKey insights:")
print("- simulate_data() provides multiple data generation methods")
print("- Block data shows clear temporal transitions in correlation structure")
print("- Dynamic correlations capture temporal changes in relationships")
print("- Ground truth comparison helps validate timecorr performance")
print("- Different kernels and parameters affect correlation estimation quality")

print(f"\nNext steps:")
print("- Try different kernel functions (Laplace, Mexican Hat)")
print("- Experiment with different kernel widths")
print("- Test with multi-subject data (S > 1)")
print("- Explore higher-order correlations using rfun parameter")

