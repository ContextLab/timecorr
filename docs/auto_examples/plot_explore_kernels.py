# -*- coding: utf-8 -*-
"""
=============================
Explore kernels
=============================

This example demonstrates and compares different kernel functions used in timecorr
for temporal weighting. Kernels control how much influence nearby timepoints have
on correlation estimates at each moment.

Different kernels are suited for different types of temporal analysis:
- **Gaussian**: Smooth, bell-shaped weighting (most common)
- **Laplace**: Exponential decay, more sparse and temporally precise
- **Mexican Hat**: Derivative-based, good for detecting temporal changes
- **Delta/Eye**: Uniform weighting within a window

Understanding kernel shapes helps choose the right approach for your temporal
correlation analysis.

"""
# Code source: Lucy Owen & Enhanced by Claude
# License: MIT

# Load packages
import timecorr as tc
import numpy as np
import matplotlib.pyplot as plt
import os

# Configure matplotlib for all environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

print("Exploring Timecorr Kernel Functions")
print("=" * 40)

# Define parameters
T = 100  # Number of timepoints
width = 10  # Kernel width parameter

print(f"\nParameters:")
print(f"  Number of timepoints: {T}")
print(f"  Kernel width: {width}")

# Define kernel functions with correct parameter syntax
print(f"\n1. Setting up kernel functions...")

kernels = {
    'Gaussian': {
        'name': 'Gaussian', 
        'weights': tc.gaussian_weights, 
        'params': {'var': width},
        'description': 'Smooth, bell-shaped weighting. Good for general temporal smoothing.'
    },
    'Laplace': {
        'name': 'Laplace', 
        'weights': tc.laplace_weights, 
        'params': {'scale': width},
        'description': 'Exponential decay. More sparse, better temporal precision.'
    },
    'Mexican Hat': {
        'name': 'Mexican Hat', 
        'weights': tc.mexican_hat_weights, 
        'params': {'sigma': width},
        'description': 'Derivative-based. Good for detecting temporal changes.'
    },
    'Delta/Eye': {
        'name': 'Delta/Eye', 
        'weights': tc.eye_weights, 
        'params': {'var': width},
        'description': 'Uniform weighting within a window. Sharp temporal boundaries.'
    }
}

# Test each kernel function
print(f"\n2. Testing kernel functions:")
kernel_weights = {}

for name, kernel in kernels.items():
    try:
        # Generate kernel weights
        weights = kernel['weights'](T, kernel['params'])
        kernel_weights[name] = weights
        print(f"   ✓ {name}: shape {weights.shape}, max={weights.max():.3f}")
    except Exception as e:
        print(f"   ✗ {name}: Failed - {e}")

# Create comprehensive comparison plot
if kernel_weights:
    print(f"\n3. Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Plot each kernel
    for i, (name, weights) in enumerate(kernel_weights.items()):
        ax = axes[i]
        
        # Plot kernel weights
        ax.plot(weights, linewidth=2, label=name)
        ax.set_title(f'{name} Kernel\\n{kernels[name]["description"]}', fontsize=12)
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Handle 2D weight matrices - extract diagonal or middle row
        if len(weights.shape) == 2:
            # For 2D weight matrices, show the middle row (centered on timepoint T//2)
            center_idx = T // 2
            weights_1d = weights[center_idx, :]
        else:
            weights_1d = weights
            
        # Plot the 1D weights
        ax.clear()
        ax.plot(weights_1d, linewidth=2, label=name)
        ax.set_title(f'{name} Kernel\\n{kernels[name]["description"]}', fontsize=12)
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add annotations
        max_idx = np.argmax(weights_1d)
        max_val = float(weights_1d[max_idx])
        ax.annotate(f'Max: {max_val:.3f}', 
                   xy=(max_idx, max_val), 
                   xytext=(max_idx+10, max_val+0.01),
                   arrowprops=dict(arrowstyle='->', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: kernel_comparison.png")
    
    # Create overlay comparison
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, (name, weights) in enumerate(kernel_weights.items()):
        # Handle 2D weight matrices
        if len(weights.shape) == 2:
            center_idx = T // 2
            weights_1d = weights[center_idx, :]
        else:
            weights_1d = weights
        plt.plot(weights_1d, linewidth=2, label=name, color=colors[i % len(colors)])
    
    plt.title('Kernel Function Comparison', fontsize=16)
    plt.xlabel('Timepoints', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations about characteristics
    plt.figtext(0.02, 0.02, 
               'Gaussian: Smooth, symmetric\\n'
               'Laplace: Sharp peak, exponential tails\\n'
               'Mexican Hat: Negative side lobes\\n'
               'Delta/Eye: Uniform within window', 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('kernel_overlay_comparison.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: kernel_overlay_comparison.png")

# Demonstrate kernel effects on correlation computation
print(f"\n4. Demonstrating kernel effects on correlation analysis...")

# Create synthetic data with temporal structure
np.random.seed(42)
T_demo = 50
K_demo = 5
demo_data = np.random.randn(T_demo, K_demo)

# Add some temporal structure
for i in range(1, T_demo):
    demo_data[i] = 0.7 * demo_data[i-1] + 0.3 * demo_data[i]

print(f"   Generated demo data: shape {demo_data.shape}")

# Apply different kernels to correlation computation
correlation_results = {}

for name, kernel in kernels.items():
    if name in kernel_weights:
        try:
            # Compute correlations with this kernel
            correlations = tc.timecorr(demo_data, 
                                     weights_function=kernel['weights'],
                                     weights_params=kernel['params'])
            correlation_results[name] = correlations
            print(f"   ✓ {name}: correlations shape {correlations.shape}")
        except Exception as e:
            print(f"   ✗ {name}: correlation failed - {e}")

# Visualize correlation results
if correlation_results:
    print(f"\n5. Visualizing correlation results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, correlations) in enumerate(correlation_results.items()):
        ax = axes[i]
        
        # Convert to matrix format for visualization
        corr_matrices = tc.vec2mat(correlations)
        
        # Plot correlation time series for first feature pair
        ax.plot(corr_matrices[0, 1, :], linewidth=2, label=f'{name} kernel')
        ax.set_title(f'Dynamic Correlation (Features 0-1)\\nUsing {name} Kernel')
        ax.set_xlabel('Timepoints')
        ax.set_ylabel('Correlation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add summary statistics
        mean_corr = np.mean(corr_matrices[0, 1, :])
        std_corr = np.std(corr_matrices[0, 1, :])
        ax.text(0.02, 0.98, f'Mean: {mean_corr:.3f}\\nStd: {std_corr:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig('kernel_correlation_effects.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: kernel_correlation_effects.png")

print(f"\n6. Summary of kernel characteristics:")
print("   • Gaussian: Smooth transitions, good for general use")
print("   • Laplace: Sharp focus, good for precise temporal analysis")
print("   • Mexican Hat: Emphasizes temporal changes, good for dynamic analysis")
print("   • Delta/Eye: Uniform weighting, good for windowed analysis")

print(f"\n7. Choosing the right kernel:")
print("   • Smooth temporal dynamics → Gaussian")
print("   • Precise temporal boundaries → Laplace")
print("   • Temporal change detection → Mexican Hat")
print("   • Simple windowed analysis → Delta/Eye")
print("   • Experiment with different widths to find optimal smoothing")

print(f"\n✓ Kernel exploration complete!")
print(f"Created visualizations:")
print("   - kernel_comparison.png: Individual kernel shapes")
print("   - kernel_overlay_comparison.png: All kernels overlaid")
print("   - kernel_correlation_effects.png: Effects on correlation analysis")

print(f"\nKey insights:")
print("- Different kernels produce different temporal smoothing patterns")
print("- Kernel choice affects the resulting dynamic correlations")
print("- Width parameter controls the temporal scale of analysis")
print("- Experiment with different kernels to find what works best for your data")
print("- Consider your temporal resolution and the dynamics you want to capture")