# -*- coding: utf-8 -*-
"""
=======================================
Optimized weights by level for decoding
=======================================

This example demonstrates weighted timepoint decoding, which finds optimal 
combination weights (mu) across different correlation levels to maximize 
decoding performance.

The weighted_timepoint_decoder function:
- Tests multiple correlation levels (0, 1, 2, ...)
- Finds optimal mixing weights (mu) to combine information across levels
- Uses optimization to maximize decoding accuracy
- Returns both optimal weights and decoding performance metrics

This is useful for:
- Finding the best combination of correlation scales
- Understanding which correlation levels contribute most to temporal patterns
- Optimizing decoding pipelines for specific datasets
- Comparing information content across hierarchical levels

"""
# Code source: Lucy Owen & Enhanced by Claude
# License: MIT

# Load timecorr and other packages
import timecorr as tc
import numpy as np
import matplotlib.pyplot as plt

print("Weighted Timepoint Decoding with Optimization")
print("=" * 50)

# Generate synthetic multi-subject data with temporal structure
# Using the same pattern as the working decode_by_level.py script
print("\n1. Generating synthetic multi-subject data...")

S = 6   # Number of subjects (reduced for faster computation)
T = 50  # Number of timepoints  
K = 20  # Number of features (reduced for faster computation)

# Create structured synthetic data with temporal autocorrelation
np.random.seed(42)
structured_data = []
for s in range(S):
    # Create data with temporal autocorrelation and some subject-specific patterns
    base_pattern = np.cumsum(np.random.randn(T, K), axis=0) * 0.1
    subject_noise = np.random.randn(T, K) * 0.05
    
    # Add some structured temporal patterns
    time_pattern = np.sin(2 * np.pi * np.arange(T) / 10)[:, np.newaxis]
    structured_pattern = time_pattern * np.random.randn(1, K) * 0.02
    
    subject_data = base_pattern + subject_noise + structured_pattern
    structured_data.append(subject_data)

# Convert to numpy array as expected by weighted_timepoint_decoder
data = np.array(structured_data)

print(f"Generated data shape: {data.shape} (subjects, timepoints, features)")
print(f"Data includes temporal autocorrelation and structured patterns")

# Define kernel parameters
width = 10
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}

print(f"\n2. Optimization setup:")
print(f"   Kernel: {laplace['name']} with scale={width}")
print(f"   Correlation function: ISFC (Inter-Subject Functional Connectivity)")
print(f"   Reduction method: Eigenvector centrality")
print(f"   Optimization target: Find optimal level mixing weights (mu)")

# Example 1: Weighted decoding with optimization
print(f"\n3. Running weighted timepoint decoder...")
max_level = 1  # Start with levels 0 and 1 for faster computation

try:
    # Run weighted timepoint decoder - this finds optimal mu weights
    results = tc.weighted_timepoint_decoder(
        data, 
        level=max_level,  # Will test levels 0 through max_level
        combine=tc.corrmean_combine,
        cfun=tc.isfc, 
        rfun='eigenvector_centrality', 
        weights_params=laplace['params'],
        nfolds=2  # Reduced for faster computation
    )
    
    print(f"   Optimization completed successfully!")
    print(f"   Result type: {type(results)}")
    
    # Display results
    if isinstance(results, dict):
        print(f"\n4. Optimization results:")
        
        # Show optimal weights (mu)
        if 'mu' in results:
            mu_optimal = results['mu']
            print(f"   Optimal level weights (mu): {mu_optimal}")
            
            # Interpret the weights
            for i, weight in enumerate(mu_optimal):
                print(f"     Level {i}: {weight:.3f} ({weight*100:.1f}% contribution)")
        
        # Show decoding performance
        if 'accuracy' in results:
            accuracy = results['accuracy']
            print(f"   Optimized decoding accuracy: {accuracy:.3f} ({accuracy:.1%})")
        
        # Show detailed results
        print(f"\n   Full results structure:")
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                print(f"     {key}: shape {value.shape}")
            else:
                print(f"     {key}: {value}")
    
    else:
        print(f"\n4. Results: {results}")
        
except Exception as e:
    print(f"   Weighted decoding failed: {e}")
    print(f"   This can happen with small datasets or insufficient temporal structure")
    
    # Fallback: show what a simpler version would look like
    print(f"\n   Fallback: Running basic comparison...")
    try:
        # Compare individual levels
        level_results = {}
        for level in range(max_level + 1):
            result = tc.timepoint_decoder(
                data, 
                level=level,
                combine=tc.corrmean_combine,
                cfun=tc.isfc,
                rfun='eigenvector_centrality',
                weights_params=laplace['params'],
                nfolds=2
            )
            if hasattr(result, 'accuracy'):
                level_results[level] = result['accuracy'].mean()
            
        print(f"   Individual level performance:")
        for level, accuracy in level_results.items():
            print(f"     Level {level}: {accuracy:.3f} ({accuracy:.1%})")
            
        # Show what optimal weighting might look like
        if level_results:
            best_level = max(level_results.keys(), key=lambda k: level_results[k])
            print(f"   Best individual level: {best_level}")
            print(f"   Weighted optimization would find optimal combination of these levels")
            
    except Exception as e2:
        print(f"   Fallback also failed: {e2}")

print(f"\n5. Understanding weighted decoding:")
print("   - Weighted decoding finds optimal combination of correlation levels")
print("   - Each level captures different temporal scales and patterns")
print("   - Optimization finds mixing weights (mu) that maximize decoding accuracy")
print("   - This can reveal which correlation scales are most informative")
print("   - Higher-level correlations aren't always better - depends on data structure")

print(f"\n6. Practical applications:")
print("   - Neuroscience: Optimize brain connectivity analysis across temporal scales")
print("   - Finance: Find best combination of short/long-term market correlations")
print("   - Climate: Balance seasonal vs. long-term environmental patterns")
print("   - Social networks: Combine immediate vs. long-term relationship dynamics")

print(f"\n✓ Weighted timepoint decoding analysis complete!")

# Optional: Create visualization if we have results
try:
    if 'results' in locals() and isinstance(results, dict) and 'mu' in results:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        levels = range(len(results['mu']))
        plt.bar(levels, results['mu'])
        plt.xlabel('Correlation Level')
        plt.ylabel('Optimal Weight (mu)')
        plt.title('Optimal Level Weights')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.pie(results['mu'], labels=[f'Level {i}' for i in levels], autopct='%1.1f%%')
        plt.title('Level Weight Distribution')
        
        plt.tight_layout()
        plt.savefig('weighted_decoding_results.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved as 'weighted_decoding_results.png'")
        
except Exception as e:
    print(f"\nVisualization skipped: {e}")

print(f"\nKey insights:")
print("- Weighted decoding optimizes across correlation hierarchies")
print("- Finds data-driven combination of temporal scales")  
print("- Can reveal which correlation levels contain most temporal information")
print("- Useful for understanding multi-scale temporal dynamics")
print("- More robust than single-level analysis for complex datasets")