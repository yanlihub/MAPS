"""
Basic usage example for C-MAPS framework.

This example demonstrates how to:
1. Load real and synthetic data
2. Train a fidelity classifier
3. Compute identifiability flags
4. Perform privacy-preserving sampling
5. Evaluate results
"""

import pandas as pd
import numpy as np

# Import C-MAPS components
from c_maps import FidelityClassifier, IdentifiabilityAnalyzer, SamplingEngine
from c_maps.utils import (
    validate_data_compatibility,
    visualize_data_distribution,
    comprehensive_evaluation,
    print_evaluation_summary,
    check_dependencies,
    set_random_seeds
)

def main():
    """Main function demonstrating C-MAPS usage."""
    
    # Set random seed for reproducibility
    set_random_seeds(42)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("Please install missing dependencies before proceeding.")
        return
    
    print("\n" + "="*60)
    print("C-MAPS FRAMEWORK BASIC USAGE EXAMPLE")
    print("="*60)
    
    # ============================================================================
    # Step 1: Load your data
    # ============================================================================
    print("\n1. LOADING DATA")
    print("-" * 40)
    
    # Replace these with your actual data loading
    # real_data = pd.read_csv('path/to/your/real_data.csv')
    # synthetic_data = pd.read_csv('path/to/your/synthetic_data.csv')
    
    # For this example, we'll create dummy data
    print("Loading dummy data for demonstration...")
    
    # Create dummy real data
    np.random.seed(42)
    n_real = 1000
    n_features = 10
    
    real_data = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_real) for i in range(n_features)
    })
    real_data['categorical_1'] = np.random.choice(['A', 'B', 'C'], n_real)
    real_data['binary_1'] = np.random.choice([0, 1], n_real)
    
    # Create dummy synthetic data (with some bias)
    n_synthetic = 5000
    synthetic_data = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_synthetic) * 0.8 + 0.2 for i in range(n_features)
    })
    synthetic_data['categorical_1'] = np.random.choice(['A', 'B', 'C'], n_synthetic, p=[0.4, 0.4, 0.2])
    synthetic_data['binary_1'] = np.random.choice([0, 1], n_synthetic, p=[0.6, 0.4])
    
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    
    # Validate data compatibility
    is_valid, error_msg = validate_data_compatibility(real_data, synthetic_data)
    if not is_valid:
        print(f"Data validation failed: {error_msg}")
        return
    
    # ============================================================================
    # Step 2: Train Fidelity Classifier
    # ============================================================================
    print("\n2. TRAINING FIDELITY CLASSIFIER")
    print("-" * 40)
    
    # Initialize fidelity classifier
    fidelity_classifier = FidelityClassifier(
        embedding_dim=8,  # Dimension of autoencoder embedding
        classifier_type='mlp',  # 'mlp', 'lr', or 'lgbm'
        calibration_method='isotonic',  # 'isotonic' or 'sigmoid'
        random_seed=42,
        verbose=True
    )
    
    # Train the complete pipeline (autoencoder + classifier)
    accuracy, roc_auc = fidelity_classifier.fit(
        real_data=real_data,
        synthetic_data=synthetic_data,
        # Autoencoder parameters
        autoencoder_epochs=50,  # Reduced for demo
        autoencoder_batch_size=256,
        use_synthetic_for_autoencoder=False,  # Train autoencoder on real data only
        # Classifier parameters
        classifier_test_size=0.2,
        show_calibration_plot=True
    )
    
    print(f"Fidelity classifier trained successfully!")
    print(f"Classifier accuracy: {accuracy:.4f}")
    print(f"Classifier ROC AUC: {roc_auc:.4f}")
    
    # ============================================================================
    # Step 3: Estimate Importance Weights
    # ============================================================================
    print("\n3. ESTIMATING IMPORTANCE WEIGHTS")
    print("-" * 40)
    
    # Estimate importance weights for synthetic data
    importance_weights, probabilities = fidelity_classifier.estimate_importance_weights()
    
    print(f"Estimated importance weights for {len(importance_weights)} synthetic samples")
    print(f"Weight statistics:")
    print(f"  Min: {np.min(importance_weights):.6f}")
    print(f"  Max: {np.max(importance_weights):.6f}")
    print(f"  Mean: {np.mean(importance_weights):.6f}")
    print(f"  Median: {np.median(importance_weights):.6f}")
    
    # Visualize importance weights
    fidelity_classifier.visualize_importance_weights(importance_weights)
    
    # ============================================================================
    # Step 4: Compute Identifiability Flags (Privacy Analysis)
    # ============================================================================
    print("\n4. COMPUTING IDENTIFIABILITY FLAGS")
    print("-" * 40)
    
    # Initialize identifiability analyzer
    identifiability_analyzer = IdentifiabilityAnalyzer(verbose=True)
    
    # Compute identifiability flags
    identifiability_flags = identifiability_analyzer.fit(real_data, synthetic_data)
    
    # Visualize identifiability
    identifiability_analyzer.visualize_identifiability_flags(identifiability_flags)
    
    # Analyze weights by identifiability
    identifiability_analyzer.analyze_weights_by_identifiability(
        importance_weights, identifiability_flags
    )
    
    # ============================================================================
    # Step 5: Privacy-Preserving Sampling
    # ============================================================================
    print("\n5. PRIVACY-PRESERVING SAMPLING")
    print("-" * 40)
    
    # Initialize sampling engine
    sampling_engine = SamplingEngine(random_seed=42, verbose=True)
    
    # Define sampling parameters
    n_target_samples = len(real_data)  # Target same size as real data
    epsilon_identifiability = 0.05  # Allow max 5% identifiable samples
    
    # Option 1: SIR sampling (only fidelity, no privacy constraints)
    print("\nPerforming SIR sampling (fidelity only)...")
    sir_data, sir_weights, _, sir_indices = sampling_engine.sample(
        synthetic_data=synthetic_data,
        importance_weights=importance_weights,
        n_samples=n_target_samples,
        use_identifiability_constraint=False,
        method='weighted'
    )
    
    # Option 2: SIR-IC sampling (fidelity + privacy constraints)
    print("\nPerforming SIR-IC sampling (fidelity + privacy)...")
    siric_data, siric_weights, siric_flags, siric_indices = sampling_engine.sample(
        synthetic_data=synthetic_data,
        importance_weights=importance_weights,
        n_samples=n_target_samples,
        use_identifiability_constraint=True,
        identifiability_flags=identifiability_flags,
        epsilon_identifiability=epsilon_identifiability
    )
    
    # ============================================================================
    # Step 6: Visualize Results
    # ============================================================================
    print("\n6. VISUALIZING RESULTS")
    print("-" * 40)
    
    # Get embeddings for visualization
    real_embeddings, synthetic_embeddings = fidelity_classifier.get_embeddings()
    
    # Select embeddings for refined data
    sir_embeddings = synthetic_embeddings[sir_indices]
    siric_embeddings = synthetic_embeddings[siric_indices]
    
    # Visualize distributions
    visualize_data_distribution(
        data_list=[real_embeddings, synthetic_embeddings, sir_embeddings, siric_embeddings],
        names=['Real Data', 'Original Synthetic', 'SIR Refined', 'SIR-IC Refined'],
        colors=['blue', 'red', 'green', 'purple'],
        method='pca',
        title='Data Distributions Comparison'
    )
    
    # Visualize sampling results
    sampling_engine.visualize_sampling_results(
        original_weights=importance_weights,
        selected_weights=siric_weights,
        original_flags=identifiability_flags,
        selected_flags=siric_flags
    )
    
    # ============================================================================
    # Step 7: Comprehensive Evaluation
    # ============================================================================
    print("\n7. COMPREHENSIVE EVALUATION")
    print("-" * 40)
    
    # Evaluate SIR-IC results
    evaluation = comprehensive_evaluation(
        real_data=real_data,
        synthetic_data=synthetic_data,
        identifiability_flags=identifiability_flags,
        importance_weights=importance_weights,
        selected_indices=siric_indices,
        epsilon_target=epsilon_identifiability
    )
    
    # Print evaluation summary
    print_evaluation_summary(evaluation)
    
    # ============================================================================
    # Step 8: Save Results (Optional)
    # ============================================================================
    print("\n8. SAVING RESULTS")
    print("-" * 40)
    
    # Prepare results dictionary
    results = {
        'refined_synthetic_data': siric_data,
        'importance_weights': importance_weights[siric_indices],
        'identifiability_flags': siric_flags,
        'evaluation_metrics': evaluation,
        'parameters': {
            'n_target_samples': n_target_samples,
            'epsilon_identifiability': epsilon_identifiability,
            'embedding_dim': 8,
            'classifier_type': 'mlp'
        }
    }
    
    # Save results (uncomment to actually save)
    # from c_maps.utils import save_results
    # save_results(results, 'results/c_maps_results.pkl', format='pickle')
    
    print("Example completed successfully!")
    print("\nNext steps:")
    print("- Replace dummy data with your actual datasets")
    print("- Tune hyperparameters for your specific use case")
    print("- Experiment with different classifier types and sampling methods")
    print("- Evaluate results using domain-specific metrics")


if __name__ == "__main__":
    main()