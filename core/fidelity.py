"""
Fidelity Classifier for C-MAPS framework.
Combines autoencoder feature extraction with classifier training for density ratio estimation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union
import warnings

from ..core.preprocessing import DataPreprocessorForAutoencoder
from ..models.autoencoder import TabularAutoencoderTrainer
from ..models.classifiers import FidelityClassifierTrainer

warnings.filterwarnings('ignore')


class FidelityClassifier:
    """
    Main class for fidelity classification in C-MAPS framework.
    
    This class handles:
    1. Data preprocessing for autoencoder training
    2. Autoencoder training for feature extraction
    3. Classifier training for density ratio estimation
    4. Importance weight estimation
    """
    
    def __init__(self,
                 embedding_dim: int = 10,
                 classifier_type: str = 'mlp',
                 calibration_method: str = 'isotonic',
                 random_seed: int = 42,
                 verbose: bool = True):
        """
        Initialize the FidelityClassifier.
        
        Args:
            embedding_dim: Dimension of autoencoder embedding
            classifier_type: Type of classifier ('mlp', 'lr', 'lgbm')
            calibration_method: Calibration method ('isotonic', 'sigmoid')
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.embedding_dim = embedding_dim
        self.classifier_type = classifier_type
        self.calibration_method = calibration_method
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Initialize components
        self.preprocessor = DataPreprocessorForAutoencoder(verbose=verbose)
        self.autoencoder_trainer = None
        self.classifier_trainer = FidelityClassifierTrainer(
            classifier_type=classifier_type,
            calibration_method=calibration_method,
            random_seed=random_seed,
            verbose=verbose
        )
        
        # Store processed data and embeddings
        self.processed_real_df = None
        self.processed_synthetic_df = None
        self.real_embeddings = None
        self.synthetic_embeddings = None
        
        self.is_fitted = False
        
    def fit(self,
            real_data: pd.DataFrame,
            synthetic_data: pd.DataFrame,
            # Autoencoder parameters
            autoencoder_epochs: int = 400,
            autoencoder_batch_size: int = 256,
            autoencoder_lr: float = 0.001,
            autoencoder_weight_decay: float = 1e-5,
            use_synthetic_for_autoencoder: bool = False,
            autoencoder_hidden_dims: Optional[List[int]] = None,
            # Classifier parameters
            classifier_test_size: float = 0.2,
            mlp_hidden_layer_sizes: Tuple[int, ...] = (100, 50),
            show_calibration_plot: bool = True) -> Tuple[float, float]:
        """
        Fit the complete fidelity classification pipeline.
        
        Args:
            real_data: Real data DataFrame
            synthetic_data: Synthetic data DataFrame
            autoencoder_epochs: Number of autoencoder training epochs
            autoencoder_batch_size: Batch size for autoencoder training
            autoencoder_lr: Learning rate for autoencoder
            autoencoder_weight_decay: Weight decay for autoencoder
            use_synthetic_for_autoencoder: Whether to use synthetic data for autoencoder training
            autoencoder_hidden_dims: Hidden dimensions for autoencoder
            classifier_test_size: Test size for classifier evaluation
            mlp_hidden_layer_sizes: Hidden layer sizes for MLP classifier
            show_calibration_plot: Whether to show calibration plot
            
        Returns:
            Tuple of (classifier_accuracy, classifier_roc_auc)
        """
        if self.verbose:
            print("=" * 60)
            print("STARTING C-MAPS FIDELITY CLASSIFIER TRAINING")
            print("=" * 60)
        
        # Step 1: Preprocess data for autoencoder
        if self.verbose:
            print("\n1. PREPROCESSING DATA FOR AUTOENCODER")
            print("-" * 40)
        
        (self.processed_real_df, 
         self.processed_synthetic_df, 
         encoders, 
         numerical_cols, 
         encoded_cols) = self.preprocessor.fit_transform(real_data, synthetic_data)
        
        # Step 2: Train autoencoder and get embeddings
        if self.verbose:
            print("\n2. TRAINING AUTOENCODER FOR FEATURE EXTRACTION")
            print("-" * 40)
        
        # Adjust embedding dimension based on data complexity
        adjusted_embedding_dim = min(self.embedding_dim, 
                                   max(8, self.processed_real_df.shape[1] // 3))
        
        self.autoencoder_trainer = TabularAutoencoderTrainer(
            input_dim=self.processed_real_df.shape[1],
            embedding_dim=adjusted_embedding_dim,
            hidden_dims=autoencoder_hidden_dims,
            verbose=self.verbose
        )
        
        # Adjust batch size based on data size
        adjusted_batch_size = min(autoencoder_batch_size, len(self.processed_real_df))
        
        self.real_embeddings, self.synthetic_embeddings = self.autoencoder_trainer.train(
            real_data=self.processed_real_df,
            synthetic_data=self.processed_synthetic_df if use_synthetic_for_autoencoder else None,
            epochs=autoencoder_epochs,
            batch_size=adjusted_batch_size,
            learning_rate=autoencoder_lr,
            weight_decay=autoencoder_weight_decay,
            use_synthetic_for_training=use_synthetic_for_autoencoder
        )
        
        # If not using synthetic data for training, get synthetic embeddings separately
        if not use_synthetic_for_autoencoder:
            self.synthetic_embeddings = self.autoencoder_trainer.get_embeddings(self.processed_synthetic_df)
        
        if self.verbose:
            print(f"Generated embeddings - Real: {self.real_embeddings.shape}, Synthetic: {self.synthetic_embeddings.shape}")
        
        # Step 3: Train classifier for density ratio estimation
        if self.verbose:
            print("\n3. TRAINING CLASSIFIER FOR DENSITY RATIO ESTIMATION")
            print("-" * 40)
        
        # Use subset of synthetic data for classifier training to avoid overfitting
        synthetic_subset_size = min(5 * len(self.real_embeddings), len(self.synthetic_embeddings))
        if synthetic_subset_size < len(self.synthetic_embeddings):
            synthetic_indices = np.random.choice(
                len(self.synthetic_embeddings), 
                synthetic_subset_size, 
                replace=False
            )
            synthetic_embeddings_for_classifier = self.synthetic_embeddings[synthetic_indices]
        else:
            synthetic_embeddings_for_classifier = self.synthetic_embeddings
        
        # Adjust MLP hidden layer sizes based on embedding dimension
        adjusted_mlp_sizes = tuple(min(size, adjusted_embedding_dim * 4) 
                                 for size in mlp_hidden_layer_sizes)
        
        accuracy, roc_auc = self.classifier_trainer.train(
            real_data=self.real_embeddings,
            synthetic_data=synthetic_embeddings_for_classifier,
            test_size=classifier_test_size,
            mlp_hidden_layer_sizes=adjusted_mlp_sizes,
            show_calibration_plot=show_calibration_plot
        )
        
        self.is_fitted = True
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("FIDELITY CLASSIFIER TRAINING COMPLETE!")
            print("=" * 60)
        
        return accuracy, roc_auc
    
    def estimate_importance_weights(self, 
                                   synthetic_data: Optional[pd.DataFrame] = None,
                                   epsilon: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate importance weights for synthetic data.
        
        Args:
            synthetic_data: Synthetic data to estimate weights for (if None, uses training synthetic data)
            epsilon: Small constant for numerical stability
            
        Returns:
            Tuple of (importance_weights, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("FidelityClassifier must be fitted before estimating weights")
        
        if synthetic_data is not None:
            # Process new synthetic data
            processed_synthetic_df, _, _, _, _ = self.preprocessor.fit_transform(
                self.processed_real_df.iloc[:1],  # Dummy real data for consistent processing
                synthetic_data
            )
            synthetic_embeddings = self.autoencoder_trainer.get_embeddings(processed_synthetic_df)
        else:
            # Use training synthetic embeddings
            synthetic_embeddings = self.synthetic_embeddings
        
        return self.classifier_trainer.estimate_importance_weights(synthetic_embeddings, epsilon)
    
    def visualize_importance_weights(self, 
                                    importance_weights: np.ndarray,
                                    title: str = "Distribution of Importance Weights"):
        """
        Visualize the distribution of importance weights.
        
        Args:
            importance_weights: Array of importance weights
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        plt.hist(importance_weights, bins=50, alpha=0.7, color='steelblue')
        plt.axvline(np.median(importance_weights), color='red', linestyle='--',
                   label=f'Median: {np.median(importance_weights):.4f}')
        plt.title(title, fontsize=16)
        plt.xlabel('Importance Weight', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        if self.verbose:
            print(f"Weight statistics:")
            print(f"  Min: {np.min(importance_weights):.6f}")
            print(f"  Max: {np.max(importance_weights):.6f}")
            print(f"  Mean: {np.mean(importance_weights):.6f}")
            print(f"  Median: {np.median(importance_weights):.6f}")
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the computed embeddings."""
        if not self.is_fitted:
            raise ValueError("FidelityClassifier must be fitted first")
        return self.real_embeddings, self.synthetic_embeddings
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get the processed data."""
        if not self.is_fitted:
            raise ValueError("FidelityClassifier must be fitted first")
        return self.processed_real_df, self.processed_synthetic_df