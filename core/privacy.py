"""
Privacy and Identifiability Analysis for C-MAPS framework.
Updated to align with source code logic.

This module identifies synthetic samples that violate privacy by being 
too close to real samples (closer than the real sample's nearest real neighbor).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from typing import Tuple, Optional
import warnings

from ..core.preprocessing import DataPreprocessorForIdentifiability

warnings.filterwarnings('ignore')


class IdentifiabilityAnalyzer:
    """
    Analyzer for computing privacy-violating synthetic samples and identifiability metrics.
    
    This class handles:
    1. Data preprocessing for identifiability analysis
    2. Feature weight computation based on entropy
    3. Real sample distinctness threshold calculation
    4. Privacy flag computation for synthetic samples (identifies which synthetic samples
       violate privacy by being too close to real samples)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the IdentifiabilityAnalyzer.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.preprocessor = DataPreprocessorForIdentifiability(verbose=verbose)
        
        # Store processed data and computed values
        self.processed_real_df = None
        self.processed_synthetic_df = None
        self.feature_weights = None
        self.real_distinctness_thresholds = None
        
        self.is_fitted = False
    
    def fit(self, 
            real_data: pd.DataFrame, 
            synthetic_data: pd.DataFrame) -> np.ndarray:
        """
        Fit the identifiability analyzer and compute identifiability flags.
        
        Args:
            real_data: Real data DataFrame
            synthetic_data: Synthetic data DataFrame
            
        Returns:
            Array of privacy flags for synthetic samples (1=violates privacy, 0=safe)
        """
        if self.verbose:
            print("=" * 60)
            print("STARTING IDENTIFIABILITY ANALYSIS")
            print("=" * 60)
        
        # Step 1: Preprocess data for identifiability analysis
        if self.verbose:
            print("\n1. PREPROCESSING DATA FOR IDENTIFIABILITY ANALYSIS")
            print("-" * 40)
        
        (self.processed_real_df,
         self.processed_synthetic_df,
         encoders,
         numerical_cols,
         encoded_cols) = self.preprocessor.fit_transform(real_data, synthetic_data)
        
        # Step 2: Compute feature weights based on entropy
        if self.verbose:
            print("\n2. COMPUTING FEATURE WEIGHTS")
            print("-" * 40)
        
        self.feature_weights = self._compute_feature_weights(self.processed_real_df)
        
        if self.verbose:
            print(f"Computed feature weights for {len(self.feature_weights)} features")
        
        # Step 3: Calculate real sample distinctness thresholds
        if self.verbose:
            print("\n3. CALCULATING REAL SAMPLE DISTINCTNESS THRESHOLDS")
            print("-" * 40)
        
        self.real_distinctness_thresholds = self._calculate_real_distinctness_thresholds(
            self.processed_real_df, self.feature_weights
        )
        
        if self.verbose:
            print(f"Calculated distinctness thresholds for {len(self.real_distinctness_thresholds)} real samples")
        
        # Step 4: Compute identifiability flags for synthetic samples
        if self.verbose:
            print("\n4. COMPUTING IDENTIFIABILITY FLAGS FOR SYNTHETIC SAMPLES")
            print("-" * 40)
        
        identifiability_flags = self._compute_identifiability_flags(
            self.processed_synthetic_df,
            self.processed_real_df,
            self.real_distinctness_thresholds,
            self.feature_weights
        )
        
        if self.verbose:
            identifiable_syn_count = np.sum(identifiability_flags)
            total_syn_count = len(identifiability_flags)
            identifiable_syn_ratio = identifiable_syn_count / total_syn_count
            
            print(f"Identifiability analysis complete:")
            print(f"  Total synthetic samples: {total_syn_count}")
            print(f"  Identifiable synthetic samples: {identifiable_syn_count}")
            print(f"  Synthetic identifiability ratio: {identifiable_syn_ratio:.4f}")
            
            print("\n" + "=" * 60)
            print("IDENTIFIABILITY ANALYSIS COMPLETE!")
            print("=" * 60)
        
        self.is_fitted = True
        
        # Compute overall identifiability score for comparison (after setting is_fitted)
        if self.verbose:
            overall_score = self.compute_overall_identifiability_score()
            print(f"  Overall real identifiability score: {overall_score:.4f}")
            print(f"  (Overall score = ratio of real samples identifiable through synthetic data)")
        
        return identifiability_flags
    
    def compute_identifiability_flags(self, 
                                    real_data: Optional[pd.DataFrame] = None,
                                    synthetic_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Compute identifiability flags for synthetic data.
        
        Args:
            real_data: Real data (if None, uses fitted real data)
            synthetic_data: Synthetic data (if None, uses fitted synthetic data)
            
        Returns:
            Array of privacy flags (1=violates privacy, 0=safe)
        """
        if not self.is_fitted and (real_data is None or synthetic_data is None):
            raise ValueError("IdentifiabilityAnalyzer must be fitted first or provide both real and synthetic data")
        
        if real_data is not None and synthetic_data is not None:
            # Process new data
            processed_real_df, processed_synthetic_df, _, _, _ = self.preprocessor.fit_transform(
                real_data, synthetic_data
            )
            feature_weights = self._compute_feature_weights(processed_real_df)
            real_thresholds = self._calculate_real_distinctness_thresholds(processed_real_df, feature_weights)
        else:
            # Use fitted data
            processed_real_df = self.processed_real_df
            processed_synthetic_df = self.processed_synthetic_df
            feature_weights = self.feature_weights
            real_thresholds = self.real_distinctness_thresholds
        
        return self._compute_identifiability_flags(
            processed_synthetic_df, processed_real_df, real_thresholds, feature_weights
        )
    
    def visualize_identifiability_flags(self, identifiability_flags: np.ndarray):
        """
        Visualize the distribution of privacy-violating synthetic samples.
        
        Args:
            identifiability_flags: Array of privacy flags (1 = synthetic sample violates privacy, 0 = safe)
        """
        plt.figure(figsize=(8, 5))
        sns.countplot(x=identifiability_flags, palette=['royalblue', 'orangered'])
        plt.title('Privacy-Violating Synthetic Samples')
        plt.xlabel('Privacy Flag (0: Safe, 1: Violates Privacy)')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Safe Synthetic\nSamples', 'Privacy-Violating\nSynthetic Samples'])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        if self.verbose:
            identifiable_count = np.sum(identifiability_flags)
            total_count = len(identifiability_flags)
            print(f"Identifiable synthetic samples: {identifiable_count}/{total_count} ({identifiable_count/total_count:.2%})")
            print(f"These are synthetic samples that violate privacy by being too close to real samples.")
    
    def analyze_weights_by_identifiability(self, 
                                          importance_weights: np.ndarray,
                                          identifiability_flags: np.ndarray):
        """
        Analyze importance weights by privacy status.
        
        Args:
            importance_weights: Array of importance weights
            identifiability_flags: Array of privacy flags (1 = violates privacy, 0 = safe)
        """
        plt.figure(figsize=(10, 6))
        
        # Split weights by identifiability
        weights_not_identifiable = importance_weights[identifiability_flags == 0]
        weights_identifiable = importance_weights[identifiability_flags == 1]
        
        plt.hist(weights_not_identifiable, bins=50, alpha=0.7, color='steelblue', 
                label='Safe Synthetic Samples')
        plt.hist(weights_identifiable, bins=50, alpha=0.7, color='orangered', 
                label='Privacy-Violating Synthetic Samples')
        
        # Add median lines
        if len(weights_not_identifiable) > 0:
            plt.axvline(np.median(weights_not_identifiable), color='blue', linestyle='--',
                       label=f'Median Safe: {np.median(weights_not_identifiable):.4f}')
        
        if len(weights_identifiable) > 0:
            plt.axvline(np.median(weights_identifiable), color='red', linestyle='--',
                       label=f'Median Privacy-Violating: {np.median(weights_identifiable):.4f}')
        
        plt.title('Distribution of Importance Weights by Privacy Status', fontsize=16)
        plt.xlabel('Importance Weight', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _compute_feature_weights(self, dataset: pd.DataFrame) -> np.ndarray:
        """
        Compute feature weights based on entropy for each feature.
        Follows the exact logic from source code.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Array of feature weights
        """
        def compute_entropy(labels: np.ndarray) -> np.ndarray:
            """Compute entropy of labels - exact function from source code."""
            value, counts = np.unique(np.round(labels), return_counts=True)
            return entropy(counts)
        
        # Convert to numpy array like source code
        X_gt_ = np.asarray(dataset)
        no, x_dim = X_gt_.shape
        
        # Initialize weights array
        W = np.zeros([x_dim,])
        
        # Compute entropy-based weight for each feature
        for i in range(x_dim):
            W[i] = compute_entropy(X_gt_[:, i])
        
        # Following source code logic - set weights to ones
        W = np.ones_like(W)
        
        return W
    
    def _calculate_real_distinctness_thresholds(self, 
                                              processed_real_df: pd.DataFrame,
                                              feature_weights: np.ndarray) -> np.ndarray:
        """
        Calculate distinctness thresholds r_i for each real sample.
        Follows the exact logic from source code.
        
        Args:
            processed_real_df: Processed real data
            feature_weights: Feature weights for scaling
            
        Returns:
            Array of distinctness thresholds
        """
        if processed_real_df.empty:
            return np.array([])
        
        # Convert to numpy array like source code
        X_gt_ = processed_real_df.values
        no, x_dim = X_gt_.shape
        
        # Normalization following source code logic
        X_hat = X_gt_.copy()
        eps = 1e-16
        
        for i in range(x_dim):
            X_hat[:, i] = X_gt_[:, i] * 1.0 / (feature_weights[i] + eps)
        
        # r_i computation - exact from source code
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
        distance, _ = nbrs.kneighbors(X_hat)
        
        # Return distance to second nearest neighbor (distance[:, 1])
        return distance[:, 1]
    
    def _compute_identifiability_flags(self,
                                     processed_synthetic_df: pd.DataFrame,
                                     processed_real_df: pd.DataFrame,
                                     real_thresholds: np.ndarray,
                                     feature_weights: np.ndarray) -> np.ndarray:
        """
        Determine if each synthetic sample is identifiable.
        A synthetic sample is identifiable if it's closer to any real sample
        than that real sample's nearest real neighbor.
        
        OPTIMIZED VERSION: Uses vectorized operations for speed.
        
        Args:
            processed_synthetic_df: Processed synthetic data
            processed_real_df: Processed real data
            real_thresholds: Distinctness thresholds for real samples (r_i values)
            feature_weights: Feature weights for scaling
            
        Returns:
            Array of identifiability flags (1 for each synthetic sample that violates privacy)
        """
        KN = len(processed_synthetic_df)
        if KN == 0 or processed_real_df.empty:
            return np.array([], dtype=int)
        
        # Convert to numpy arrays like source code
        X_gt_ = processed_real_df.values
        X_syn_ = processed_synthetic_df.values
        no, x_dim = X_gt_.shape
        
        # Normalization following source code logic
        X_hat = X_gt_.copy()
        X_syn_hat = X_syn_.copy()
        eps = 1e-16
        
        for i in range(x_dim):
            X_hat[:, i] = X_gt_[:, i] * 1.0 / (feature_weights[i] + eps)
            X_syn_hat[:, i] = X_syn_[:, i] * 1.0 / (feature_weights[i] + eps)
        
        if self.verbose:
            print(f"  Computing distances between {no} real and {KN} synthetic samples...")
        
        # OPTIMIZATION: Compute all pairwise distances at once using vectorized operations
        # This is much faster than the loop-based approach
        distances_real_to_syn = pairwise_distances(X_hat, X_syn_hat, metric='euclidean')
        # Shape: (no, KN) - distances_real_to_syn[i, j] = distance from real_i to syn_j
        
        if self.verbose:
            print(f"  Distance computation complete. Finding privacy violations...")
        
        # OPTIMIZATION: Vectorized comparison using broadcasting
        # real_thresholds shape: (no,) -> reshape to (no, 1) for broadcasting
        # distances_real_to_syn shape: (no, KN)
        # Result shape: (no, KN) - True where syn sample j violates privacy of real sample i
        violation_matrix = distances_real_to_syn < real_thresholds.reshape(-1, 1)
        
        # OPTIMIZATION: Find synthetic samples that violate ANY real sample's privacy
        # Use np.any along axis 0 (real samples axis) to get per-synthetic-sample flags
        synthetic_identifiable_flags = np.any(violation_matrix, axis=0).astype(int)
        
        # Final progress update
        if self.verbose:
            identifiable_syn_count = np.sum(synthetic_identifiable_flags)
            print(f"  Privacy analysis complete:")
            print(f"  Synthetic samples marked as identifiable: {identifiable_syn_count}/{KN}")
            print(f"  Synthetic identifiability ratio: {identifiable_syn_count/KN:.4f}")
            
            # Additional statistics
            real_samples_violated = np.sum(np.any(violation_matrix, axis=1))
            print(f"  Real samples with privacy violations: {real_samples_violated}/{no}")
        
        return synthetic_identifiable_flags
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get the processed data."""
        if not self.is_fitted:
            raise ValueError("IdentifiabilityAnalyzer must be fitted first")
        return self.processed_real_df, self.processed_synthetic_df
    
    def get_feature_weights(self) -> np.ndarray:
        """Get the computed feature weights."""
        if not self.is_fitted:
            raise ValueError("IdentifiabilityAnalyzer must be fitted first")
        return self.feature_weights
    
    def get_real_distinctness_thresholds(self) -> np.ndarray:
        """Get the real sample distinctness thresholds."""
        if not self.is_fitted:
            raise ValueError("IdentifiabilityAnalyzer must be fitted first")
        return self.real_distinctness_thresholds
    
    def compute_overall_identifiability_score(self) -> float:
        """
        Compute the overall identifiability score as in the source code.
        
        Returns:
            Overall identifiability score (ratio of identifiable real samples)
        """
        if not self.is_fitted:
            raise ValueError("IdentifiabilityAnalyzer must be fitted first")
        
        # Convert to numpy arrays like source code
        X_gt_ = self.processed_real_df.values
        X_syn_ = self.processed_synthetic_df.values
        no, x_dim = X_gt_.shape
        
        # Normalization following source code logic
        X_hat = X_gt_.copy()
        X_syn_hat = X_syn_.copy()
        eps = 1e-16
        W = self.feature_weights
        
        for i in range(x_dim):
            X_hat[:, i] = X_gt_[:, i] * 1.0 / (W[i] + eps)
            X_syn_hat[:, i] = X_syn_[:, i] * 1.0 / (W[i] + eps)
        
        # r_i computation - exact from source code
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
        distance, _ = nbrs.kneighbors(X_hat)
        
        # hat{r_i} computation - exact from source code
        nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
        distance_hat, _ = nbrs_hat.kneighbors(X_hat)
        
        # R_Diff computation - exact from source code
        R_Diff = distance_hat[:, 0] - distance[:, 1]
        identifiability_value = np.sum(R_Diff < 0) / float(no)
        
        return identifiability_value