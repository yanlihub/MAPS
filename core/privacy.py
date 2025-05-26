"""
Privacy and Identifiability Analysis for C-MAPS framework.
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
    Analyzer for computing identifiability flags and privacy metrics.
    
    This class handles:
    1. Data preprocessing for identifiability analysis
    2. Feature weight computation based on entropy
    3. Real sample distinctness threshold calculation
    4. Synthetic sample identifiability flag computation
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
            Array of identifiability flags for synthetic samples (0=not identifiable, 1=identifiable)
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
            identifiable_count = np.sum(identifiability_flags)
            total_count = len(identifiability_flags)
            identifiable_ratio = identifiable_count / total_count
            
            print(f"Identifiability analysis complete:")
            print(f"  Total synthetic samples: {total_count}")
            print(f"  Identifiable samples: {identifiable_count}")
            print(f"  Identifiable ratio: {identifiable_ratio:.4f}")
            
            print("\n" + "=" * 60)
            print("IDENTIFIABILITY ANALYSIS COMPLETE!")
            print("=" * 60)
        
        self.is_fitted = True
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
            Array of identifiability flags
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
        Visualize the distribution of identifiability flags.
        
        Args:
            identifiability_flags: Array of identifiability flags
        """
        plt.figure(figsize=(8, 5))
        sns.countplot(x=identifiability_flags, palette=['royalblue', 'orangered'])
        plt.title('Identifiability Flags for Synthetic Samples')
        plt.xlabel('Identifiability Flag (0: Not Identifiable, 1: Identifiable)')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Not Identifiable', 'Identifiable'])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        if self.verbose:
            identifiable_count = np.sum(identifiability_flags)
            total_count = len(identifiability_flags)
            print(f"Identifiable samples: {identifiable_count}/{total_count} ({identifiable_count/total_count:.2%})")
    
    def analyze_weights_by_identifiability(self, 
                                          importance_weights: np.ndarray,
                                          identifiability_flags: np.ndarray):
        """
        Analyze importance weights by identifiability status.
        
        Args:
            importance_weights: Array of importance weights
            identifiability_flags: Array of identifiability flags
        """
        plt.figure(figsize=(10, 6))
        
        # Split weights by identifiability
        weights_not_identifiable = importance_weights[identifiability_flags == 0]
        weights_identifiable = importance_weights[identifiability_flags == 1]
        
        plt.hist(weights_not_identifiable, bins=50, alpha=0.7, color='steelblue', 
                label='Not Identifiable')
        plt.hist(weights_identifiable, bins=50, alpha=0.7, color='orangered', 
                label='Identifiable')
        
        # Add median lines
        if len(weights_not_identifiable) > 0:
            plt.axvline(np.median(weights_not_identifiable), color='blue', linestyle='--',
                       label=f'Median Not Identifiable: {np.median(weights_not_identifiable):.4f}')
        
        if len(weights_identifiable) > 0:
            plt.axvline(np.median(weights_identifiable), color='red', linestyle='--',
                       label=f'Median Identifiable: {np.median(weights_identifiable):.4f}')
        
        plt.title('Distribution of Importance Weights by Identifiability Flag', fontsize=16)
        plt.xlabel('Importance Weight', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _compute_feature_weights(self, dataset: pd.DataFrame) -> np.ndarray:
        """
        Compute feature weights based on entropy for each feature.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Array of feature weights
        """
        def compute_entropy(labels: np.ndarray) -> float:
            """Compute entropy of labels."""
            value, counts = np.unique(np.round(labels), return_counts=True)
            return entropy(counts)
        
        X = np.asarray(dataset)
        no, x_dim = X.shape
        
        W = np.zeros(x_dim)
        
        # Compute entropy-based weight for each feature
        for i in range(x_dim):
            W[i] = compute_entropy(X[:, i])
        
        # Set to ones as per the paper's approach and add small epsilon
        eps = 1e-16
        W = np.ones_like(W)
        
        return 1.0 / (W + eps)
    
    def _calculate_real_distinctness_thresholds(self, 
                                              processed_real_df: pd.DataFrame,
                                              feature_weights: np.ndarray) -> np.ndarray:
        """
        Calculate distinctness thresholds r_i for each real sample.
        
        Args:
            processed_real_df: Processed real data
            feature_weights: Feature weights for scaling
            
        Returns:
            Array of distinctness thresholds
        """
        if processed_real_df.empty:
            return np.array([])
        
        real_data_values = processed_real_df.values
        
        # Scale data by feature weights
        if feature_weights.ndim == 1 and feature_weights.shape[0] == real_data_values.shape[1]:
            real_data_scaled = real_data_values * feature_weights.reshape(1, -1)
        else:
            real_data_scaled = real_data_values * feature_weights
        
        # Use NearestNeighbors to find distance to closest other sample
        nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean', algorithm='auto').fit(real_data_scaled)
        distances, _ = nbrs.kneighbors(real_data_scaled)
        
        # Return distance to second nearest neighbor (first is the point itself)
        return distances[:, 1]
    
    def _compute_identifiability_flags(self,
                                     processed_synthetic_df: pd.DataFrame,
                                     processed_real_df: pd.DataFrame,
                                     real_thresholds: np.ndarray,
                                     feature_weights: np.ndarray) -> np.ndarray:
        """
        Determine if each synthetic sample is identifiable.
        
        Args:
            processed_synthetic_df: Processed synthetic data
            processed_real_df: Processed real data
            real_thresholds: Distinctness thresholds for real samples
            feature_weights: Feature weights for scaling
            
        Returns:
            Array of identifiability flags
        """
        KN = len(processed_synthetic_df)
        if KN == 0 or processed_real_df.empty:
            return np.array([], dtype=int)
        
        synthetic_data_values = processed_synthetic_df.values
        real_data_values = processed_real_df.values
        
        # Scale data by feature weights
        if feature_weights.ndim == 1:
            w = feature_weights.reshape(1, -1)
        else:
            w = feature_weights
        
        synthetic_data_scaled = synthetic_data_values * w
        real_data_scaled = real_data_values * w
        
        identifiable_flags = np.zeros(KN, dtype=int)
        
        # Calculate pairwise distances
        dist_matrix = pairwise_distances(synthetic_data_scaled, real_data_scaled, metric='euclidean')
        
        for k in range(KN):
            # Check if any distance is less than corresponding threshold
            if np.any(dist_matrix[k, :] < real_thresholds):
                identifiable_flags[k] = 1
            
            # Progress update
            if k > 0 and k % (KN // 20 if KN >= 20 else 1) == 0:
                if self.verbose:
                    print(f"  Checked {k+1}/{KN} synthetic samples")
        
        if self.verbose:
            print(f"  Finished checking {KN}/{KN} synthetic samples")
        
        return identifiable_flags
    
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