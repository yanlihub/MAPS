"""
Sampling Engine for C-MAPS framework.
Implements both SIR and SIR-IC (with identifiability constraints) sampling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class SamplingEngine:
    """
    Unified sampling engine for C-MAPS framework.
    
    Supports two sampling methods:
    1. SIR (Sampling-Importance-Resampling): Based only on fidelity weights
    2. SIR-IC (SIR with Identifiability Constraints): Considers both fidelity and privacy
    """
    
    def __init__(self, random_seed: int = 42, verbose: bool = True):
        """
        Initialize the SamplingEngine.
        
        Args:
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.random_seed = random_seed
        self.verbose = verbose
        np.random.seed(random_seed)
        
    def sir_sampler(self,
                   synthetic_data: Union[pd.DataFrame, np.ndarray],
                   importance_weights: np.ndarray,
                   n_samples: int,
                   method: str = 'weighted') -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, np.ndarray]:
        """
        Sampling-Importance-Resampling (SIR) algorithm.
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Estimated importance weights
            n_samples: Number of samples to draw
            method: Sampling method ('weighted' or 'top_k')
            
        Returns:
            Tuple of (resampled_data, resampled_weights, resampled_indices)
        """
        if self.verbose:
            print(f"Running SIR sampler to select {n_samples} samples using {method} method")
        
        np.random.seed(self.random_seed)
        
        if method == 'weighted':
            # Normalize weights for probability sampling
            normalized_weights = importance_weights / np.sum(importance_weights)
            
            # Sample indices with replacement according to normalized weights
            resampled_indices = np.random.choice(
                len(synthetic_data),
                size=n_samples,
                replace=True,
                p=normalized_weights
            )
            
        elif method == 'top_k':
            # Select top k samples by importance weight
            top_indices = np.argsort(importance_weights)[-n_samples:]
            resampled_indices = top_indices
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'weighted' or 'top_k'.")
        
        # Extract resampled data
        if isinstance(synthetic_data, pd.DataFrame):
            resampled_data = synthetic_data.iloc[resampled_indices].reset_index(drop=True)
        else:
            resampled_data = synthetic_data[resampled_indices]
        
        resampled_weights = importance_weights[resampled_indices]
        
        if self.verbose:
            print(f"SIR sampling complete. Selected {len(resampled_data)} samples.")
            if method == 'weighted':
                print(f"Weight statistics - Min: {np.min(resampled_weights):.4f}, "
                      f"Max: {np.max(resampled_weights):.4f}, "
                      f"Mean: {np.mean(resampled_weights):.4f}")
        
        return resampled_data, resampled_weights, resampled_indices
    
    def sir_ic_sampler(self,
                      synthetic_data: Union[pd.DataFrame, np.ndarray],
                      importance_weights: np.ndarray,
                      identifiability_flags: np.ndarray,
                      n_samples: int,
                      epsilon_identifiability: float = 0.05) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Sampling-Importance-Resampling with Identifiability Constraints (SIR-IC).
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Fidelity importance weights
            identifiability_flags: Binary flags indicating identifiability (0=not identifiable, 1=identifiable)
            n_samples: Desired number of final samples
            epsilon_identifiability: Maximum proportion of identifiable samples allowed
            
        Returns:
            Tuple of (selected_data, selected_weights, selected_flags, selected_indices)
        """
        if self.verbose:
            print(f"Running SIR-IC sampler with epsilon={epsilon_identifiability}")
            print(f"Target: {n_samples} samples with max {epsilon_identifiability:.1%} identifiable")
        
        np.random.seed(self.random_seed)
        
        KN = len(synthetic_data)
        if KN == 0:
            if isinstance(synthetic_data, pd.DataFrame):
                return pd.DataFrame(), np.array([]), np.array([]), np.array([])
            else:
                return np.array([]), np.array([]), np.array([]), np.array([])
        
        selected_indices = []
        selected_weights = []
        selected_flags = []
        
        num_identifiable_selected = 0
        max_identifiable_allowed = int(np.floor(epsilon_identifiability * n_samples))
        
        available_indices = list(range(KN))
        current_weights = np.maximum(importance_weights.copy(), 1e-9)  # Avoid zero weights
        
        if self.verbose:
            print(f"Max identifiable samples allowed: {max_identifiable_allowed}")
        
        for step in range(n_samples):
            if not available_indices:
                if self.verbose:
                    print(f"Warning: Ran out of samples at step {step+1}")
                break
            
            # Compute sampling weights for available samples
            temp_weights = np.zeros(len(available_indices))
            
            for i, idx in enumerate(available_indices):
                # If sample is identifiable and quota is full, assign minimal weight
                if (identifiability_flags[idx] == 1 and 
                    num_identifiable_selected >= max_identifiable_allowed):
                    temp_weights[i] = 1e-9
                else:
                    temp_weights[i] = current_weights[idx]
            
            # Check if we have valid weights
            if np.sum(temp_weights) < 1e-8:
                # Fallback: uniform sampling from eligible samples
                eligible_indices = []
                for i, idx in enumerate(available_indices):
                    if not (identifiability_flags[idx] == 1 and 
                           num_identifiable_selected >= max_identifiable_allowed):
                        eligible_indices.append(i)
                
                if eligible_indices:
                    temp_weights = np.zeros(len(available_indices))
                    for i in eligible_indices:
                        temp_weights[i] = 1.0 / len(eligible_indices)
                else:
                    # No eligible samples - may need to violate constraint
                    if self.verbose:
                        print(f"Warning: May exceed identifiability constraint at step {step+1}")
                    temp_weights = np.ones(len(available_indices)) / len(available_indices)
            
            # Normalize weights
            temp_weights = temp_weights / np.sum(temp_weights)
            
            # Sample one index
            chosen_pool_idx = np.random.choice(len(available_indices), p=temp_weights)
            chosen_original_idx = available_indices[chosen_pool_idx]
            
            # Store selection
            selected_indices.append(chosen_original_idx)
            selected_weights.append(importance_weights[chosen_original_idx])
            selected_flags.append(identifiability_flags[chosen_original_idx])
            
            # Update identifiable count
            if identifiability_flags[chosen_original_idx] == 1:
                num_identifiable_selected += 1
            
            # Remove from available pool
            available_indices.pop(chosen_pool_idx)
            
            # Progress update
            if self.verbose and (step + 1) % max(1, n_samples // 10) == 0:
                print(f"  Selected {step + 1}/{n_samples} samples, "
                      f"identifiable: {num_identifiable_selected}")
        
        # Extract selected data
        if isinstance(synthetic_data, pd.DataFrame):
            selected_data = synthetic_data.iloc[selected_indices].reset_index(drop=True)
        else:
            selected_data = synthetic_data[selected_indices]
        
        selected_weights = np.array(selected_weights)
        selected_flags = np.array(selected_flags)
        selected_indices = np.array(selected_indices)
        
        # Report results
        if self.verbose:
            actual_identifiable_count = np.sum(selected_flags)
            actual_identifiable_ratio = np.mean(selected_flags)
            
            print(f"\nSIR-IC sampling complete:")
            print(f"  Selected samples: {len(selected_data)}")
            print(f"  Identifiable samples: {actual_identifiable_count}/{len(selected_data)}")
            print(f"  Identifiable ratio: {actual_identifiable_ratio:.4f} (target: ≤{epsilon_identifiability:.4f})")
            
            if actual_identifiable_ratio > epsilon_identifiability:
                print(f"  Warning: Identifiability constraint exceeded!")
        
        return selected_data, selected_weights, selected_flags, selected_indices
    
    def sample(self,
               synthetic_data: Union[pd.DataFrame, np.ndarray],
               importance_weights: np.ndarray,
               n_samples: int,
               use_identifiability_constraint: bool = False,
               identifiability_flags: Optional[np.ndarray] = None,
               epsilon_identifiability: float = 0.05,
               method: str = 'weighted') -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Unified sampling interface.
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Importance weights from fidelity classifier
            n_samples: Number of samples to select
            use_identifiability_constraint: Whether to use identifiability constraints
            identifiability_flags: Identifiability flags (required if use_identifiability_constraint=True)
            epsilon_identifiability: Maximum proportion of identifiable samples
            method: Sampling method for SIR ('weighted' or 'top_k')
            
        Returns:
            Tuple of (selected_data, selected_weights, selected_flags, selected_indices)
            Note: selected_flags is None if not using identifiability constraints
        """
        if use_identifiability_constraint:
            if identifiability_flags is None:
                raise ValueError("identifiability_flags must be provided when use_identifiability_constraint=True")
            
            return self.sir_ic_sampler(
                synthetic_data=synthetic_data,
                importance_weights=importance_weights,
                identifiability_flags=identifiability_flags,
                n_samples=n_samples,
                epsilon_identifiability=epsilon_identifiability
            )
        else:
            selected_data, selected_weights, selected_indices = self.sir_sampler(
                synthetic_data=synthetic_data,
                importance_weights=importance_weights,
                n_samples=n_samples,
                method=method
            )
            return selected_data, selected_weights, None, selected_indices
    
    def visualize_sampling_results(self,
                                  original_weights: np.ndarray,
                                  selected_weights: np.ndarray,
                                  original_flags: Optional[np.ndarray] = None,
                                  selected_flags: Optional[np.ndarray] = None):
        """
        Visualize sampling results.
        
        Args:
            original_weights: Original importance weights
            selected_weights: Selected importance weights
            original_flags: Original identifiability flags (optional)
            selected_flags: Selected identifiability flags (optional)
        """
        if original_flags is not None and selected_flags is not None:
            # Plot with identifiability information
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Weight distributions
            axes[0, 0].hist(original_weights, bins=50, alpha=0.7, color='lightblue', label='Original')
            axes[0, 0].hist(selected_weights, bins=50, alpha=0.7, color='darkblue', label='Selected')
            axes[0, 0].set_title('Weight Distributions')
            axes[0, 0].set_xlabel('Importance Weight')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
            
            # Identifiability flags
            flag_data = [
                ('Original', original_flags),
                ('Selected', selected_flags)
            ]
            
            for i, (label, flags) in enumerate(flag_data):
                identifiable_count = np.sum(flags)
                total_count = len(flags)
                axes[0, 1].bar([f'{label}\nNot Identifiable', f'{label}\nIdentifiable'], 
                              [total_count - identifiable_count, identifiable_count],
                              alpha=0.7, 
                              color=['steelblue', 'orangered'][i % 2])
            
            axes[0, 1].set_title('Identifiability Distribution')
            axes[0, 1].set_ylabel('Count')
            
            # Weight by identifiability for original data
            orig_not_ident = original_weights[original_flags == 0]
            orig_ident = original_weights[original_flags == 1]
            
            axes[1, 0].hist(orig_not_ident, bins=30, alpha=0.7, color='steelblue', label='Not Identifiable')
            axes[1, 0].hist(orig_ident, bins=30, alpha=0.7, color='orangered', label='Identifiable')
            axes[1, 0].set_title('Original: Weights by Identifiability')
            axes[1, 0].set_xlabel('Importance Weight')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
            
            # Weight by identifiability for selected data
            sel_not_ident = selected_weights[selected_flags == 0]
            sel_ident = selected_weights[selected_flags == 1]
            
            if len(sel_not_ident) > 0:
                axes[1, 1].hist(sel_not_ident, bins=30, alpha=0.7, color='steelblue', label='Not Identifiable')
            if len(sel_ident) > 0:
                axes[1, 1].hist(sel_ident, bins=30, alpha=0.7, color='orangered', label='Identifiable')
            
            axes[1, 1].set_title('Selected: Weights by Identifiability')
            axes[1, 1].set_xlabel('Importance Weight')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')
            
        else:
            # Simple weight comparison plot
            plt.figure(figsize=(10, 6))
            plt.hist(original_weights, bins=50, alpha=0.7, color='lightblue', label='Original')
            plt.hist(selected_weights, bins=50, alpha=0.7, color='darkblue', label='Selected')
            plt.title('Importance Weight Distributions')
            plt.xlabel('Importance Weight')
            plt.ylabel('Frequency')
            plt.legend()
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()