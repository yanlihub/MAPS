"""
Sampling Engine for C-MAPS framework.
Implements both SIR and SIR-IC (with identifiability constraints) sampling.
Enhanced with configurable weight processing methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class SamplingEngine:
    """
    Unified sampling engine for C-MAPS framework.
    
    Supports two sampling methods:
    1. SIR (Sampling-Importance-Resampling): Based only on fidelity weights
    2. SIR-IC (SIR with Identifiability Constraints): Considers both fidelity and privacy
    
    Enhanced with configurable weight processing methods to reduce variance.
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
    
    def _process_importance_weights(self, 
                                   importance_weights: np.ndarray,
                                   weight_processing: str = 'raw',
                                   alpha: float = 1.0,
                                   min_clip: float = 1e-9) -> np.ndarray:
        """
        Process importance weights according to specified method.
        
        Args:
            importance_weights: Original importance weights
            weight_processing: Processing method ('raw', 'flatten', 'clipped')
            alpha: Alpha parameter for flatten method (importance_weights ** alpha)
            min_clip: Minimum clip value for clipped method
            
        Returns:
            Processed importance weights
        """
        if weight_processing == 'raw':
            processed_weights = importance_weights.copy()
        elif weight_processing == 'flatten':
            processed_weights = importance_weights ** alpha
        elif weight_processing == 'clipped':
            processed_weights = np.maximum(importance_weights, min_clip)
        else:
            raise ValueError(f"Unknown weight_processing method: {weight_processing}. "
                           "Use 'raw', 'flatten', or 'clipped'.")
        
        if self.verbose:
            print(f"Weight processing: {weight_processing}")
            if weight_processing == 'flatten':
                print(f"  Alpha parameter: {alpha}")
            elif weight_processing == 'clipped':
                print(f"  Min clip value: {min_clip}")
            print(f"  Original weights - Min: {np.min(importance_weights):.6f}, "
                  f"Max: {np.max(importance_weights):.6f}, "
                  f"Std: {np.std(importance_weights):.6f}, "
                  f"Mean: {np.mean(importance_weights):.6f}")
            print(f"  Processed weights - Min: {np.min(processed_weights):.6f}, "
                  f"Max: {np.max(processed_weights):.6f}, "
                  f"Std: {np.std(processed_weights):.6f}, "
                  f"Mean: {np.mean(processed_weights):.6f}")
        
        return processed_weights
        
    def sir_sampler(self,
                synthetic_data: Union[pd.DataFrame, np.ndarray],
                importance_weights: np.ndarray,
                n_samples: int,
                method: str = 'weighted',
                identifiability_flags: Optional[np.ndarray] = None,
                weight_processing: str = 'raw',
                alpha: float = 1.0,
                min_clip: float = 1e-9) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Sampling-Importance-Resampling (SIR) algorithm.
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Estimated importance weights
            n_samples: Number of samples to draw
            method: Sampling method ('weighted', 'top_k', 'reverse_weighted', 'reverse_top_k')
                - 'weighted': Sample with probability proportional to weights
                - 'top_k': Select k samples with highest weights
                - 'reverse_weighted': Sample with probability inversely proportional to weights
                - 'reverse_top_k': Select k samples with lowest weights
            identifiability_flags: Optional identifiability flags for tracking privacy metrics
            weight_processing: Weight processing method ('raw', 'flatten', 'clipped')
            alpha: Alpha parameter for flatten processing
            min_clip: Minimum clip value for clipped processing
            
        Returns:
            Tuple of (resampled_data, resampled_weights, resampled_indices, stats_dict)
        """
        if self.verbose:
            print(f"Running SIR sampler to select {n_samples} samples using {method} method")
        
        np.random.seed(self.random_seed)
        
        processed_weights = self._process_importance_weights(
            importance_weights, weight_processing, alpha, min_clip
        )
        
        if method == 'weighted':
            # Normalize weights for probability sampling
            normalized_weights = processed_weights / np.sum(processed_weights)
            
            # Sample indices with replacement according to normalized weights
            resampled_indices = np.random.choice(
                len(synthetic_data),
                size=n_samples,
                replace=True,
                p=normalized_weights
            )
            
        elif method == 'reverse_weighted':
            # Invert weights for reverse sampling (low weight = high probability)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            inverse_weights = 1.0 / (processed_weights + epsilon)
            
            # Normalize inverse weights for probability sampling
            normalized_inverse_weights = inverse_weights / np.sum(inverse_weights)
            
            # Sample indices with replacement according to normalized inverse weights
            resampled_indices = np.random.choice(
                len(synthetic_data),
                size=n_samples,
                replace=True,
                p=normalized_inverse_weights
            )
            
            if self.verbose:
                print(f"  Using inverse weights - Min: {np.min(inverse_weights):.6f}, "
                      f"Max: {np.max(inverse_weights):.6f}, "
                      f"Mean: {np.mean(inverse_weights):.6f}")
            
        elif method == 'top_k':
            # Select top k samples by processed importance weight
            top_indices = np.argsort(processed_weights)[-n_samples:]
            resampled_indices = top_indices
            
        elif method == 'reverse_top_k':
            # Select bottom k samples by processed importance weight (lowest weights)
            bottom_indices = np.argsort(processed_weights)[:n_samples]
            resampled_indices = bottom_indices
            
            if self.verbose:
                print(f"  Selecting {n_samples} samples with lowest weights")
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'weighted', 'top_k', 'reverse_weighted', or 'reverse_top_k'.")
        
        # Extract resampled data
        if isinstance(synthetic_data, pd.DataFrame):
            resampled_data = synthetic_data.iloc[resampled_indices].reset_index(drop=True)
        else:
            resampled_data = synthetic_data[resampled_indices]
        
        resampled_weights = importance_weights[resampled_indices]  # Return original weights
        
        stats_dict = {
            'method': method,
            'weight_processing': weight_processing,
            'selected_samples': len(resampled_data)
        }
        
        if identifiability_flags is not None:
            resampled_flags = identifiability_flags[resampled_indices]
            identifiable_count = np.sum(resampled_flags)
            identifiable_percentage = identifiable_count / len(resampled_flags)
            
            stats_dict.update({
                'identifiable_samples': identifiable_count,
                'identifiable_percentage': identifiable_percentage,
                'resampled_flags': resampled_flags
            })
        
        if self.verbose:
            print(f"SIR sampling complete. Selected {len(resampled_data)} samples.")
            # Show weight statistics for both methods
            print(f"Weight statistics - Min: {np.min(resampled_weights):.4f}, "
                f"Max: {np.max(resampled_weights):.4f}, "
                f"Mean: {np.mean(resampled_weights):.4f}")
            
            if identifiability_flags is not None:
                print(f"Identifiable samples: {stats_dict['identifiable_samples']}/{len(resampled_data)} "
                    f"({stats_dict['identifiable_percentage']:.2%})")
        
        return resampled_data, resampled_weights, resampled_indices, stats_dict

    def sir_ic_sampler(self,
                    synthetic_data: Union[pd.DataFrame, np.ndarray],
                    importance_weights: np.ndarray,
                    identifiability_flags: np.ndarray,
                    n_samples: int,
                    epsilon_identifiability: float = 0.05,
                    weight_processing: str = 'raw',
                    alpha: float = 1.0,
                    min_clip: float = 1e-9) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Improved Sampling-Importance-Resampling with Identifiability Constraints (SIR-IC).
        
        This version uses a budget-based approach:
        1. Calculate maximum identifiable samples allowed
        2. Sample with replacement based on importance weights from all samples
        3. Once identifiable budget is exhausted, only sample from non-identifiable samples
        4. Maintains natural importance sampling properties throughout
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Fidelity importance weights
            identifiability_flags: Binary flags indicating identifiability (0=not identifiable, 1=identifiable)
            n_samples: Desired number of final samples
            epsilon_identifiability: Maximum proportion of identifiable samples allowed
            weight_processing: Weight processing method ('raw', 'flatten', 'clipped')
            alpha: Alpha parameter for flatten processing
            min_clip: Minimum clip value for clipped processing
            
        Returns:
            Tuple of (selected_data, selected_weights, selected_flags, selected_indices, stats_dict)
        """
        if self.verbose:
            print(f"Running improved SIR-IC sampler with epsilon={epsilon_identifiability}")
            print(f"Target: {n_samples} samples with max {epsilon_identifiability:.1%} identifiable")
        
        np.random.seed(self.random_seed)
        
        processed_weights = self._process_importance_weights(
            importance_weights, weight_processing, alpha, min_clip
        )
        
        KN = len(synthetic_data)
        if KN == 0:
            empty_stats = {
                'method': 'sir_ic_improved',
                'weight_processing': weight_processing,
                'epsilon_identifiability': epsilon_identifiability,
                'selected_samples': 0,
                'identifiable_samples': 0,
                'identifiable_percentage': 0.0
            }
            if isinstance(synthetic_data, pd.DataFrame):
                return pd.DataFrame(), np.array([]), np.array([]), np.array([]), empty_stats
            else:
                return np.array([]), np.array([]), np.array([]), np.array([]), empty_stats
        
        # Calculate identifiable budget
        max_identifiable_allowed = int(np.floor(epsilon_identifiability * n_samples))
        
        # Prepare sampling pools
        identifiable_indices = np.where(identifiability_flags == 1)[0]
        non_identifiable_indices = np.where(identifiability_flags == 0)[0]
        
        # Extract weights for each pool
        if len(identifiable_indices) > 0:
            identifiable_weights = processed_weights[identifiable_indices]
            identifiable_weights_normalized = identifiable_weights / np.sum(identifiable_weights)
        else:
            identifiable_weights_normalized = np.array([])
        
        if len(non_identifiable_indices) > 0:
            non_identifiable_weights = processed_weights[non_identifiable_indices]
            non_identifiable_weights_normalized = non_identifiable_weights / np.sum(non_identifiable_weights)
        else:
            non_identifiable_weights_normalized = np.array([])
        
        # Prepare all samples weights for initial sampling
        all_weights_normalized = processed_weights / np.sum(processed_weights)
        
        selected_indices = []
        num_identifiable_selected = 0
        
        if self.verbose:
            print(f"Max identifiable samples allowed: {max_identifiable_allowed}")
            print(f"Identifiable pool size: {len(identifiable_indices)}")
            print(f"Non-identifiable pool size: {len(non_identifiable_indices)}")
        
        for step in range(n_samples):
            # Decide which pool to sample from
            if num_identifiable_selected < max_identifiable_allowed and len(identifiable_indices) > 0:
                # Budget not exhausted, sample from all samples
                chosen_idx = np.random.choice(KN, p=all_weights_normalized)
            else:
                # Budget exhausted or no identifiable samples, sample only from non-identifiable
                if len(non_identifiable_indices) == 0:
                    if self.verbose:
                        print(f"Warning: No non-identifiable samples available at step {step+1}")
                    # Fallback: must sample from identifiable (will violate constraint)
                    if len(identifiable_indices) > 0:
                        chosen_idx = np.random.choice(identifiable_indices, p=identifiable_weights_normalized)
                    else:
                        # This should not happen in practice
                        chosen_idx = np.random.choice(KN)
                else:
                    # Sample from non-identifiable pool using their importance weights
                    chosen_pool_idx = np.random.choice(len(non_identifiable_indices), p=non_identifiable_weights_normalized)
                    chosen_idx = non_identifiable_indices[chosen_pool_idx]
            
            # Record selection
            selected_indices.append(chosen_idx)
            
            # Update identifiable count
            if identifiability_flags[chosen_idx] == 1:
                num_identifiable_selected += 1
            
            # Progress update
            if self.verbose and (step + 1) % max(1, n_samples // 10) == 0:
                print(f"  Selected {step + 1}/{n_samples} samples, "
                    f"identifiable: {num_identifiable_selected}")
        
        # Convert to arrays
        selected_indices = np.array(selected_indices)
        
        # Extract selected data
        if isinstance(synthetic_data, pd.DataFrame):
            selected_data = synthetic_data.iloc[selected_indices].reset_index(drop=True)
        else:
            selected_data = synthetic_data[selected_indices]
        
        selected_weights = importance_weights[selected_indices]  # Return original weights
        selected_flags = identifiability_flags[selected_indices]
        
        actual_identifiable_count = np.sum(selected_flags)
        actual_identifiable_ratio = actual_identifiable_count / len(selected_flags)
        
        stats_dict = {
            'method': 'sir_ic_improved',
            'weight_processing': weight_processing,
            'epsilon_identifiability': epsilon_identifiability,
            'max_identifiable_allowed': max_identifiable_allowed,
            'selected_samples': len(selected_data),
            'identifiable_samples': actual_identifiable_count,
            'identifiable_percentage': actual_identifiable_ratio,
            'constraint_violated': actual_identifiable_ratio > epsilon_identifiability,
            'identifiable_pool_size': len(identifiable_indices),
            'non_identifiable_pool_size': len(non_identifiable_indices)
        }
        
        # Report results
        if self.verbose:
            print(f"\nImproved SIR-IC sampling complete:")
            print(f"  Selected samples: {len(selected_data)}")
            print(f"  Identifiable samples: {actual_identifiable_count}/{len(selected_data)}")
            print(f"  Identifiable ratio: {actual_identifiable_ratio:.4f} (target: ≤{epsilon_identifiability:.4f})")
            print(f"  Weight statistics - Min: {np.min(selected_weights):.4f}, "
                f"Max: {np.max(selected_weights):.4f}, "
                f"Mean: {np.mean(selected_weights):.4f}")
            
            if actual_identifiable_ratio > epsilon_identifiability:
                print(f"  Warning: Identifiability constraint exceeded!")
                print(f"  This may happen when non-identifiable pool is insufficient")
        
        return selected_data, selected_weights, selected_flags, selected_indices, stats_dict
    
    def sample(self,
               synthetic_data: Union[pd.DataFrame, np.ndarray],
               importance_weights: np.ndarray,
               n_samples: int,
               use_identifiability_constraint: bool = False,
               identifiability_flags: Optional[np.ndarray] = None,
               epsilon_identifiability: float = 0.05,
               method: str = 'weighted',
               weight_processing: str = 'raw',
               alpha: float = 1.0,
               min_clip: float = 1e-9) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
        """
        Unified sampling interface with configurable weight processing.
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Importance weights from fidelity classifier
            n_samples: Number of samples to select
            use_identifiability_constraint: Whether to use identifiability constraints
            identifiability_flags: Identifiability flags (required if use_identifiability_constraint=True)
            epsilon_identifiability: Maximum proportion of identifiable samples
            method: Sampling method for SIR ('weighted', 'top_k', 'reverse_weighted', 'reverse_top_k')
            weight_processing: Weight processing method ('raw', 'flatten', 'clipped')
            alpha: Alpha parameter for flatten processing (importance_weights ** alpha)
            min_clip: Minimum clip value for clipped processing
            
        Returns:
            Tuple of (selected_data, selected_weights, selected_flags, selected_indices, stats_dict)
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
                epsilon_identifiability=epsilon_identifiability,
                weight_processing=weight_processing,
                alpha=alpha,
                min_clip=min_clip
            )
        else:
            selected_data, selected_weights, selected_indices, stats_dict = self.sir_sampler(
                synthetic_data=synthetic_data,
                importance_weights=importance_weights,
                n_samples=n_samples,
                method=method,
                identifiability_flags=identifiability_flags,
                weight_processing=weight_processing,
                alpha=alpha,
                min_clip=min_clip
            )
            selected_flags = stats_dict.get('resampled_flags', None)
            return selected_data, selected_weights, selected_flags, selected_indices, stats_dict
    
    def visualize_sampling_results(self,
                                  original_weights: np.ndarray,
                                  selected_weights: np.ndarray,
                                  original_flags: Optional[np.ndarray] = None,
                                  selected_flags: Optional[np.ndarray] = None,
                                  stats_dict: Optional[Dict[str, Any]] = None):
        """
        Visualize sampling results with enhanced statistics display.
        
        Args:
            original_weights: Original importance weights
            selected_weights: Selected importance weights
            original_flags: Original identifiability flags (optional)
            selected_flags: Selected identifiability flags (optional)
            stats_dict: Statistics dictionary from sampling (optional)
        """
        if stats_dict and self.verbose:
            print("\n" + "="*50)
            print("SAMPLING STATISTICS")
            print("="*50)
            for key, value in stats_dict.items():
                if key != 'resampled_flags':  # Skip array data
                    print(f"{key}: {value}")
            print("="*50)
        
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
    
    def analyze_weight_processing_effects(self,
                                        importance_weights: np.ndarray,
                                        alpha_values: list = [0.5, 0.8, 1.0, 1.2, 1.5],
                                        min_clip_values: list = [1e-6, 1e-4, 1e-2, 0.1]):
        """
        Analyze the effects of different weight processing methods.
        
        Args:
            importance_weights: Original importance weights
            alpha_values: List of alpha values to test for flatten method
            min_clip_values: List of min_clip values to test for clipped method
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original weights
        axes[0, 0].hist(importance_weights, bins=50, alpha=0.7, color='gray', label='Original')
        axes[0, 0].set_title('Original Weights')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        
        # Flatten method effects
        for alpha in alpha_values:
            flattened = importance_weights ** alpha
            axes[0, 1].hist(flattened, bins=50, alpha=0.5, label=f'α={alpha}')
        axes[0, 1].set_title('Flatten Method Effects')
        axes[0, 1].set_xlabel('Weight Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        
        # Clipped method effects
        for min_clip in min_clip_values:
            clipped = np.maximum(importance_weights, min_clip)
            axes[1, 0].hist(clipped, bins=50, alpha=0.5, label=f'clip={min_clip}')
        axes[1, 0].set_title('Clipped Method Effects')
        axes[1, 0].set_xlabel('Weight Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        
        # Variance comparison
        methods = ['Original']
        variances = [np.var(importance_weights)]
        
        for alpha in alpha_values:
            methods.append(f'Flatten α={alpha}')
            variances.append(np.var(importance_weights ** alpha))
        
        for min_clip in min_clip_values:
            methods.append(f'Clip {min_clip}')
            variances.append(np.var(np.maximum(importance_weights, min_clip)))
        
        axes[1, 1].bar(range(len(variances)), variances)
        axes[1, 1].set_title('Weight Variance Comparison')
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].set_xticks(range(len(methods)))
        axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        if self.verbose:
            print("Weight Processing Analysis:")
            print(f"Original - Mean: {np.mean(importance_weights):.4f}, Std: {np.std(importance_weights):.4f}")
            for alpha in alpha_values:
                flattened = importance_weights ** alpha
                print(f"Flatten α={alpha} - Mean: {np.mean(flattened):.4f}, Std: {np.std(flattened):.4f}")
            for min_clip in min_clip_values:
                clipped = np.maximum(importance_weights, min_clip)
                print(f"Clipped {min_clip} - Mean: {np.mean(clipped):.4f}, Std: {np.std(clipped):.4f}")