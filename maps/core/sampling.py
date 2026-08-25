"""
Sampling Engine for MAPS framework.
Implements both Standard SIR and DP-SIR (Differentially Private SIR) sampling.
Enhanced with configurable weight processing methods and identifiability tracking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class SamplingEngine:
    """
    Unified sampling engine for MAPS framework.
    
    Supports two sampling methods:
    1. SIR (Sampling-Importance-Resampling): Standard importance sampling
    2. DP-SIR (Differentially Private SIR): Uses exponential mechanism for privacy
    
    Enhanced with configurable weight processing methods to reduce variance and
    identifiability constraint tracking for privacy analysis.
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
    
    def _analyze_identifiability(self, 
                                identifiability_flags: Optional[np.ndarray],
                                selected_indices: np.ndarray) -> Dict[str, Any]:
        """
        Analyze identifiability statistics for the dataset and selected samples.
        
        Args:
            identifiability_flags: Boolean array where True means identifiable (privacy violation)
            selected_indices: Indices of selected samples
            
        Returns:
            Dictionary with identifiability statistics
        """
        identifiability_stats = {}
        
        if identifiability_flags is not None:
            # Overall dataset statistics
            total_samples = len(identifiability_flags)
            identifiable_samples = np.sum(identifiability_flags)
            identifiable_ratio = identifiable_samples / total_samples
            
            # Selected samples statistics
            selected_flags = identifiability_flags[selected_indices]
            selected_identifiable = np.sum(selected_flags)
            selected_identifiable_ratio = selected_identifiable / len(selected_indices)
            
            identifiability_stats = {
                'total_samples': total_samples,
                'identifiable_samples': identifiable_samples,
                'identifiable_ratio': identifiable_ratio,
                'selected_samples': len(selected_indices),
                'selected_identifiable_samples': selected_identifiable,
                'selected_identifiable_ratio': selected_identifiable_ratio,
                'privacy_preserved_samples': total_samples - identifiable_samples,
                'privacy_preserved_ratio': 1 - identifiable_ratio,
                'selected_privacy_preserved_samples': len(selected_indices) - selected_identifiable,
                'selected_privacy_preserved_ratio': 1 - selected_identifiable_ratio
            }
            
            if self.verbose:
                print(f"Identifiability Analysis:")
                print(f"  Total samples: {total_samples}")
                print(f"  Identifiable samples (privacy violations): {identifiable_samples} ({identifiable_ratio:.2%})")
                print(f"  Privacy-preserved samples: {total_samples - identifiable_samples} ({1-identifiable_ratio:.2%})")
                print(f"  Selected identifiable samples: {selected_identifiable} ({selected_identifiable_ratio:.2%})")
                print(f"  Selected privacy-preserved samples: {len(selected_indices) - selected_identifiable} ({1-selected_identifiable_ratio:.2%})")
        
        return identifiability_stats
        
    def sir_sampler(self,
                   synthetic_data: Union[pd.DataFrame, np.ndarray],
                   importance_weights: np.ndarray,
                   n_samples: int,
                   method: str = 'weighted',
                   weight_processing: str = 'raw',
                   alpha: float = 1.0,
                   min_clip: float = 1e-9,
                   privacy_alpha: Optional[float] = None,
                   identifiability_flags: Optional[np.ndarray] = None,
                   replacement: bool = True) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
        """
        Sampling-Importance-Resampling (SIR) algorithm with optional Differential Privacy.
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Estimated importance weights from fidelity classifier
            n_samples: Number of samples to draw
            method: Sampling method ('weighted', 'top_k', 'reverse_weighted', 'reverse_top_k')
                - 'weighted': Sample with probability proportional to weights
                - 'top_k': Select k samples with highest weights
                - 'reverse_weighted': Sample with probability inversely proportional to weights
                - 'reverse_top_k': Select k samples with lowest weights
            weight_processing: Weight processing method ('raw', 'flatten', 'clipped')
            alpha: Alpha parameter for flatten processing
            min_clip: Minimum clip value for clipped processing
            privacy_alpha: If provided, use DP-SIR with this privacy parameter
                         Larger alpha = less privacy, None = standard SIR
            identifiability_flags: Optional boolean array indicating which samples violate 
                                 identifiability constraints (True = identifiable/violation)
            replacement: If True (default), sample with replacement, matching the SIR
                         formulation; if False, sample without replacement (breaks the
                         i.i.d. assumption -- see Appendix B.5 of the paper)
            
        Returns:
            Tuple of (resampled_data, resampled_weights, resampled_flags, resampled_indices, stats_dict)
        """
        # Determine sampling mode
        is_dp_sir = privacy_alpha is not None
        sampling_mode = "DP-SIR" if is_dp_sir else "Standard SIR"
        
        if self.verbose:
            print(f"Running {sampling_mode} sampler to select {n_samples} samples using {method} method")
            if is_dp_sir:
                print(f"Privacy parameter α = {privacy_alpha}")
        
        np.random.seed(self.random_seed)
        
        # Process importance weights
        processed_weights = self._process_importance_weights(
            importance_weights, weight_processing, alpha, min_clip
        )
        
        # Perform sampling based on method
        if is_dp_sir:
            # DP-SIR: Apply exponential mechanism for each sample draw
            # Step 1: Normalize weights
            w_tilde = processed_weights / np.sum(processed_weights)
            
            # Step 2: Calculate privacy parameter ε = α/(2N)
            epsilon = privacy_alpha / (2 * n_samples)
            
            # Calculate privacy guarantee
            privacy_guarantee = privacy_alpha / (2 * n_samples)
            
            if self.verbose:
                print(f"  ε = {epsilon:.6f}")
                print(f"  Privacy guarantee: ({privacy_guarantee:.6f})-differential privacy")
            
            # DP-SIR sampling: Apply exponential mechanism for each sample
            resampled_indices = []
            available_indices = np.arange(len(synthetic_data))
            remaining_weights = w_tilde.copy()
            
            for i in range(n_samples):
                if len(available_indices) == 0:
                    break
                
                # Apply exponential mechanism for this sample
                log_weights = epsilon * remaining_weights
                max_log_weight = np.max(log_weights)  # For numerical stability
                stabilized_log_weights = log_weights - max_log_weight
                exp_weights = np.exp(stabilized_log_weights)
                sampling_probs = exp_weights / np.sum(exp_weights)
                
                # Avoid zeros that break sampling without replacement by clipping and renormalizing
                sampling_probs = np.maximum(sampling_probs, min_clip)
                sampling_probs = sampling_probs / np.sum(sampling_probs)
                
                # Sample one index
                if method == 'weighted':
                    chosen_idx = np.random.choice(len(available_indices), p=sampling_probs)
                elif method == 'top_k':
                    chosen_idx = np.argmax(sampling_probs)
                elif method == 'reverse_weighted':
                    reverse_probs = (1 / sampling_probs)
                    reverse_probs = reverse_probs / np.sum(reverse_probs)
                    reverse_probs = np.maximum(reverse_probs, min_clip)
                    reverse_probs = reverse_probs / np.sum(reverse_probs)
                    chosen_idx = np.random.choice(len(available_indices), p=reverse_probs)
                elif method == 'reverse_top_k':
                    chosen_idx = np.argmin(sampling_probs)
                else:
                    raise ValueError(f"Unknown method: {method}. "
                                   "Use 'weighted', 'top_k', 'reverse_weighted', or 'reverse_top_k'.")
                
                # Add the chosen index to results
                actual_idx = available_indices[chosen_idx]
                resampled_indices.append(actual_idx)
                
                # Update for next iteration (if sampling without replacement)
                if not replacement:
                    available_indices = np.delete(available_indices, chosen_idx)
                    remaining_weights = np.delete(remaining_weights, chosen_idx)
                    if len(remaining_weights) > 0:
                        remaining_weights = remaining_weights / np.sum(remaining_weights)
            
            resampled_indices = np.array(resampled_indices)
            
            # Store the final sampling probabilities for the last sample (for compatibility)
            if len(remaining_weights) > 0:
                log_weights = epsilon * remaining_weights
                max_log_weight = np.max(log_weights)
                stabilized_log_weights = log_weights - max_log_weight
                exp_weights = np.exp(stabilized_log_weights)
                self.sampling_probs = exp_weights / np.sum(exp_weights)
            else:
                self.sampling_probs = np.ones(len(synthetic_data)) / len(synthetic_data)
                
        else:
            # Standard SIR: Direct normalization
            sampling_probs = processed_weights / np.sum(processed_weights)
            # Avoid zeros that break sampling without replacement by clipping and renormalizing
            sampling_probs = np.maximum(sampling_probs, min_clip)
            sampling_probs = sampling_probs / np.sum(sampling_probs)
            # save the sampling_probs to a atribute of the class
            self.sampling_probs = sampling_probs
            privacy_guarantee = None
            
            # Perform standard sampling
            if method == 'weighted':
                resampled_indices = np.random.choice(
                    len(synthetic_data), 
                    size=n_samples, 
                    p=sampling_probs, 
                    replace=replacement
                )
            elif method == 'top_k':
                resampled_indices = np.argsort(sampling_probs)[-n_samples:]
            elif method == 'reverse_weighted':
                reverse_probs = (1 / sampling_probs)
                reverse_probs = reverse_probs / np.sum(reverse_probs)
                resampled_indices = np.random.choice(
                    len(synthetic_data), 
                    size=n_samples, 
                    p=reverse_probs, 
                    replace=replacement
                )
            elif method == 'reverse_top_k':
                resampled_indices = np.argsort(sampling_probs)[:n_samples]
            else:
                raise ValueError(f"Unknown method: {method}. "
                               "Use 'weighted', 'top_k', 'reverse_weighted', or 'reverse_top_k'.")
        
        # Extract resampled data
        if isinstance(synthetic_data, pd.DataFrame):
            resampled_data = synthetic_data.iloc[resampled_indices].reset_index(drop=True)
        else:
            resampled_data = synthetic_data[resampled_indices]
        
        resampled_weights = importance_weights[resampled_indices]  # Return original weights
        
        # Extract resampled identifiability flags if provided
        resampled_flags = None
        if identifiability_flags is not None:
            resampled_flags = identifiability_flags[resampled_indices]
        
        # Analyze identifiability if flags provided
        identifiability_stats = self._analyze_identifiability(identifiability_flags, resampled_indices)
        
        # Duplicate statistics. With replacement=True a record may be drawn more than
        # once, so the refined dataset is a multiset. Reported in Appendix B.5.
        n_unique = int(len(np.unique(resampled_indices)))
        duplicate_rate = 1.0 - n_unique / len(resampled_indices) if len(resampled_indices) else 0.0

        # Effective sample size of the selection distribution, ESS = 1 / sum(p^2). This is
        # what determines how large a pool has to be: if the weights concentrate, a pool of
        # M records behaves like one of ESS records, and the draw starts repeating itself.
        # It is the quantitative form of the pool-size requirement.
        probs = getattr(self, "sampling_probs", None)
        if probs is not None and len(probs):
            p_norm = np.asarray(probs, dtype=float)
            p_norm = p_norm / p_norm.sum()
            ess = float(1.0 / np.sum(p_norm**2))
            ess_fraction = ess / len(p_norm)
            top1 = float(np.sort(p_norm)[::-1][: max(1, len(p_norm) // 100)].sum())
        else:
            ess = ess_fraction = top1 = float("nan")

        # Create statistics dictionary
        stats_dict = {
            'method': method,
            'sampling_mode': sampling_mode,
            'weight_processing': weight_processing,
            'alpha': alpha,
            'replacement': replacement,
            'selected_samples': len(resampled_data),
            'n_unique_samples': n_unique,
            'duplicate_rate': duplicate_rate,
            'effective_sample_size': ess,
            'ess_fraction_of_pool': ess_fraction,
            'prob_mass_top_1pct': top1,
            'pool_size': int(len(synthetic_data)),
            'privacy_alpha': privacy_alpha,
            'privacy_guarantee': privacy_guarantee,
            'identifiability_stats': identifiability_stats
        }
        
        if is_dp_sir:
            stats_dict.update({
                'epsilon': epsilon,
                'differential_privacy': f"({privacy_guarantee:.6f})-DP"
            })
        
        if self.verbose:
            print(f"{sampling_mode} sampling complete. Selected {len(resampled_data)} samples "
                  f"({n_unique} unique, duplicate rate {duplicate_rate:.3f}, "
                  f"ESS {ess:.0f} = {100 * ess_fraction:.1f}% of pool, "
                  f"replacement={replacement}).")
            print(f"Weight statistics - Min: {np.min(resampled_weights):.4f}, "
                  f"Max: {np.max(resampled_weights):.4f}, "
                  f"Mean: {np.mean(resampled_weights):.4f}")
            
            if is_dp_sir:
                print(f"Privacy guarantee: {stats_dict['differential_privacy']}")
        
        return resampled_data, resampled_weights, resampled_flags, resampled_indices, stats_dict
    
    def sample(self,
               synthetic_data: Union[pd.DataFrame, np.ndarray],
               importance_weights: np.ndarray,
               n_samples: int,
               method: str = 'weighted',
               weight_processing: str = 'raw',
               alpha: float = 1.0,
               min_clip: float = 1e-9,
               privacy_alpha: Optional[float] = None,
               identifiability_flags: Optional[np.ndarray] = None,
               replacement: bool = True) -> Tuple[Union[pd.DataFrame, np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
        """
        Unified sampling interface for both Standard SIR and DP-SIR.
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Importance weights from fidelity classifier
            n_samples: Number of samples to select
            method: Sampling method ('weighted', 'top_k', 'reverse_weighted', 'reverse_top_k')
            weight_processing: Weight processing method ('raw', 'flatten', 'clipped')
            alpha: Alpha parameter for flatten processing (importance_weights ** alpha)
            min_clip: Minimum clip value for clipped processing
            privacy_alpha: Privacy parameter for DP-SIR (None = standard SIR, 
                          larger values = less privacy)
            identifiability_flags: Optional boolean array indicating which samples violate 
                                 identifiability constraints (True = identifiable/violation)
            replacement: If True (default), sample with replacement, matching the SIR
                         formulation; if False, sample without replacement (breaks the
                         i.i.d. assumption -- see Appendix B.5 of the paper)
            
        Returns:
            Tuple of (selected_data, selected_weights, selected_flags, selected_indices, stats_dict)
        """
        return self.sir_sampler(
            synthetic_data=synthetic_data,
            importance_weights=importance_weights,
            n_samples=n_samples,
            method=method,
            weight_processing=weight_processing,
            alpha=alpha,
            min_clip=min_clip,
            privacy_alpha=privacy_alpha,
            identifiability_flags=identifiability_flags,
            replacement=replacement
        )
    
    def visualize_sampling_results(self,
                                  original_weights: np.ndarray,
                                  selected_weights: np.ndarray,
                                  stats_dict: Optional[Dict[str, Any]] = None):
        """
        Visualize sampling results with enhanced statistics display including identifiability analysis.
        
        Args:
            original_weights: Original importance weights
            selected_weights: Selected importance weights after sampling
            stats_dict: Statistics dictionary from sampling
        """
        # Determine if we have identifiability stats
        has_identifiability = (stats_dict is not None and 
                              stats_dict.get('identifiability_stats') and 
                              len(stats_dict['identifiability_stats']) > 0)
        
        if has_identifiability:
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Original weight distribution
        axes[0, 0].hist(original_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Original Importance Weights Distribution')
        axes[0, 0].set_xlabel('Importance Weight')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Selected weight distribution
        axes[0, 1].hist(selected_weights, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Selected Importance Weights Distribution')
        axes[0, 1].set_xlabel('Importance Weight')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Weight comparison (box plots)
        data_to_plot = [original_weights, selected_weights]
        axes[1, 0].boxplot(data_to_plot, labels=['Original', 'Selected'])
        axes[1, 0].set_title('Weight Distribution Comparison')
        axes[1, 0].set_ylabel('Importance Weight')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative distributions
        original_sorted = np.sort(original_weights)
        selected_sorted = np.sort(selected_weights)
        axes[1, 1].plot(original_sorted, np.linspace(0, 1, len(original_sorted)), 
                       label='Original', linewidth=2)
        axes[1, 1].plot(selected_sorted, np.linspace(0, 1, len(selected_sorted)), 
                       label='Selected', linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution Comparison')
        axes[1, 1].set_xlabel('Importance Weight')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Additional identifiability plots if available
        if has_identifiability:
            id_stats = stats_dict['identifiability_stats']
            
            # Plot 5: Identifiability comparison (pie charts)
            labels = ['Privacy Preserved', 'Identifiable']
            original_sizes = [id_stats['privacy_preserved_samples'], id_stats['identifiable_samples']]
            selected_sizes = [id_stats['selected_privacy_preserved_samples'], id_stats['selected_identifiable_samples']]
            colors = ['lightgreen', 'lightcoral']
            
            axes[2, 0].pie(original_sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[2, 0].set_title('Original Dataset: Privacy vs Identifiability')
            
            axes[2, 1].pie(selected_sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[2, 1].set_title('Selected Samples: Privacy vs Identifiability')
        
        plt.tight_layout()
        
        # Print summary statistics
        if stats_dict:
            print("\n" + "="*60)
            print("SAMPLING RESULTS SUMMARY")
            print("="*60)
            print(f"Sampling Mode: {stats_dict.get('sampling_mode', 'Unknown')}")
            print(f"Method: {stats_dict.get('method', 'Unknown')}")
            print(f"Weight Processing: {stats_dict.get('weight_processing', 'Unknown')}")
            print(f"Selected Samples: {stats_dict.get('selected_samples', 'Unknown')}")
            
            if stats_dict.get('privacy_alpha') is not None:
                print(f"Privacy Parameter α: {stats_dict.get('privacy_alpha')}")
                print(f"Privacy Guarantee: {stats_dict.get('differential_privacy', 'Unknown')}")
            
            print("\nWeight Statistics:")
            print(f"  Original - Mean: {np.mean(original_weights):.4f}, "
                  f"Std: {np.std(original_weights):.4f}, "
                  f"Min: {np.min(original_weights):.4f}, "
                  f"Max: {np.max(original_weights):.4f}")
            print(f"  Selected - Mean: {np.mean(selected_weights):.4f}, "
                  f"Std: {np.std(selected_weights):.4f}, "
                  f"Min: {np.min(selected_weights):.4f}, "
                  f"Max: {np.max(selected_weights):.4f}")
            
            # Print identifiability statistics
            if has_identifiability:
                id_stats = stats_dict['identifiability_stats']
                print("\nIdentifiability Analysis:")
                print(f"  Original Dataset:")
                print(f"    Total samples: {id_stats['total_samples']}")
                print(f"    Privacy-preserved: {id_stats['privacy_preserved_samples']} ({id_stats['privacy_preserved_ratio']:.2%})")
                print(f"    Identifiable: {id_stats['identifiable_samples']} ({id_stats['identifiable_ratio']:.2%})")
                print(f"  Selected Samples:")
                print(f"    Privacy-preserved: {id_stats['selected_privacy_preserved_samples']} ({id_stats['selected_privacy_preserved_ratio']:.2%})")
                print(f"    Identifiable: {id_stats['selected_identifiable_samples']} ({id_stats['selected_identifiable_ratio']:.2%})")
            
            print("="*60)
        
        plt.show()
    
    def compare_sir_vs_dp_sir(self,
                             synthetic_data: Union[pd.DataFrame, np.ndarray],
                             importance_weights: np.ndarray,
                             n_samples: int,
                             privacy_alphas: list = [0.1, 1.0, 10.0],
                             method: str = 'weighted',
                             weight_processing: str = 'raw',
                             identifiability_flags: Optional[np.ndarray] = None):
        """
        Compare Standard SIR vs DP-SIR with different privacy parameters.
        
        Args:
            synthetic_data: Synthetic data samples
            importance_weights: Importance weights from fidelity classifier
            n_samples: Number of samples to select
            privacy_alphas: List of privacy parameters to test
            method: Sampling method
            weight_processing: Weight processing method
            identifiability_flags: Optional boolean array for identifiability analysis
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        # Standard SIR
        sir_data, sir_weights, sir_flags, sir_indices, sir_stats = self.sample(
            synthetic_data, importance_weights, n_samples, 
            method=method, weight_processing=weight_processing,
            privacy_alpha=None, identifiability_flags=identifiability_flags
        )
        results['standard_sir'] = {
            'weights': sir_weights,
            'flags': sir_flags,
            'indices': sir_indices,
            'stats': sir_stats
        }
        
        # DP-SIR with different privacy parameters
        for alpha in privacy_alphas:
            dp_data, dp_weights, dp_flags, dp_indices, dp_stats = self.sample(
                synthetic_data, importance_weights, n_samples,
                method=method, weight_processing=weight_processing,
                privacy_alpha=alpha, identifiability_flags=identifiability_flags
            )
            results[f'dp_sir_alpha_{alpha}'] = {
                'weights': dp_weights,
                'flags': dp_flags,
                'indices': dp_indices,
                'stats': dp_stats
            }
        
        # Determine figure size based on whether we have identifiability data
        has_identifiability = (identifiability_flags is not None)
        fig_height = 15 if has_identifiability else 10
        n_rows = 3 if has_identifiability else 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, fig_height))
        
        # Plot weight distributions
        axes[0, 0].hist(sir_weights, bins=30, alpha=0.7, label='Standard SIR', color='blue')
        for i, alpha in enumerate(privacy_alphas):
            dp_weights = results[f'dp_sir_alpha_{alpha}']['weights']
            axes[0, 0].hist(dp_weights, bins=30, alpha=0.5, 
                           label=f'DP-SIR (α={alpha})', color=f'C{i+1}')
        axes[0, 0].set_title('Weight Distributions Comparison')
        axes[0, 0].set_xlabel('Importance Weight')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot sample diversity (unique indices count)
        methods = ['Standard SIR'] + [f'DP-SIR (α={alpha})' for alpha in privacy_alphas]
        unique_counts = [len(np.unique(sir_indices))]
        unique_counts.extend([len(np.unique(results[f'dp_sir_alpha_{alpha}']['indices'])) 
                             for alpha in privacy_alphas])
        
        axes[0, 1].bar(range(len(methods)), unique_counts, color=['blue'] + [f'C{i+1}' for i in range(len(privacy_alphas))])
        axes[0, 1].set_title('Sample Diversity (Unique Samples)')
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Number of Unique Samples')
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot privacy vs utility trade-off
        privacy_levels = [0] + [alpha/n_samples for alpha in privacy_alphas]  # Approximate privacy levels
        weight_variances = [np.var(sir_weights)]
        weight_variances.extend([np.var(results[f'dp_sir_alpha_{alpha}']['weights']) 
                                for alpha in privacy_alphas])
        
        axes[1, 0].plot(privacy_levels, weight_variances, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Privacy vs Utility Trade-off')
        axes[1, 0].set_xlabel('Privacy Level (ε)')
        axes[1, 0].set_ylabel('Weight Variance (Utility Proxy)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table_data = []
        for method in methods:
            if method == 'Standard SIR':
                weights = sir_weights
                unique_samples = len(np.unique(sir_indices))
                privacy = "None"
                stats = sir_stats
            else:
                alpha = float(method.split('α=')[1].rstrip(')'))
                weights = results[f'dp_sir_alpha_{alpha}']['weights']
                unique_samples = len(np.unique(results[f'dp_sir_alpha_{alpha}']['indices']))
                privacy = f"{alpha/n_samples:.4f}"
                stats = results[f'dp_sir_alpha_{alpha}']['stats']
            
            row = [
                method,
                f"{np.mean(weights):.4f}",
                f"{np.std(weights):.4f}",
                f"{unique_samples}",
                privacy
            ]
            
            # Add identifiability info if available
            if has_identifiability and stats.get('identifiability_stats'):
                id_stats = stats['identifiability_stats']
                row.append(f"{id_stats['selected_identifiable_ratio']:.2%}")
            
            table_data.append(row)
        
        # Create column labels
        col_labels = ['Method', 'Mean Weight', 'Std Weight', 'Unique Samples', 'Privacy (ε)']
        if has_identifiability:
            col_labels.append('Identifiable %')
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=col_labels,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        axes[1, 1].set_title('Summary Statistics')
        
        # Additional identifiability comparison plot
        if has_identifiability:
            identifiable_ratios = []
            for method in methods:
                if method == 'Standard SIR':
                    stats = sir_stats
                else:
                    alpha = float(method.split('α=')[1].rstrip(')'))
                    stats = results[f'dp_sir_alpha_{alpha}']['stats']
                
                if stats.get('identifiability_stats'):
                    identifiable_ratios.append(stats['identifiability_stats']['selected_identifiable_ratio'])
                else:
                    identifiable_ratios.append(0)
            
            axes[2, 0].bar(range(len(methods)), identifiable_ratios, 
                          color=['blue'] + [f'C{i+1}' for i in range(len(privacy_alphas))])
            axes[2, 0].set_title('Identifiable Sample Ratios by Method')
            axes[2, 0].set_xlabel('Method')
            axes[2, 0].set_ylabel('Identifiable Ratio')
            axes[2, 0].set_xticks(range(len(methods)))
            axes[2, 0].set_xticklabels(methods, rotation=45, ha='right')
            axes[2, 0].grid(True, alpha=0.3)
            
            # Privacy vs identifiability trade-off
            axes[2, 1].plot(privacy_levels, identifiable_ratios, 'o-', linewidth=2, markersize=8, color='red')
            axes[2, 1].set_title('Privacy vs Identifiability Trade-off')
            axes[2, 1].set_xlabel('Privacy Level (ε)')
            axes[2, 1].set_ylabel('Identifiable Sample Ratio')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results