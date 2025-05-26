"""
Metrics utilities for evaluating synthetic data quality in C-MAPS framework.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Union, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel: str = 'rbf', gamma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two datasets.
    
    Args:
        X: First dataset (n_samples_X, n_features)
        Y: Second dataset (n_samples_Y, n_features)
        kernel: Kernel type ('rbf', 'linear')
        gamma: Kernel parameter for RBF kernel
        
    Returns:
        MMD value
    """
    def kernel_matrix(A: np.ndarray, B: np.ndarray, kernel: str, gamma: float) -> np.ndarray:
        """Compute kernel matrix between A and B."""
        if kernel == 'rbf':
            # RBF kernel: exp(-gamma * ||x - y||^2)
            sq_dists = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * np.dot(A, B.T)
            return np.exp(-gamma * sq_dists)
        elif kernel == 'linear':
            # Linear kernel: <x, y>
            return np.dot(A, B.T)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    
    # Ensure arrays are 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    m, n = len(X), len(Y)
    
    # Compute kernel matrices
    K_XX = kernel_matrix(X, X, kernel, gamma)
    K_YY = kernel_matrix(Y, Y, kernel, gamma)
    K_XY = kernel_matrix(X, Y, kernel, gamma)
    
    # Compute MMD^2
    mmd_squared = (np.sum(K_XX) / (m * m) + 
                   np.sum(K_YY) / (n * n) - 
                   2 * np.sum(K_XY) / (m * n))
    
    return np.sqrt(max(0, mmd_squared))


def compute_wasserstein_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute 1-Wasserstein distance between two 1D distributions.
    
    Args:
        X: First dataset
        Y: Second dataset
        
    Returns:
        Wasserstein distance
    """
    return stats.wasserstein_distance(X.flatten(), Y.flatten())


def compute_correlation_metrics(data1: Union[pd.DataFrame, np.ndarray],
                               data2: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
    """
    Compute correlation-based metrics between two datasets.
    
    Args:
        data1: First dataset (real data)
        data2: Second dataset (synthetic data)
        
    Returns:
        Dictionary of correlation metrics
    """
    if isinstance(data1, pd.DataFrame):
        data1 = data1.values
    if isinstance(data2, pd.DataFrame):
        data2 = data2.values
    
    # Ensure same number of features
    min_features = min(data1.shape[1], data2.shape[1])
    data1 = data1[:, :min_features]
    data2 = data2[:, :min_features]
    
    # Compute correlation matrices
    corr1 = np.corrcoef(data1.T)
    corr2 = np.corrcoef(data2.T)
    
    # Handle NaN values
    corr1 = np.nan_to_num(corr1, nan=0.0)
    corr2 = np.nan_to_num(corr2, nan=0.0)
    
    # Correlation matrix difference
    corr_diff = np.abs(corr1 - corr2)
    
    metrics = {
        'correlation_matrix_mse': mean_squared_error(corr1.flatten(), corr2.flatten()),
        'correlation_matrix_mae': np.mean(corr_diff),
        'correlation_matrix_max_diff': np.max(corr_diff),
        'correlation_frobenius_norm': np.linalg.norm(corr1 - corr2, 'fro')
    }
    
    return metrics


def compute_marginal_distribution_metrics(data1: Union[pd.DataFrame, np.ndarray],
                                         data2: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
    """
    Compute marginal distribution comparison metrics.
    
    Args:
        data1: First dataset (real data)
        data2: Second dataset (synthetic data)
        
    Returns:
        Dictionary of marginal distribution metrics
    """
    if isinstance(data1, pd.DataFrame):
        data1 = data1.values
    if isinstance(data2, pd.DataFrame):
        data2 = data2.values
    
    min_features = min(data1.shape[1], data2.shape[1])
    data1 = data1[:, :min_features]
    data2 = data2[:, :min_features]
    
    ks_statistics = []
    wasserstein_distances = []
    
    for i in range(min_features):
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(data1[:, i], data2[:, i])
        ks_statistics.append(ks_stat)
        
        # Wasserstein distance
        wd = compute_wasserstein_distance(data1[:, i], data2[:, i])
        wasserstein_distances.append(wd)
    
    metrics = {
        'ks_statistics': ks_statistics,
        'mean_ks_statistic': np.mean(ks_statistics),
        'max_ks_statistic': np.max(ks_statistics),
        'wasserstein_distances': wasserstein_distances,
        'mean_wasserstein_distance': np.mean(wasserstein_distances),
        'max_wasserstein_distance': np.max(wasserstein_distances)
    }
    
    return metrics


def compute_statistical_moments(data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute statistical moments for each feature.
    
    Args:
        data: Input dataset
        
    Returns:
        Dictionary of statistical moments
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    moments = {
        'mean': np.mean(data, axis=0),
        'std': np.std(data, axis=0),
        'skewness': stats.skew(data, axis=0),
        'kurtosis': stats.kurtosis(data, axis=0),
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0),
        'median': np.median(data, axis=0),
        'q25': np.percentile(data, 25, axis=0),
        'q75': np.percentile(data, 75, axis=0)
    }
    
    return moments


def compare_statistical_moments(data1: Union[pd.DataFrame, np.ndarray],
                               data2: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
    """
    Compare statistical moments between two datasets.
    
    Args:
        data1: First dataset (real data)
        data2: Second dataset (synthetic data)
        
    Returns:
        Dictionary of moment comparison metrics
    """
    moments1 = compute_statistical_moments(data1)
    moments2 = compute_statistical_moments(data2)
    
    comparisons = {}
    
    for moment_name in ['mean', 'std', 'skewness', 'kurtosis']:
        if moment_name in moments1 and moment_name in moments2:
            diff = np.abs(moments1[moment_name] - moments2[moment_name])
            comparisons[f'{moment_name}_mae'] = np.mean(diff)
            comparisons[f'{moment_name}_max_diff'] = np.max(diff)
            comparisons[f'{moment_name}_mse'] = mean_squared_error(moments1[moment_name], moments2[moment_name])
    
    return comparisons


def compute_privacy_metrics(identifiability_flags: np.ndarray, 
                           epsilon_target: float = 0.05) -> Dict[str, Any]:
    """
    Compute privacy-related metrics.
    
    Args:
        identifiability_flags: Array of identifiability flags
        epsilon_target: Target identifiability ratio
        
    Returns:
        Dictionary of privacy metrics
    """
    total_samples = len(identifiability_flags)
    identifiable_samples = np.sum(identifiability_flags)
    identifiable_ratio = identifiable_samples / total_samples if total_samples > 0 else 0
    
    metrics = {
        'total_samples': total_samples,
        'identifiable_samples': identifiable_samples,
        'non_identifiable_samples': total_samples - identifiable_samples,
        'identifiable_ratio': identifiable_ratio,
        'privacy_preserved_ratio': 1 - identifiable_ratio,
        'epsilon_target': epsilon_target,
        'privacy_constraint_satisfied': identifiable_ratio <= epsilon_target,
        'privacy_violation_degree': max(0, identifiable_ratio - epsilon_target)
    }
    
    return metrics


def compute_utility_preservation_metrics(original_weights: np.ndarray,
                                        selected_weights: np.ndarray) -> Dict[str, float]:
    """
    Compute utility preservation metrics after sampling.
    
    Args:
        original_weights: Original importance weights
        selected_weights: Selected importance weights after sampling
        
    Returns:
        Dictionary of utility metrics
    """
    metrics = {
        'original_weight_mean': np.mean(original_weights),
        'selected_weight_mean': np.mean(selected_weights),
        'weight_mean_preservation_ratio': np.mean(selected_weights) / np.mean(original_weights),
        'original_weight_std': np.std(original_weights),
        'selected_weight_std': np.std(selected_weights),
        'high_quality_sample_ratio': np.mean(selected_weights > np.median(original_weights))
    }
    
    return metrics


def comprehensive_evaluation(real_data: Union[pd.DataFrame, np.ndarray],
                           synthetic_data: Union[pd.DataFrame, np.ndarray],
                           identifiability_flags: np.ndarray,
                           importance_weights: np.ndarray,
                           selected_indices: np.ndarray,
                           epsilon_target: float = 0.05) -> Dict[str, Any]:
    """
    Comprehensive evaluation of synthetic data quality and privacy.
    
    Args:
        real_data: Real dataset
        synthetic_data: Full synthetic dataset
        identifiability_flags: Identifiability flags for synthetic data
        importance_weights: Importance weights for synthetic data
        selected_indices: Indices of selected samples after refinement
        epsilon_target: Target privacy parameter
        
    Returns:
        Comprehensive evaluation metrics
    """
    # Convert to numpy arrays
    if isinstance(real_data, pd.DataFrame):
        real_data_array = real_data.values
    else:
        real_data_array = real_data
        
    if isinstance(synthetic_data, pd.DataFrame):
        synthetic_data_array = synthetic_data.values
    else:
        synthetic_data_array = synthetic_data
    
    # Get selected synthetic data
    selected_synthetic_data = synthetic_data_array[selected_indices]
    selected_identifiability_flags = identifiability_flags[selected_indices]
    selected_weights = importance_weights[selected_indices]
    
    evaluation = {}
    
    # Fidelity metrics (original vs selected synthetic)
    evaluation['fidelity_metrics'] = {
        'mmd_rbf': compute_mmd(real_data_array, selected_synthetic_data, kernel='rbf'),
        'mmd_linear': compute_mmd(real_data_array, selected_synthetic_data, kernel='linear'),
        **compute_correlation_metrics(real_data_array, selected_synthetic_data),
        **compute_marginal_distribution_metrics(real_data_array, selected_synthetic_data),
        **compare_statistical_moments(real_data_array, selected_synthetic_data)
    }
    
    # Privacy metrics
    evaluation['privacy_metrics'] = compute_privacy_metrics(selected_identifiability_flags, epsilon_target)
    
    # Utility preservation metrics
    evaluation['utility_metrics'] = compute_utility_preservation_metrics(importance_weights, selected_weights)
    
    # Overall metrics
    evaluation['overall_metrics'] = {
        'original_synthetic_samples': len(synthetic_data_array),
        'selected_samples': len(selected_synthetic_data),
        'selection_ratio': len(selected_synthetic_data) / len(synthetic_data_array),
        'privacy_utility_tradeoff': (1 - evaluation['privacy_metrics']['identifiable_ratio']) * 
                                   evaluation['utility_metrics']['weight_mean_preservation_ratio']
    }
    
    return evaluation


def print_evaluation_summary(evaluation: Dict[str, Any]):
    """
    Print a summary of the evaluation results.
    
    Args:
        evaluation: Evaluation dictionary from comprehensive_evaluation
    """
    print("=" * 60)
    print("C-MAPS EVALUATION SUMMARY")
    print("=" * 60)
    
    # Overall metrics
    overall = evaluation['overall_metrics']
    print(f"\nOVERALL METRICS:")
    print(f"  Original synthetic samples: {overall['original_synthetic_samples']}")
    print(f"  Selected samples: {overall['selected_samples']}")
    print(f"  Selection ratio: {overall['selection_ratio']:.2%}")
    print(f"  Privacy-Utility tradeoff: {overall['privacy_utility_tradeoff']:.4f}")
    
    # Privacy metrics
    privacy = evaluation['privacy_metrics']
    print(f"\nPRIVACY METRICS:")
    print(f"  Identifiable samples: {privacy['identifiable_samples']}/{privacy['total_samples']}")
    print(f"  Identifiable ratio: {privacy['identifiable_ratio']:.4f}")
    print(f"  Target epsilon: {privacy['epsilon_target']:.4f}")
    print(f"  Privacy constraint satisfied: {privacy['privacy_constraint_satisfied']}")
    if privacy['privacy_violation_degree'] > 0:
        print(f"  Privacy violation degree: {privacy['privacy_violation_degree']:.4f}")
    
    # Utility metrics
    utility = evaluation['utility_metrics']
    print(f"\nUTILITY METRICS:")
    print(f"  Weight preservation ratio: {utility['weight_mean_preservation_ratio']:.4f}")
    print(f"  High-quality sample ratio: {utility['high_quality_sample_ratio']:.2%}")
    
    # Fidelity metrics (key ones)
    fidelity = evaluation['fidelity_metrics']
    print(f"\nFIDELITY METRICS:")
    print(f"  MMD (RBF): {fidelity['mmd_rbf']:.6f}")
    print(f"  MMD (Linear): {fidelity['mmd_linear']:.6f}")
    print(f"  Correlation matrix MSE: {fidelity['correlation_matrix_mse']:.6f}")
    print(f"  Mean KS statistic: {fidelity['mean_ks_statistic']:.6f}")
    print(f"  Mean Wasserstein distance: {fidelity['mean_wasserstein_distance']:.6f}")
    
    print("\n" + "=" * 60)