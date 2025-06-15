import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, ks_2samp, chisquare
from sklearn.metrics import pairwise
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def identify_column_types(df: pd.DataFrame, 
                         categorical_threshold: float = 0.05, 
                         max_categories: int = 50) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        categorical_threshold: If unique_values/total_values < threshold, treat as categorical
        max_categories: Maximum number of categories to consider as categorical
        
    Returns:
        Tuple of (numerical_columns, categorical_columns)
    """
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)
            unique_count = df[col].nunique()
            
            # Treat as categorical if few unique values
            if unique_ratio < categorical_threshold and unique_count <= max_categories:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numerical_cols, categorical_cols


def get_frequency_categorical(df_real: pd.DataFrame, df_syn: pd.DataFrame) -> Dict:
    """
    Get frequency distributions for categorical columns only.
    """
    freqs = {}
    
    for col in df_real.columns:
        if col not in df_syn.columns:
            continue
            
        real_col = df_real[col].dropna()
        syn_col = df_syn[col].dropna()
        
        # For categorical columns, use value counts
        real_counts = real_col.value_counts(normalize=True, dropna=False)
        syn_counts = syn_col.value_counts(normalize=True, dropna=False)
        
        # Align the series to have same index
        real_aligned, syn_aligned = real_counts.align(syn_counts, fill_value=0)
        
        # Add small epsilon
        real_freq = real_aligned.values + 1e-10
        syn_freq = syn_aligned.values + 1e-10
        
        freqs[col] = (real_freq, syn_freq)
    
    return freqs


def get_frequency_with_discretization(df_real: pd.DataFrame, df_syn: pd.DataFrame, n_histogram_bins: int = 50) -> Dict:
    """
    Get frequency distributions for each column with discretization for numerical columns.
    """
    freqs = {}
    
    for col in df_real.columns:
        if col not in df_syn.columns:
            continue
            
        real_col = df_real[col].dropna()
        syn_col = df_syn[col].dropna()
        
        if pd.api.types.is_numeric_dtype(real_col):
            # For numerical columns, use histogram binning
            local_bins = min(n_histogram_bins, len(real_col.unique()))
            if local_bins <= 1:
                freqs[col] = (np.array([1.0]), np.array([1.0]))
                continue
                
            try:
                real_bin, bins = pd.cut(real_col, bins=local_bins, retbins=True)
                syn_bin = pd.cut(syn_col, bins=bins)
                
                real_freq, syn_freq = real_bin.value_counts(
                    dropna=False, normalize=True
                ).align(
                    syn_bin.value_counts(dropna=False, normalize=True),
                    join="outer", axis=0, fill_value=0
                )
                
                # Add small epsilon to avoid zeros
                real_freq = real_freq.values + 1e-10
                syn_freq = syn_freq.values + 1e-10
                
                freqs[col] = (real_freq, syn_freq)
            except:
                freqs[col] = (np.array([1.0]), np.array([1.0]))
        else:
            # For categorical columns, use value counts
            real_counts = real_col.value_counts(normalize=True, dropna=False)
            syn_counts = syn_col.value_counts(normalize=True, dropna=False)
            
            # Align the series to have same index
            real_aligned, syn_aligned = real_counts.align(syn_counts, fill_value=0)
            
            # Add small epsilon
            real_freq = real_aligned.values + 1e-10
            syn_freq = syn_aligned.values + 1e-10
            
            freqs[col] = (real_freq, syn_freq)
    
    return freqs


def compute_jensen_shannon_distance(df_real: pd.DataFrame, df_syn: pd.DataFrame, 
                                   discretize_numerical: bool = False,
                                   n_histogram_bins: int = 50) -> float:
    """
    Compute Jensen-Shannon distance per column and return mean.
    Only works on categorical columns unless discretize_numerical=True.
    """
    try:
        if discretize_numerical:
            freqs = get_frequency_with_discretization(df_real, df_syn, n_histogram_bins)
        else:
            # Identify categorical columns
            _, categorical_cols = identify_column_types(df_real)
            if not categorical_cols:
                return np.nan
            freqs = get_frequency_categorical(df_real[categorical_cols], df_syn[categorical_cols])
        
        res = []
        
        for col in freqs:
            real_freq, syn_freq = freqs[col]
            
            # Normalize frequencies
            real_freq = real_freq / np.sum(real_freq)
            syn_freq = syn_freq / np.sum(syn_freq)
            
            js_dist = jensenshannon(real_freq, syn_freq)
            if not np.isnan(js_dist):
                res.append(js_dist)
        
        return np.mean(res) if res else np.nan
        
    except Exception as e:
        print(f"Error computing Jensen-Shannon distance: {e}")
        return np.nan


def compute_ks_statistic(df_real: pd.DataFrame, df_syn: pd.DataFrame) -> float:
    """
    Compute KS statistic per column and return mean (1 - statistic for better interpretation).
    """
    try:
        res = []
        
        for col in df_real.columns:
            if col in df_syn.columns:
                real_col = df_real[col].dropna()
                syn_col = df_syn[col].dropna()
                
                if len(real_col) > 0 and len(syn_col) > 0:
                    statistic, _ = ks_2samp(real_col, syn_col)
                    res.append(1 - statistic)  # Higher is better
        
        return np.mean(res) if res else np.nan
        
    except Exception as e:
        print(f"Error computing KS statistic: {e}")
        return np.nan


def compute_wasserstein_distance(df_real: pd.DataFrame, df_syn: pd.DataFrame) -> float:
    """
    Compute Wasserstein distance per column and return mean.
    """
    try:
        res = []
        
        for col in df_real.columns:
            if col in df_syn.columns:
                real_col = df_real[col].dropna()
                syn_col = df_syn[col].dropna()
                
                if len(real_col) > 0 and len(syn_col) > 0:
                    wd = stats.wasserstein_distance(real_col, syn_col)
                    res.append(wd)
        
        return np.mean(res) if res else np.nan
        
    except Exception as e:
        print(f"Error computing Wasserstein distance: {e}")
        return np.nan


def compute_mmd(df_real: pd.DataFrame, df_syn: pd.DataFrame, kernel: str = "rbf") -> float:
    """
    Compute Maximum Mean Discrepancy with configurable kernels (column-wise joint computation).
    """
    try:
        # Only use numeric columns and handle NaN values
        real_numeric = df_real.select_dtypes(include=[np.number]).fillna(0)
        syn_numeric = df_syn.select_dtypes(include=[np.number]).fillna(0)
        
        if real_numeric.shape[1] == 0 or syn_numeric.shape[1] == 0:
            return np.nan
        
        # Ensure same columns
        common_cols = list(set(real_numeric.columns) & set(syn_numeric.columns))
        if not common_cols:
            return np.nan
            
        X_real = real_numeric[common_cols].values
        X_syn = syn_numeric[common_cols].values
        
        if kernel == "linear":
            # MMD using linear kernel
            delta = X_real.mean(axis=0) - X_syn.mean(axis=0)
            score = delta.dot(delta.T)
            
        elif kernel == "rbf":
            # MMD using RBF kernel
            gamma = 1.0
            XX = pairwise.rbf_kernel(X_real, X_real, gamma)
            YY = pairwise.rbf_kernel(X_syn, X_syn, gamma)
            XY = pairwise.rbf_kernel(X_real, X_syn, gamma)
            score = XX.mean() + YY.mean() - 2 * XY.mean()
            
        elif kernel == "polynomial":
            # MMD using polynomial kernel
            degree = 2
            gamma = 1
            coef0 = 0
            XX = pairwise.polynomial_kernel(X_real, X_real, degree, gamma, coef0)
            YY = pairwise.polynomial_kernel(X_syn, X_syn, degree, gamma, coef0)
            XY = pairwise.polynomial_kernel(X_real, X_syn, degree, gamma, coef0)
            score = XX.mean() + YY.mean() - 2 * XY.mean()
            
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
        return float(score)
        
    except Exception as e:
        print(f"Error computing MMD with {kernel} kernel: {e}")
        return np.nan


def compute_inverse_kl_divergence(df_real: pd.DataFrame, df_syn: pd.DataFrame,
                                 discretize_numerical: bool = False,
                                 n_histogram_bins: int = 50) -> float:
    """
    Compute inverse KL divergence per column and return mean.
    Only works on categorical columns unless discretize_numerical=True.
    """
    try:
        if discretize_numerical:
            freqs = get_frequency_with_discretization(df_real, df_syn, n_histogram_bins)
        else:
            # Identify categorical columns
            _, categorical_cols = identify_column_types(df_real)
            if not categorical_cols:
                return np.nan
            freqs = get_frequency_categorical(df_real[categorical_cols], df_syn[categorical_cols])
        
        res = []
        
        for col in freqs:
            real_freq, syn_freq = freqs[col]
            
            # Normalize frequencies
            real_freq = real_freq / np.sum(real_freq)
            syn_freq = syn_freq / np.sum(syn_freq)
            
            kl_div = entropy(real_freq, syn_freq)
            inv_kl = 1 / (1 + kl_div)
            res.append(inv_kl)
        
        return np.mean(res) if res else np.nan
        
    except Exception as e:
        print(f"Error computing inverse KL divergence: {e}")
        return np.nan


def compute_chi_squared_test(df_real: pd.DataFrame, df_syn: pd.DataFrame,
                            discretize_numerical: bool = False,
                            n_histogram_bins: int = 50) -> float:
    """
    Compute Chi-squared test p-value per column and return mean.
    Only works on categorical columns unless discretize_numerical=True.
    """
    try:
        if discretize_numerical:
            freqs = get_frequency_with_discretization(df_real, df_syn, n_histogram_bins)
        else:
            # Identify categorical columns
            _, categorical_cols = identify_column_types(df_real)
            if not categorical_cols:
                return np.nan
            freqs = get_frequency_categorical(df_real[categorical_cols], df_syn[categorical_cols])
        
        res = []
        
        for col in freqs:
            real_freq, syn_freq = freqs[col]
            
            try:
                _, pvalue = chisquare(real_freq, syn_freq)
                if np.isnan(pvalue):
                    pvalue = 0
                res.append(pvalue)
            except:
                continue
        
        return np.mean(res) if res else np.nan
        
    except Exception as e:
        print(f"Error computing Chi-squared test: {e}")
        return np.nan


def compute_total_variation_distance(df_real: pd.DataFrame, df_syn: pd.DataFrame,
                                   discretize_numerical: bool = False,
                                   n_histogram_bins: int = 50) -> float:
    """
    Compute Total Variation Distance per column and return mean.
    Only works on categorical columns unless discretize_numerical=True.
    """
    try:
        if discretize_numerical:
            freqs = get_frequency_with_discretization(df_real, df_syn, n_histogram_bins)
        else:
            # Identify categorical columns
            _, categorical_cols = identify_column_types(df_real)
            if not categorical_cols:
                return np.nan
            freqs = get_frequency_categorical(df_real[categorical_cols], df_syn[categorical_cols])
        
        res = []
        
        for col in freqs:
            real_freq, syn_freq = freqs[col]
            
            # Normalize frequencies
            real_freq = real_freq / np.sum(real_freq)
            syn_freq = syn_freq / np.sum(syn_freq)
            
            tv_distance = 0.5 * np.sum(np.abs(real_freq - syn_freq))
            res.append(tv_distance)
        
        return np.mean(res) if res else np.nan
        
    except Exception as e:
        print(f"Error computing Total Variation Distance: {e}")
        return np.nan


def compute_hellinger_distance(df_real: pd.DataFrame, df_syn: pd.DataFrame,
                              discretize_numerical: bool = False,
                              n_histogram_bins: int = 50) -> float:
    """
    Compute Hellinger Distance per column and return mean.
    Only works on categorical columns unless discretize_numerical=True.
    """
    try:
        if discretize_numerical:
            freqs = get_frequency_with_discretization(df_real, df_syn, n_histogram_bins)
        else:
            # Identify categorical columns
            _, categorical_cols = identify_column_types(df_real)
            if not categorical_cols:
                return np.nan
            freqs = get_frequency_categorical(df_real[categorical_cols], df_syn[categorical_cols])
        
        res = []
        
        for col in freqs:
            real_freq, syn_freq = freqs[col]
            
            # Normalize frequencies
            real_freq = real_freq / np.sum(real_freq)
            syn_freq = syn_freq / np.sum(syn_freq)
            
            hellinger_dist = np.sqrt(1 - np.sum(np.sqrt(real_freq * syn_freq)))
            res.append(hellinger_dist)
        
        return np.mean(res) if res else np.nan
        
    except Exception as e:
        print(f"Error computing Hellinger distance: {e}")
        return np.nan


def evaluate_distribution_similarity(real_data: pd.DataFrame, 
                                   raw_synthetic_data: pd.DataFrame, 
                                   refined_synthetic_data: pd.DataFrame,
                                   metrics: Optional[List[str]] = None,
                                   mmd_kernel: str = "rbf",
                                   discretize_numerical: bool = False,
                                   n_histogram_bins: int = 50,
                                   verbose: bool = True) -> pd.DataFrame:
    """
    Comprehensive evaluation of distribution similarity between real and synthetic datasets.
    
    Args:
        real_data: Real dataset
        raw_synthetic_data: Raw synthetic dataset (before refinement)
        refined_synthetic_data: Refined synthetic dataset (after C-MAPS)
        metrics: List of metrics to compute. If None, compute all available metrics.
                Available metrics: ["jensen_shannon_distance", "inverse_kl_divergence", 
                "chi_squared_test", "total_variation_distance", "ks_test", 
                "wasserstein_distance", "maximum_mean_discrepancy", "hellinger_distance"]
        mmd_kernel: Kernel for MMD computation ("rbf", "linear", "polynomial")
        discretize_numerical: Whether to discretize numerical variables for categorical metrics
        n_histogram_bins: Number of bins for histogram-based metrics
        verbose: Whether to print progress
        
    Returns:
        DataFrame with similarity metrics
    """
    if verbose:
        print("=" * 80)
        print("DISTRIBUTION SIMILARITY EVALUATION")
        print("=" * 80)
    
    # Ensure all dataframes have the same columns
    common_cols = list(set(real_data.columns) & 
                      set(raw_synthetic_data.columns) & 
                      set(refined_synthetic_data.columns))
    
    if len(common_cols) < len(real_data.columns):
        if verbose:
            print(f"Warning: Using {len(common_cols)} common columns out of {len(real_data.columns)} total")
    
    real_data = real_data[common_cols]
    raw_synthetic_data = raw_synthetic_data[common_cols]
    refined_synthetic_data = refined_synthetic_data[common_cols]
    
    # Identify column types
    numerical_cols, categorical_cols = identify_column_types(real_data)
    
    if verbose:
        print(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns")
        print(f"Numerical columns: {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}")
        print(f"Categorical columns: {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        if discretize_numerical:
            print("Discretization enabled: Categorical metrics will include discretized numerical columns")
    
    # Define all available metrics
    all_metrics = {
        "jensen_shannon_distance": {
            "name": "Jensen-Shannon Distance",
            "data_type": "Categorical" if not discretize_numerical else "All",
            "func": lambda df_r, df_s: compute_jensen_shannon_distance(df_r, df_s, discretize_numerical, n_histogram_bins),
            "direction": "minimize",
            "description": "Lower is better"
        },
        "inverse_kl_divergence": {
            "name": "Inverse KL Divergence",
            "data_type": "Categorical" if not discretize_numerical else "All", 
            "func": lambda df_r, df_s: compute_inverse_kl_divergence(df_r, df_s, discretize_numerical, n_histogram_bins),
            "direction": "maximize",
            "description": "Higher is better"
        },
        "chi_squared_test": {
            "name": "Chi-squared Test p-value",
            "data_type": "Categorical" if not discretize_numerical else "All",
            "func": lambda df_r, df_s: compute_chi_squared_test(df_r, df_s, discretize_numerical, n_histogram_bins),
            "direction": "maximize", 
            "description": "Higher is better"
        },
        "total_variation_distance": {
            "name": "Total Variation Distance",
            "data_type": "Categorical" if not discretize_numerical else "All",
            "func": lambda df_r, df_s: compute_total_variation_distance(df_r, df_s, discretize_numerical, n_histogram_bins),
            "direction": "minimize",
            "description": "Lower is better"
        },
        "ks_test": {
            "name": "Kolmogorov-Smirnov Test",
            "data_type": "Numerical",
            "func": lambda df_r, df_s: compute_ks_statistic(df_r[numerical_cols], df_s[numerical_cols]) if numerical_cols else np.nan,
            "direction": "maximize",
            "description": "Higher is better"
        },
        "wasserstein_distance": {
            "name": "Wasserstein Distance",
            "data_type": "Numerical",
            "func": lambda df_r, df_s: compute_wasserstein_distance(df_r[numerical_cols], df_s[numerical_cols]) if numerical_cols else np.nan,
            "direction": "minimize",
            "description": "Lower is better"
        },
        "maximum_mean_discrepancy": {
            "name": f"Maximum Mean Discrepancy ({mmd_kernel})",
            "data_type": "Numerical",
            "func": lambda df_r, df_s: compute_mmd(df_r, df_s, mmd_kernel),
            "direction": "minimize", 
            "description": "Lower is better"
        },
        "hellinger_distance": {
            "name": "Hellinger Distance",
            "data_type": "Categorical" if not discretize_numerical else "All",
            "func": lambda df_r, df_s: compute_hellinger_distance(df_r, df_s, discretize_numerical, n_histogram_bins),
            "direction": "minimize",
            "description": "Lower is better"
        }
    }
    
    # Select metrics to compute
    if metrics is None:
        metrics_to_compute = all_metrics
    else:
        metrics_to_compute = {k: v for k, v in all_metrics.items() if k in metrics}
        if not metrics_to_compute:
            available_metrics = list(all_metrics.keys())
            raise ValueError(f"No valid metrics found. Available metrics: {available_metrics}")
    
    results = []
    
    # Evaluate each metric
    for metric_key, metric_info in metrics_to_compute.items():
        if verbose:
            print(f"\nComputing {metric_info['name']}...")
        
        try:
            # Skip if no appropriate columns for specific data type metrics
            if metric_info['data_type'] == "Numerical" and not numerical_cols:
                if verbose:
                    print(f"  Skipping - no numerical columns available")
                continue
            elif metric_info['data_type'] == "Categorical" and not categorical_cols:
                if verbose:
                    print(f"  Skipping - no categorical columns available")
                continue
                
            # Compute scores
            raw_score = metric_info['func'](real_data, raw_synthetic_data)
            refined_score = metric_info['func'](real_data, refined_synthetic_data)
            
            if np.isnan(raw_score) or np.isnan(refined_score):
                if verbose:
                    print(f"  Could not compute {metric_info['name']} - invalid scores")
                continue
            
            # Determine improvement
            if metric_info['direction'] == "maximize":
                improvement = "✓" if refined_score > raw_score else "✗"
            else:
                improvement = "✓" if refined_score < raw_score else "✗"
            
            results.append({
                "Metric": metric_info['name'],
                "Data Type": metric_info['data_type'],
                "Raw Synthetic": f"{raw_score:.6f}",
                "Refined Synthetic": f"{refined_score:.6f}",
                "Improvement": improvement,
                "Interpretation": metric_info['description']
            })
            
            if verbose:
                print(f"  {metric_info['name']}: Raw={raw_score:.6f}, Refined={refined_score:.6f} {improvement}")
                
        except Exception as e:
            if verbose:
                print(f"  Error computing {metric_info['name']}: {e}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n{'='*80}")
        print("SUMMARY RESULTS")
        print('='*80)
        if not results_df.empty:
            print(results_df.to_string(index=False))
            
            # Summary statistics
            total_metrics = len(results_df)
            improved_metrics = len(results_df[results_df['Improvement'] == '✓'])
            
            print(f"\nOVERALL SUMMARY:")
            print(f"Total metrics computed: {total_metrics}")
            print(f"Metrics improved by refinement: {improved_metrics}/{total_metrics} ({improved_metrics/total_metrics*100:.1f}%)")
        else:
            print("No metrics could be computed successfully.")
    
    return results_df