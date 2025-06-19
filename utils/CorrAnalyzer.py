import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

def _cramers_v(var1, var2):
    """Calculate Cramer's V between two categorical variables."""
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape) - 1
    return np.sqrt(stat / (obs * mini + 1e-16))

def _correlation_ratio(categories, measurements):
    """Calculate the correlation ratio (eta squared) between categorical and numerical variables."""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    
    for i in range(cat_num):
        cat_measures = measurements[fcat == i]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures) if len(cat_measures) > 0 else 0
    
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    
    if numerator == 0 or denominator == 0:
        eta = 0.0
    else:
        eta = numerator / denominator
    
    return eta

def _apply_mat(data, func, labs1, labs2):
    """Apply a function across combinations of labels to create a matrix."""
    res = [func(data[lab1], data[lab2]) for lab1 in labs1 for lab2 in labs2]
    return pd.DataFrame(
        np.array(res).reshape(len(labs1), len(labs2)), 
        columns=labs2, 
        index=labs1
    )

def mixed_correlation(data, num_cols, cat_cols, method='spearman'):
    """Calculate mixed correlation matrix using appropriate correlation measures."""
    # Correlation for numerical-numerical using specified method
    if len(num_cols) > 0:
        corr_num_num = data[num_cols].corr(method=method)
    else:
        corr_num_num = pd.DataFrame()
    
    # Cramer's V for categorical-categorical
    if len(cat_cols) > 0:
        corr_cat_cat = _apply_mat(data, _cramers_v, cat_cols, cat_cols)
    else:
        corr_cat_cat = pd.DataFrame()
    
    # Correlation ratio for categorical-numerical
    if len(cat_cols) > 0 and len(num_cols) > 0:
        corr_cat_num = _apply_mat(data, _correlation_ratio, cat_cols, num_cols)
    else:
        corr_cat_num = pd.DataFrame()
    
    # Combine matrices
    if corr_cat_cat.empty and corr_num_num.empty:
        return pd.DataFrame()
    elif corr_cat_cat.empty:
        return corr_num_num
    elif corr_num_num.empty:
        return corr_cat_cat
    else:
        top_row = pd.concat([corr_cat_cat, corr_cat_num], axis=1)
        bot_row = pd.concat([corr_cat_num.transpose(), corr_num_num], axis=1)
        corr = pd.concat([top_row, bot_row], axis=0)
        return corr + np.diag(1 - np.diag(corr))

def identify_column_types(data):
    """Automatically identify numerical and categorical columns."""
    num_cols = []
    cat_cols = []
    
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.05 and data[col].nunique() < 20:
                cat_cols.append(col)
            else:
                num_cols.append(col)
        else:
            cat_cols.append(col)
    
    return num_cols, cat_cols

def analyze_correlation_structure(corr_matrix, num_cols, cat_cols):
    """
    Analyze the structure of a mixed correlation matrix to understand potential biases.
    
    Returns detailed statistics about different correlation types.
    """
    n_num = len(num_cols)
    n_cat = len(cat_cols)
    
    # Extract different types of correlations
    if n_num > 0 and n_cat > 0:
        # Numerical-numerical correlations (excluding diagonal)
        num_num_corr = corr_matrix.loc[num_cols, num_cols].values
        num_num_mask = ~np.eye(n_num, dtype=bool)
        num_num_values = num_num_corr[num_num_mask]
        
        # Categorical-categorical correlations (excluding diagonal) 
        cat_cat_corr = corr_matrix.loc[cat_cols, cat_cols].values
        cat_cat_mask = ~np.eye(n_cat, dtype=bool)
        cat_cat_values = cat_cat_corr[cat_cat_mask]
        
        # Categorical-numerical correlations
        cat_num_values = corr_matrix.loc[cat_cols, num_cols].values.flatten()
        
        structure_analysis = {
            'num_num': {
                'count': len(num_num_values),
                'mean': np.mean(np.abs(num_num_values)),
                'std': np.std(np.abs(num_num_values)),
                'range': [np.min(num_num_values), np.max(num_num_values)],
                'abs_mean': np.mean(np.abs(num_num_values))
            },
            'cat_cat': {
                'count': len(cat_cat_values),
                'mean': np.mean(cat_cat_values),
                'std': np.std(cat_cat_values),
                'range': [np.min(cat_cat_values), np.max(cat_cat_values)],
                'abs_mean': np.mean(np.abs(cat_cat_values))
            },
            'cat_num': {
                'count': len(cat_num_values),
                'mean': np.mean(cat_num_values),
                'std': np.std(cat_num_values),
                'range': [np.min(cat_num_values), np.max(cat_num_values)],
                'abs_mean': np.mean(np.abs(cat_num_values))
            }
        }
    else:
        structure_analysis = {'warning': 'Mixed analysis requires both numerical and categorical columns'}
    
    return structure_analysis

def compute_multiple_similarity_scores(diff_matrix, corr_real, num_cols, cat_cols):
    """
    Compute multiple similarity scores to address the scale bias concern.
    
    Returns different approaches to quantify similarity that handle scale differences.
    """
    scores = {}
    
    # 1. Standard Frobenius norm
    scores['frobenius_raw'] = np.linalg.norm(diff_matrix.values, ord='fro')
    scores['frobenius_normalized'] = scores['frobenius_raw'] / np.sqrt(diff_matrix.size)
    
    # 2. Element-wise approach: separate scores by correlation type
    if len(num_cols) > 0 and len(cat_cols) > 0:
        n_num = len(num_cols)
        n_cat = len(cat_cols)
        
        # Extract different parts
        num_num_diff = diff_matrix.loc[num_cols, num_cols].values
        cat_cat_diff = diff_matrix.loc[cat_cols, cat_cols].values  
        cat_num_diff = diff_matrix.loc[cat_cols, num_cols].values
        
        # Compute separate scores (excluding diagonal)
        num_num_mask = ~np.eye(n_num, dtype=bool)
        cat_cat_mask = ~np.eye(n_cat, dtype=bool)
        
        scores['num_num_mae'] = np.mean(np.abs(num_num_diff[num_num_mask]))
        scores['cat_cat_mae'] = np.mean(np.abs(cat_cat_diff[cat_cat_mask]))
        scores['cat_num_mae'] = np.mean(np.abs(cat_num_diff))
        
        scores['num_num_mse'] = np.mean(num_num_diff[num_num_mask]**2)
        scores['cat_cat_mse'] = np.mean(cat_cat_diff[cat_cat_mask]**2)
        scores['cat_num_mse'] = np.mean(cat_num_diff**2)
        
        # Weighted average (equal weight to each correlation type)
        scores['weighted_mae'] = (scores['num_num_mae'] + scores['cat_cat_mae'] + scores['cat_num_mae']) / 3
        scores['weighted_mse'] = (scores['num_num_mse'] + scores['cat_cat_mse'] + scores['cat_num_mse']) / 3
    
    # 3. Normalized by expected variance approach
    # For numerical correlations: range is [-1,1], so max diff is 2
    # For categorical correlations: range is [0,1], so max diff is 1
    total_elements = diff_matrix.size - len(diff_matrix)  # Exclude diagonal
    
    if len(num_cols) > 0 and len(cat_cols) > 0:
        num_num_elements = n_num * (n_num - 1)  # Off-diagonal elements
        cat_cat_elements = n_cat * (n_cat - 1)
        cat_num_elements = n_cat * n_num * 2  # Both directions
        
        # Theoretical maximum differences
        max_num_num_diff = 2.0  # Range [-1,1] 
        max_cat_cat_diff = 1.0  # Range [0,1]
        max_cat_num_diff = 1.0  # Range [0,1]
        
        # Normalize each part by its theoretical maximum
        if num_num_elements > 0:
            normalized_num_num = np.sum(np.abs(num_num_diff[num_num_mask])) / (num_num_elements * max_num_num_diff)
        else:
            normalized_num_num = 0
            
        if cat_cat_elements > 0:
            normalized_cat_cat = np.sum(np.abs(cat_cat_diff[cat_cat_mask])) / (cat_cat_elements * max_cat_cat_diff)
        else:
            normalized_cat_cat = 0
            
        if cat_num_elements > 0:
            normalized_cat_num = np.sum(np.abs(cat_num_diff)) / (cat_num_elements * max_cat_num_diff)
        else:
            normalized_cat_num = 0
        
        scores['scale_normalized_score'] = (normalized_num_num + normalized_cat_cat + normalized_cat_num) / 3
    
    # 4. Relative to real correlation magnitude
    real_abs_mean = np.mean(np.abs(corr_real.values[~np.eye(len(corr_real), dtype=bool)]))
    if real_abs_mean > 0:
        scores['relative_to_magnitude'] = scores['frobenius_normalized'] / real_abs_mean
    
    return scores

def mixed_correlation_analysis(
    real_data: pd.DataFrame,
    raw_synthetic_data: pd.DataFrame,
    refined_synthetic_data: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    use_mixed: bool = True,
    method: str = 'spearman',
    detailed_analysis: bool = True,
    show_plots: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[str, Any]:
    """
    Mixed correlation analysis that addresses scale bias concerns.
    
    This function provides multiple validation approaches to ensure reliable results
    despite different correlation measure ranges.
    """
    
    # Setup and validation
    if num_cols is None or cat_cols is None:
        auto_num_cols, auto_cat_cols = identify_column_types(real_data)
        num_cols = num_cols or auto_num_cols
        cat_cols = cat_cols or auto_cat_cols
    
    print("MIXED CORRELATION ANALYSIS")
    print("="*60)
    print(f"Addressing the scale bias concern in mixed correlation matrices")
    print(f"Configuration: use_mixed={use_mixed}, method={method}")
    print(f"Numerical columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print()
    
    # Select columns and prepare data
    if use_mixed:
        analysis_cols = num_cols + cat_cols
        analysis_num_cols = num_cols
        analysis_cat_cols = cat_cols
        correlation_type = f"Mixed Correlation ({method.title()} for numerical)"
    else:
        analysis_cols = num_cols
        analysis_num_cols = num_cols
        analysis_cat_cols = []
        correlation_type = f"{method.title()} Correlation (Numerical Only)"
    
    # Ensure common columns
    common_cols = list(set(real_data.columns) & 
                      set(raw_synthetic_data.columns) & 
                      set(refined_synthetic_data.columns))
    analysis_cols = [col for col in analysis_cols if col in common_cols]
    
    if not analysis_cols:
        raise ValueError("No common columns found across all datasets")
    
    # Subset data
    real_subset = real_data[analysis_cols].copy()
    raw_subset = raw_synthetic_data[analysis_cols].copy()
    refined_subset = refined_synthetic_data[analysis_cols].copy()
    
    # Calculate correlation matrices
    if use_mixed:
        corr_real = mixed_correlation(real_subset, analysis_num_cols, analysis_cat_cols, method)
        corr_raw = mixed_correlation(raw_subset, analysis_num_cols, analysis_cat_cols, method)
        corr_refined = mixed_correlation(refined_subset, analysis_num_cols, analysis_cat_cols, method)
    else:
        corr_real = real_subset.corr(method=method)
        corr_raw = raw_subset.corr(method=method)
        corr_refined = refined_subset.corr(method=method)
    
    # Calculate difference matrices
    diff_raw = corr_real - corr_raw
    diff_refined = corr_real - corr_refined
    
    # DETAILED ANALYSIS: Address the scale bias concern
    print("SCALE BIAS ANALYSIS")
    print("-"*30)
    
    if detailed_analysis and use_mixed and len(num_cols) > 0 and len(cat_cols) > 0:
        # Analyze correlation structure
        real_structure = analyze_correlation_structure(corr_real, analysis_num_cols, analysis_cat_cols)
        
        print("Value Ranges and Typical Magnitudes in Real Data:")
        print(f"  Numerical-Numerical: range {real_structure['num_num']['range']}, mean(|r|) = {real_structure['num_num']['abs_mean']:.3f}")
        print(f"  Categorical-Categorical: range {real_structure['cat_cat']['range']}, mean(|V|) = {real_structure['cat_cat']['abs_mean']:.3f}")
        print(f"  Categorical-Numerical: range {real_structure['cat_num']['range']}, mean(|η|) = {real_structure['cat_num']['abs_mean']:.3f}")
        print()
        
        # Check for potential bias
        ranges = {
            'num_num': real_structure['num_num']['range'][1] - real_structure['num_num']['range'][0],
            'cat_cat': real_structure['cat_cat']['range'][1] - real_structure['cat_cat']['range'][0],
            'cat_num': real_structure['cat_num']['range'][1] - real_structure['cat_num']['range'][0]
        }
        
        if max(ranges.values()) / min(ranges.values()) > 1.5:
            print("WARNING: Significant range differences detected!")
            print("   This could potentially bias the Frobenius norm analysis.")
            print("   Using multiple validation metrics below...")
        else:
            print("Range differences are moderate - standard analysis should be reliable.")
        print()
    
    # MULTIPLE SIMILARITY SCORES
    scores_raw = compute_multiple_similarity_scores(diff_raw, corr_real, analysis_num_cols, analysis_cat_cols)
    scores_refined = compute_multiple_similarity_scores(diff_refined, corr_real, analysis_num_cols, analysis_cat_cols)
    
    print("MULTIPLE SIMILARITY METRICS")
    print("-"*35)
    print(f"{'Metric':<25} {'Raw':<12} {'Refined':<12} {'Improvement':<12}")
    print("-"*65)
    
    improvement_summary = {}
    
    for metric in ['frobenius_normalized', 'weighted_mae', 'scale_normalized_score']:
        if metric in scores_raw and metric in scores_refined:
            raw_score = scores_raw[metric]
            refined_score = scores_refined[metric]
            improvement = ((raw_score - refined_score) / raw_score * 100) if raw_score > 0 else 0
            improvement_summary[metric] = improvement
            
            print(f"{metric:<25} {raw_score:<12.4f} {refined_score:<12.4f} {improvement:>+11.1f}%")
    
    print("-"*65)
    
    # CONSENSUS ASSESSMENT
    improvements = list(improvement_summary.values())
    if len(improvements) > 0:
        avg_improvement = np.mean(improvements)
        improvement_std = np.std(improvements)
        
        print(f"CONSENSUS ASSESSMENT:")
        print(f"Average improvement across metrics: {avg_improvement:+.1f}% ± {improvement_std:.1f}%")
        
        if improvement_std < 5:
            print("High consensus - all metrics agree!")
        elif improvement_std < 15:
            print("Moderate consensus - some variation between metrics")
        else:
            print("Low consensus - significant disagreement between metrics")
    
    # DETAILED BREAKDOWN (if mixed correlation is used)
    if detailed_analysis and use_mixed and len(analysis_num_cols) > 0 and len(analysis_cat_cols) > 0:
        print(f"DETAILED BREAKDOWN BY CORRELATION TYPE:")
        print("-"*45)
        
        breakdown_metrics = ['num_num_mae', 'cat_cat_mae', 'cat_num_mae']
        for metric in breakdown_metrics:
            if metric in scores_raw and metric in scores_refined:
                raw_score = scores_raw[metric]
                refined_score = scores_refined[metric]
                improvement = ((raw_score - refined_score) / raw_score * 100) if raw_score > 0 else 0
                
                corr_type = metric.replace('_mae', '').replace('_', '-').title()
                print(f"{corr_type:<20}: {raw_score:.4f} → {refined_score:.4f} ({improvement:+.1f}%)")
    
    # PLOTTING (with consistent scales)
    if show_plots:
        print(f"Generating visualizations...")
        
        # Plot correlation matrices
        plt.figure(figsize=figsize)
        sns.heatmap(corr_real, annot=True, cmap='coolwarm', fmt=".2f", 
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title(f"Real Data - {correlation_type}", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_raw, annot=True, cmap='coolwarm', fmt=".2f",
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title(f"Raw Synthetic - {correlation_type}", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_refined, annot=True, cmap='coolwarm', fmt=".2f",
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title(f"Refined Synthetic - {correlation_type}", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Plot difference matrices (with shared scale)
        global_max_abs_val = max(np.abs(diff_raw.values).max(), np.abs(diff_refined.values).max())
        
        plt.figure(figsize=figsize)
        sns.heatmap(diff_raw, annot=True, cmap='RdBu_r', fmt=".3f", center=0,
                   vmin=-global_max_abs_val, vmax=global_max_abs_val, square=True)
        plt.title(f"Difference: Real vs Raw Synthetic", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=figsize)
        sns.heatmap(diff_refined, annot=True, cmap='RdBu_r', fmt=".3f", center=0,
                   vmin=-global_max_abs_val, vmax=global_max_abs_val, square=True)
        plt.title(f"Difference: Real vs Refined Synthetic", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    # RETURN COMPREHENSIVE RESULTS
    results = {
        'correlation_matrices': {
            'real': corr_real,
            'raw_synthetic': corr_raw,
            'refined_synthetic': corr_refined
        },
        'difference_matrices': {
            'real_vs_raw': diff_raw,
            'real_vs_refined': diff_refined
        },
        'similarity_scores': {
            'raw_synthetic': scores_raw,
            'refined_synthetic': scores_refined,
            'improvement_summary': improvement_summary,
            'consensus_improvement': np.mean(improvements) if improvements else None,
            'consensus_std': np.std(improvements) if improvements else None
        },
        'scale_analysis': {
            'real_structure': real_structure if detailed_analysis and use_mixed else None,
            'potential_bias_detected': max(ranges.values()) / min(ranges.values()) > 1.5 if detailed_analysis and use_mixed and len(analysis_num_cols) > 0 and len(analysis_cat_cols) > 0 else False
        },
        'configuration': {
            'use_mixed': use_mixed,
            'correlation_method': method,
            'correlation_type': correlation_type,
            'matrix_dimensions': corr_real.shape,
            'numerical_columns': analysis_num_cols,
            'categorical_columns': analysis_cat_cols,
            'analysis_columns': analysis_cols
        },
        'methodological_notes': {
            'scale_bias_addressed': True,
            'multiple_metrics_used': True,
            'consensus_method': 'average_across_normalized_metrics',
            'validation_approach': 'separate_analysis_by_correlation_type'
        }
    }
    
    print(f"Analysis complete! Check results['methodological_notes'] for validation details.")
    
    return results

###################### below version handled the asymmetric correlation ratio issue ######################

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import chi2_contingency
# from typing import List, Optional, Tuple, Dict, Any
# import warnings
# warnings.filterwarnings('ignore')

# def _cramers_v(var1, var2):
#     """Calculate Cramer's V between two categorical variables."""
#     crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
#     stat = chi2_contingency(crosstab)[0]
#     obs = np.sum(crosstab)
#     mini = min(crosstab.shape) - 1
#     return np.sqrt(stat / (obs * mini + 1e-16))

# def _correlation_ratio(categories, measurements):
#     """Calculate the correlation ratio (eta squared) between categorical and numerical variables."""
#     fcat, _ = pd.factorize(categories)
#     cat_num = np.max(fcat) + 1
#     y_avg_array = np.zeros(cat_num)
#     n_array = np.zeros(cat_num)
    
#     for i in range(cat_num):
#         cat_measures = measurements[fcat == i]
#         n_array[i] = len(cat_measures)
#         y_avg_array[i] = np.average(cat_measures) if len(cat_measures) > 0 else 0
    
#     y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
#     numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
#     denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    
#     if numerator == 0 or denominator == 0:
#         eta = 0.0
#     else:
#         eta = numerator / denominator
    
#     return eta

# def _apply_mat(data, func, labs1, labs2):
#     """Apply a function across combinations of labels to create a matrix."""
#     res = [func(data[lab1], data[lab2]) for lab1 in labs1 for lab2 in labs2]
#     return pd.DataFrame(
#         np.array(res).reshape(len(labs1), len(labs2)), 
#         columns=labs2, 
#         index=labs1
#     )

# def mixed_correlation(data, num_cols, cat_cols, method='spearman', symmetrize='average'):
#     """
#     Calculate mixed correlation matrix using appropriate correlation measures.
    
#     Args:
#         data: Input DataFrame
#         num_cols: Numerical column names
#         cat_cols: Categorical column names  
#         method: Correlation method for numerical variables
#         symmetrize: How to handle asymmetric correlation ratio ('average', 'max', 'asymmetric')
#     """
#     # Correlation for numerical-numerical using specified method
#     if len(num_cols) > 0:
#         corr_num_num = data[num_cols].corr(method=method)
#     else:
#         corr_num_num = pd.DataFrame()
    
#     # Cramer's V for categorical-categorical (already symmetric)
#     if len(cat_cols) > 0:
#         corr_cat_cat = _apply_mat(data, _cramers_v, cat_cols, cat_cols)
#     else:
#         corr_cat_cat = pd.DataFrame()
    
#     # Handle categorical-numerical correlations (ASYMMETRIC!)
#     if len(cat_cols) > 0 and len(num_cols) > 0:
#         # Categorical → Numerical: η²(numerical | categorical)
#         corr_cat_num = _apply_mat(data, _correlation_ratio, cat_cols, num_cols)
        
#         # Numerical → Categorical: η²(categorical | numerical) 
#         # Note: This is conceptually different but we need a symmetric matrix
#         corr_num_cat = _apply_mat(data, _correlation_ratio, num_cols, cat_cols)
        
#         # Handle asymmetry based on chosen method
#         if symmetrize == 'average':
#             # Take average of both directions
#             corr_mixed = (corr_cat_num + corr_num_cat.transpose()) / 2
#         elif symmetrize == 'max':
#             # Take maximum (more conservative)
#             corr_mixed = np.maximum(corr_cat_num, corr_num_cat.transpose())
#         elif symmetrize == 'asymmetric':
#             # Keep asymmetric (upper triangle = cat→num, lower = num→cat)
#             corr_mixed = corr_cat_num.copy()
#             # Note: This would require special handling in visualization
#         else:
#             raise ValueError("symmetrize must be 'average', 'max', or 'asymmetric'")
            
#     else:
#         corr_mixed = pd.DataFrame()
    
#     # Combine matrices
#     if corr_cat_cat.empty and corr_num_num.empty:
#         return pd.DataFrame()
#     elif corr_cat_cat.empty:
#         return corr_num_num
#     elif corr_num_num.empty:
#         return corr_cat_cat
#     else:
#         if not corr_mixed.empty:
#             top_row = pd.concat([corr_cat_cat, corr_mixed], axis=1)
#             bot_row = pd.concat([corr_mixed.transpose(), corr_num_num], axis=1)
#         else:
#             # Fallback if no mixed correlations
#             top_row = corr_cat_cat
#             bot_row = corr_num_num
            
#         corr = pd.concat([top_row, bot_row], axis=0)
#         return corr + np.diag(1 - np.diag(corr))

# def identify_column_types(data):
#     """Automatically identify numerical and categorical columns."""
#     num_cols = []
#     cat_cols = []
    
#     for col in data.columns:
#         if pd.api.types.is_numeric_dtype(data[col]):
#             unique_ratio = data[col].nunique() / len(data)
#             if unique_ratio < 0.05 and data[col].nunique() < 20:
#                 cat_cols.append(col)
#             else:
#                 num_cols.append(col)
#         else:
#             cat_cols.append(col)
    
#     return num_cols, cat_cols

# def analyze_correlation_structure(corr_matrix, num_cols, cat_cols):
#     """
#     Analyze the structure of a mixed correlation matrix to understand potential biases.
    
#     Returns detailed statistics about different correlation types.
#     """
#     n_num = len(num_cols)
#     n_cat = len(cat_cols)
    
#     # Extract different types of correlations
#     if n_num > 0 and n_cat > 0:
#         # Numerical-numerical correlations (excluding diagonal)
#         num_num_corr = corr_matrix.loc[num_cols, num_cols].values
#         num_num_mask = ~np.eye(n_num, dtype=bool)
#         num_num_values = num_num_corr[num_num_mask]
        
#         # Categorical-categorical correlations (excluding diagonal) 
#         cat_cat_corr = corr_matrix.loc[cat_cols, cat_cols].values
#         cat_cat_mask = ~np.eye(n_cat, dtype=bool)
#         cat_cat_values = cat_cat_corr[cat_cat_mask]
        
#         # Categorical-numerical correlations
#         cat_num_values = corr_matrix.loc[cat_cols, num_cols].values.flatten()
        
#         structure_analysis = {
#             'num_num': {
#                 'count': len(num_num_values),
#                 'mean': np.mean(np.abs(num_num_values)),
#                 'std': np.std(np.abs(num_num_values)),
#                 'range': [np.min(num_num_values), np.max(num_num_values)],
#                 'abs_mean': np.mean(np.abs(num_num_values))
#             },
#             'cat_cat': {
#                 'count': len(cat_cat_values),
#                 'mean': np.mean(cat_cat_values),
#                 'std': np.std(cat_cat_values),
#                 'range': [np.min(cat_cat_values), np.max(cat_cat_values)],
#                 'abs_mean': np.mean(np.abs(cat_cat_values))
#             },
#             'cat_num': {
#                 'count': len(cat_num_values),
#                 'mean': np.mean(cat_num_values),
#                 'std': np.std(cat_num_values),
#                 'range': [np.min(cat_num_values), np.max(cat_num_values)],
#                 'abs_mean': np.mean(np.abs(cat_num_values))
#             }
#         }
#     else:
#         structure_analysis = {'warning': 'Mixed analysis requires both numerical and categorical columns'}
    
#     return structure_analysis

# def compute_multiple_similarity_scores(diff_matrix, corr_real, num_cols, cat_cols):
#     """
#     Compute multiple similarity scores to address the scale bias concern.
    
#     Returns different approaches to quantify similarity that handle scale differences.
#     """
#     scores = {}
    
#     # 1. Standard Frobenius norm (your current approach)
#     scores['frobenius_raw'] = np.linalg.norm(diff_matrix.values, ord='fro')
#     scores['frobenius_normalized'] = scores['frobenius_raw'] / np.sqrt(diff_matrix.size)
    
#     # 2. Element-wise approach: separate scores by correlation type
#     if len(num_cols) > 0 and len(cat_cols) > 0:
#         n_num = len(num_cols)
#         n_cat = len(cat_cols)
        
#         # Extract different parts
#         num_num_diff = diff_matrix.loc[num_cols, num_cols].values
#         cat_cat_diff = diff_matrix.loc[cat_cols, cat_cols].values  
#         cat_num_diff = diff_matrix.loc[cat_cols, num_cols].values
        
#         # Compute separate scores (excluding diagonal)
#         num_num_mask = ~np.eye(n_num, dtype=bool)
#         cat_cat_mask = ~np.eye(n_cat, dtype=bool)
        
#         scores['num_num_mae'] = np.mean(np.abs(num_num_diff[num_num_mask]))
#         scores['cat_cat_mae'] = np.mean(np.abs(cat_cat_diff[cat_cat_mask]))
#         scores['cat_num_mae'] = np.mean(np.abs(cat_num_diff))
        
#         scores['num_num_mse'] = np.mean(num_num_diff[num_num_mask]**2)
#         scores['cat_cat_mse'] = np.mean(cat_cat_diff[cat_cat_mask]**2)
#         scores['cat_num_mse'] = np.mean(cat_num_diff**2)
        
#         # Weighted average (equal weight to each correlation type)
#         scores['weighted_mae'] = (scores['num_num_mae'] + scores['cat_cat_mae'] + scores['cat_num_mae']) / 3
#         scores['weighted_mse'] = (scores['num_num_mse'] + scores['cat_cat_mse'] + scores['cat_num_mse']) / 3
    
#     # 3. Normalized by expected variance approach
#     # For numerical correlations: range is [-1,1], so max diff is 2
#     # For categorical correlations: range is [0,1], so max diff is 1
#     total_elements = diff_matrix.size - len(diff_matrix)  # Exclude diagonal
    
#     if len(num_cols) > 0 and len(cat_cols) > 0:
#         num_num_elements = n_num * (n_num - 1)  # Off-diagonal elements
#         cat_cat_elements = n_cat * (n_cat - 1)
#         cat_num_elements = n_cat * n_num * 2  # Both directions
        
#         # Theoretical maximum differences
#         max_num_num_diff = 2.0  # Range [-1,1] 
#         max_cat_cat_diff = 1.0  # Range [0,1]
#         max_cat_num_diff = 1.0  # Range [0,1]
        
#         # Normalize each part by its theoretical maximum
#         if num_num_elements > 0:
#             normalized_num_num = np.sum(np.abs(num_num_diff[num_num_mask])) / (num_num_elements * max_num_num_diff)
#         else:
#             normalized_num_num = 0
            
#         if cat_cat_elements > 0:
#             normalized_cat_cat = np.sum(np.abs(cat_cat_diff[cat_cat_mask])) / (cat_cat_elements * max_cat_cat_diff)
#         else:
#             normalized_cat_cat = 0
            
#         if cat_num_elements > 0:
#             normalized_cat_num = np.sum(np.abs(cat_num_diff)) / (cat_num_elements * max_cat_num_diff)
#         else:
#             normalized_cat_num = 0
        
#         scores['scale_normalized_score'] = (normalized_num_num + normalized_cat_cat + normalized_cat_num) / 3
    
#     # 4. Relative to real correlation magnitude
#     real_abs_mean = np.mean(np.abs(corr_real.values[~np.eye(len(corr_real), dtype=bool)]))
#     if real_abs_mean > 0:
#         scores['relative_to_magnitude'] = scores['frobenius_normalized'] / real_abs_mean
    
#     return scores

# def robust_mixed_correlation_analysis(
#     real_data: pd.DataFrame,
#     raw_synthetic_data: pd.DataFrame,
#     refined_synthetic_data: pd.DataFrame,
#     num_cols: Optional[List[str]] = None,
#     cat_cols: Optional[List[str]] = None,
#     use_mixed: bool = True,
#     method: str = 'spearman',
#     symmetrize: str = 'average',
#     detailed_analysis: bool = True,
#     show_plots: bool = True,
#     figsize: Tuple[int, int] = (12, 8)
# ) -> Dict[str, Any]:
#     """
#     Robust mixed correlation analysis that addresses scale bias concerns.
    
#     Args:
#         symmetrize: How to handle asymmetric correlation ratio ('average', 'max', 'asymmetric')
#                    'average' - takes average of η²(Y|X) and η²(X|Y) 
#                    'max' - takes maximum (more conservative)
#                    'asymmetric' - keeps asymmetric (not recommended for visualization)
#     """
    
#     # Setup and validation
#     if num_cols is None or cat_cols is None:
#         auto_num_cols, auto_cat_cols = identify_column_types(real_data)
#         num_cols = num_cols or auto_num_cols
#         cat_cols = cat_cols or auto_cat_cols
    
#     print("🔍 ROBUST MIXED CORRELATION ANALYSIS")
#     print("="*60)
#     print(f"Addressing the scale bias concern in mixed correlation matrices")
#     print(f"Configuration: use_mixed={use_mixed}, method={method}, symmetrize={symmetrize}")
#     print(f"Numerical columns ({len(num_cols)}): {num_cols}")
#     print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    
#     if use_mixed and len(num_cols) > 0 and len(cat_cols) > 0:
#         print(f"\n⚠️  IMPORTANT: Correlation ratio η² is asymmetric!")
#         print(f"   η²(numerical|categorical) ≠ η²(categorical|numerical)")
#         print(f"   Using '{symmetrize}' method to create symmetric matrix")
#     print()
    
#     # Select columns and prepare data
#     if use_mixed:
#         analysis_cols = num_cols + cat_cols
#         analysis_num_cols = num_cols
#         analysis_cat_cols = cat_cols
#         correlation_type = f"Mixed Correlation ({method.title()} for numerical, {symmetrize} for asymmetric)"
#     else:
#         analysis_cols = num_cols
#         analysis_num_cols = num_cols
#         analysis_cat_cols = []
#         correlation_type = f"{method.title()} Correlation (Numerical Only)"
    
#     # Ensure common columns
#     common_cols = list(set(real_data.columns) & 
#                       set(raw_synthetic_data.columns) & 
#                       set(refined_synthetic_data.columns))
#     analysis_cols = [col for col in analysis_cols if col in common_cols]
    
#     if not analysis_cols:
#         raise ValueError("No common columns found across all datasets")
    
#     # Subset data
#     real_subset = real_data[analysis_cols].copy()
#     raw_subset = raw_synthetic_data[analysis_cols].copy()
#     refined_subset = refined_synthetic_data[analysis_cols].copy()
    
#     # Calculate correlation matrices
#     if use_mixed:
#         corr_real = mixed_correlation(real_subset, analysis_num_cols, analysis_cat_cols, method, symmetrize)
#         corr_raw = mixed_correlation(raw_subset, analysis_num_cols, analysis_cat_cols, method, symmetrize)
#         corr_refined = mixed_correlation(refined_subset, analysis_num_cols, analysis_cat_cols, method, symmetrize)
#     else:
#         corr_real = real_subset.corr(method=method)
#         corr_raw = raw_subset.corr(method=method)
#         corr_refined = refined_subset.corr(method=method)
    
#     # Calculate difference matrices
#     diff_raw = corr_real - corr_raw
#     diff_refined = corr_real - corr_refined
    
#     # DETAILED ANALYSIS: Address the scale bias concern
#     print("📊 SCALE BIAS ANALYSIS")
#     print("-"*30)
    
#     if detailed_analysis and use_mixed and len(num_cols) > 0 and len(cat_cols) > 0:
#         # Analyze correlation structure
#         real_structure = analyze_correlation_structure(corr_real, analysis_num_cols, analysis_cat_cols)
        
#         print("Value Ranges and Typical Magnitudes in Real Data:")
#         print(f"  Numerical-Numerical: range {real_structure['num_num']['range']}, mean(|r|) = {real_structure['num_num']['abs_mean']:.3f}")
#         print(f"  Categorical-Categorical: range {real_structure['cat_cat']['range']}, mean(|V|) = {real_structure['cat_cat']['abs_mean']:.3f}")
#         print(f"  Categorical-Numerical: range {real_structure['cat_num']['range']}, mean(|η|) = {real_structure['cat_num']['abs_mean']:.3f}")
#         print()
        
#         # Check for potential bias
#         ranges = {
#             'num_num': real_structure['num_num']['range'][1] - real_structure['num_num']['range'][0],
#             'cat_cat': real_structure['cat_cat']['range'][1] - real_structure['cat_cat']['range'][0],
#             'cat_num': real_structure['cat_num']['range'][1] - real_structure['cat_num']['range'][0]
#         }
        
#         if max(ranges.values()) / min(ranges.values()) > 1.5:
#             print("⚠️  WARNING: Significant range differences detected!")
#             print("   This could potentially bias the Frobenius norm analysis.")
#             print("   Using multiple validation metrics below...")
#         else:
#             print("✅ Range differences are moderate - standard analysis should be reliable.")
#         print()
    
#     # MULTIPLE SIMILARITY SCORES
#     scores_raw = compute_multiple_similarity_scores(diff_raw, corr_real, analysis_num_cols, analysis_cat_cols)
#     scores_refined = compute_multiple_similarity_scores(diff_refined, corr_real, analysis_num_cols, analysis_cat_cols)
    
#     print("📈 MULTIPLE SIMILARITY METRICS")
#     print("-"*35)
#     print(f"{'Metric':<25} {'Raw':<12} {'Refined':<12} {'Improvement':<12}")
#     print("-"*65)
    
#     improvement_summary = {}
    
#     for metric in ['frobenius_normalized', 'weighted_mae', 'scale_normalized_score']:
#         if metric in scores_raw and metric in scores_refined:
#             raw_score = scores_raw[metric]
#             refined_score = scores_refined[metric]
#             improvement = ((raw_score - refined_score) / raw_score * 100) if raw_score > 0 else 0
#             improvement_summary[metric] = improvement
            
#             print(f"{metric:<25} {raw_score:<12.4f} {refined_score:<12.4f} {improvement:>+11.1f}%")
    
#     print("-"*65)
    
#     # CONSENSUS ASSESSMENT
#     improvements = list(improvement_summary.values())
#     if len(improvements) > 0:
#         avg_improvement = np.mean(improvements)
#         improvement_std = np.std(improvements)
        
#         print(f"\n🎯 CONSENSUS ASSESSMENT:")
#         print(f"Average improvement across metrics: {avg_improvement:+.1f}% ± {improvement_std:.1f}%")
        
#         if improvement_std < 5:
#             print("✅ High consensus - all metrics agree!")
#         elif improvement_std < 15:
#             print("⚠️  Moderate consensus - some variation between metrics")
#         else:
#             print("❌ Low consensus - significant disagreement between metrics")
    
#     # DETAILED BREAKDOWN (if mixed correlation is used)
#     if detailed_analysis and use_mixed and len(analysis_num_cols) > 0 and len(analysis_cat_cols) > 0:
#         print(f"\n🔍 DETAILED BREAKDOWN BY CORRELATION TYPE:")
#         print("-"*45)
        
#         breakdown_metrics = ['num_num_mae', 'cat_cat_mae', 'cat_num_mae']
#         for metric in breakdown_metrics:
#             if metric in scores_raw and metric in scores_refined:
#                 raw_score = scores_raw[metric]
#                 refined_score = scores_refined[metric]
#                 improvement = ((raw_score - refined_score) / raw_score * 100) if raw_score > 0 else 0
                
#                 corr_type = metric.replace('_mae', '').replace('_', '-').title()
#                 print(f"{corr_type:<20}: {raw_score:.4f} → {refined_score:.4f} ({improvement:+.1f}%)")
    
#     # PLOTTING (with consistent scales)
#     if show_plots:
#         print(f"\n📊 Generating visualizations...")
        
#         # Plot correlation matrices
#         plt.figure(figsize=figsize)
#         sns.heatmap(corr_real, annot=True, cmap='coolwarm', fmt=".2f", 
#                    center=0, vmin=-1, vmax=1, square=True)
#         plt.title(f"Real Data - {correlation_type}", fontsize=14, pad=20)
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
        
#         plt.figure(figsize=figsize)
#         sns.heatmap(corr_raw, annot=True, cmap='coolwarm', fmt=".2f",
#                    center=0, vmin=-1, vmax=1, square=True)
#         plt.title(f"Raw Synthetic - {correlation_type}", fontsize=14, pad=20)
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
        
#         plt.figure(figsize=figsize)
#         sns.heatmap(corr_refined, annot=True, cmap='coolwarm', fmt=".2f",
#                    center=0, vmin=-1, vmax=1, square=True)
#         plt.title(f"Refined Synthetic - {correlation_type}", fontsize=14, pad=20)
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
        
#         # Plot difference matrices (with shared scale)
#         global_max_abs_val = max(np.abs(diff_raw.values).max(), np.abs(diff_refined.values).max())
        
#         plt.figure(figsize=figsize)
#         sns.heatmap(diff_raw, annot=True, cmap='RdBu_r', fmt=".3f", center=0,
#                    vmin=-global_max_abs_val, vmax=global_max_abs_val, square=True)
#         plt.title(f"Difference: Real vs Raw Synthetic", fontsize=14, pad=20)
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
        
#         plt.figure(figsize=figsize)
#         sns.heatmap(diff_refined, annot=True, cmap='RdBu_r', fmt=".3f", center=0,
#                    vmin=-global_max_abs_val, vmax=global_max_abs_val, square=True)
#         plt.title(f"Difference: Real vs Refined Synthetic", fontsize=14, pad=20)
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()
    
#     # RETURN COMPREHENSIVE RESULTS
#     results = {
#         'correlation_matrices': {
#             'real': corr_real,
#             'raw_synthetic': corr_raw,
#             'refined_synthetic': corr_refined
#         },
#         'difference_matrices': {
#             'real_vs_raw': diff_raw,
#             'real_vs_refined': diff_refined
#         },
#         'similarity_scores': {
#             'raw_synthetic': scores_raw,
#             'refined_synthetic': scores_refined,
#             'improvement_summary': improvement_summary,
#             'consensus_improvement': np.mean(improvements) if improvements else None,
#             'consensus_std': np.std(improvements) if improvements else None
#         },
#         'scale_analysis': {
#             'real_structure': real_structure if detailed_analysis and use_mixed else None,
#             'potential_bias_detected': max(ranges.values()) / min(ranges.values()) > 1.5 if detailed_analysis and use_mixed and len(analysis_num_cols) > 0 and len(analysis_cat_cols) > 0 else False
#         },
#         'configuration': {
#             'use_mixed': use_mixed,
#             'correlation_method': method,
#             'symmetrize_method': symmetrize,
#             'correlation_type': correlation_type,
#             'matrix_dimensions': corr_real.shape,
#             'numerical_columns': analysis_num_cols,
#             'categorical_columns': analysis_cat_cols,
#             'analysis_columns': analysis_cols
#         },
#         'methodological_notes': {
#             'scale_bias_addressed': True,
#             'multiple_metrics_used': True,
#             'consensus_method': 'average_across_normalized_metrics',
#             'validation_approach': 'separate_analysis_by_correlation_type'
#         }
#     }
    
#     print(f"\n✅ Analysis complete! Check results['methodological_notes'] for validation details.")
    
#     return results