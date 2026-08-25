import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

def mixed_correlation_lower_triangle(data, num_cols, cat_cols, method='spearman'):
    """
    Calculate mixed correlation matrix with proper handling of asymmetry.
    Returns only the lower triangle with categorical->numerical correlation ratios.
    
    Matrix structure (lower triangle only):
    - Top-left: Categorical-Categorical (Cramer's V) - symmetric
    - Bottom-left: Categorical->Numerical (Correlation Ratio η²) - asymmetric, only cat->num
    - Bottom-right: Numerical-Numerical (Spearman/Pearson) - symmetric
    """
    
    # Reorder columns: categorical first, then numerical
    ordered_cols = cat_cols + num_cols
    ordered_data = data[ordered_cols]
    
    n_cat = len(cat_cols)
    n_num = len(num_cols)
    n_total = n_cat + n_num
    
    # Initialize correlation matrix
    corr_matrix = pd.DataFrame(
        np.eye(n_total), 
        index=ordered_cols, 
        columns=ordered_cols
    )
    
    # 1. Categorical-Categorical correlations (Cramer's V) - upper-left block
    if n_cat > 0:
        corr_cat_cat = _apply_mat(ordered_data, _cramers_v, cat_cols, cat_cols)
        corr_matrix.loc[cat_cols, cat_cols] = corr_cat_cat
    
    # 2. Numerical-Numerical correlations - bottom-right block
    if n_num > 0:
        corr_num_num = ordered_data[num_cols].corr(method=method)
        corr_matrix.loc[num_cols, num_cols] = corr_num_num
    
    # 3. Categorical->Numerical correlations (Correlation Ratio) - bottom-left block only
    if n_cat > 0 and n_num > 0:
        for cat_var in cat_cols:
            for num_var in num_cols:
                # Only compute categorical -> numerical (asymmetric)
                eta_squared = _correlation_ratio(ordered_data[cat_var], ordered_data[num_var])
                corr_matrix.loc[num_var, cat_var] = eta_squared
    
    # Convert to lower triangle only
    lower_triangle_mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix_lower = corr_matrix.where(lower_triangle_mask)
    
    return corr_matrix_lower, n_cat, n_num

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

def analyze_correlation_structure_asymmetric(corr_matrix, n_cat, n_num):
    """
    Analyze the structure of an asymmetric mixed correlation matrix.
    """
    
    # Extract different correlation types from lower triangle
    structure_analysis = {}
    
    if n_cat > 0:
        # Categorical-categorical correlations (excluding diagonal)
        cat_indices = list(range(n_cat))
        cat_cat_values = []
        for i in range(n_cat):
            for j in range(i):  # Lower triangle only
                cat_cat_values.append(corr_matrix.iloc[i, j])
        
        if cat_cat_values:
            structure_analysis['cat_cat'] = {
                'count': len(cat_cat_values),
                'mean': np.mean(cat_cat_values),
                'std': np.std(cat_cat_values),
                'range': [np.min(cat_cat_values), np.max(cat_cat_values)],
                'abs_mean': np.mean(np.abs(cat_cat_values))
            }
    
    if n_num > 0:
        # Numerical-numerical correlations (excluding diagonal)
        num_indices = list(range(n_cat, n_cat + n_num))
        num_num_values = []
        for i in range(n_cat, n_cat + n_num):
            for j in range(n_cat, i):  # Lower triangle only
                num_num_values.append(corr_matrix.iloc[i, j])
        
        if num_num_values:
            structure_analysis['num_num'] = {
                'count': len(num_num_values),
                'mean': np.mean(num_num_values),
                'std': np.std(num_num_values),
                'range': [np.min(num_num_values), np.max(num_num_values)],
                'abs_mean': np.mean(np.abs(num_num_values))
            }
    
    if n_cat > 0 and n_num > 0:
        # Categorical->numerical correlations
        cat_num_values = []
        for i in range(n_cat, n_cat + n_num):  # Numerical variables (rows)
            for j in range(n_cat):  # Categorical variables (columns)
                cat_num_values.append(corr_matrix.iloc[i, j])
        
        if cat_num_values:
            structure_analysis['cat_num'] = {
                'count': len(cat_num_values),
                'mean': np.mean(cat_num_values),
                'std': np.std(cat_num_values),
                'range': [np.min(cat_num_values), np.max(cat_num_values)],
                'abs_mean': np.mean(np.abs(cat_num_values))
            }
    
    return structure_analysis

def compute_similarity_scores_asymmetric(diff_matrix, n_cat, n_num):
    """
    Compute similarity scores for asymmetric correlation matrices.
    """
    scores = {}
    
    # Overall scores
    lower_triangle_mask = np.tril(np.ones_like(diff_matrix, dtype=bool), k=-1)
    lower_triangle_values = diff_matrix.values[lower_triangle_mask]
    
    scores['frobenius_raw'] = np.linalg.norm(lower_triangle_values)
    scores['frobenius_normalized'] = scores['frobenius_raw'] / np.sqrt(len(lower_triangle_values))
    scores['mae_overall'] = np.mean(np.abs(lower_triangle_values))
    scores['mse_overall'] = np.mean(lower_triangle_values**2)
    
    # Separate scores by correlation type
    if n_cat > 0 and n_num > 0:
        # Cat-cat (lower triangle)
        cat_cat_values = []
        for i in range(n_cat):
            for j in range(i):
                cat_cat_values.append(diff_matrix.iloc[i, j])
        
        # Num-num (lower triangle)
        num_num_values = []
        for i in range(n_cat, n_cat + n_num):
            for j in range(n_cat, i):
                num_num_values.append(diff_matrix.iloc[i, j])
        
        # Cat->num
        cat_num_values = []
        for i in range(n_cat, n_cat + n_num):
            for j in range(n_cat):
                cat_num_values.append(diff_matrix.iloc[i, j])
        
        if cat_cat_values:
            scores['cat_cat_mae'] = np.mean(np.abs(cat_cat_values))
            scores['cat_cat_mse'] = np.mean(np.array(cat_cat_values)**2)
        
        if num_num_values:
            scores['num_num_mae'] = np.mean(np.abs(num_num_values))
            scores['num_num_mse'] = np.mean(np.array(num_num_values)**2)
        
        if cat_num_values:
            scores['cat_num_mae'] = np.mean(np.abs(cat_num_values))
            scores['cat_num_mse'] = np.mean(np.array(cat_num_values)**2)
        
        # Weighted average
        available_maes = [scores.get('cat_cat_mae'), scores.get('num_num_mae'), scores.get('cat_num_mae')]
        available_maes = [x for x in available_maes if x is not None]
        if available_maes:
            scores['weighted_mae'] = np.mean(available_maes)
    
    return scores

def plot_asymmetric_correlation_matrix(corr_matrix, n_cat, n_num, title, figsize=(10, 8)):
    """
    Plot asymmetric correlation matrix with colored boxes indicating different correlation types.
    Fixed version with non-overlapping boxes and consistent line widths.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle (set to NaN)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix_masked = corr_matrix.copy()
    corr_matrix_masked = corr_matrix_masked.where(~mask)
    
    # Plot heatmap
    sns.heatmap(corr_matrix_masked, annot=True, cmap='coolwarm', fmt=".2f", 
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={'label': 'Correlation Strength'},
                linewidths=0, linecolor='none')
    
    # Draw non-overlapping colored boxes using Rectangle patches
    if n_cat > 0 and n_num > 0:
        line_width = 3  # Consistent line width
        offset = 0.05   # Small offset to prevent overlap
        
        # Import Rectangle here if not already imported
        from matplotlib.patches import Rectangle
        
        # Red box for categorical-categorical (Cramer's V)
        if n_cat > 1:
            rect_red = Rectangle((offset, offset), 
                               n_cat - 2*offset, n_cat - 2*offset,
                               linewidth=line_width, edgecolor='red', 
                               facecolor='none', linestyle='--')
            ax.add_patch(rect_red)
        
        # Green box for categorical->numerical (Correlation Ratio)
        rect_green = Rectangle((offset, n_cat + offset), 
                             n_cat - 2*offset, n_num - 2*offset,
                             linewidth=line_width, edgecolor='green', 
                             facecolor='none', linestyle='--')
        ax.add_patch(rect_green)
        
        # Blue box for numerical-numerical (Spearman/Pearson)
        if n_num > 1:
            rect_blue = Rectangle((n_cat + offset, n_cat + offset), 
                                n_num - 2*offset, n_num - 2*offset,
                                linewidth=line_width, edgecolor='blue', 
                                facecolor='none', linestyle='--')
            ax.add_patch(rect_blue)
    
    # Add legend for colored boxes
    from matplotlib.patches import Patch
    legend_elements = []
    
    if n_cat > 1:
        legend_elements.append(Patch(facecolor='none', edgecolor='red', linewidth=2, 
                                   label='Categorical-Categorical (Cramer\'s V)'))
    
    if n_cat > 0 and n_num > 0:
        legend_elements.append(Patch(facecolor='none', edgecolor='green', linewidth=2, 
                                   label='Categorical→Numerical (η²)'))
    
    if n_num > 1:
        legend_elements.append(Patch(facecolor='none', edgecolor='blue', linewidth=2, 
                                   label='Numerical-Numerical (Correlation)'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Variables (Rows)', fontsize=12)
    plt.xlabel('Variables (Columns)', fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    return fig, ax

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
    Asymmetric mixed correlation analysis with proper handling of correlation ratio asymmetry.
    
    This function addresses the asymmetry issue by:
    1. Showing only the lower triangle of correlation matrices
    2. Grouping variables (categorical first, then numerical)
    3. Using colored boxes to distinguish correlation types
    4. Only computing categorical->numerical correlation ratios (not bidirectional)
    """
    
    # Setup and validation
    if num_cols is None or cat_cols is None:
        auto_num_cols, auto_cat_cols = identify_column_types(real_data)
        num_cols = num_cols or auto_num_cols
        cat_cols = cat_cols or auto_cat_cols
    
    print("ASYMMETRIC MIXED CORRELATION ANALYSIS")
    print("="*60)
    print("Properly handling correlation ratio asymmetry with lower triangle display")
    print(f"Configuration: use_mixed={use_mixed}, method={method}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical columns ({len(num_cols)}): {num_cols}")
    print("\nMatrix Structure (Lower Triangle Only):")
    print("• Categorical-Categorical: Cramer's V (symmetric, red box)")
    print("• Categorical→Numerical: Correlation Ratio η² (asymmetric, green box)")
    print("• Numerical-Numerical: Correlation coefficient (symmetric, blue box)")
    print()
    
    # Ensure common columns
    common_cols = list(set(real_data.columns) & 
                      set(raw_synthetic_data.columns) & 
                      set(refined_synthetic_data.columns))
    
    # Filter to common columns
    cat_cols_common = [col for col in cat_cols if col in common_cols]
    num_cols_common = [col for col in num_cols if col in common_cols]
    
    if not cat_cols_common and not num_cols_common:
        raise ValueError("No common columns found across all datasets")
    
    # Calculate correlation matrices
    if use_mixed:
        corr_real, n_cat, n_num = mixed_correlation_lower_triangle(
            real_data, num_cols_common, cat_cols_common, method)
        corr_raw, _, _ = mixed_correlation_lower_triangle(
            raw_synthetic_data, num_cols_common, cat_cols_common, method)
        corr_refined, _, _ = mixed_correlation_lower_triangle(
            refined_synthetic_data, num_cols_common, cat_cols_common, method)
        correlation_type = f"Mixed Correlation ({method.title()} for numerical)"
    else:
        # Standard correlation for numerical only
        analysis_cols = num_cols_common
        real_subset = real_data[analysis_cols]
        raw_subset = raw_synthetic_data[analysis_cols]
        refined_subset = refined_synthetic_data[analysis_cols]
        
        corr_real = real_subset.corr(method=method)
        corr_raw = raw_subset.corr(method=method)
        corr_refined = refined_subset.corr(method=method)
        
        # Apply lower triangle mask
        mask = np.triu(np.ones_like(corr_real, dtype=bool))
        corr_real = corr_real.where(~mask)
        corr_raw = corr_raw.where(~mask)
        corr_refined = corr_refined.where(~mask)
        
        n_cat, n_num = 0, len(num_cols_common)
        correlation_type = f"{method.title()} Correlation (Numerical Only)"
    
    # Calculate difference matrices
    diff_raw = corr_real - corr_raw
    diff_refined = corr_real - corr_refined
    
    # DETAILED ANALYSIS
    if detailed_analysis and use_mixed and n_cat > 0 and n_num > 0:
        print("ASYMMETRIC CORRELATION STRUCTURE ANALYSIS")
        print("-"*45)
        
        real_structure = analyze_correlation_structure_asymmetric(corr_real, n_cat, n_num)
        
        if 'cat_cat' in real_structure:
            print(f"Categorical-Categorical (Cramer's V): "
                  f"range {real_structure['cat_cat']['range']}, "
                  f"mean = {real_structure['cat_cat']['mean']:.3f}")
        
        if 'cat_num' in real_structure:
            print(f"Categorical→Numerical (η²): "
                  f"range {real_structure['cat_num']['range']}, "
                  f"mean = {real_structure['cat_num']['mean']:.3f}")
        
        if 'num_num' in real_structure:
            print(f"Numerical-Numerical ({method}): "
                  f"range {real_structure['num_num']['range']}, "
                  f"mean = {real_structure['num_num']['mean']:.3f}")
        print()
    
    # SIMILARITY SCORES
    scores_raw = compute_similarity_scores_asymmetric(diff_raw, n_cat, n_num)
    scores_refined = compute_similarity_scores_asymmetric(diff_refined, n_cat, n_num)
    
    print("SIMILARITY METRICS (Lower Triangle Only)")
    print("-"*45)
    print(f"{'Metric':<25} {'Raw':<12} {'Refined':<12} {'Improvement':<12}")
    print("-"*65)
    
    improvement_summary = {}
    
    for metric in ['frobenius_normalized', 'mae_overall', 'weighted_mae']:
        if metric in scores_raw and metric in scores_refined:
            raw_score = scores_raw[metric]
            refined_score = scores_refined[metric]
            improvement = ((raw_score - refined_score) / raw_score * 100) if raw_score > 0 else 0
            improvement_summary[metric] = improvement
            
            print(f"{metric:<25} {raw_score:<12.4f} {refined_score:<12.4f} {improvement:>+11.1f}%")
    
    print("-"*65)
    
    # NEW FEATURE: DETAILED BREAKDOWN BY CORRELATION TYPE
    print("\nDETAILED BREAKDOWN BY CORRELATION TYPE:")
    print("-"*45)
    
    # Check which correlation types are available and display them
    correlation_breakdown = {}
    
    if 'num_num_mae' in scores_raw and 'num_num_mae' in scores_refined:
        raw_score = scores_raw['num_num_mae']
        refined_score = scores_refined['num_num_mae']
        improvement = ((raw_score - refined_score) / raw_score * 100) if raw_score > 0 else 0
        correlation_breakdown['num_num'] = improvement
        print(f"Num-Num : {raw_score:.4f} → {refined_score:.4f} ({improvement:+.1f}%)")
    
    if 'cat_cat_mae' in scores_raw and 'cat_cat_mae' in scores_refined:
        raw_score = scores_raw['cat_cat_mae']
        refined_score = scores_refined['cat_cat_mae']
        improvement = ((raw_score - refined_score) / raw_score * 100) if raw_score > 0 else 0
        correlation_breakdown['cat_cat'] = improvement
        print(f"Cat-Cat : {raw_score:.4f} → {refined_score:.4f} ({improvement:+.1f}%)")
    
    if 'cat_num_mae' in scores_raw and 'cat_num_mae' in scores_refined:
        raw_score = scores_raw['cat_num_mae']
        refined_score = scores_refined['cat_num_mae']
        improvement = ((raw_score - refined_score) / raw_score * 100) if raw_score > 0 else 0
        correlation_breakdown['cat_num'] = improvement
        print(f"Cat-Num : {raw_score:.4f} → {refined_score:.4f} ({improvement:+.1f}%)")
    
    # CONSENSUS ASSESSMENT
    improvements = list(improvement_summary.values())
    if len(improvements) > 0:
        avg_improvement = np.mean(improvements)
        improvement_std = np.std(improvements)
        
        print(f"\nCONSENSUS ASSESSMENT:")
        print(f"Average improvement across metrics: {avg_improvement:+.1f}% ± {improvement_std:.1f}%")
    
    # PLOTTING
    if show_plots:
        print(f"Generating asymmetric correlation visualizations...")
        
        plot_asymmetric_correlation_matrix(
            corr_real, n_cat, n_num, 
            f"Real Data - {correlation_type}", figsize)
        plt.show()
        
        plot_asymmetric_correlation_matrix(
            corr_raw, n_cat, n_num, 
            f"Raw Synthetic - {correlation_type}", figsize)
        plt.show()
        
        plot_asymmetric_correlation_matrix(
            corr_refined, n_cat, n_num, 
            f"Refined Synthetic - {correlation_type}", figsize)
        plt.show()
        
        # Plot difference matrices
        plot_asymmetric_correlation_matrix(
            diff_raw, n_cat, n_num, 
            f"Difference: Real vs Raw Synthetic", figsize)
        plt.show()
        
        plot_asymmetric_correlation_matrix(
            diff_refined, n_cat, n_num, 
            f"Difference: Real vs Refined Synthetic", figsize)
        plt.show()
    
    # RETURN RESULTS
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
            'correlation_breakdown': correlation_breakdown,  # NEW: Added breakdown to results
            'consensus_improvement': np.mean(improvements) if improvements else None,
            'consensus_std': np.std(improvements) if improvements else None
        },
        'matrix_structure': {
            'n_categorical': n_cat,
            'n_numerical': n_num,
            'categorical_columns': cat_cols_common,
            'numerical_columns': num_cols_common,
            'variable_order': cat_cols_common + num_cols_common
        },
        'configuration': {
            'use_mixed': use_mixed,
            'correlation_method': method,
            'correlation_type': correlation_type,
            'matrix_dimensions': corr_real.shape,
            'asymmetric_handling': True,
            'display_mode': 'lower_triangle_only'
        },
        'methodological_notes': {
            'asymmetry_addressed': True,
            'correlation_ratio_direction': 'categorical_to_numerical_only',
            'visual_indicators': 'colored_boxes_by_correlation_type',
            'matrix_interpretation': 'lower_triangle_due_to_asymmetric_correlation_ratios'
        }
    }
    
    print(f"Asymmetric analysis complete! Lower triangle display resolves correlation ratio asymmetry.")
    
    return results