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

# def plot_asymmetric_correlation_matrix(corr_matrix, n_cat, n_num, title, figsize=(10, 8)):
#     """
#     Plot asymmetric correlation matrix with colored boxes indicating different correlation types.
#     """
    
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Create mask for upper triangle (set to NaN)
#     mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#     corr_matrix_masked = corr_matrix.copy()
#     corr_matrix_masked = corr_matrix_masked.where(~mask)
    
#     # Plot heatmap
#     sns.heatmap(corr_matrix_masked, annot=True, cmap='coolwarm', fmt=".2f", 
#                 center=0, vmin=-1, vmax=1, square=True, ax=ax,
#                 cbar_kws={'label': 'Correlation Strength'},
#                 linewidths=0, linecolor='none')
    
#     # Add colored boxes to indicate different correlation types
#     # if n_cat > 0 and n_num > 0:
#     #     # Red box for categorical-categorical (Cramer's V)
#     #     if n_cat > 1:
#     #         rect_cat_cat = Rectangle((0, 0), n_cat, n_cat, 
#     #                                linewidth=5, edgecolor='red', facecolor='none',linestyle='--')
#     #         ax.add_patch(rect_cat_cat)
        
#     #     # Green box for categorical->numerical (Correlation Ratio)
#     #     rect_cat_num = Rectangle((0, n_cat), n_cat, n_num, 
#     #                            linewidth=5, edgecolor='green', facecolor='none',linestyle='--')
#     #     ax.add_patch(rect_cat_num)
        
#     #     # Blue box for numerical-numerical (Spearman/Pearson)
#     #     if n_num > 1:
#     #         rect_num_num = Rectangle((n_cat, n_cat), n_num, n_num, 
#     #                                linewidth=5, edgecolor='blue', facecolor='none',linestyle='--')
#     #         ax.add_patch(rect_num_num)
#     if n_cat > 0 and n_num > 0:
#         # Define consistent dash pattern for all rectangles
#         # dash_pattern = [8, 4]  # 8 units line, 4 units gap
        
#         # Red dashed box for categorical-categorical
#         if n_cat > 1:
#             rect_cat_cat = Rectangle((0, 0), n_cat, n_cat, 
#                                 linewidth=4, edgecolor='red', facecolor='none',
#                                 linestyle='-', zorder=10)  # Use solid line with custom dashes
#             # rect_cat_cat.set_dashes(dash_pattern)  # Apply custom dash pattern
#             ax.add_patch(rect_cat_cat)
        
#         # Green dashed box for categorical->numerical
#         rect_cat_num = Rectangle((0, n_cat), n_cat, n_num, 
#                             linewidth=4, edgecolor='green', facecolor='none',
#                             linestyle='-', zorder=10)
#         # rect_cat_num.set_dashes(dash_pattern)  # Same dash pattern
#         ax.add_patch(rect_cat_num)
        
#         # Blue dashed box for numerical-numerical
#         if n_num > 1:
#             rect_num_num = Rectangle((n_cat, n_cat), n_num, n_num, 
#                                 linewidth=4, edgecolor='blue', facecolor='none',
#                                 linestyle='-', zorder=10)
#             # rect_num_num.set_dashes(dash_pattern)  # Same dash pattern
#             ax.add_patch(rect_num_num)
    
#     # Add legend for colored boxes
#     from matplotlib.patches import Patch
#     legend_elements = []
    
#     if n_cat > 1:
#         legend_elements.append(Patch(facecolor='none', edgecolor='red', linewidth=2, 
#                                    label='Categorical-Categorical (Cramer\'s V)'))
    
#     if n_cat > 0 and n_num > 0:
#         legend_elements.append(Patch(facecolor='none', edgecolor='green', linewidth=2, 
#                                    label='Categorical→Numerical (η²)'))
    
#     if n_num > 1:
#         legend_elements.append(Patch(facecolor='none', edgecolor='blue', linewidth=2, 
#                                    label='Numerical-Numerical (Correlation)'))
    
#     if legend_elements:
#         ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
#     plt.title(title, fontsize=14, pad=20)
#     plt.xticks(rotation=45, ha='right')
#     plt.ylabel('Variables (Rows)', fontsize=12)
#     plt.xlabel('Variables (Columns)', fontsize=12)

#     # remove the grid of the heatmap
#     plt.grid(False)
    
#     # Add explanatory text
#     # plt.figtext(0.02, 0.02, 
#     #             'Note: Only lower triangle shown due to asymmetric correlation ratios.\n'
#     #             'Categorical→Numerical shows η² (categorical explains numerical variance).',
#     #             fontsize=10, style='italic')
    
#     plt.tight_layout()
#     return fig, ax

def plot_asymmetric_correlation_matrix(corr_matrix, n_cat, n_num, title, figsize=(10, 8)):
    """
    Plot asymmetric correlation matrix with colored boxes indicating different correlation types.
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
    
    # Draw colored boxes using plot() with consistent line width
    if n_cat > 0 and n_num > 0:
        line_width = 4
        
        # Red box for categorical-categorical (Cramer's V)
        if n_cat > 1:
            # Draw rectangle using plot
            ax.plot([0, n_cat, n_cat, 0, 0], [0, 0, n_cat, n_cat, 0], 
                   color='red', linewidth=line_width, linestyle='--')
        
        # Green box for categorical->numerical (Correlation Ratio)
        ax.plot([0, n_cat, n_cat, 0, 0], [n_cat, n_cat, n_cat+n_num, n_cat+n_num, n_cat], 
               color='green', linewidth=line_width, linestyle='--')
        
        # Blue box for numerical-numerical (Spearman/Pearson)
        if n_num > 1:
            ax.plot([n_cat, n_cat+n_num, n_cat+n_num, n_cat, n_cat], 
                   [n_cat, n_cat, n_cat+n_num, n_cat+n_num, n_cat], 
                   color='blue', linewidth=line_width, linestyle='--')
    
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
    
    # CONSENSUS ASSESSMENT
    improvements = list(improvement_summary.values())
    if len(improvements) > 0:
        avg_improvement = np.mean(improvements)
        improvement_std = np.std(improvements)
        
        print(f"CONSENSUS ASSESSMENT:")
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

# def mixed_correlation(data, num_cols, cat_cols, method='spearman'):
#     """Calculate mixed correlation matrix using appropriate correlation measures."""
#     # Correlation for numerical-numerical using specified method
#     if len(num_cols) > 0:
#         corr_num_num = data[num_cols].corr(method=method)
#     else:
#         corr_num_num = pd.DataFrame()
    
#     # Cramer's V for categorical-categorical
#     if len(cat_cols) > 0:
#         corr_cat_cat = _apply_mat(data, _cramers_v, cat_cols, cat_cols)
#     else:
#         corr_cat_cat = pd.DataFrame()
    
#     # Correlation ratio for categorical-numerical
#     if len(cat_cols) > 0 and len(num_cols) > 0:
#         corr_cat_num = _apply_mat(data, _correlation_ratio, cat_cols, num_cols)
#     else:
#         corr_cat_num = pd.DataFrame()
    
#     # Combine matrices
#     if corr_cat_cat.empty and corr_num_num.empty:
#         return pd.DataFrame()
#     elif corr_cat_cat.empty:
#         return corr_num_num
#     elif corr_num_num.empty:
#         return corr_cat_cat
#     else:
#         top_row = pd.concat([corr_cat_cat, corr_cat_num], axis=1)
#         bot_row = pd.concat([corr_cat_num.transpose(), corr_num_num], axis=1)
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
    
#     # 1. Standard Frobenius norm
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

# def mixed_correlation_analysis(
#     real_data: pd.DataFrame,
#     raw_synthetic_data: pd.DataFrame,
#     refined_synthetic_data: pd.DataFrame,
#     num_cols: Optional[List[str]] = None,
#     cat_cols: Optional[List[str]] = None,
#     use_mixed: bool = True,
#     method: str = 'spearman',
#     detailed_analysis: bool = True,
#     show_plots: bool = True,
#     figsize: Tuple[int, int] = (12, 8)
# ) -> Dict[str, Any]:
#     """
#     Mixed correlation analysis that addresses scale bias concerns.
    
#     This function provides multiple validation approaches to ensure reliable results
#     despite different correlation measure ranges.
#     """
    
#     # Setup and validation
#     if num_cols is None or cat_cols is None:
#         auto_num_cols, auto_cat_cols = identify_column_types(real_data)
#         num_cols = num_cols or auto_num_cols
#         cat_cols = cat_cols or auto_cat_cols
    
#     print("MIXED CORRELATION ANALYSIS")
#     print("="*60)
#     print(f"Addressing the scale bias concern in mixed correlation matrices")
#     print(f"Configuration: use_mixed={use_mixed}, method={method}")
#     print(f"Numerical columns ({len(num_cols)}): {num_cols}")
#     print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
#     print()
    
#     # Select columns and prepare data
#     if use_mixed:
#         analysis_cols = num_cols + cat_cols
#         analysis_num_cols = num_cols
#         analysis_cat_cols = cat_cols
#         correlation_type = f"Mixed Correlation ({method.title()} for numerical)"
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
#         corr_real = mixed_correlation(real_subset, analysis_num_cols, analysis_cat_cols, method)
#         corr_raw = mixed_correlation(raw_subset, analysis_num_cols, analysis_cat_cols, method)
#         corr_refined = mixed_correlation(refined_subset, analysis_num_cols, analysis_cat_cols, method)
#     else:
#         corr_real = real_subset.corr(method=method)
#         corr_raw = raw_subset.corr(method=method)
#         corr_refined = refined_subset.corr(method=method)
    
#     # Calculate difference matrices
#     diff_raw = corr_real - corr_raw
#     diff_refined = corr_real - corr_refined
    
#     # DETAILED ANALYSIS: Address the scale bias concern
#     print("SCALE BIAS ANALYSIS")
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
#             print("WARNING: Significant range differences detected!")
#             print("   This could potentially bias the Frobenius norm analysis.")
#             print("   Using multiple validation metrics below...")
#         else:
#             print("Range differences are moderate - standard analysis should be reliable.")
#         print()
    
#     # MULTIPLE SIMILARITY SCORES
#     scores_raw = compute_multiple_similarity_scores(diff_raw, corr_real, analysis_num_cols, analysis_cat_cols)
#     scores_refined = compute_multiple_similarity_scores(diff_refined, corr_real, analysis_num_cols, analysis_cat_cols)
    
#     print("MULTIPLE SIMILARITY METRICS")
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
        
#         print(f"CONSENSUS ASSESSMENT:")
#         print(f"Average improvement across metrics: {avg_improvement:+.1f}% ± {improvement_std:.1f}%")
        
#         if improvement_std < 5:
#             print("High consensus - all metrics agree!")
#         elif improvement_std < 15:
#             print("Moderate consensus - some variation between metrics")
#         else:
#             print("Low consensus - significant disagreement between metrics")
    
#     # DETAILED BREAKDOWN (if mixed correlation is used)
#     if detailed_analysis and use_mixed and len(analysis_num_cols) > 0 and len(analysis_cat_cols) > 0:
#         print(f"DETAILED BREAKDOWN BY CORRELATION TYPE:")
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
#         print(f"Generating visualizations...")
        
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
    
#     print(f"Analysis complete! Check results['methodological_notes'] for validation details.")
    
#     return results

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