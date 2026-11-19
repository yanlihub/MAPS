import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import wasserstein_distance
import warnings

def visualize_distributions(real_data, baseline_data, filtered_data=None, 
                          columns=None, figsize=None, separate_figures=False):
    """
    Visualize distributions of real, baseline, and filtered data in a single column
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    baseline_data : pd.DataFrame
        Baseline synthetic dataset (random sampling without filtering)
    filtered_data : pd.DataFrame, optional
        Filtered synthetic dataset
    columns : list, optional
        List of columns to plot (default is to select a sample)
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    if columns is None:
        # Get all columns that exist in both real and baseline data
        common_cols = [col for col in real_data.columns if col in baseline_data.columns]
        
        # Categorize columns by data type
        continuous_cols = []
        binary_cols = []
        categorical_cols = []
        
        for col in common_cols:
            n_unique = real_data[col].nunique()
            if n_unique > 10:
                continuous_cols.append(col)
            elif n_unique <= 2:
                binary_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Select sample columns from each type
        selected_cols = []
        
        # Add continuous columns
        if continuous_cols:
            selected_cols.extend(continuous_cols[:min(20, len(continuous_cols))])
        
        # Add binary columns
        if binary_cols:
            selected_cols.extend(binary_cols[:min(20, len(binary_cols))])
        
        # Add categorical columns
        if categorical_cols:
            selected_cols.extend(categorical_cols[:min(20, len(categorical_cols))])
        
        # If we have no columns, raise a helpful error
        if not selected_cols:
            raise ValueError("No common columns found between real and baseline data")
        
        columns = selected_cols
    
    if separate_figures:
        # Create separate figures for each subplot
        figures = []
        
        for col in columns:
            # Create individual figure for this column
            fig, ax = plt.subplots(1, 1, figsize=(8, 6) if figsize is None else figsize)
            
            # Check if column exists in all datasets
            if col not in real_data.columns or col not in baseline_data.columns:
                ax.text(0.5, 0.5, f"Column '{col}' not found", ha='center', va='center')
                continue
                
            if filtered_data is not None and col not in filtered_data.columns:
                filtered_data_for_col = None
            else:
                filtered_data_for_col = filtered_data
            
            # Get number of unique values for column type detection
            n_unique = real_data[col].nunique()
            
            if n_unique <= 2:  # Binary feature
                # Get the categories (0, 1 or actual values if different)
                categories = sorted(pd.concat([real_data[col], baseline_data[col]]).unique())
                real_counts = real_data[col].value_counts().reindex(categories, fill_value=0)
                baseline_counts = baseline_data[col].value_counts().reindex(categories, fill_value=0)
                
                x = np.arange(len(categories))
                width = 0.33
                
                ax.bar(x - width/2, real_counts / len(real_data), width, label='Real', alpha=0.7)
                ax.bar(x + width/2, baseline_counts / len(baseline_data), width, label='Raw', alpha=0.7)
                
                if filtered_data_for_col is not None:
                    filtered_counts = filtered_data_for_col[col].value_counts().reindex(categories, fill_value=0)
                    ax.bar(x + width*1.5, filtered_counts / len(filtered_data_for_col), width, label='Refined', alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels([str(cat) for cat in categories])
                ax.set_ylabel('Frequency')
                
            elif n_unique > 10:  # Continuous feature
                # Try KDE plot, but fall back to histogram if KDE fails
                try:
                    sns.kdeplot(real_data[col], ax=ax, label='Real', alpha=0.7)
                    sns.kdeplot(baseline_data[col], ax=ax, label='Raw', alpha=0.7)
                    
                    if filtered_data_for_col is not None:
                        sns.kdeplot(filtered_data_for_col[col], ax=ax, label='Refined', alpha=0.7)
                except:
                    # Fall back to histograms
                    ax.hist(real_data[col], alpha=0.5, label='Real', density=True, bins=20)
                    ax.hist(baseline_data[col], alpha=0.5, label='Raw', density=True, bins=20)
                    
                    if filtered_data_for_col is not None:
                        ax.hist(filtered_data_for_col[col], alpha=0.5, label='Refined', density=True, bins=20)
                    
            else:  # Categorical feature
                # Get top categories (up to 10)
                top_cats = real_data[col].value_counts().nlargest(10).index.tolist()
                
                x = np.arange(len(top_cats))
                width = 0.35
                
                real_counts = real_data[col].value_counts().reindex(top_cats, fill_value=0).values
                baseline_counts = baseline_data[col].value_counts().reindex(top_cats, fill_value=0).values
                
                ax.bar(x - width/2, real_counts / len(real_data), width, label='Real', alpha=0.7)
                ax.bar(x + width/2, baseline_counts / len(baseline_data), width, label='Raw', alpha=0.7)
                
                if filtered_data_for_col is not None:
                    filtered_counts = filtered_data_for_col[col].value_counts().reindex(top_cats, fill_value=0).values
                    ax.bar(x + width*1.5, filtered_counts / len(filtered_data_for_col), width, label='Refined', alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels([str(cat) for cat in top_cats], rotation=45, ha='right')
                ax.set_ylabel('Frequency')
            
            ax.set_title(col, fontsize=14)
            ax.legend(loc='upper right', framealpha=0.8)
            ax.grid(False)
            
            plt.tight_layout()
            figures.append(fig)
            plt.show()  # Display each figure separately
        
        return figures
    
    else:
        n_plots = len(columns)
        
        # Use a single column layout
        if figsize is None:
            # Make each subplot larger and account for total height
            figsize = (12, 5 * n_plots)
        
        # Create a figure with n_plots rows and 1 column
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        
        # Make axes iterable even when n_plots = 1
        if n_plots == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            ax = axes[i]
            
            # Check if column exists in all datasets
            if col not in real_data.columns or col not in baseline_data.columns:
                ax.text(0.5, 0.5, f"Column '{col}' not found", ha='center', va='center')
                continue
                
            if filtered_data is not None and col not in filtered_data.columns:
                filtered_data_for_col = None
            else:
                filtered_data_for_col = filtered_data
            
            # Get number of unique values for column type detection
            n_unique = real_data[col].nunique()
            
            if n_unique <= 2:  # Binary feature
                # Get the categories (0, 1 or actual values if different)
                categories = sorted(pd.concat([real_data[col], baseline_data[col]]).unique())
                real_counts = real_data[col].value_counts().reindex(categories, fill_value=0)
                baseline_counts = baseline_data[col].value_counts().reindex(categories, fill_value=0)
                
                x = np.arange(len(categories))
                width = 0.33
                
                ax.bar(x - width/2, real_counts / len(real_data), width, label='Real', alpha=0.7)
                ax.bar(x + width/2, baseline_counts / len(baseline_data), width, label='Baseline', alpha=0.7)
                
                if filtered_data_for_col is not None:
                    filtered_counts = filtered_data_for_col[col].value_counts().reindex(categories, fill_value=0)
                    ax.bar(x + width*1.5, filtered_counts / len(filtered_data_for_col), width, label='Filtered', alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels([str(cat) for cat in categories])
                ax.set_ylabel('Frequency')
                
            elif n_unique > 10:  # Continuous feature
                # Try KDE plot, but fall back to histogram if KDE fails
                try:
                    sns.kdeplot(real_data[col], ax=ax, label='Real', alpha=0.7)
                    sns.kdeplot(baseline_data[col], ax=ax, label='Baseline', alpha=0.7)
                    
                    if filtered_data_for_col is not None:
                        sns.kdeplot(filtered_data_for_col[col], ax=ax, label='Filtered', alpha=0.7)
                except:
                    # Fall back to histograms
                    ax.hist(real_data[col], alpha=0.5, label='Real', density=True, bins=20)
                    ax.hist(baseline_data[col], alpha=0.5, label='Baseline', density=True, bins=20)
                    
                    if filtered_data_for_col is not None:
                        ax.hist(filtered_data_for_col[col], alpha=0.5, label='Filtered', density=True, bins=20)
                    
            else:  # Categorical feature
                # Get top categories (up to 10)
                top_cats = real_data[col].value_counts().nlargest(10).index.tolist()
                
                x = np.arange(len(top_cats))
                width = 0.35
                
                real_counts = real_data[col].value_counts().reindex(top_cats, fill_value=0).values
                baseline_counts = baseline_data[col].value_counts().reindex(top_cats, fill_value=0).values
                
                ax.bar(x - width/2, real_counts / len(real_data), width, label='Real', alpha=0.7)
                ax.bar(x + width/2, baseline_counts / len(baseline_data), width, label='Baseline', alpha=0.7)
                
                if filtered_data_for_col is not None:
                    filtered_counts = filtered_data_for_col[col].value_counts().reindex(top_cats, fill_value=0).values
                    ax.bar(x + width*1.5, filtered_counts / len(filtered_data_for_col), width, label='Filtered', alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels([str(cat) for cat in top_cats], rotation=45, ha='right')
                ax.set_ylabel('Frequency')
            
            ax.set_title(col, fontsize=14)
            ax.legend(loc='upper right', framealpha=0.8)
            ax.grid(False)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)  # Add more space between subplots
        return fig

def plot_correlation_matrices(real_data, baseline_data, filtered_data=None, 
                             method='spearman', figsize=None, mask_upper=True,
                             cmap="coolwarm", vmin=-1, vmax=1):
    """
    Plot correlation matrices for real, baseline, and filtered data in a vertical column
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    baseline_data : pd.DataFrame
        Baseline synthetic dataset (random sampling without filtering)
    filtered_data : pd.DataFrame, optional
        Filtered synthetic dataset
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'
    figsize : tuple, optional
        Figure size
    mask_upper : bool, default=True
        Whether to mask the upper triangular portion
    cmap : str, default="coolwarm"
        Colormap to use
    vmin, vmax : float, default=-1, 1
        Color scale limits
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Get common columns that are numeric
    common_cols = [col for col in real_data.columns 
                  if col in baseline_data.columns 
                  and pd.api.types.is_numeric_dtype(real_data[col])
                  and pd.api.types.is_numeric_dtype(baseline_data[col])]
    
    # Check if filtered data is provided and update common columns
    if filtered_data is not None:
        common_cols = [col for col in common_cols 
                      if col in filtered_data.columns 
                      and pd.api.types.is_numeric_dtype(filtered_data[col])]
    
    # Create a mask for the upper triangle
    if mask_upper:
        mask = np.triu(np.ones_like(real_data[common_cols].corr(), dtype=bool))
    else:
        mask = None
    
    # Calculate correlation matrices
    real_corr = real_data[common_cols].corr(method=method)
    baseline_corr = baseline_data[common_cols].corr(method=method)
    
    # Determine number of plots and figure size
    n_plots = 3 if filtered_data is not None else 2
    if figsize is None:
        # For vertical layout: width, height = (8, 8 * n_plots)
        figsize = (10, 8 * n_plots)  # Larger plot size
    
    # Create figure and axes - with n_plots rows and 1 column
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    # Plot correlation matrices
    sns.heatmap(real_corr, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax, 
               mask=mask, annot=True, fmt=".2f", square=True)
    axes[0].set_title(f"Real Data\n{method.capitalize()} Correlation", fontsize=14)
    
    sns.heatmap(baseline_corr, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax, 
               mask=mask, annot=True, fmt=".2f", square=True)
    axes[1].set_title(f"Baseline Data\n{method.capitalize()} Correlation", fontsize=14)
    
    if filtered_data is not None:
        filtered_corr = filtered_data[common_cols].corr(method=method)
        sns.heatmap(filtered_corr, ax=axes[2], cmap=cmap, vmin=vmin, vmax=vmax, 
                   mask=mask, annot=True, fmt=".2f", square=True)
        axes[2].set_title(f"Filtered Data\n{method.capitalize()} Correlation", fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Add more space between subplots
    return fig

def calculate_mutual_info_matrix(df):
    """
    Calculate mutual information matrix for a dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    pd.DataFrame
        Mutual information matrix
    """
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Initialize mutual information matrix with zeros
    n_cols = len(df.columns)
    mi_values = np.zeros((n_cols, n_cols))
    
    # Fill matrix
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i == j:
                mi_values[i, j] = 1.0
            else:
                try:
                    # Choose appropriate MI function based on column types
                    if col1 in numeric_cols and col2 in numeric_cols:
                        # Both numeric
                        mi = mutual_info_regression(df[[col1]], df[col2], random_state=42)[0]
                    elif col1 in categorical_cols and col2 in categorical_cols:
                        # Both categorical
                        mi = mutual_info_classif(
                            df[col1].astype('category').cat.codes.values.reshape(-1, 1), 
                            df[col2].astype('category').cat.codes, 
                            random_state=42
                        )[0]
                    elif col1 in numeric_cols and col2 in categorical_cols:
                        # col1 numeric, col2 categorical
                        mi = mutual_info_classif(
                            df[[col1]], 
                            df[col2].astype('category').cat.codes, 
                            random_state=42
                        )[0]
                    else:
                        # col1 categorical, col2 numeric
                        mi = mutual_info_regression(
                            df[col1].astype('category').cat.codes.values.reshape(-1, 1), 
                            df[col2], 
                            random_state=42
                        )[0]
                    
                    # Ensure numeric value
                    mi_values[i, j] = float(mi)
                    mi_values[j, i] = float(mi)  # MI is symmetric
                except Exception as e:
                    warnings.warn(f"Error calculating MI between {col1} and {col2}: {str(e)}")
                    mi_values[i, j] = 0.0
                    mi_values[j, i] = 0.0
    
    # Create dataframe with numeric values
    mi_matrix = pd.DataFrame(mi_values, index=df.columns, columns=df.columns)
    
    # Normalize to [0, 1]
    for i in range(n_cols):
        max_mi = mi_matrix.iloc[i].max()
        if max_mi > 0:
            mi_matrix.iloc[i] = mi_matrix.iloc[i] / max_mi
    
    return mi_matrix

def plot_mutual_info_matrices(real_data, baseline_data, filtered_data=None, 
                             figsize=None, mask_upper=True, cmap="coolwarm"):
    """
    Plot mutual information matrices for real, baseline, and filtered data in a vertical column
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    baseline_data : pd.DataFrame
        Baseline synthetic dataset (random sampling without filtering)
    filtered_data : pd.DataFrame, optional
        Filtered synthetic dataset
    figsize : tuple, optional
        Figure size
    mask_upper : bool, default=True
        Whether to mask the upper triangular portion
    cmap : str, default="viridis"
        Colormap to use
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Get common columns
    common_cols = [col for col in real_data.columns if col in baseline_data.columns]
    
    # Check if filtered data is provided and update common columns
    if filtered_data is not None:
        common_cols = [col for col in common_cols if col in filtered_data.columns]
    
    # Create a mask for the upper triangle
    if mask_upper:
        mask = np.triu(np.ones((len(common_cols), len(common_cols))), k=1)
    else:
        mask = None
    
    # Calculate mutual information matrices
    real_mi = calculate_mutual_info_matrix(real_data[common_cols])
    baseline_mi = calculate_mutual_info_matrix(baseline_data[common_cols])
    
    # Determine number of plots and figure size
    n_plots = 3 if filtered_data is not None else 2
    if figsize is None:
        # For vertical layout: width, height = (10, 8 * n_plots)
        figsize = (16, 8 * n_plots)  # Larger plot size #
    
    # Create figure and axes - with n_plots rows and 1 column
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    # Plot mutual information matrices
    sns.heatmap(real_mi.astype(float), ax=axes[0], cmap=cmap, vmin=0, vmax=1, 
                mask=mask, annot=True, fmt=".2f", square=True)
    axes[0].set_title("Real Data\nMutual Information", fontsize=14)
    
    sns.heatmap(baseline_mi.astype(float), ax=axes[1], cmap=cmap, vmin=0, vmax=1, 
                mask=mask, annot=True, fmt=".2f", square=True)
    axes[1].set_title("Baseline Data\nMutual Information", fontsize=14)
    
    if filtered_data is not None:
        filtered_mi = calculate_mutual_info_matrix(filtered_data[common_cols])
        sns.heatmap(filtered_mi.astype(float), ax=axes[2], cmap=cmap, vmin=0, vmax=1, 
                    mask=mask, annot=True, fmt=".2f", square=True)
        axes[2].set_title("Filtered Data\nMutual Information", fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Add more space between subplots
    return fig

def calculate_distribution_metrics(real_data, baseline_data, filtered_data=None):
    """
    Calculate distribution similarity metrics between datasets
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    baseline_data : pd.DataFrame
        Baseline synthetic dataset (random sampling without filtering)
    filtered_data : pd.DataFrame, optional
        Filtered synthetic dataset
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics for each column and dataset comparison
    """
    # Get common columns
    common_cols = [col for col in real_data.columns if col in baseline_data.columns]
    
    # Check if filtered data is provided and update common columns
    if filtered_data is not None:
        common_cols = [col for col in common_cols if col in filtered_data.columns]
    
    results = []
    
    for col in common_cols:
        col_metrics = {'Column': col}
        
        # Skip columns with all missing values
        if real_data[col].isna().all() or baseline_data[col].isna().all():
            continue
            
        # For numerical columns
        if pd.api.types.is_numeric_dtype(real_data[col]) and pd.api.types.is_numeric_dtype(baseline_data[col]):
            # Wasserstein distance (Earth Mover's Distance)
            real_values = real_data[col].dropna().values
            baseline_values = baseline_data[col].dropna().values
            
            if len(real_values) > 0 and len(baseline_values) > 0:
                try:
                    # Calculate Wasserstein distance between real and baseline
                    wd_baseline = wasserstein_distance(real_values, baseline_values)
                    col_metrics['Wasserstein_Baseline'] = wd_baseline
                    
                    # Calculate Wasserstein distance between real and filtered (if provided)
                    if filtered_data is not None:
                        filtered_values = filtered_data[col].dropna().values
                        if len(filtered_values) > 0:
                            wd_filtered = wasserstein_distance(real_values, filtered_values)
                            col_metrics['Wasserstein_Filtered'] = wd_filtered
                except:
                    # Skip if Wasserstein calculation fails
                    pass
        
        # For categorical/binary columns
        else:
            # Calculate Jensen-Shannon divergence or KL divergence for categorical data
            real_probs = real_data[col].value_counts(normalize=True)
            baseline_probs = baseline_data[col].value_counts(normalize=True)
            
            # Reindex to have the same categories
            all_categories = set(real_probs.index) | set(baseline_probs.index)
            real_probs = real_probs.reindex(all_categories, fill_value=0)
            baseline_probs = baseline_probs.reindex(all_categories, fill_value=0)
            
            # Calculate proportional difference
            prop_diff_baseline = np.mean(np.abs(real_probs - baseline_probs))
            col_metrics['PropDiff_Baseline'] = prop_diff_baseline
            
            # Calculate for filtered data if provided
            if filtered_data is not None:
                filtered_probs = filtered_data[col].value_counts(normalize=True)
                filtered_probs = filtered_probs.reindex(all_categories, fill_value=0)
                prop_diff_filtered = np.mean(np.abs(real_probs - filtered_probs))
                col_metrics['PropDiff_Filtered'] = prop_diff_filtered
        
        results.append(col_metrics)
    
    return pd.DataFrame(results)

def plot_distribution_metrics(real_data, baseline_data, filtered_data=None, figsize=None):
    """
    Plot distribution metrics comparing baseline and filtered data to real data
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    baseline_data : pd.DataFrame
        Baseline synthetic dataset (random sampling without filtering)
    filtered_data : pd.DataFrame, optional
        Filtered synthetic dataset
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    # Calculate metrics
    metrics_df = calculate_distribution_metrics(real_data, baseline_data, filtered_data)
    
    if len(metrics_df) == 0:
        # No metrics were calculated
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No metrics available for comparison", 
                ha='center', va='center', fontsize=14)
        return fig
    
    # Determine if we have Wasserstein or PropDiff metrics
    has_wasserstein = 'Wasserstein_Baseline' in metrics_df.columns
    has_propdiff = 'PropDiff_Baseline' in metrics_df.columns
    
    # Set up figure
    if figsize is None:
        figsize = (12, 6 * (has_wasserstein + has_propdiff))
    
    fig, axes = plt.subplots(has_wasserstein + has_propdiff, 1, figsize=figsize)
    
    if has_wasserstein + has_propdiff == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot Wasserstein distances
    if has_wasserstein:
        wasserstein_df = metrics_df[metrics_df['Wasserstein_Baseline'].notna()]
        
        # Prepare data for plotting
        columns = []
        baseline_values = []
        filtered_values = []
        
        for _, row in wasserstein_df.iterrows():
            columns.append(row['Column'])
            baseline_values.append(row['Wasserstein_Baseline'])
            if filtered_data is not None and 'Wasserstein_Filtered' in row:
                filtered_values.append(row['Wasserstein_Filtered'])
        
        # Create bar plot
        x = np.arange(len(columns))
        width = 0.35
        
        axes[plot_idx].bar(x - width/2, baseline_values, width, label='Baseline', color='blue', alpha=0.7)
        
        if filtered_data is not None and len(filtered_values) == len(columns):
            axes[plot_idx].bar(x + width/2, filtered_values, width, label='Filtered', color='red', alpha=0.7)
        
        # Add labels
        axes[plot_idx].set_xlabel('Features')
        axes[plot_idx].set_ylabel('Wasserstein Distance\n(lower is better)')
        axes[plot_idx].set_title('Wasserstein Distance Between Real and Synthetic Distributions')
        axes[plot_idx].set_xticks(x)
        axes[plot_idx].set_xticklabels(columns, rotation=45, ha='right')
        axes[plot_idx].legend()
        axes[plot_idx].grid(axis='y', alpha=0.3)
        
        plot_idx += 1
    
    # Plot Proportional Differences
    if has_propdiff:
        propdiff_df = metrics_df[metrics_df['PropDiff_Baseline'].notna()]
        
        # Prepare data for plotting
        columns = []
        baseline_values = []
        filtered_values = []
        
        for _, row in propdiff_df.iterrows():
            columns.append(row['Column'])
            baseline_values.append(row['PropDiff_Baseline'])
            if filtered_data is not None and 'PropDiff_Filtered' in row:
                filtered_values.append(row['PropDiff_Filtered'])
        
        # Create bar plot
        x = np.arange(len(columns))
        width = 0.35
        
        axes[plot_idx].bar(x - width/2, baseline_values, width, label='Baseline', color='blue', alpha=0.7)
        
        if filtered_data is not None and len(filtered_values) == len(columns):
            axes[plot_idx].bar(x + width/2, filtered_values, width, label='Filtered', color='red', alpha=0.7)
        
        # Add labels
        axes[plot_idx].set_xlabel('Features')
        axes[plot_idx].set_ylabel('Proportional Difference\n(lower is better)')
        axes[plot_idx].set_title('Proportional Difference Between Real and Synthetic Distributions')
        axes[plot_idx].set_xticks(x)
        axes[plot_idx].set_xticklabels(columns, rotation=45, ha='right')
        axes[plot_idx].legend()
        axes[plot_idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_datasets_comparison(real_data, baseline_data, filtered_data=None, 
                                 selected_features=None, figsize=(15, 12)):
    """
    Comprehensive visualization dashboard comparing real, baseline, and filtered datasets
    
    Parameters:
    -----------
    real_data : pd.DataFrame
        Real dataset
    baseline_data : pd.DataFrame
        Baseline synthetic dataset (random sampling without filtering)
    filtered_data : pd.DataFrame, optional
        Filtered synthetic dataset
    selected_features : list, optional
        List of features to visualize distributions for (if None, auto-selected)
    figsize : tuple, default=(15, 12)
        Figure size for the main figure
        
    Returns:
    --------
    dict
        Dictionary of matplotlib figures
    """
    # Create figures dictionary to store all visualizations
    figures = {}
    
    # 1. Distribution visualizations for selected features
    figures['distributions'] = visualize_distributions(
        real_data, baseline_data, filtered_data, 
        columns=selected_features
    )
    
    # 2. Correlation matrices
    figures['correlations'] = plot_correlation_matrices(
        real_data, baseline_data, filtered_data
    )
    
    # 3. Mutual information matrices
    figures['mutual_info'] = plot_mutual_info_matrices(
        real_data, baseline_data, filtered_data
    )
    
    # 4. Distribution metrics
    figures['distribution_metrics'] = plot_distribution_metrics(
        real_data, baseline_data, filtered_data
    )
    
    # Create a combined figure for overall comparison
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Add title to the combined figure
    fig.suptitle('Comprehensive Dataset Comparison Dashboard', fontsize=16, y=0.98)
    
    # Create subfigures for different visualizations
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))  # Top left
    axes.append(fig.add_subplot(gs[0, 1]))  # Top right
    axes.append(fig.add_subplot(gs[1, 0]))  # Bottom left
    axes.append(fig.add_subplot(gs[1, 1]))  # Bottom right
    
    # 1. Summary statistics comparison (top left)
    # Select numerical columns for summary statistics
    numeric_cols = real_data.select_dtypes(include=['number']).columns
    common_numeric_cols = [col for col in numeric_cols if col in baseline_data.columns]
    
    if filtered_data is not None:
        common_numeric_cols = [col for col in common_numeric_cols if col in filtered_data.columns]
    
    if len(common_numeric_cols) > 0:
        # Calculate basic statistics
        real_stats = real_data[common_numeric_cols].describe()
        baseline_stats = baseline_data[common_numeric_cols].describe()
        
        if filtered_data is not None:
            filtered_stats = filtered_data[common_numeric_cols].describe()
            
            # Calculate improvement percentage
            improvement = (np.abs(baseline_stats - real_stats) - np.abs(filtered_stats - real_stats)) / np.abs(baseline_stats - real_stats)
            improvement = improvement.fillna(0)
            
            # Plot heatmap of improvements for key statistics
            key_stats = ['mean', 'std', '50%']
            improvement_subset = improvement.loc[key_stats]
            
            sns.heatmap(improvement_subset, cmap='RdYlGn', center=0, annot=True, fmt='.2f', ax=axes[0])
            axes[0].set_title('Filtering Improvement\n(% reduction in error to real data)')
        else:
            # Just show comparison between real and baseline
            diff_pct = (baseline_stats - real_stats) / real_stats * 100
            key_stats = ['mean', 'std', '50%']
            diff_subset = diff_pct.loc[key_stats]
            
            sns.heatmap(diff_subset, cmap='RdYlGn', center=0, annot=True, fmt='.2f', ax=axes[0])
            axes[0].set_title('Baseline % Difference from Real Data')
    else:
        axes[0].text(0.5, 0.5, "No common numeric columns available", 
                    ha='center', va='center', fontsize=12)
        axes[0].set_title('Summary Statistics Comparison')
    
    # 2. Sample correlation difference matrix (top right)
    numeric_cols = real_data.select_dtypes(include=['number']).columns
    common_numeric_cols = [col for col in numeric_cols if col in baseline_data.columns]
    
    if filtered_data is not None:
        common_numeric_cols = [col for col in common_numeric_cols if col in filtered_data.columns]
    
    if len(common_numeric_cols) >= 2:  # Need at least 2 columns for correlation
        real_corr = real_data[common_numeric_cols].corr()
        baseline_corr = baseline_data[common_numeric_cols].corr()
        
        if filtered_data is not None:
            filtered_corr = filtered_data[common_numeric_cols].corr()
            # Show improvement in correlation matrix
            baseline_diff = np.abs(real_corr - baseline_corr)
            filtered_diff = np.abs(real_corr - filtered_corr)
            improvement = (baseline_diff - filtered_diff)
            
            # Mask the diagonal
            mask = np.zeros_like(improvement, dtype=bool)
            np.fill_diagonal(mask, True)
            
            sns.heatmap(improvement, cmap='RdYlGn', center=0, annot=True, fmt='.2f', 
                       mask=mask, vmin=-1, vmax=1, ax=axes[1])
            axes[1].set_title('Correlation Improvement\n(Filtered vs Baseline)')
        else:
            # Show difference between real and baseline correlation
            diff = real_corr - baseline_corr
            
            # Mask the diagonal
            mask = np.zeros_like(diff, dtype=bool)
            np.fill_diagonal(mask, True)
            
            sns.heatmap(diff, cmap='RdYlGn', center=0, annot=True, fmt='.2f', 
                       mask=mask, vmin=-1, vmax=1, ax=axes[1])
            axes[1].set_title('Correlation Difference\n(Real - Baseline)')
    else:
        axes[1].text(0.5, 0.5, "Insufficient numeric columns for correlation", 
                    ha='center', va='center', fontsize=12)
        axes[1].set_title('Correlation Comparison')
    
    # 3. Distribution comparison for a selected feature (bottom left)
    if len(common_numeric_cols) > 0:
        # Select the first numeric column for visualization
        selected_col = common_numeric_cols[0]
        
        # Plot distributions
        sns.kdeplot(real_data[selected_col], ax=axes[2], label='Real', alpha=0.7)
        sns.kdeplot(baseline_data[selected_col], ax=axes[2], label='Baseline', alpha=0.7)
        
        if filtered_data is not None:
            sns.kdeplot(filtered_data[selected_col], ax=axes[2], label='Filtered', alpha=0.7)
            
        axes[2].set_title(f'Distribution Comparison\n{selected_col}')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No suitable columns for distribution visualization", 
                    ha='center', va='center', fontsize=12)
        axes[2].set_title('Distribution Comparison')
    
    # 4. Retention/filtering statistics (bottom right)
    axes[3].axis('off')  # Turn off the axes
    
    # Create a table with statistics
    total_samples = len(baseline_data)
    stats_text = f"Total baseline samples: {total_samples:,}\n\n"
    
    if filtered_data is not None:
        retained_samples = len(filtered_data)
        retention_rate = retained_samples / total_samples * 100
        stats_text += f"Retained samples: {retained_samples:,} ({retention_rate:.1f}%)\n"
        stats_text += f"Filtered out: {total_samples - retained_samples:,} ({100 - retention_rate:.1f}%)\n\n"
    
    # Add data shape information
    stats_text += f"Real data shape: {real_data.shape}\n"
    stats_text += f"Baseline data shape: {baseline_data.shape}\n"
    if filtered_data is not None:
        stats_text += f"Filtered data shape: {filtered_data.shape}\n"
    
    axes[3].text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12, 
                transform=axes[3].transAxes)
    axes[3].set_title('Dataset Statistics')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    figures['dashboard'] = fig
    
    return figures