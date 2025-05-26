"""
Visualization utilities for C-MAPS framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Union
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")


def visualize_data_distribution(data_list: List[Union[pd.DataFrame, np.ndarray]],
                               names: List[str],
                               colors: List[str],
                               method: str = 'pca',
                               n_components: int = 2,
                               alpha_list: Optional[List[float]] = None,
                               title: str = "Data Distribution",
                               sample_size: int = 1000,
                               random_seed: int = 42) -> tuple:
    """
    Visualize high-dimensional data distribution using dimensionality reduction.
    
    Args:
        data_list: List of DataFrames or arrays to visualize
        names: List of names for each dataset
        colors: List of colors for each dataset
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of components for dimensionality reduction
        alpha_list: List of alpha values for transparency
        title: Plot title
        sample_size: Max number of points to plot from each dataset
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (figure, axes)
    """
    np.random.seed(random_seed)
    
    if alpha_list is None:
        alpha_list = [0.7] * len(data_list)
    
    # Convert to numpy arrays and sample if needed
    processed_data = []
    for df in data_list:
        if isinstance(df, pd.DataFrame):
            data = df.values
        else:
            data = df
            
        # Sample if dataset is large
        if len(data) > sample_size:
            indices = np.random.choice(len(data), sample_size, replace=False)
            data = data[indices]
        
        processed_data.append(data)
    
    # Combine all data for consistent dimensionality reduction
    combined_data = np.vstack(processed_data)
    
    # Standardize the data
    scaler = StandardScaler()
    combined_data_scaled = scaler.fit_transform(combined_data)
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_seed)
        reduced_data = reducer.fit_transform(combined_data_scaled)
        x_label = f"PCA Component 1 (Variance: {reducer.explained_variance_ratio_[0]:.2%})"
        y_label = f"PCA Component 2 (Variance: {reducer.explained_variance_ratio_[1]:.2%})" if n_components > 1 else ""
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_seed, perplexity=min(30, len(combined_data)//4))
        reduced_data = reducer.fit_transform(combined_data_scaled)
        x_label = "t-SNE Component 1"
        y_label = "t-SNE Component 2" if n_components > 1 else ""
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
    
    # Split the reduced data back
    start_idx = 0
    reduced_data_list = []
    for data in processed_data:
        end_idx = start_idx + len(data)
        reduced_data_list.append(reduced_data[start_idx:end_idx])
        start_idx = end_idx
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i, (data, name, color, alpha) in enumerate(zip(reduced_data_list, names, colors, alpha_list)):
        ax.scatter(data[:, 0], data[:, 1], c=color, alpha=alpha, 
                  label=f"{name} (n={len(data_list[i])})", s=30)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    return fig, ax


def plot_weight_distribution(weights: np.ndarray,
                           title: str = "Importance Weight Distribution",
                           bins: int = 50,
                           color: str = 'steelblue',
                           show_stats: bool = True) -> tuple:
    """
    Plot distribution of importance weights.
    
    Args:
        weights: Array of importance weights
        title: Plot title
        bins: Number of histogram bins
        color: Color for the histogram
        show_stats: Whether to show statistics on the plot
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(weights, bins=bins, alpha=0.7, color=color)
    
    if show_stats:
        ax.axvline(np.median(weights), color='red', linestyle='--',
                  label=f'Median: {np.median(weights):.4f}')
        ax.axvline(np.mean(weights), color='orange', linestyle='--',
                  label=f'Mean: {np.mean(weights):.4f}')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Importance Weight', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_yscale('log')
    
    if show_stats:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_weights_by_identifiability(weights: np.ndarray,
                                   identifiability_flags: np.ndarray,
                                   title: str = "Weights by Identifiability Status") -> tuple:
    """
    Plot importance weights grouped by identifiability status.
    
    Args:
        weights: Array of importance weights
        identifiability_flags: Array of identifiability flags
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weights_not_identifiable = weights[identifiability_flags == 0]
    weights_identifiable = weights[identifiability_flags == 1]
    
    if len(weights_not_identifiable) > 0:
        ax.hist(weights_not_identifiable, bins=50, alpha=0.7, color='steelblue', 
               label='Not Identifiable')
        ax.axvline(np.median(weights_not_identifiable), color='blue', linestyle='--',
                  label=f'Median Not Identifiable: {np.median(weights_not_identifiable):.4f}')
    
    if len(weights_identifiable) > 0:
        ax.hist(weights_identifiable, bins=50, alpha=0.7, color='orangered', 
               label='Identifiable')
        ax.axvline(np.median(weights_identifiable), color='red', linestyle='--',
                  label=f'Median Identifiable: {np.median(weights_identifiable):.4f}')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Importance Weight', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_identifiability_distribution(identifiability_flags: np.ndarray,
                                     title: str = "Identifiability Distribution") -> tuple:
    """
    Plot distribution of identifiability flags.
    
    Args:
        identifiability_flags: Array of identifiability flags
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.countplot(x=identifiability_flags, palette=['royalblue', 'orangered'], ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Identifiability Flag (0: Not Identifiable, 1: Identifiable)')
    ax.set_ylabel('Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Identifiable', 'Identifiable'])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_correlation_heatmap(data1: Union[pd.DataFrame, np.ndarray],
                           data2: Union[pd.DataFrame, np.ndarray],
                           names: List[str] = ['Real', 'Synthetic'],
                           title: str = "Feature Correlation Heatmap") -> tuple:
    """
    Plot correlation heatmaps for two datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        names: Names for the datasets
        title: Plot title
        
    Returns:
        Tuple of (figure, axes)
    """
    if isinstance(data1, np.ndarray):
        data1 = pd.DataFrame(data1, columns=[f'Feature_{i}' for i in range(data1.shape[1])])
    if isinstance(data2, np.ndarray):
        data2 = pd.DataFrame(data2, columns=[f'Feature_{i}' for i in range(data2.shape[1])])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot correlation matrix for first dataset
    corr1 = data1.corr()
    sns.heatmap(corr1, annot=False, cmap='coolwarm', center=0, ax=axes[0])
    axes[0].set_title(f'{names[0]} Data Correlation')
    
    # Plot correlation matrix for second dataset
    corr2 = data2.corr()
    sns.heatmap(corr2, annot=False, cmap='coolwarm', center=0, ax=axes[1])
    axes[1].set_title(f'{names[1]} Data Correlation')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig, axes


def plot_feature_distributions(data1: Union[pd.DataFrame, np.ndarray],
                              data2: Union[pd.DataFrame, np.ndarray],
                              feature_names: Optional[List[str]] = None,
                              names: List[str] = ['Real', 'Synthetic'],
                              max_features: int = 6) -> tuple:
    """
    Plot feature distributions comparison between two datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        feature_names: Names of features to plot
        names: Names for the datasets
        max_features: Maximum number of features to plot
        
    Returns:
        Tuple of (figure, axes)
    """
    if isinstance(data1, np.ndarray):
        data1 = pd.DataFrame(data1)
    if isinstance(data2, np.ndarray):
        data2 = pd.DataFrame(data2)
    
    if feature_names is None:
        feature_names = data1.columns[:max_features]
    else:
        feature_names = feature_names[:max_features]
    
    n_features = len(feature_names)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(feature_names):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        if feature in data1.columns and feature in data2.columns:
            ax.hist(data1[feature], bins=30, alpha=0.7, label=names[0], color='steelblue')
            ax.hist(data2[feature], bins=30, alpha=0.7, label=names[1], color='orangered')
            ax.set_title(f'{feature}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    return fig, axes