"""
Helper utilities for C-MAPS framework.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Any
import warnings
import os
import pickle
import json

warnings.filterwarnings('ignore')


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_data_compatibility(real_data: pd.DataFrame, 
                               synthetic_data: pd.DataFrame,
                               verbose: bool = True) -> Tuple[bool, str]:
    """
    Validate that real and synthetic data are compatible.
    
    Args:
        real_data: Real data DataFrame
        synthetic_data: Synthetic data DataFrame
        verbose: Whether to print validation details
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    # Check if both are DataFrames
    if not isinstance(real_data, pd.DataFrame):
        errors.append("real_data must be a pandas DataFrame")
    if not isinstance(synthetic_data, pd.DataFrame):
        errors.append("synthetic_data must be a pandas DataFrame")
    
    if errors:
        return False, "; ".join(errors)
    
    # Check for empty data
    if len(real_data) == 0:
        errors.append("real_data is empty")
    if len(synthetic_data) == 0:
        errors.append("synthetic_data is empty")
    
    # Check column compatibility
    real_cols = set(real_data.columns)
    synthetic_cols = set(synthetic_data.columns)
    
    if real_cols != synthetic_cols:
        missing_in_synthetic = real_cols - synthetic_cols
        missing_in_real = synthetic_cols - real_cols
        
        if missing_in_synthetic:
            errors.append(f"Columns missing in synthetic data: {missing_in_synthetic}")
        if missing_in_real:
            errors.append(f"Extra columns in synthetic data: {missing_in_real}")
    
    # Check for sufficient data
    if len(real_data) < 10:
        errors.append("real_data has fewer than 10 samples (too small)")
    if len(synthetic_data) < 10:
        errors.append("synthetic_data has fewer than 10 samples (too small)")
    
    # Check for data types compatibility
    common_cols = real_cols & synthetic_cols
    for col in common_cols:
        real_dtype = real_data[col].dtype
        synthetic_dtype = synthetic_data[col].dtype
        
        # Check if both are numeric or both are object/string
        real_is_numeric = pd.api.types.is_numeric_dtype(real_dtype)
        synthetic_is_numeric = pd.api.types.is_numeric_dtype(synthetic_dtype)
        
        if real_is_numeric != synthetic_is_numeric:
            errors.append(f"Column '{col}' has incompatible types: "
                         f"real={real_dtype}, synthetic={synthetic_dtype}")
    
    if verbose and not errors:
        print("✓ Data validation passed")
        print(f"  Real data: {real_data.shape}")
        print(f"  Synthetic data: {synthetic_data.shape}")
        print(f"  Common columns: {len(common_cols)}")
    
    if errors:
        error_message = "; ".join(errors)
        if verbose:
            print("✗ Data validation failed:")
            for error in errors:
                print(f"  - {error}")
        return False, error_message
    
    return True, ""


def save_results(results: dict, 
                filepath: str,
                format: str = 'pickle') -> bool:
    """
    Save results to file.
    
    Args:
        results: Results dictionary to save
        filepath: Path to save file
        format: Save format ('pickle' or 'json')
        
    Returns:
        Success status
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_results = convert_numpy_to_json_serializable(results)
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved to: {filepath}")
        return True
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def load_results(filepath: str, format: str = 'pickle') -> Optional[dict]:
    """
    Load results from file.
    
    Args:
        filepath: Path to load file
        format: Load format ('pickle' or 'json')
        
    Returns:
        Loaded results dictionary or None if failed
    """
    try:
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
        elif format == 'json':
            with open(filepath, 'r') as f:
                results = json.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results loaded from: {filepath}")
        return results
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def convert_numpy_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy arrays and other non-JSON-serializable objects to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_json_serializable(item) for item in obj)
    else:
        return obj


def create_data_summary(data: pd.DataFrame, name: str = "Dataset") -> dict:
    """
    Create a summary of dataset characteristics.
    
    Args:
        data: Input DataFrame
        name: Name for the dataset
        
    Returns:
        Dictionary with dataset summary
    """
    summary = {
        'name': name,
        'shape': data.shape,
        'n_samples': len(data),
        'n_features': len(data.columns),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_values': data.isnull().sum().sum(),
        'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
        'duplicate_rows': data.duplicated().sum(),
        'feature_types': {}
    }
    
    # Analyze feature types
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            if data[col].nunique() <= 2:
                summary['feature_types'][col] = 'binary'
            elif data[col].nunique() < 20 and data[col].nunique() / len(data) < 0.05:
                summary['feature_types'][col] = 'categorical_numeric'
            else:
                summary['feature_types'][col] = 'continuous'
        else:
            if data[col].nunique() <= 2:
                summary['feature_types'][col] = 'binary_categorical'
            else:
                summary['feature_types'][col] = 'categorical'
    
    # Count feature types
    feature_type_counts = {}
    for ftype in summary['feature_types'].values():
        feature_type_counts[ftype] = feature_type_counts.get(ftype, 0) + 1
    
    summary['feature_type_counts'] = feature_type_counts
    
    return summary


def print_data_summary(real_data: pd.DataFrame, 
                      synthetic_data: pd.DataFrame,
                      selected_data: Optional[pd.DataFrame] = None):
    """
    Print a formatted summary of datasets.
    
    Args:
        real_data: Real dataset
        synthetic_data: Synthetic dataset  
        selected_data: Selected/refined synthetic dataset (optional)
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    real_summary = create_data_summary(real_data, "Real Data")
    synthetic_summary = create_data_summary(synthetic_data, "Synthetic Data")
    
    print(f"\nREAL DATA:")
    print(f"  Shape: {real_summary['shape']}")
    print(f"  Memory: {real_summary['memory_usage_mb']:.2f} MB")
    print(f"  Missing values: {real_summary['missing_values']} ({real_summary['missing_percentage']:.2f}%)")
    print(f"  Duplicate rows: {real_summary['duplicate_rows']}")
    print(f"  Feature types: {real_summary['feature_type_counts']}")
    
    print(f"\nSYNTHETIC DATA:")
    print(f"  Shape: {synthetic_summary['shape']}")
    print(f"  Memory: {synthetic_summary['memory_usage_mb']:.2f} MB")
    print(f"  Missing values: {synthetic_summary['missing_values']} ({synthetic_summary['missing_percentage']:.2f}%)")
    print(f"  Duplicate rows: {synthetic_summary['duplicate_rows']}")
    print(f"  Feature types: {synthetic_summary['feature_type_counts']}")
    
    if selected_data is not None:
        selected_summary = create_data_summary(selected_data, "Selected Data")
        print(f"\nSELECTED DATA:")
        print(f"  Shape: {selected_summary['shape']}")
        print(f"  Memory: {selected_summary['memory_usage_mb']:.2f} MB")
        print(f"  Missing values: {selected_summary['missing_values']} ({selected_summary['missing_percentage']:.2f}%)")
        print(f"  Duplicate rows: {selected_summary['duplicate_rows']}")
        print(f"  Selection ratio: {len(selected_data) / len(synthetic_data):.2%}")
    
    print("\n" + "=" * 60)


def check_dependencies():
    """
    Check if all required dependencies are available.
    """
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'torch': 'torch',
        'scipy': 'scipy'
    }
    
    optional_packages = {
        'lightgbm': 'lightgbm'
    }
    
    print("Checking dependencies...")
    
    missing_required = []
    missing_optional = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (REQUIRED)")
            missing_required.append(package_name)
    
    for package_name, import_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} (optional)")
        except ImportError:
            print(f"- {package_name} (optional)")
            missing_optional.append(package_name)
    
    if missing_required:
        print(f"\nMissing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\nMissing optional packages: {missing_optional}")
        print("Install with: pip install " + " ".join(missing_optional))
    
    print("\nAll required dependencies are available!")
    return True


def estimate_memory_usage(real_data: pd.DataFrame, 
                         synthetic_data: pd.DataFrame,
                         embedding_dim: int = 10) -> dict:
    """
    Estimate memory usage for C-MAPS processing.
    
    Args:
        real_data: Real dataset
        synthetic_data: Synthetic dataset
        embedding_dim: Embedding dimension
        
    Returns:
        Dictionary with memory estimates
    """
    n_real = len(real_data)
    n_synthetic = len(synthetic_data)
    n_features = len(real_data.columns)
    
    # Data storage (in MB)
    real_data_mb = real_data.memory_usage(deep=True).sum() / 1024 / 1024
    synthetic_data_mb = synthetic_data.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Processed data (float32)
    processed_data_mb = (n_real + n_synthetic) * n_features * 4 / 1024 / 1024
    
    # Embeddings
    embeddings_mb = (n_real + n_synthetic) * embedding_dim * 4 / 1024 / 1024
    
    # Distance matrices (for identifiability)
    distance_matrix_mb = n_synthetic * n_real * 4 / 1024 / 1024
    
    estimates = {
        'real_data_mb': real_data_mb,
        'synthetic_data_mb': synthetic_data_mb,
        'processed_data_mb': processed_data_mb,
        'embeddings_mb': embeddings_mb,
        'distance_matrix_mb': distance_matrix_mb,
        'total_estimated_mb': (real_data_mb + synthetic_data_mb + processed_data_mb + 
                              embeddings_mb + distance_matrix_mb),
        'peak_memory_gb': (real_data_mb + synthetic_data_mb + processed_data_mb + 
                          embeddings_mb + distance_matrix_mb) / 1024
    }
    
    return estimates


def print_memory_estimate(estimates: dict):
    """Print memory usage estimates."""
    print("\nMEMORY USAGE ESTIMATES:")
    print(f"  Real data: {estimates['real_data_mb']:.2f} MB")
    print(f"  Synthetic data: {estimates['synthetic_data_mb']:.2f} MB")
    print(f"  Processed data: {estimates['processed_data_mb']:.2f} MB")
    print(f"  Embeddings: {estimates['embeddings_mb']:.2f} MB")
    print(f"  Distance matrix: {estimates['distance_matrix_mb']:.2f} MB")
    print(f"  Total estimated: {estimates['total_estimated_mb']:.2f} MB")
    print(f"  Peak memory: {estimates['peak_memory_gb']:.2f} GB")
    
    if estimates['peak_memory_gb'] > 8:
        print("  ⚠️  Warning: High memory usage expected (>8GB)")
    elif estimates['peak_memory_gb'] > 4:
        print("  ⚠️  Moderate memory usage expected (>4GB)")