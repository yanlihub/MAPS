"""
Data preprocessing utilities for C-MAPS framework.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Tuple, Dict, List, Any
import warnings

warnings.filterwarnings('ignore')


def identify_column_types(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify column types in the dataframes using smart detection.
    
    Args:
        df1: First dataframe (typically real data)
        df2: Second dataframe (typically synthetic data)
        
    Returns:
        Tuple of (numerical_cols, categorical_cols, binary_cols)
    """
    numerical_cols = []
    categorical_cols = []
    binary_cols = []
    
    for col in df1.columns:
        # Check if column is numerical in both datasets
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
            # Additional check: if it looks like categorical (few unique values relative to size)
            real_unique = df1[col].nunique()
            syn_unique = df2[col].nunique()
            real_ratio = real_unique / len(df1)
            syn_ratio = syn_unique / len(df2)
            
            # Check if binary first
            if real_unique <= 2 and syn_unique <= 2:
                binary_cols.append(col)
            # If less than 5% unique values, might be categorical encoded as numbers
            elif real_ratio < 0.05 and syn_ratio < 0.05 and real_unique < 20:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            # Non-numeric columns
            real_unique = df1[col].nunique()
            syn_unique = df2[col].nunique()
            
            if real_unique <= 2 and syn_unique <= 2:
                binary_cols.append(col)
            else:
                categorical_cols.append(col)
    
    return numerical_cols, categorical_cols, binary_cols


# class DataPreprocessorForAutoencoder:
#     """
#     Data preprocessor for autoencoder training and fidelity classification.
#     Uses MinMaxScaler and one-hot encoding.
#     """
    
#     def __init__(self, verbose: bool = True):
#         self.verbose = verbose
#         self.encoders = {}
#         self.numerical_cols = []
#         self.categorical_cols = []
#         self.binary_cols = []
#         self.encoded_cols = []
#         self.fitted = False

#     def transform(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Transform data using already fitted encoders.
        
#         Args:
#             real_df: Real data DataFrame
#             synthetic_df: Synthetic data DataFrame
            
#         Returns:
#             Tuple of (processed_real_df, processed_synthetic_df)
#         """
#         if not self.fitted:
#             raise ValueError("Preprocessor must be fitted before transforming")
        
#         # Apply the same transformations using stored encoders
#         # This is a simplified version - adapt to your actual preprocessing logic
#         processed_real_df = self._apply_transformations(real_df)
#         processed_synthetic_df = self._apply_transformations(synthetic_df)
        
#         return processed_real_df, processed_synthetic_df
    
#     def fit_transform(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str], List[str]]:
#         """
#         Fit preprocessor and transform both dataframes.
        
#         Args:
#             real_df: Real data DataFrame
#             synthetic_df: Synthetic data DataFrame
            
#         Returns:
#             Tuple of (processed_real_df, processed_synthetic_df, encoders, numerical_cols, encoded_cols)
#         """
#         if self.verbose:
#             print("Starting data preprocessing for autoencoder...")
#             print(f"Real data shape: {real_df.shape}")
#             print(f"Synthetic data shape: {synthetic_df.shape}")
        
#         # Ensure same columns
#         common_cols = list(set(real_df.columns) & set(synthetic_df.columns))
#         if len(common_cols) < len(real_df.columns) or len(common_cols) < len(synthetic_df.columns):
#             if self.verbose:
#                 print(f"Using {len(common_cols)} common columns")
        
#         real_df = real_df[common_cols].copy()
#         synthetic_df = synthetic_df[common_cols].copy()
        
#         # Identify column types
#         self.numerical_cols, self.categorical_cols, self.binary_cols = identify_column_types(real_df, synthetic_df)
        
#         if self.verbose:
#             print(f"Identified {len(self.numerical_cols)} numerical, {len(self.categorical_cols)} categorical, {len(self.binary_cols)} binary columns")
        
#         # Initialize processed dataframes
#         processed_real_df = pd.DataFrame()
#         processed_synthetic_df = pd.DataFrame()
#         self.encoders = {}
#         self.encoded_cols = []
        
#         # Handle numerical columns with imputation and MinMaxScaling
#         for col in self.numerical_cols:
#             median_value = real_df[col].median()
#             processed_real_df[col] = real_df[col].fillna(median_value)
#             processed_synthetic_df[col] = synthetic_df[col].fillna(median_value)
#             self.encoded_cols.append(col)
        
#         # MinMaxScale numerical columns
#         if self.numerical_cols:
#             scaler = MinMaxScaler()
#             scaled_real = scaler.fit_transform(processed_real_df[self.numerical_cols])
#             scaled_synthetic = scaler.transform(processed_synthetic_df[self.numerical_cols])
            
#             processed_real_df[self.numerical_cols] = scaled_real
#             processed_synthetic_df[self.numerical_cols] = scaled_synthetic
#             self.encoders['numerical_scaler'] = scaler
        
#         # Handle binary columns
#         for col in self.binary_cols:
#             unique_values = pd.concat([real_df[col], synthetic_df[col]]).dropna().unique()
            
#             if len(unique_values) <= 2:
#                 if len(unique_values) == 1:
#                     mapping = {unique_values[0]: 0}
#                 else:
#                     sorted_values = sorted(unique_values)
#                     mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
                
#                 processed_real_df[col] = real_df[col].map(mapping).fillna(0)
#                 processed_synthetic_df[col] = synthetic_df[col].map(mapping).fillna(0)
#                 self.encoders[f'binary_mapping_{col}'] = mapping
#                 self.encoded_cols.append(col)
        
#         # Handle categorical columns with one-hot encoding
#         for col in self.categorical_cols:
#             unique_categories = pd.concat([real_df[col], synthetic_df[col]]).dropna().unique()
            
#             for category in unique_categories:
#                 new_col = f"{col}_{category}"
#                 processed_real_df[new_col] = (real_df[col] == category).astype(int)
#                 processed_synthetic_df[new_col] = (synthetic_df[col] == category).astype(int)
#                 self.encoded_cols.append(new_col)
            
#             self.encoders[f'categories_{col}'] = list(unique_categories)
        
#         # Fill any remaining NaN and ensure float32
#         processed_real_df = processed_real_df.fillna(0).astype(np.float32)
#         processed_synthetic_df = processed_synthetic_df.fillna(0).astype(np.float32)
        
#         if self.verbose:
#             print(f"Preprocessing complete. Shape: {processed_real_df.shape}")
        
#         self.fitted = True
#         return processed_real_df, processed_synthetic_df, self.encoders, self.numerical_cols, self.encoded_cols

class DataPreprocessorForAutoencoder:
    """
    Data preprocessor for autoencoder training and fidelity classification.
    Uses MinMaxScaler and one-hot encoding.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.encoded_cols = []
        self.fitted = False
        self.common_cols = []  # Store common columns from fitting
        self.median_values = {}  # Store median values for numerical columns

    def transform(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform data using already fitted encoders.
        
        Args:
            real_df: Real data DataFrame
            synthetic_df: Synthetic data DataFrame
            
        Returns:
            Tuple of (processed_real_df, processed_synthetic_df)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        # Apply the same transformations using stored encoders
        processed_real_df = self._apply_transformations(real_df)
        processed_synthetic_df = self._apply_transformations(synthetic_df)
        
        return processed_real_df, processed_synthetic_df
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to a dataframe using stored encoders.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        # Use only the common columns identified during fitting
        df = df[self.common_cols].copy()
        
        # Initialize processed dataframe
        processed_df = pd.DataFrame()
        
        # Handle numerical columns
        for col in self.numerical_cols:
            # Use stored median for imputation
            median_value = self.median_values.get(col, 0)
            processed_df[col] = df[col].fillna(median_value)
        
        # Apply MinMaxScaler to numerical columns
        if self.numerical_cols and 'numerical_scaler' in self.encoders:
            scaler = self.encoders['numerical_scaler']
            scaled_values = scaler.transform(processed_df[self.numerical_cols])
            processed_df[self.numerical_cols] = scaled_values
        
        # Handle binary columns
        for col in self.binary_cols:
            mapping_key = f'binary_mapping_{col}'
            if mapping_key in self.encoders:
                mapping = self.encoders[mapping_key]
                # Apply mapping, use 0 for unknown values
                processed_df[col] = df[col].map(mapping).fillna(0)
        
        # Handle categorical columns with one-hot encoding
        for col in self.categorical_cols:
            categories_key = f'categories_{col}'
            if categories_key in self.encoders:
                unique_categories = self.encoders[categories_key]
                
                for category in unique_categories:
                    new_col = f"{col}_{category}"
                    processed_df[new_col] = (df[col] == category).astype(int)
        
        # Ensure all encoded columns are present (fill missing with 0)
        for col in self.encoded_cols:
            if col not in processed_df.columns:
                processed_df[col] = 0
        
        # Ensure correct column order
        processed_df = processed_df[self.encoded_cols]
        
        # Fill any remaining NaN and ensure float32
        processed_df = processed_df.fillna(0).astype(np.float32)
        
        return processed_df
    
    def fit_transform(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str], List[str]]:
        """
        Fit preprocessor and transform both dataframes.
        
        Args:
            real_df: Real data DataFrame
            synthetic_df: Synthetic data DataFrame
            
        Returns:
            Tuple of (processed_real_df, processed_synthetic_df, encoders, numerical_cols, encoded_cols)
        """
        if self.verbose:
            print("Starting data preprocessing for autoencoder...")
            print(f"Real data shape: {real_df.shape}")
            print(f"Synthetic data shape: {synthetic_df.shape}")
        
        # Ensure same columns
        common_cols = list(set(real_df.columns) & set(synthetic_df.columns))
        if len(common_cols) < len(real_df.columns) or len(common_cols) < len(synthetic_df.columns):
            if self.verbose:
                print(f"Using {len(common_cols)} common columns")
        
        self.common_cols = common_cols  # Store for later use
        real_df = real_df[common_cols].copy()
        synthetic_df = synthetic_df[common_cols].copy()
        
        # Identify column types
        self.numerical_cols, self.categorical_cols, self.binary_cols = identify_column_types(real_df, synthetic_df)
        
        if self.verbose:
            print(f"Identified {len(self.numerical_cols)} numerical, {len(self.categorical_cols)} categorical, {len(self.binary_cols)} binary columns")
        
        # Initialize processed dataframes
        processed_real_df = pd.DataFrame()
        processed_synthetic_df = pd.DataFrame()
        self.encoders = {}
        self.encoded_cols = []
        self.median_values = {}  # Reset median values
        
        # Handle numerical columns with imputation and MinMaxScaling
        for col in self.numerical_cols:
            median_value = real_df[col].median()
            self.median_values[col] = median_value  # Store median for transform
            processed_real_df[col] = real_df[col].fillna(median_value)
            processed_synthetic_df[col] = synthetic_df[col].fillna(median_value)
            self.encoded_cols.append(col)
        
        # MinMaxScale numerical columns
        if self.numerical_cols:
            scaler = MinMaxScaler()
            scaled_real = scaler.fit_transform(processed_real_df[self.numerical_cols])
            scaled_synthetic = scaler.transform(processed_synthetic_df[self.numerical_cols])
            
            processed_real_df[self.numerical_cols] = scaled_real
            processed_synthetic_df[self.numerical_cols] = scaled_synthetic
            self.encoders['numerical_scaler'] = scaler
        
        # Handle binary columns
        for col in self.binary_cols:
            unique_values = pd.concat([real_df[col], synthetic_df[col]]).dropna().unique()
            
            if len(unique_values) <= 2:
                if len(unique_values) == 1:
                    mapping = {unique_values[0]: 0}
                else:
                    sorted_values = sorted(unique_values)
                    mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
                
                processed_real_df[col] = real_df[col].map(mapping).fillna(0)
                processed_synthetic_df[col] = synthetic_df[col].map(mapping).fillna(0)
                self.encoders[f'binary_mapping_{col}'] = mapping
                self.encoded_cols.append(col)
        
        # Handle categorical columns with one-hot encoding
        for col in self.categorical_cols:
            unique_categories = pd.concat([real_df[col], synthetic_df[col]]).dropna().unique()
            
            for category in unique_categories:
                new_col = f"{col}_{category}"
                processed_real_df[new_col] = (real_df[col] == category).astype(int)
                processed_synthetic_df[new_col] = (synthetic_df[col] == category).astype(int)
                self.encoded_cols.append(new_col)
            
            self.encoders[f'categories_{col}'] = list(unique_categories)
        
        # Fill any remaining NaN and ensure float32
        processed_real_df = processed_real_df.fillna(0).astype(np.float32)
        processed_synthetic_df = processed_synthetic_df.fillna(0).astype(np.float32)
        
        if self.verbose:
            print(f"Preprocessing complete. Shape: {processed_real_df.shape}")
        
        self.fitted = True
        return processed_real_df, processed_synthetic_df, self.encoders, self.numerical_cols, self.encoded_cols


class DataPreprocessorForIdentifiability:
    """
    Data preprocessor for identifiability analysis.
    Uses LabelEncoder for categorical variables and different scaling approach.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.fitted = False
        
    def fit_transform(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str], List[str]]:
        """
        Fit preprocessor and transform both dataframes for identifiability analysis.
        
        Args:
            real_df: Real data DataFrame
            synthetic_df: Synthetic data DataFrame
            
        Returns:
            Tuple of (processed_real_df, processed_synthetic_df, encoders, numerical_cols, encoded_cols)
        """
        if self.verbose:
            print("Starting data preprocessing for identifiability analysis...")
            print(f"Real data shape: {real_df.shape}")
            print(f"Synthetic data shape: {synthetic_df.shape}")
        
        # Ensure same columns
        if list(real_df.columns) != list(synthetic_df.columns):
            common_cols = list(set(real_df.columns) & set(synthetic_df.columns))
            real_df = real_df[common_cols].copy()
            synthetic_df = synthetic_df[common_cols].copy()
            if self.verbose:
                print(f"Using {len(common_cols)} common columns")
        else:
            real_df = real_df.copy()
            synthetic_df = synthetic_df.copy()
        
        # Identify column types
        self.numerical_cols, self.categorical_cols, self.binary_cols = identify_column_types(real_df, synthetic_df)
        
        if self.verbose:
            print(f"Numerical: {len(self.numerical_cols)}, Categorical: {len(self.categorical_cols)}, Binary: {len(self.binary_cols)}")
        
        processed_real_df = real_df.copy()
        processed_synthetic_df = synthetic_df.copy()
        
        self.encoders = {
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'binary_cols': self.binary_cols,
            'label_encoders': {},
            'binary_mappings': {}
        }
        
        # Process numerical columns (impute missing values)
        for col in self.numerical_cols:
            median_value = real_df[col].median()
            processed_real_df[col] = real_df[col].fillna(median_value)
            processed_synthetic_df[col] = synthetic_df[col].fillna(median_value)
        
        # Handle binary columns
        for col in self.binary_cols:
            real_col = real_df[col].fillna('__MISSING__')
            syn_col = synthetic_df[col].fillna('__MISSING__')
            all_values = pd.concat([real_col, syn_col]).unique()
            
            if len(all_values) <= 2:
                if len(all_values) == 1:
                    mapping = {all_values[0]: 0}
                else:
                    sorted_values = sorted(all_values)
                    mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
                
                processed_real_df[col] = real_col.map(mapping)
                processed_synthetic_df[col] = syn_col.map(mapping)
                self.encoders['binary_mappings'][col] = mapping
        
        # Handle categorical columns with LabelEncoder
        for col in self.categorical_cols:
            real_col = real_df[col].fillna('__MISSING__')
            syn_col = synthetic_df[col].fillna('__MISSING__')
            all_values = pd.concat([real_col, syn_col]).unique()
            
            le = LabelEncoder()
            le.fit(all_values)
            
            processed_real_df[col] = le.transform(real_col)
            processed_synthetic_df[col] = le.transform(syn_col)
            self.encoders['label_encoders'][col] = le
        
        # Convert to float32
        processed_real_df = processed_real_df.astype(np.float32)
        processed_synthetic_df = processed_synthetic_df.astype(np.float32)
        
        if self.verbose:
            print(f"Preprocessing complete. Shape: {processed_real_df.shape}")
        
        self.fitted = True
        encoded_cols = list(processed_real_df.columns)
        return processed_real_df, processed_synthetic_df, self.encoders, self.numerical_cols, encoded_cols