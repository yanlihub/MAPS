"""
Utility Evaluation for C-MAPS Framework
Train on Synthetic, Test on Real approach with Random Forest
Fixed to properly handle cross-validation: Train on Synthetic, Validate on Real
FIXED: Synthetic datasets are now properly split to match real training size
"""
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, roc_auc_score,
    adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
)
from scipy import stats
from sklearn.cluster import KMeans


warnings.filterwarnings('ignore')

def prepare_data_for_classification(data, target_column='cancer_types', feature_encoders=None):
    """
    Prepare data for classification by separating features and target.
    
    Args:
        data: DataFrame with features and target
        target_column: Name of the target column
        feature_encoders: Dict of fitted encoders for features (None to fit new ones)
        
    Returns:
        X: Features (encoded)
        y: Target (encoded if categorical)
        target_encoder: Fitted label encoder for target (None if target is already numeric)
        feature_encoders: Dict of fitted encoders for categorical features
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    X = data.drop(columns=[target_column]).copy()
    y = data[target_column].copy()
    
    # Encode target if it's categorical
    target_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        target_encoder = LabelEncoder()
        # Handle any missing values in target
        y = y.fillna('missing')
        y = target_encoder.fit_transform(y.astype(str))
    
    # Handle categorical features
    if feature_encoders is None:
        # Fit new encoders
        feature_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                # Handle missing values
                X_col_filled = X[col].fillna('missing').astype(str)
                X[col] = le.fit_transform(X_col_filled)
                feature_encoders[col] = le
                print(f"  Encoded categorical feature '{col}': {len(le.classes_)} categories")
            else:
                # Ensure numeric columns are float and handle missing values
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    else:
        # Use existing encoders
        for col in X.columns:
            if col in feature_encoders:
                # Apply existing encoder
                X_col_filled = X[col].fillna('missing').astype(str)
                try:
                    X[col] = feature_encoders[col].transform(X_col_filled)
                except ValueError as e:
                    # Handle unseen categories by mapping them to a default value
                    print(f"  Warning: Unseen categories in '{col}', using fallback encoding")
                    # Get known categories
                    known_categories = set(feature_encoders[col].classes_)
                    # Map unknown categories to the first known category
                    X_col_mapped = X_col_filled.apply(
                        lambda x: x if x in known_categories else feature_encoders[col].classes_[0]
                    )
                    X[col] = feature_encoders[col].transform(X_col_mapped)
            else:
                # Ensure numeric columns are float and handle missing values
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Final check: ensure all data is numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"  Warning: Column '{col}' still contains non-numeric data, converting to numeric")
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Ensure no infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"  Final feature matrix shape: {X.shape}")
    print(f"  Final feature matrix dtypes: {X.dtypes.nunique()} unique types: {X.dtypes.unique()}")
    
    return X, y, target_encoder, feature_encoders

def train_synthetic_validate_real_cv(X_synthetic, y_synthetic, X_real_val, y_real_val, 
                                   n_estimators=100, cv_folds=5, random_state=42):
    """
    Custom cross-validation: Train on synthetic data, validate on real data.
    Now returns all three metrics: accuracy, F1, and ROC-AUC.
    
    Args:
        X_synthetic: Synthetic training features
        y_synthetic: Synthetic training targets
        X_real_val: Real validation features
        y_real_val: Real validation targets
        n_estimators: Number of trees in Random Forest
        cv_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        cv_accuracy_scores: List of accuracy scores for each fold
        cv_f1_scores: List of F1 scores for each fold
        cv_roc_auc_scores: List of ROC-AUC scores for each fold
    """
    # Split real validation data into folds
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_accuracy_scores = []
    cv_f1_scores = []
    cv_roc_auc_scores = []
    
    # For each fold, train on all synthetic data and validate on a portion of real data
    for fold_idx, (_, val_indices) in enumerate(cv_strategy.split(X_real_val, y_real_val)):
        # Use the validation indices to get the validation set for this fold
        X_val_fold = X_real_val.iloc[val_indices] if hasattr(X_real_val, 'iloc') else X_real_val[val_indices]
        y_val_fold = y_real_val[val_indices]
        
        # Train on all synthetic data
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rf.fit(X_synthetic, y_synthetic)
        
        # Predict on real data fold
        y_pred = rf.predict(X_val_fold)
        y_pred_proba = rf.predict_proba(X_val_fold)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        
        # Calculate ROC AUC
        try:
            n_classes = len(np.unique(y_val_fold))
            n_classes_pred = y_pred_proba.shape[1]
            
            if n_classes == 2 and n_classes_pred == 2:
                roc_auc = roc_auc_score(y_val_fold, y_pred_proba[:, 1])
            elif n_classes == n_classes_pred:
                roc_auc = roc_auc_score(y_val_fold, y_pred_proba, multi_class='ovr')
            else:
                roc_auc = roc_auc_score(y_val_fold, y_pred_proba, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"    Warning: Could not calculate ROC AUC for fold {fold_idx+1}: {e}")
            roc_auc = np.nan
        
        cv_accuracy_scores.append(accuracy)
        cv_f1_scores.append(f1)
        cv_roc_auc_scores.append(roc_auc)
        
        roc_auc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
        print(f"    Fold {fold_idx+1}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc_str}")
    
    return cv_accuracy_scores, cv_f1_scores, cv_roc_auc_scores

def evaluate_model_with_cv(X_train, y_train, X_test, y_test, 
                          model_name="Random Forest", 
                          n_estimators=100, 
                          cv_folds=5, 
                          random_state=42,
                          is_synthetic_train=False,
                          X_real_for_cv=None, 
                          y_real_for_cv=None):
    """
    Train a Random Forest with cross-validation and evaluate on test set.
    Now includes CV for all metrics and feature importance analysis.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features  
        y_test: Test target
        model_name: Name for the model (for reporting)
        n_estimators: Number of trees in Random Forest
        cv_folds: Number of CV folds
        random_state: Random seed
        is_synthetic_train: Whether training data is synthetic
        X_real_for_cv: Real data for CV validation (when training on synthetic)
        y_real_for_cv: Real targets for CV validation (when training on synthetic)
        
    Returns:
        Dictionary with evaluation results including feature importances
    """
    print(f"\nEvaluating {model_name}:")
    print(f"  Training data shape: {X_train.shape}")
    print(f"  Test data shape: {X_test.shape}")
    
    # Initialize Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Perform cross-validation
    if is_synthetic_train and X_real_for_cv is not None and y_real_for_cv is not None:
        # Custom CV: Train on synthetic, validate on real
        print(f"  Performing custom CV: Train on synthetic, validate on real")
        print(f"  Real validation data shape: {X_real_for_cv.shape}")
        
        cv_accuracy_scores, cv_f1_scores, cv_roc_auc_scores = train_synthetic_validate_real_cv(
            X_train, y_train, X_real_for_cv, y_real_for_cv, 
            n_estimators, cv_folds, random_state
        )
    else:
        # Standard CV: Train and validate on same type of data
        print(f"  Performing standard CV: Train and validate on same data type")
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_accuracy_scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='accuracy')
        cv_f1_scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='f1_macro')
        
        # For ROC AUC, we need to handle it manually
        cv_roc_auc_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
            X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            rf_fold = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
            rf_fold.fit(X_train_fold, y_train_fold)
            
            try:
                y_pred_proba = rf_fold.predict_proba(X_val_fold)
                n_classes = len(np.unique(y_val_fold))
                n_classes_pred = y_pred_proba.shape[1]
                
                if n_classes == 2 and n_classes_pred == 2:
                    roc_auc = roc_auc_score(y_val_fold, y_pred_proba[:, 1])
                elif n_classes == n_classes_pred:
                    roc_auc = roc_auc_score(y_val_fold, y_pred_proba, multi_class='ovr')
                else:
                    roc_auc = roc_auc_score(y_val_fold, y_pred_proba, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"    Warning: Could not calculate ROC AUC for fold {fold_idx+1}: {e}")
                roc_auc = np.nan
            
            cv_roc_auc_scores.append(roc_auc)
    
    # Train on full training set for final test evaluation
    rf.fit(X_train, y_train)
    
    # Extract feature importances
    feature_importances = rf.feature_importances_
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # Predict on test set
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    # Calculate ROC AUC (handle binary vs multiclass and class mismatch)
    try:
        n_classes_test = len(np.unique(y_test))
        n_classes_pred = y_pred_proba.shape[1]
        
        if n_classes_test == 2 and n_classes_pred == 2:
            test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        elif n_classes_test == n_classes_pred:
            test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            # Class mismatch - use macro average with available classes
            print(f"    Warning: Class mismatch - test has {n_classes_test} classes, model predicts {n_classes_pred}")
            test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"    Warning: Could not calculate ROC AUC: {e}")
        test_roc_auc = np.nan
    
    # Calculate CV means and stds, handling NaN values for ROC AUC
    valid_roc_auc_scores = [score for score in cv_roc_auc_scores if not np.isnan(score)]
    cv_roc_auc_mean = np.mean(valid_roc_auc_scores) if valid_roc_auc_scores else np.nan
    cv_roc_auc_std = np.std(valid_roc_auc_scores) if valid_roc_auc_scores else np.nan
    
    results = {
        'model_name': model_name,
        'cv_accuracy_mean': np.mean(cv_accuracy_scores),
        'cv_accuracy_std': np.std(cv_accuracy_scores),
        'cv_f1_mean': np.mean(cv_f1_scores),
        'cv_f1_std': np.std(cv_f1_scores),
        'cv_roc_auc_mean': cv_roc_auc_mean,
        'cv_roc_auc_std': cv_roc_auc_std,
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1,
        'test_roc_auc': test_roc_auc,
        'cv_scores': cv_accuracy_scores,
        'cv_f1_scores': cv_f1_scores,
        'cv_roc_auc_scores': cv_roc_auc_scores,
        'feature_importances': feature_importances,
        'feature_names': feature_names,
        'classification_report': classification_report(y_test, y_pred),
        'trained_model': rf
    }
    
    return results

def plot_feature_importance_comparison(results, top_n=10, figsize=(15, 8)):
    """
    Plot and compare feature importances across different scenarios.
    
    Args:
        results: Results dictionary from run_utility_evaluation
        top_n: Number of top features to show
        figsize: Figure size
    """
    scenarios = ['raw_synthetic', 'refined_synthetic', 'real_baseline']
    scenario_names = ['Raw Synthetic → Real', 'Refined Synthetic → Real', 'Real Train → Real Test']
    colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']
    
    # Get feature names (should be the same across all scenarios)
    feature_names = results['real_baseline']['feature_names']
    
    # Create DataFrame with all feature importances
    importance_data = {}
    for i, scenario in enumerate(scenarios):
        importance_data[scenario_names[i]] = results[scenario]['feature_importances']
    
    importance_df = pd.DataFrame(importance_data, index=feature_names)
    
    # Get top features based on average importance across all scenarios
    importance_df['average'] = importance_df.mean(axis=1)
    top_features = importance_df.nlargest(top_n, 'average').index
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Side-by-side bar plot for top features
    top_importance_df = importance_df.loc[top_features, scenario_names].sort_values(by='Real Train → Real Test', ascending=True)
    
    x_pos = np.arange(len(top_importance_df))
    width = 0.25
    
    for i, scenario_name in enumerate(scenario_names):
        axes[0].barh(x_pos + i * width, top_importance_df[scenario_name], width, 
                    label=scenario_name, color=colors[i], alpha=0.7)
    
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title(f'Top {top_n} Feature Importances Comparison')
    axes[0].set_yticks(x_pos + width)
    axes[0].set_yticklabels(top_importance_df.index)
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Correlation between feature importances (only Real vs Raw and Real vs Refined)
    # Calculate correlation coefficients first
    corr_real_raw = np.corrcoef(importance_df['Real Train → Real Test'], importance_df['Raw Synthetic → Real'])[0,1]
    corr_real_refined = np.corrcoef(importance_df['Real Train → Real Test'], importance_df['Refined Synthetic → Real'])[0,1]
    
    # Plot Real vs Raw (first)
    axes[1].scatter(importance_df['Real Train → Real Test'], importance_df['Raw Synthetic → Real'], 
                   c='green', alpha=0.6, s=30, label=f'Real vs Raw (r={corr_real_raw:.3f})')
    
    # Plot Real vs Refined (second)
    axes[1].scatter(importance_df['Real Train → Real Test'], importance_df['Refined Synthetic → Real'], 
                   c='blue', alpha=0.6, s=30, label=f'Real vs Refined (r={corr_real_refined:.3f})')
    
    # Add diagonal line
    max_importance = importance_df[scenario_names].max().max()
    axes[1].plot([0, max_importance], [0, max_importance], 'k--', alpha=0.5)
    
    axes[1].set_xlabel('Real Train → Real Test Feature Importance')
    axes[1].set_ylabel('Synthetic → Real Feature Importance')
    axes[1].set_title('Feature Importance Correlations')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print correlation coefficients
    print("\nFeature Importance Correlation Analysis:")
    print("=" * 50)
    
    print(f"Real vs Raw Synthetic: {corr_real_raw:.4f}")
    print(f"Real vs Refined Synthetic: {corr_real_refined:.4f}")
    
    # Print top features for each scenario
    print(f"\nTop {min(5, top_n)} Features by Importance:")
    print("=" * 50)
    for scenario_name in scenario_names:
        top_5_features = importance_df.nlargest(5, scenario_name).index.tolist()
        print(f"{scenario_name}:")
        for i, feature in enumerate(top_5_features):
            importance = importance_df.loc[feature, scenario_name]
            print(f"  {i+1}. {feature}: {importance:.4f}")
        print()

def run_utility_evaluation(real_data, raw_synthetic_data, refined_synthetic_data, 
                          target_column='cancer_types', test_size=0.2, cv_val_size=0.2, random_state=42):
    """
    Run complete utility evaluation comparing three scenarios.
    
    Args:
        real_data: Real dataset
        raw_synthetic_data: Original synthetic dataset
        refined_synthetic_data: Refined synthetic dataset (after C-MAPS)
        target_column: Name of the target column
        test_size: Proportion of real data to use for testing
        cv_val_size: Proportion of remaining real data to use for CV validation (when training on synthetic)
        random_state: Random seed
        
    Returns:
        Dictionary with all evaluation results
    """
    print("=" * 80)
    print("ENHANCED UTILITY EVALUATION: TRAIN ON SYNTHETIC, TEST ON REAL")
    print("Cross-Validation: All Metrics (Accuracy, F1-Macro, ROC-AUC)")
    print("Feature Importance Analysis Included")
    print("FIXED: Synthetic datasets properly split to match real training size")
    print("=" * 80)
    
    # First, check what classes are available in each dataset
    print("\nAnalyzing target classes across datasets...")
    all_targets = pd.concat([
        real_data[target_column],
        raw_synthetic_data[target_column], 
        refined_synthetic_data[target_column]
    ])
    unique_classes = sorted(all_targets.unique())
    print(f"All unique classes found: {unique_classes}")
    
    # Fit target encoder on all possible classes
    target_encoder = LabelEncoder()
    target_encoder.fit(all_targets.astype(str))
    print(f"Target encoder classes: {target_encoder.classes_}")
    
    # Split real data: test_size for final testing, cv_val_size of remainder for CV validation, rest for baseline training
    print(f"\nSplitting real data for proper evaluation...")
    print(f"Test size: {test_size:.0%}, CV validation size: {cv_val_size:.0%} of remaining data")
    
    # First split: separate test data
    real_temp, real_test = train_test_split(
        real_data, 
        test_size=test_size, 
        stratify=real_data[target_column], 
        random_state=random_state
    )
    
    # Second split: separate CV validation from training data
    real_train, real_cv_val = train_test_split(
        real_temp,
        test_size=cv_val_size,
        stratify=real_temp[target_column],
        random_state=random_state
    )
    
    print(f"Real train set: {len(real_train)} samples")
    print(f"Real CV validation set: {len(real_cv_val)} samples") 
    print(f"Real test set: {len(real_test)} samples")
    
    # FIXED: Sample synthetic datasets to match real training size for fair comparison
    real_train_size = len(real_train)
    print(f"\nSampling synthetic datasets to match real training size: {real_train_size} samples")
    
    if len(raw_synthetic_data) < real_train_size:
        print(f"Warning: Raw synthetic data ({len(raw_synthetic_data)}) smaller than real train size, using all")
        raw_synthetic_train = raw_synthetic_data.copy()
    else:
        raw_synthetic_train = raw_synthetic_data.sample(n=real_train_size, random_state=random_state)
    
    if len(refined_synthetic_data) < real_train_size:
        print(f"Warning: Refined synthetic data ({len(refined_synthetic_data)}) smaller than real train size, using all")
        refined_synthetic_train = refined_synthetic_data.copy()
    else:
        refined_synthetic_train = refined_synthetic_data.sample(n=real_train_size, random_state=random_state)
    
    print(f"Raw synthetic train set: {len(raw_synthetic_train)} samples")
    print(f"Refined synthetic train set: {len(refined_synthetic_train)} samples")
    
    # Prepare all datasets with consistent encoding
    print("\nPreparing real test data...")
    X_real_test, y_real_test_temp, _, feature_encoders = prepare_data_for_classification(
        real_test, target_column
    )
    y_real_test = target_encoder.transform(real_test[target_column].astype(str))
    
    print("\nPreparing real train data...")
    X_real_train, y_real_train_temp, _, _ = prepare_data_for_classification(
        real_train, target_column, feature_encoders
    )
    y_real_train = target_encoder.transform(real_train[target_column].astype(str))
    
    print("\nPreparing real CV validation data...")
    X_real_cv_val, y_real_cv_val_temp, _, _ = prepare_data_for_classification(
        real_cv_val, target_column, feature_encoders
    )
    y_real_cv_val = target_encoder.transform(real_cv_val[target_column].astype(str))
    
    print("\nPreparing raw synthetic training data...")
    X_raw_syn, y_raw_syn_temp, _, _ = prepare_data_for_classification(
        raw_synthetic_train, target_column, feature_encoders
    )
    y_raw_syn = target_encoder.transform(raw_synthetic_train[target_column].astype(str))
    
    print("\nPreparing refined synthetic training data...")
    X_refined_syn, y_refined_syn_temp, _, _ = prepare_data_for_classification(
        refined_synthetic_train, target_column, feature_encoders
    )
    y_refined_syn = target_encoder.transform(refined_synthetic_train[target_column].astype(str))
    
    print(f"\nData preparation complete:")
    print(f"Real test data shape: {X_real_test.shape}")
    print(f"Real train data shape: {X_real_train.shape}")
    print(f"Real CV validation data shape: {X_real_cv_val.shape}")
    print(f"Raw synthetic training data shape: {X_raw_syn.shape}")
    print(f"Refined synthetic training data shape: {X_refined_syn.shape}")
    print(f"Number of encoded categorical features: {len(feature_encoders)}")
    
    # Check class distributions
    print(f"\nClass distributions:")
    print(f"Real test classes: {np.bincount(y_real_test)}")
    print(f"Real train classes: {np.bincount(y_real_train)}")
    print(f"Real CV val classes: {np.bincount(y_real_cv_val)}")
    print(f"Raw synthetic training classes: {np.bincount(y_raw_syn)}")
    print(f"Refined synthetic training classes: {np.bincount(y_refined_syn)}")
    
    results = {}
    
    # Scenario 1: Train on raw synthetic, test on real (with proper CV: train synthetic, validate real)
    print(f"\n{'='*50}")
    print("SCENARIO 1: Train on Raw Synthetic, Test on Real")
    print("             CV: Train on Raw Synthetic, Validate on Real")
    print('='*50)
    
    results['raw_synthetic'] = evaluate_model_with_cv(
        X_raw_syn, y_raw_syn, X_real_test, y_real_test,
        model_name="Raw Synthetic → Real",
        random_state=random_state,
        is_synthetic_train=True,
        X_real_for_cv=X_real_cv_val,
        y_real_for_cv=y_real_cv_val
    )
    
    print(f"CV Accuracy: {results['raw_synthetic']['cv_accuracy_mean']:.4f} ± {results['raw_synthetic']['cv_accuracy_std']:.4f}")
    print(f"CV F1-Macro: {results['raw_synthetic']['cv_f1_mean']:.4f} ± {results['raw_synthetic']['cv_f1_std']:.4f}")
    if not np.isnan(results['raw_synthetic']['cv_roc_auc_mean']):
        print(f"CV ROC-AUC: {results['raw_synthetic']['cv_roc_auc_mean']:.4f} ± {results['raw_synthetic']['cv_roc_auc_std']:.4f}")
    print(f"Test Accuracy: {results['raw_synthetic']['test_accuracy']:.4f}")
    print(f"Test F1-Macro: {results['raw_synthetic']['test_f1_macro']:.4f}")
    if not np.isnan(results['raw_synthetic']['test_roc_auc']):
        print(f"Test ROC-AUC: {results['raw_synthetic']['test_roc_auc']:.4f}")
    
    # Scenario 2: Train on refined synthetic, test on real (with proper CV: train synthetic, validate real)
    print(f"\n{'='*50}")
    print("SCENARIO 2: Train on Refined Synthetic, Test on Real")
    print("             CV: Train on Refined Synthetic, Validate on Real")
    print('='*50)
    
    results['refined_synthetic'] = evaluate_model_with_cv(
        X_refined_syn, y_refined_syn, X_real_test, y_real_test,
        model_name="Refined Synthetic → Real",
        random_state=random_state,
        is_synthetic_train=True,
        X_real_for_cv=X_real_cv_val,
        y_real_for_cv=y_real_cv_val
    )
    
    print(f"CV Accuracy: {results['refined_synthetic']['cv_accuracy_mean']:.4f} ± {results['refined_synthetic']['cv_accuracy_std']:.4f}")
    print(f"CV F1-Macro: {results['refined_synthetic']['cv_f1_mean']:.4f} ± {results['refined_synthetic']['cv_f1_std']:.4f}")
    if not np.isnan(results['refined_synthetic']['cv_roc_auc_mean']):
        print(f"CV ROC-AUC: {results['refined_synthetic']['cv_roc_auc_mean']:.4f} ± {results['refined_synthetic']['cv_roc_auc_std']:.4f}")
    print(f"Test Accuracy: {results['refined_synthetic']['test_accuracy']:.4f}")
    print(f"Test F1-Macro: {results['refined_synthetic']['test_f1_macro']:.4f}")
    if not np.isnan(results['refined_synthetic']['test_roc_auc']):
        print(f"Test ROC-AUC: {results['refined_synthetic']['test_roc_auc']:.4f}")
    
    # Scenario 3: Train on real (train split), test on real (test split) - PROPER BASELINE
    print(f"\n{'='*50}")
    print("SCENARIO 3: Train on Real (Train Split), Test on Real (Test Split)")
    print("             CV: Train on Real, Validate on Real")
    print("             [PROPER BASELINE - NO DATA LEAKAGE]")
    print('='*50)
    
    results['real_baseline'] = evaluate_model_with_cv(
        X_real_train, y_real_train, X_real_test, y_real_test,
        model_name="Real (Train) → Real (Test)",
        random_state=random_state,
        is_synthetic_train=False  # Standard CV
    )
    
    print(f"CV Accuracy: {results['real_baseline']['cv_accuracy_mean']:.4f} ± {results['real_baseline']['cv_accuracy_std']:.4f}")
    print(f"CV F1-Macro: {results['real_baseline']['cv_f1_mean']:.4f} ± {results['real_baseline']['cv_f1_std']:.4f}")
    if not np.isnan(results['real_baseline']['cv_roc_auc_mean']):
        print(f"CV ROC-AUC: {results['real_baseline']['cv_roc_auc_mean']:.4f} ± {results['real_baseline']['cv_roc_auc_std']:.4f}")
    print(f"Test Accuracy: {results['real_baseline']['test_accuracy']:.4f}")
    print(f"Test F1-Macro: {results['real_baseline']['test_f1_macro']:.4f}")
    if not np.isnan(results['real_baseline']['test_roc_auc']):
        print(f"Test ROC-AUC: {results['real_baseline']['test_roc_auc']:.4f}")
    
    # Feature importance comparison
    print(f"\n{'='*50}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print('='*50)
    plot_feature_importance_comparison(results)
    
    return results

def plot_evaluation_results(results, separate_figures=False, figsize=None):
    """
    Plot comparison of evaluation results with all CV metrics.
    
    Args:
        results: Results dictionary from run_utility_evaluation
        separate_figures: bool, whether to create separate figures for each plot
        figsize: tuple, optional, figure size for individual plots
    """
    # Prepare data for plotting
    scenarios = ['Raw Synthetic\n→ Real', 'Refined Synthetic\n→ Real', 'Real Train\n→ Real Test\n("Oracle")']
    
    # Test metrics
    test_accuracies = [
        results['raw_synthetic']['test_accuracy'],
        results['refined_synthetic']['test_accuracy'],
        results['real_baseline']['test_accuracy']
    ]
    test_f1_scores = [
        results['raw_synthetic']['test_f1_macro'],
        results['refined_synthetic']['test_f1_macro'],
        results['real_baseline']['test_f1_macro']
    ]
    
    # Handle potential NaN ROC AUC values
    test_roc_aucs = []
    roc_auc_available = True
    for scenario_key in ['raw_synthetic', 'refined_synthetic', 'real_baseline']:
        roc_auc = results[scenario_key]['test_roc_auc']
        if np.isnan(roc_auc):
            roc_auc_available = False
            test_roc_aucs.append(0)  # Placeholder
        else:
            test_roc_aucs.append(roc_auc)
    
    # CV metrics
    cv_accuracies = [
        results['raw_synthetic']['cv_accuracy_mean'],
        results['refined_synthetic']['cv_accuracy_mean'],
        results['real_baseline']['cv_accuracy_mean']
    ]
    cv_accuracy_stds = [
        results['raw_synthetic']['cv_accuracy_std'],
        results['refined_synthetic']['cv_accuracy_std'],
        results['real_baseline']['cv_accuracy_std']
    ]
    
    cv_f1_scores = [
        results['raw_synthetic']['cv_f1_mean'],
        results['refined_synthetic']['cv_f1_mean'],
        results['real_baseline']['cv_f1_mean']
    ]
    cv_f1_stds = [
        results['raw_synthetic']['cv_f1_std'],
        results['refined_synthetic']['cv_f1_std'],
        results['real_baseline']['cv_f1_std']
    ]
    
    # CV ROC AUC (handle NaN values)
    cv_roc_aucs = []
    cv_roc_auc_stds = []
    cv_roc_auc_available = True
    for scenario_key in ['raw_synthetic', 'refined_synthetic', 'real_baseline']:
        roc_auc_mean = results[scenario_key]['cv_roc_auc_mean']
        roc_auc_std = results[scenario_key]['cv_roc_auc_std']
        if np.isnan(roc_auc_mean):
            cv_roc_auc_available = False
            cv_roc_aucs.append(0)
            cv_roc_auc_stds.append(0)
        else:
            cv_roc_aucs.append(roc_auc_mean)
            cv_roc_auc_stds.append(roc_auc_std)
    
    # Colors for different scenarios
    colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']  # Light red, green, blue
    
    if separate_figures:
        # Create separate figures for each plot
        figures = []
        individual_figsize = (10, 6) if figsize is None else figsize
        
        # Plot 1: Test Metrics
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
        
        # Test Accuracy
        axes1[0].bar(scenarios, test_accuracies, color=colors, alpha=0.7)
        axes1[0].set_title('Test Accuracy Comparison')
        axes1[0].set_ylabel('Accuracy')
        axes1[0].set_ylim(0, 1)
        axes1[0].grid(False)
        for i, v in enumerate(test_accuracies):
            axes1[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Test F1-Macro
        axes1[1].bar(scenarios, test_f1_scores, color=colors, alpha=0.7)
        axes1[1].set_title('Test F1-Macro Comparison')
        axes1[1].set_ylabel('F1-Macro Score')
        axes1[1].set_ylim(0, 1)
        axes1[1].grid(False)
        for i, v in enumerate(test_f1_scores):
            axes1[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Test ROC-AUC
        if roc_auc_available:
            axes1[2].bar(scenarios, test_roc_aucs, color=colors, alpha=0.7)
            axes1[2].set_title('Test ROC-AUC Comparison')
            axes1[2].set_ylabel('ROC-AUC Score')
            axes1[2].set_ylim(0, 1.05)
            axes1[2].grid(False)
            for i, v in enumerate(test_roc_aucs):
                axes1[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        else:
            axes1[2].text(0.5, 0.5, 'ROC-AUC\nNot Available', ha='center', va='center', transform=axes1[2].transAxes)
            axes1[2].set_title('Test ROC-AUC Comparison')
        
        plt.tight_layout()
        figures.append(fig1)
        plt.show()
        
        # Plot 2: CV Metrics
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        
        # CV Accuracy
        axes2[0].bar(scenarios, cv_accuracies, yerr=cv_accuracy_stds, 
                    color=colors, alpha=0.7, capsize=5)
        axes2[0].set_title('Cross-Validation Accuracy')
        axes2[0].set_ylabel('CV Accuracy')
        axes2[0].set_ylim(0, 1)
        axes2[0].grid(False)
        for i, (v, std) in enumerate(zip(cv_accuracies, cv_accuracy_stds)):
            axes2[0].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        
        # CV F1-Macro
        axes2[1].bar(scenarios, cv_f1_scores, yerr=cv_f1_stds, 
                    color=colors, alpha=0.7, capsize=5)
        axes2[1].set_title('Cross-Validation F1-Macro')
        axes2[1].set_ylabel('CV F1-Macro')
        axes2[1].set_ylim(0, 1)
        axes2[1].grid(False)
        for i, (v, std) in enumerate(zip(cv_f1_scores, cv_f1_stds)):
            axes2[1].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        
        # CV ROC-AUC
        if cv_roc_auc_available:
            axes2[2].bar(scenarios, cv_roc_aucs, yerr=cv_roc_auc_stds, 
                        color=colors, alpha=0.7, capsize=5)
            axes2[2].set_title('Cross-Validation ROC-AUC')
            axes2[2].set_ylabel('CV ROC-AUC')
            axes2[2].set_ylim(0, 1.05)
            axes2[2].grid(False)
            for i, (v, std) in enumerate(zip(cv_roc_aucs, cv_roc_auc_stds)):
                axes2[2].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        else:
            axes2[2].text(0.5, 0.5, 'CV ROC-AUC\nNot Available', ha='center', va='center', transform=axes2[2].transAxes)
            axes2[2].set_title('Cross-Validation ROC-AUC')
        
        plt.tight_layout()
        figures.append(fig2)
        plt.show()
        
        return figures
    
    else:
        # Create combined figure with subplots
        n_plots = 6 if (roc_auc_available and cv_roc_auc_available) else 4
        if n_plots == 6:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test Accuracy
        axes[0, 0].bar(scenarios, test_accuracies, color=colors, alpha=0.7)
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(False)
        for i, v in enumerate(test_accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Test F1-Macro
        axes[0, 1].bar(scenarios, test_f1_scores, color=colors, alpha=0.7)
        axes[0, 1].set_title('Test F1-Macro Comparison')
        axes[0, 1].set_ylabel('F1-Macro Score')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(False)
        for i, v in enumerate(test_f1_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # CV Accuracy
        axes[1, 0].bar(scenarios, cv_accuracies, yerr=cv_accuracy_stds, 
                      color=colors, alpha=0.7, capsize=5)
        axes[1, 0].set_title('Cross-Validation Accuracy')
        axes[1, 0].set_ylabel('CV Accuracy')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(False)
        for i, (v, std) in enumerate(zip(cv_accuracies, cv_accuracy_stds)):
            axes[1, 0].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        
        # CV F1-Macro
        axes[1, 1].bar(scenarios, cv_f1_scores, yerr=cv_f1_stds, 
                      color=colors, alpha=0.7, capsize=5)
        axes[1, 1].set_title('Cross-Validation F1-Macro')
        axes[1, 1].set_ylabel('CV F1-Macro')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(False)
        for i, (v, std) in enumerate(zip(cv_f1_scores, cv_f1_stds)):
            axes[1, 1].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        
        # Test and CV ROC-AUC (if available)
        if n_plots == 6:
            # Test ROC-AUC
            axes[0, 2].bar(scenarios, test_roc_aucs, color=colors, alpha=0.7)
            axes[0, 2].set_title('Test ROC-AUC Comparison')
            axes[0, 2].set_ylabel('ROC-AUC Score')
            axes[0, 2].set_ylim(0, 1.05)
            axes[0, 2].grid(False)
            for i, v in enumerate(test_roc_aucs):
                axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # CV ROC-AUC
            axes[1, 2].bar(scenarios, cv_roc_aucs, yerr=cv_roc_auc_stds, 
                          color=colors, alpha=0.7, capsize=5)
            axes[1, 2].set_title('Cross-Validation ROC-AUC')
            axes[1, 2].set_ylabel('CV ROC-AUC')
            axes[1, 2].set_ylim(0, 1.05)
            axes[1, 2].grid(False)
            for i, (v, std) in enumerate(zip(cv_roc_aucs, cv_roc_auc_stds)):
                axes[1, 2].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def print_summary_table(results):
    """
    Print a comprehensive summary table of all results including all CV metrics.
    
    Args:
        results: Results dictionary from run_utility_evaluation
    """
    print("\n" + "=" * 120)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("=" * 120)
    
    print(f"{'Scenario':<30} {'CV Acc':<12} {'CV F1':<12} {'CV AUC':<12} {'Test Acc':<12} {'Test F1':<12} {'Test AUC':<12}")
    print("-" * 120)
    
    scenarios = [
        ('Raw Synthetic → Real', results['raw_synthetic']),
        ('Refined Synthetic → Real', results['refined_synthetic']),
        ('Real Train → Real Test', results['real_baseline'])
    ]
    
    for name, result in scenarios:
        cv_acc = f"{result['cv_accuracy_mean']:.4f}±{result['cv_accuracy_std']:.3f}"
        cv_f1 = f"{result['cv_f1_mean']:.4f}±{result['cv_f1_std']:.3f}"
        
        if np.isnan(result['cv_roc_auc_mean']):
            cv_auc = "N/A"
        else:
            cv_auc = f"{result['cv_roc_auc_mean']:.4f}±{result['cv_roc_auc_std']:.3f}"
        
        test_acc = f"{result['test_accuracy']:.4f}"
        test_f1 = f"{result['test_f1_macro']:.4f}"
        
        if np.isnan(result['test_roc_auc']):
            test_auc = "N/A"
        else:
            test_auc = f"{result['test_roc_auc']:.4f}"
        
        print(f"{name:<30} {cv_acc:<12} {cv_f1:<12} {cv_auc:<12} {test_acc:<12} {test_f1:<12} {test_auc:<12}")
    
    # Calculate improvements
    raw_acc = results['raw_synthetic']['test_accuracy']
    refined_acc = results['refined_synthetic']['test_accuracy']
    baseline_acc = results['real_baseline']['test_accuracy']
    
    raw_f1 = results['raw_synthetic']['test_f1_macro']
    refined_f1 = results['refined_synthetic']['test_f1_macro']
    baseline_f1 = results['real_baseline']['test_f1_macro']
    
    raw_cv_acc = results['raw_synthetic']['cv_accuracy_mean']
    refined_cv_acc = results['refined_synthetic']['cv_accuracy_mean']
    baseline_cv_acc = results['real_baseline']['cv_accuracy_mean']
    
    print("\n" + "=" * 120)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 120)
    
    # Test metric improvements
    test_acc_improvement = ((refined_acc - raw_acc) / raw_acc) * 100 if raw_acc > 0 else 0
    test_f1_improvement = ((refined_f1 - raw_f1) / raw_f1) * 100 if raw_f1 > 0 else 0
    
    # CV metric improvements
    cv_acc_improvement = ((refined_cv_acc - raw_cv_acc) / raw_cv_acc) * 100 if raw_cv_acc > 0 else 0
    
    print(f"Test Accuracy improvement from C-MAPS refinement: {test_acc_improvement:+.2f}%")
    print(f"Test F1-Macro improvement from C-MAPS refinement: {test_f1_improvement:+.2f}%")
    print(f"CV Accuracy improvement from C-MAPS refinement: {cv_acc_improvement:+.2f}%")
    
    # Gap to baseline analysis
    baseline_gap_raw_acc = ((baseline_acc - raw_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
    baseline_gap_refined_acc = ((baseline_acc - refined_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
    
    print(f"\nGap to baseline (Raw Synthetic): {baseline_gap_raw_acc:.2f}%")
    print(f"Gap to baseline (Refined Synthetic): {baseline_gap_refined_acc:.2f}%")
    print(f"Refinement reduces gap by: {baseline_gap_raw_acc - baseline_gap_refined_acc:.2f} percentage points")
    
    print("\nNote: CV for synthetic scenarios uses 'Train on Synthetic, Validate on Real'")
    print("      CV for real scenario uses standard 'Train on Real, Validate on Real'")
    print("      All training sets now have equal size for fair comparison")

########################### below is the clustering evaluation function ###########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from scipy import stats


def run_clustering_evaluation(real_data, raw_synthetic_data, refined_synthetic_data,
                              n_clusters_range=range(2, 16), selected_k=5, 
                              target_column=None, exclude_columns=None, 
                              test_size=0.2, cv_folds=5, random_state=42):
    """
    Evaluate K-means clustering with proper train/test splits, BIC analysis, and cross-validation.
    
    Args:
        real_data: Real dataset
        raw_synthetic_data: Original synthetic dataset
        refined_synthetic_data: Refined synthetic dataset (post C-MAPS)
        n_clusters_range: Range of cluster numbers for BIC analysis
        selected_k: Number of clusters for detailed evaluation
        target_column: Column name containing true labels for ARI/AMI calculation (optional)
        exclude_columns: Columns to exclude from clustering features
        test_size: Proportion for train/test split
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        
    Note:
        If target_column is provided, uses it as ground truth for ARI/AMI.
        If target_column is None, uses real data clustering predictions as reference.
        Cross-validation is always performed to get standard deviations.
    """
    print("=" * 80)
    print("K-MEANS CLUSTERING EVALUATION (CONFIGURABLE LABELS & CV)")
    print("FIXED: Synthetic datasets properly split to match real training size")
    print("=" * 80)
    
    # Prepare data
    if exclude_columns is None:
        exclude_columns = []
    
    # Check for target column (now optional)
    use_true_labels = False
    if target_column is not None:
        if target_column in real_data.columns:
            use_true_labels = True
            print(f"Using '{target_column}' as ground truth for ARI/AMI calculation")
            # Add target column to exclude list for clustering features
            if target_column not in exclude_columns:
                exclude_columns = exclude_columns + [target_column]
        else:
            print(f"Warning: Target column '{target_column}' not found in real_data")
            print("Will use real data clustering predictions as reference instead")
    else:
        print("No target_column specified. Will use real data clustering predictions as reference")
    
    # Get numeric columns only (excluding target and other excluded columns)
    numeric_cols = []
    for col in real_data.columns:
        if col not in exclude_columns and pd.api.types.is_numeric_dtype(real_data[col]):
            numeric_cols.append(col)
    
    print(f"Using {len(numeric_cols)} numeric features for clustering")
    print(f"Excluded columns: {exclude_columns}")
    
    # Prepare datasets
    real_features = real_data[numeric_cols].fillna(0)
    real_labels = real_data[target_column] if use_true_labels else None
    raw_syn_features = raw_synthetic_data[numeric_cols].fillna(0)
    refined_syn_features = refined_synthetic_data[numeric_cols].fillna(0)
    
    # Single train/test split for BIC analysis
    from sklearn.model_selection import train_test_split, StratifiedKFold
    
    if use_true_labels:
        # Use stratified split if we have true labels
        real_train, real_test, labels_train, labels_test = train_test_split(
            real_features, real_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=real_labels if len(real_labels.unique()) < len(real_labels) * 0.5 else None
        )
    else:
        # Simple split without labels
        real_train, real_test = train_test_split(
            real_features, 
            test_size=test_size, 
            random_state=random_state
        )
        labels_train, labels_test = None, None
    
    print(f"Real data split: {len(real_train)} train, {len(real_test)} test")
    
    # FIXED: Sample synthetic datasets to match real training size for fair comparison
    real_train_size = len(real_train)
    print(f"Sampling synthetic datasets to match real training size: {real_train_size} samples")
    
    if len(raw_syn_features) < real_train_size:
        print(f"Warning: Raw synthetic data ({len(raw_syn_features)}) smaller than real train size, using all")
        raw_syn_train = raw_syn_features.copy()
    else:
        raw_syn_train = raw_syn_features.sample(n=real_train_size, random_state=random_state)
    
    if len(refined_syn_features) < real_train_size:
        print(f"Warning: Refined synthetic data ({len(refined_syn_features)}) smaller than real train size, using all")
        refined_syn_train = refined_syn_features.copy()
    else:
        refined_syn_train = refined_syn_features.sample(n=real_train_size, random_state=random_state)
    
    print(f"Raw synthetic training set: {len(raw_syn_train)} samples")
    print(f"Refined synthetic training set: {len(refined_syn_train)} samples")
    
    # Standardize features
    scaler = StandardScaler()
    real_train_scaled = scaler.fit_transform(real_train)
    real_test_scaled = scaler.transform(real_test)
    
    # Scale synthetic data using same scaler
    raw_syn_scaled = scaler.transform(raw_syn_train)
    refined_syn_scaled = scaler.transform(refined_syn_train)
    
    results = {
        'bic_analysis': {},
        'clustering_agreement': {},
        'clustering_agreement_cv': {},
        'cluster_quality': {}
    }
    
    # 1. BIC Analysis (like in the paper)
    print("\nPerforming BIC analysis...")
    
    datasets = {
        'real_train': real_train_scaled,
        'raw_synthetic': raw_syn_scaled,
        'refined_synthetic': refined_syn_scaled
    }
    
    for name, data in datasets.items():
        bic_scores = []
        aic_scores = []
        inertias = []
        
        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(data)
            
            # Calculate BIC and AIC
            n_samples = len(data)
            bic = k * np.log(n_samples) + kmeans.inertia_
            aic = 2 * k + kmeans.inertia_
            
            bic_scores.append(bic)
            aic_scores.append(aic)
            inertias.append(kmeans.inertia_)
        
        results['bic_analysis'][name] = {
            'n_clusters': list(n_clusters_range),
            'bic_scores': bic_scores,
            'aic_scores': aic_scores,
            'inertias': inertias
        }
    
    # 2. Single Train/Test Evaluation
    print(f"\nEvaluating clustering agreement with k={selected_k} (single split)...")
    
    # Fit models on training data
    kmeans_real = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    kmeans_real.fit(real_train_scaled)
    
    kmeans_raw = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    kmeans_raw.fit(raw_syn_scaled)
    
    kmeans_refined = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
    kmeans_refined.fit(refined_syn_scaled)
    
    # Predict on real test set
    real_test_pred = kmeans_real.predict(real_test_scaled)
    raw_test_pred = kmeans_raw.predict(real_test_scaled)
    refined_test_pred = kmeans_refined.predict(real_test_scaled)
    
    # Calculate agreement metrics
    if use_true_labels:
        # Compare with true labels
        reference_labels = labels_test
        print("Computing ARI/AMI against true labels...")
    else:
        # Use real data clustering predictions as reference
        reference_labels = real_test_pred
        print("Computing ARI/AMI against real data clustering predictions...")
    
    real_ari = adjusted_rand_score(reference_labels, real_test_pred)
    raw_ari = adjusted_rand_score(reference_labels, raw_test_pred)
    refined_ari = adjusted_rand_score(reference_labels, refined_test_pred)
    
    real_ami = adjusted_mutual_info_score(reference_labels, real_test_pred)
    raw_ami = adjusted_mutual_info_score(reference_labels, raw_test_pred)
    refined_ami = adjusted_mutual_info_score(reference_labels, refined_test_pred)
    
    results['clustering_agreement'] = {
        'real': {'ari': real_ari, 'ami': real_ami},
        'raw_synthetic': {'ari': raw_ari, 'ami': raw_ami},
        'refined_synthetic': {'ari': refined_ari, 'ami': refined_ami}
    }
    
    # 3. Cross-Validation for Robust Statistics (always performed)
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    # Prepare for CV
    if use_true_labels and len(real_labels.unique()) < len(real_labels) * 0.5:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_splits = list(cv.split(real_features, real_labels))
        print("Using stratified CV based on true labels")
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_splits = list(cv.split(real_features))
        print("Using regular CV")
    
    # Storage for CV results
    cv_results = {
        'real_ari': [], 'real_ami': [],
        'raw_ari': [], 'raw_ami': [],
        'refined_ari': [], 'refined_ami': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        if fold % 2 == 0:  # Progress indicator
            print(f"  Processing fold {fold + 1}/{cv_folds}...")
        
        # Split data for this fold
        X_train_fold = real_features.iloc[train_idx]
        X_test_fold = real_features.iloc[test_idx]
        
        if use_true_labels:
            y_test_fold = real_labels.iloc[test_idx]
        
        # FIXED: Sample synthetic data to match this fold's training size
        fold_train_size = len(X_train_fold)
        
        if len(raw_syn_features) < fold_train_size:
            raw_syn_fold = raw_syn_features.copy()
        else:
            raw_syn_fold = raw_syn_features.sample(n=fold_train_size, random_state=random_state + fold)
        
        if len(refined_syn_features) < fold_train_size:
            refined_syn_fold = refined_syn_features.copy()
        else:
            refined_syn_fold = refined_syn_features.sample(n=fold_train_size, random_state=random_state + fold)
        
        # Scale data for this fold
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_scaled = scaler_fold.transform(X_test_fold)
        
        # Scale synthetic data with this fold's scaler
        raw_syn_scaled_fold = scaler_fold.transform(raw_syn_fold)
        refined_syn_scaled_fold = scaler_fold.transform(refined_syn_fold)
        
        # Fit clustering models
        kmeans_real_fold = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        kmeans_real_fold.fit(X_train_scaled)
        
        kmeans_raw_fold = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        kmeans_raw_fold.fit(raw_syn_scaled_fold)
        
        kmeans_refined_fold = KMeans(n_clusters=selected_k, random_state=random_state, n_init=10)
        kmeans_refined_fold.fit(refined_syn_scaled_fold)
        
        # Predict on test set
        real_pred_fold = kmeans_real_fold.predict(X_test_scaled)
        raw_pred_fold = kmeans_raw_fold.predict(X_test_scaled)
        refined_pred_fold = kmeans_refined_fold.predict(X_test_scaled)
        
        # Determine reference labels for this fold
        if use_true_labels:
            ref_labels_fold = y_test_fold
        else:
            ref_labels_fold = real_pred_fold
        
        # Calculate metrics
        cv_results['real_ari'].append(adjusted_rand_score(ref_labels_fold, real_pred_fold))
        cv_results['real_ami'].append(adjusted_mutual_info_score(ref_labels_fold, real_pred_fold))
        
        cv_results['raw_ari'].append(adjusted_rand_score(ref_labels_fold, raw_pred_fold))
        cv_results['raw_ami'].append(adjusted_mutual_info_score(ref_labels_fold, raw_pred_fold))
        
        cv_results['refined_ari'].append(adjusted_rand_score(ref_labels_fold, refined_pred_fold))
        cv_results['refined_ami'].append(adjusted_mutual_info_score(ref_labels_fold, refined_pred_fold))
    
    # Calculate CV statistics
    cv_stats = {}
    for metric in cv_results:
        cv_stats[metric] = {
            'mean': np.mean(cv_results[metric]),
            'std': np.std(cv_results[metric], ddof=1),
            'values': cv_results[metric]
        }
    
    results['clustering_agreement_cv'] = cv_stats
    
    # 4. Cluster Quality Metrics (on test set from single split)
    real_silhouette = silhouette_score(real_test_scaled, real_test_pred)
    
    # For synthetic models, evaluate silhouette on their own test portions
    raw_test_size = min(len(raw_syn_scaled), len(real_test_scaled))
    refined_test_size = min(len(refined_syn_scaled), len(real_test_scaled))
    
    raw_test_indices = np.random.RandomState(random_state).choice(
        len(raw_syn_scaled), raw_test_size, replace=False)
    refined_test_indices = np.random.RandomState(random_state).choice(
        len(refined_syn_scaled), refined_test_size, replace=False)
    
    raw_test_data = raw_syn_scaled[raw_test_indices]
    refined_test_data = refined_syn_scaled[refined_test_indices]
    
    raw_test_labels_own = kmeans_raw.predict(raw_test_data)
    refined_test_labels_own = kmeans_refined.predict(refined_test_data)
    
    raw_silhouette = silhouette_score(raw_test_data, raw_test_labels_own)
    refined_silhouette = silhouette_score(refined_test_data, refined_test_labels_own)
    
    results['cluster_quality'] = {
        'real': real_silhouette,
        'raw_synthetic': raw_silhouette,
        'refined_synthetic': refined_silhouette
    }
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("CLUSTERING EVALUATION RESULTS")
    print('='*80)
    
    label_type = "true labels" if use_true_labels else "real clustering predictions"
    
    print(f"\nSINGLE TRAIN/TEST SPLIT RESULTS:")
    print(f"{'Metric':<25} {'Real':<15} {'Raw Synthetic':<15} {'Refined Synthetic':<15}")
    print("-" * 75)
    print(f"{'ARI (vs ' + label_type + ')':<25} {real_ari:<15.4f} {raw_ari:<15.4f} {refined_ari:<15.4f}")
    print(f"{'AMI (vs ' + label_type + ')':<25} {real_ami:<15.4f} {raw_ami:<15.4f} {refined_ami:<15.4f}")
    print(f"{'Silhouette Score':<25} {real_silhouette:<15.4f} {raw_silhouette:<15.4f} {refined_silhouette:<15.4f}")
    
    print(f"\n{cv_folds}-FOLD CROSS-VALIDATION RESULTS (Mean ± Std):")
    print(f"{'Metric':<25} {'Real':<20} {'Raw Synthetic':<20} {'Refined Synthetic':<20}")
    print("-" * 90)
    
    real_ari_cv = f"{cv_stats['real_ari']['mean']:.4f} ± {cv_stats['real_ari']['std']:.4f}"
    raw_ari_cv = f"{cv_stats['raw_ari']['mean']:.4f} ± {cv_stats['raw_ari']['std']:.4f}"
    refined_ari_cv = f"{cv_stats['refined_ari']['mean']:.4f} ± {cv_stats['refined_ari']['std']:.4f}"
    
    real_ami_cv = f"{cv_stats['real_ami']['mean']:.4f} ± {cv_stats['real_ami']['std']:.4f}"
    raw_ami_cv = f"{cv_stats['raw_ami']['mean']:.4f} ± {cv_stats['raw_ami']['std']:.4f}"
    refined_ami_cv = f"{cv_stats['refined_ami']['mean']:.4f} ± {cv_stats['refined_ami']['std']:.4f}"
    
    print(f"{'ARI (vs ' + label_type + ')':<25} {real_ari_cv:<20} {raw_ari_cv:<20} {refined_ari_cv:<20}")
    print(f"{'AMI (vs ' + label_type + ')':<25} {real_ami_cv:<20} {raw_ami_cv:<20} {refined_ami_cv:<20}")
    
    # Calculate improvements
    ari_improvement = cv_stats['refined_ari']['mean'] - cv_stats['raw_ari']['mean']
    ami_improvement = cv_stats['refined_ami']['mean'] - cv_stats['raw_ami']['mean']
    
    print(f"\nIMPROVEMENT FROM C-MAPS REFINEMENT:")
    print(f"ARI improvement: {ari_improvement:+.4f}")
    print(f"AMI improvement: {ami_improvement:+.4f}")
    
    # Statistical significance test (if you want to add)
    from scipy import stats
    if cv_folds >= 3:
        ari_ttest = stats.ttest_rel(cv_stats['refined_ari']['values'], cv_stats['raw_ari']['values'])
        ami_ttest = stats.ttest_rel(cv_stats['refined_ami']['values'], cv_stats['raw_ami']['values'])
        
        print(f"\nSTATISTICAL SIGNIFICANCE (paired t-test):")
        print(f"ARI improvement p-value: {ari_ttest.pvalue:.4f}")
        print(f"AMI improvement p-value: {ami_ttest.pvalue:.4f}")
    
    print(f"\nAll training sets had equal size ({real_train_size} samples) for fair comparison")
    
    # Store configuration info in results
    results['config'] = {
        'use_true_labels': use_true_labels,
        'target_column': target_column,
        'label_type': label_type
    }
    
    return results

def plot_clustering_results(results):
    """
    Visualize corrected clustering evaluation results with error bars.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. BIC Analysis (like in the paper)
    for name, data in results['bic_analysis'].items():
        label = name.replace('_', ' ').title()
        axes[0, 0].plot(data['n_clusters'], data['bic_scores'], 
                       marker='o', label=label)
    
    axes[0, 0].set_xlabel('Number of Clusters')
    axes[0, 0].set_ylabel('BIC Score')
    axes[0, 0].set_title('Bayesian Information Criterion')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Inertia (Elbow Method)
    for name, data in results['bic_analysis'].items():
        label = name.replace('_', ' ').title()
        axes[0, 1].plot(data['n_clusters'], data['inertias'], 
                       marker='o', label=label)
    
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Inertia')
    axes[0, 1].set_title('Elbow Method Analysis')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Clustering Agreement Metrics with Error Bars (FROM CV RESULTS)
    cv_stats = results['clustering_agreement_cv']
    label_type = results['config']['label_type']
    
    metrics = ['ARI', 'AMI']
    
    # Get CV means and stds
    real_means = [cv_stats['real_ari']['mean'], cv_stats['real_ami']['mean']]
    real_stds = [cv_stats['real_ari']['std'], cv_stats['real_ami']['std']]
    
    raw_means = [cv_stats['raw_ari']['mean'], cv_stats['raw_ami']['mean']]
    raw_stds = [cv_stats['raw_ari']['std'], cv_stats['raw_ami']['std']]
    
    refined_means = [cv_stats['refined_ari']['mean'], cv_stats['refined_ami']['mean']]
    refined_stds = [cv_stats['refined_ari']['std'], cv_stats['refined_ami']['std']]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    # Plot bars with error bars
    axes[1, 0].bar(x - width, real_means, width, yerr=real_stds, 
                   label='Real', color='#7f7fff', alpha=0.7, capsize=5)
    axes[1, 0].bar(x, raw_means, width, yerr=raw_stds,
                   label='Raw Synthetic', color='#ff7f7f', alpha=0.7, capsize=5)
    axes[1, 0].bar(x + width, refined_means, width, yerr=refined_stds,
                   label='Refined Synthetic', color='#7fbf7f', alpha=0.7, capsize=5)
    
    axes[1, 0].set_ylabel('Score (Mean ± Std)')
    axes[1, 0].set_title(f'Clustering Agreement (vs {label_type})')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, min(1.0, max(max(real_means), max(raw_means), max(refined_means)) * 1.2))
    
    # Add value labels on bars (mean ± std)
    for i, (metric_name, real_m, real_s, raw_m, raw_s, ref_m, ref_s) in enumerate(
        zip(metrics, real_means, real_stds, raw_means, raw_stds, refined_means, refined_stds)):
        
        # Real values
        axes[1, 0].text(i - width, real_m + real_s + 0.01, f'{real_m:.3f}±{real_s:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
        # Raw values  
        axes[1, 0].text(i, raw_m + raw_s + 0.01, f'{raw_m:.3f}±{raw_s:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
        # Refined values
        axes[1, 0].text(i + width, ref_m + ref_s + 0.01, f'{ref_m:.3f}±{ref_s:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    # 4. Silhouette Score Comparison (separate plot, no error bars as it's single split)
    sil_scores = [
        results['cluster_quality']['real'],
        results['cluster_quality']['raw_synthetic'],
        results['cluster_quality']['refined_synthetic']
    ]
    labels = ['Real', 'Raw Synthetic', 'Refined Synthetic']
    colors = ['#7f7fff', '#ff7f7f', '#7fbf7f']
    
    axes[1, 1].bar(labels, sil_scores, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Silhouette Score')
    axes[1, 1].set_title('Cluster Quality (Silhouette Score)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, max(sil_scores) * 1.1)
    
    # Add value labels on bars
    for i, v in enumerate(sil_scores):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of improvements with statistical significance
    print("\n" + "="*60)
    print("CLUSTERING IMPROVEMENT SUMMARY")
    print("="*60)
    
    ari_improvement = refined_means[0] - raw_means[0]
    ami_improvement = refined_means[1] - raw_means[1]
    
    print(f"ARI improvement: {ari_improvement:+.4f} ({ari_improvement/raw_means[0]*100:+.1f}%)")
    print(f"AMI improvement: {ami_improvement:+.4f} ({ami_improvement/raw_means[1]*100:+.1f}%)")
    
    # Effect size (Cohen's d)
    ari_pooled_std = np.sqrt((raw_stds[0]**2 + refined_stds[0]**2) / 2)
    ami_pooled_std = np.sqrt((raw_stds[1]**2 + refined_stds[1]**2) / 2)
    
    ari_effect_size = ari_improvement / ari_pooled_std if ari_pooled_std > 0 else 0
    ami_effect_size = ami_improvement / ami_pooled_std if ami_pooled_std > 0 else 0
    
    print(f"ARI effect size (Cohen's d): {ari_effect_size:.3f}")
    print(f"AMI effect size (Cohen's d): {ami_effect_size:.3f}")
    
    if 'clustering_agreement_cv' in results:
        from scipy import stats
        if len(results['clustering_agreement_cv']['refined_ari']['values']) >= 3:
            ari_ttest = stats.ttest_rel(
                results['clustering_agreement_cv']['refined_ari']['values'], 
                results['clustering_agreement_cv']['raw_ari']['values']
            )
            ami_ttest = stats.ttest_rel(
                results['clustering_agreement_cv']['refined_ami']['values'], 
                results['clustering_agreement_cv']['raw_ami']['values']
            )
            
            print(f"\nStatistical significance (paired t-test):")
            print(f"ARI p-value: {ari_ttest.pvalue:.4f} {'(significant)' if ari_ttest.pvalue < 0.05 else '(not significant)'}")
            print(f"AMI p-value: {ami_ttest.pvalue:.4f} {'(significant)' if ami_ttest.pvalue < 0.05 else '(not significant)'}")