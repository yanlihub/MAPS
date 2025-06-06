# """
# Utility Evaluation for C-MAPS Framework
# Train on Synthetic, Test on Real approach with Random Forest
# """

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
# from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings

# warnings.filterwarnings('ignore')

# def prepare_data_for_classification(data, target_column='cancer_types', feature_encoders=None):
#     """
#     Prepare data for classification by separating features and target.
    
#     Args:
#         data: DataFrame with features and target
#         target_column: Name of the target column
#         feature_encoders: Dict of fitted encoders for features (None to fit new ones)
        
#     Returns:
#         X: Features (encoded)
#         y: Target (encoded if categorical)
#         target_encoder: Fitted label encoder for target (None if target is already numeric)
#         feature_encoders: Dict of fitted encoders for categorical features
#     """
#     if target_column not in data.columns:
#         raise ValueError(f"Target column '{target_column}' not found in data")
    
#     X = data.drop(columns=[target_column]).copy()
#     y = data[target_column].copy()
    
#     # Encode target if it's categorical
#     target_encoder = None
#     if y.dtype == 'object' or y.dtype.name == 'category':
#         target_encoder = LabelEncoder()
#         # Handle any missing values in target
#         y = y.fillna('missing')
#         y = target_encoder.fit_transform(y.astype(str))
    
#     # Handle categorical features
#     if feature_encoders is None:
#         # Fit new encoders
#         feature_encoders = {}
#         for col in X.columns:
#             if X[col].dtype == 'object' or X[col].dtype.name == 'category':
#                 le = LabelEncoder()
#                 # Handle missing values
#                 X_col_filled = X[col].fillna('missing').astype(str)
#                 X[col] = le.fit_transform(X_col_filled)
#                 feature_encoders[col] = le
#                 print(f"  Encoded categorical feature '{col}': {len(le.classes_)} categories")
#             else:
#                 # Ensure numeric columns are float and handle missing values
#                 X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
#     else:
#         # Use existing encoders
#         for col in X.columns:
#             if col in feature_encoders:
#                 # Apply existing encoder
#                 X_col_filled = X[col].fillna('missing').astype(str)
#                 try:
#                     X[col] = feature_encoders[col].transform(X_col_filled)
#                 except ValueError as e:
#                     # Handle unseen categories by mapping them to a default value
#                     print(f"  Warning: Unseen categories in '{col}', using fallback encoding")
#                     # Get known categories
#                     known_categories = set(feature_encoders[col].classes_)
#                     # Map unknown categories to the first known category
#                     X_col_mapped = X_col_filled.apply(
#                         lambda x: x if x in known_categories else feature_encoders[col].classes_[0]
#                     )
#                     X[col] = feature_encoders[col].transform(X_col_mapped)
#             else:
#                 # Ensure numeric columns are float and handle missing values
#                 X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
#     # Final check: ensure all data is numeric
#     for col in X.columns:
#         if X[col].dtype == 'object':
#             print(f"  Warning: Column '{col}' still contains non-numeric data, converting to numeric")
#             X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
#     # Ensure no infinite values
#     X = X.replace([np.inf, -np.inf], 0)
    
#     print(f"  Final feature matrix shape: {X.shape}")
#     print(f"  Final feature matrix dtypes: {X.dtypes.nunique()} unique types: {X.dtypes.unique()}")
    
#     return X, y, target_encoder, feature_encoders

# def evaluate_model_with_cv(X_train, y_train, X_test, y_test, 
#                           model_name="Random Forest", 
#                           n_estimators=100, 
#                           cv_folds=5, 
#                           random_state=42):
#     """
#     Train a Random Forest with cross-validation and evaluate on test set.
    
#     Args:
#         X_train: Training features
#         y_train: Training target
#         X_test: Test features  
#         y_test: Test target
#         model_name: Name for the model (for reporting)
#         n_estimators: Number of trees in Random Forest
#         cv_folds: Number of CV folds
#         random_state: Random seed
        
#     Returns:
#         Dictionary with evaluation results
#     """
#     # Initialize Random Forest
#     rf = RandomForestClassifier(
#         n_estimators=n_estimators,
#         random_state=random_state,
#         n_jobs=-1
#     )
    
#     # Cross-validation on training data
#     cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
#     cv_scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='accuracy')
#     cv_f1_scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='f1_macro')
    
#     # Train on full training set
#     rf.fit(X_train, y_train)
    
#     # Predict on test set
#     y_pred = rf.predict(X_test)
#     y_pred_proba = rf.predict_proba(X_test)
    
#     # Calculate test metrics
#     test_accuracy = accuracy_score(y_test, y_pred)
#     test_f1 = f1_score(y_test, y_pred, average='macro')
    
#     # Calculate ROC AUC (handle binary vs multiclass and class mismatch)
#     try:
#         n_classes_test = len(np.unique(y_test))
#         n_classes_pred = y_pred_proba.shape[1]
        
#         if n_classes_test == 2 and n_classes_pred == 2:
#             test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
#         elif n_classes_test == n_classes_pred:
#             test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
#         else:
#             # Class mismatch - use macro average with available classes
#             print(f"    Warning: Class mismatch - test has {n_classes_test} classes, model predicts {n_classes_pred}")
#             test_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
#     except Exception as e:
#         print(f"    Warning: Could not calculate ROC AUC: {e}")
#         test_roc_auc = np.nan
    
#     results = {
#         'model_name': model_name,
#         'cv_accuracy_mean': np.mean(cv_scores),
#         'cv_accuracy_std': np.std(cv_scores),
#         'cv_f1_mean': np.mean(cv_f1_scores),
#         'cv_f1_std': np.std(cv_f1_scores),
#         'test_accuracy': test_accuracy,
#         'test_f1_macro': test_f1,
#         'test_roc_auc': test_roc_auc,
#         'cv_scores': cv_scores,
#         'cv_f1_scores': cv_f1_scores,
#         'classification_report': classification_report(y_test, y_pred),
#         'trained_model': rf
#     }
    
#     return results

# def run_utility_evaluation(real_data, raw_synthetic_data, refined_synthetic_data, 
#                           target_column='cancer_types', test_size=0.2, random_state=42):
#     """
#     Run complete utility evaluation comparing three scenarios.
    
#     Args:
#         real_data: Real dataset
#         raw_synthetic_data: Original synthetic dataset
#         refined_synthetic_data: Refined synthetic dataset (after C-MAPS)
#         target_column: Name of the target column
#         test_size: Proportion of real data to use for testing (for Scenario 3)
#         random_state: Random seed
        
#     Returns:
#         Dictionary with all evaluation results
#     """
#     print("=" * 80)
#     print("UTILITY EVALUATION: TRAIN ON SYNTHETIC, TEST ON REAL")
#     print("=" * 80)
    
#     # First, check what classes are available in each dataset
#     print("\nAnalyzing target classes across datasets...")
#     all_targets = pd.concat([
#         real_data[target_column],
#         raw_synthetic_data[target_column], 
#         refined_synthetic_data[target_column]
#     ])
#     unique_classes = sorted(all_targets.unique())
#     print(f"All unique classes found: {unique_classes}")
    
#     # Fit target encoder on all possible classes
#     target_encoder = LabelEncoder()
#     target_encoder.fit(all_targets.astype(str))
#     print(f"Target encoder classes: {target_encoder.classes_}")
    
#     # Split real data into train and test for Scenario 3 (stratified)
#     print(f"\nSplitting real data for proper evaluation...")
#     print(f"Using {test_size:.0%} of real data for testing, {1-test_size:.0%} for training in Scenario 3")
    
#     real_train, real_test = train_test_split(
#         real_data, 
#         test_size=test_size, 
#         stratify=real_data[target_column], 
#         random_state=random_state
#     )
    
#     print(f"Real train set: {len(real_train)} samples")
#     print(f"Real test set: {len(real_test)} samples")
    
#     # Prepare all datasets with consistent encoding
#     print("\nPreparing real test data...")
#     X_real_test, y_real_test_temp, _, feature_encoders = prepare_data_for_classification(
#         real_test, target_column
#     )
#     y_real_test = target_encoder.transform(real_test[target_column].astype(str))
    
#     print("\nPreparing real train data...")
#     X_real_train, y_real_train_temp, _, _ = prepare_data_for_classification(
#         real_train, target_column, feature_encoders
#     )
#     y_real_train = target_encoder.transform(real_train[target_column].astype(str))
    
#     print("\nPreparing raw synthetic data...")
#     X_raw_syn, y_raw_syn_temp, _, _ = prepare_data_for_classification(
#         raw_synthetic_data, target_column, feature_encoders
#     )
#     y_raw_syn = target_encoder.transform(raw_synthetic_data[target_column].astype(str))
    
#     print("\nPreparing refined synthetic data...")
#     X_refined_syn, y_refined_syn_temp, _, _ = prepare_data_for_classification(
#         refined_synthetic_data, target_column, feature_encoders
#     )
#     y_refined_syn = target_encoder.transform(refined_synthetic_data[target_column].astype(str))
    
#     print(f"\nData preparation complete:")
#     print(f"Real test data shape: {X_real_test.shape}")
#     print(f"Real train data shape: {X_real_train.shape}")
#     print(f"Raw synthetic data shape: {X_raw_syn.shape}")
#     print(f"Refined synthetic data shape: {X_refined_syn.shape}")
#     print(f"Number of encoded categorical features: {len(feature_encoders)}")
    
#     # Check class distributions
#     print(f"\nClass distributions:")
#     print(f"Real test classes: {np.bincount(y_real_test)}")
#     print(f"Real train classes: {np.bincount(y_real_train)}")
#     print(f"Raw synthetic classes: {np.bincount(y_raw_syn)}")
#     print(f"Refined synthetic classes: {np.bincount(y_refined_syn)}")
    
#     # Check for any remaining non-numeric data
#     print(f"\nData types check:")
#     print(f"Real test dtypes: {X_real_test.dtypes.unique()}")
#     print(f"Raw synthetic dtypes: {X_raw_syn.dtypes.unique()}")
#     print(f"Refined synthetic dtypes: {X_refined_syn.dtypes.unique()}")
    
#     results = {}
    
#     # Scenario 1: Train on raw synthetic, test on real
#     print(f"\n{'='*50}")
#     print("SCENARIO 1: Train on Raw Synthetic, Test on Real")
#     print('='*50)
    
#     results['raw_synthetic'] = evaluate_model_with_cv(
#         X_raw_syn, y_raw_syn, X_real_test, y_real_test,
#         model_name="Raw Synthetic → Real",
#         random_state=random_state
#     )
    
#     print(f"CV Accuracy: {results['raw_synthetic']['cv_accuracy_mean']:.4f} ± {results['raw_synthetic']['cv_accuracy_std']:.4f}")
#     print(f"Test Accuracy: {results['raw_synthetic']['test_accuracy']:.4f}")
#     print(f"Test F1-Macro: {results['raw_synthetic']['test_f1_macro']:.4f}")
#     if not np.isnan(results['raw_synthetic']['test_roc_auc']):
#         print(f"Test ROC-AUC: {results['raw_synthetic']['test_roc_auc']:.4f}")
    
#     # Scenario 2: Train on refined synthetic, test on real
#     print(f"\n{'='*50}")
#     print("SCENARIO 2: Train on Refined Synthetic, Test on Real")
#     print('='*50)
    
#     results['refined_synthetic'] = evaluate_model_with_cv(
#         X_refined_syn, y_refined_syn, X_real_test, y_real_test,
#         model_name="Refined Synthetic → Real",
#         random_state=random_state
#     )
    
#     print(f"CV Accuracy: {results['refined_synthetic']['cv_accuracy_mean']:.4f} ± {results['refined_synthetic']['cv_accuracy_std']:.4f}")
#     print(f"Test Accuracy: {results['refined_synthetic']['test_accuracy']:.4f}")
#     print(f"Test F1-Macro: {results['refined_synthetic']['test_f1_macro']:.4f}")
#     if not np.isnan(results['refined_synthetic']['test_roc_auc']):
#         print(f"Test ROC-AUC: {results['refined_synthetic']['test_roc_auc']:.4f}")
    
#     # Scenario 3: Train on real (train split), test on real (test split) - PROPER BASELINE
#     print(f"\n{'='*50}")
#     print("SCENARIO 3: Train on Real (Train Split), Test on Real (Test Split)")
#     print("             [PROPER BASELINE - NO DATA LEAKAGE]")
#     print('='*50)
    
#     results['real_baseline'] = evaluate_model_with_cv(
#         X_real_train, y_real_train, X_real_test, y_real_test,
#         model_name="Real (Train) → Real (Test)",
#         random_state=random_state
#     )
    
#     print(f"CV Accuracy: {results['real_baseline']['cv_accuracy_mean']:.4f} ± {results['real_baseline']['cv_accuracy_std']:.4f}")
#     print(f"Test Accuracy: {results['real_baseline']['test_accuracy']:.4f}")
#     print(f"Test F1-Macro: {results['real_baseline']['test_f1_macro']:.4f}")
#     if not np.isnan(results['real_baseline']['test_roc_auc']):
#         print(f"Test ROC-AUC: {results['real_baseline']['test_roc_auc']:.4f}")
    
#     return results


# def plot_evaluation_results(results, separate_figures=False, figsize=None):
#     """
#     Plot comparison of evaluation results.
    
#     Args:
#         results: Results dictionary from run_utility_evaluation
#         separate_figures: bool, whether to create separate figures for each plot
#         figsize: tuple, optional, figure size for individual plots
#     """
#     # Prepare data for plotting
#     scenarios = ['Raw Synthetic\n→ Real', 'Refined Synthetic\n→ Real', 'Real Train\n→ Real Test\n(Baseline)']
#     test_accuracies = [
#         results['raw_synthetic']['test_accuracy'],
#         results['refined_synthetic']['test_accuracy'],
#         results['real_baseline']['test_accuracy']
#     ]
#     test_f1_scores = [
#         results['raw_synthetic']['test_f1_macro'],
#         results['refined_synthetic']['test_f1_macro'],
#         results['real_baseline']['test_f1_macro']
#     ]
    
#     # Handle potential NaN ROC AUC values
#     test_roc_aucs = []
#     roc_auc_available = True
#     for scenario_key in ['raw_synthetic', 'refined_synthetic', 'real_baseline']:
#         roc_auc = results[scenario_key]['test_roc_auc']
#         if np.isnan(roc_auc):
#             roc_auc_available = False
#             test_roc_aucs.append(0)  # Placeholder
#         else:
#             test_roc_aucs.append(roc_auc)
    
#     cv_accuracies = [
#         results['raw_synthetic']['cv_accuracy_mean'],
#         results['refined_synthetic']['cv_accuracy_mean'],
#         results['real_baseline']['cv_accuracy_mean']
#     ]
#     cv_accuracy_stds = [
#         results['raw_synthetic']['cv_accuracy_std'],
#         results['refined_synthetic']['cv_accuracy_std'],
#         results['real_baseline']['cv_accuracy_std']
#     ]
    
#     # Colors for different scenarios
#     colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']  # Light red, green, blue
    
#     if separate_figures:
#         # Create separate figures for each plot
#         figures = []
#         individual_figsize = (8, 6) if figsize is None else figsize
        
#         # Plot 1: Test Accuracy
#         fig1, ax1 = plt.subplots(1, 1, figsize=individual_figsize)
#         ax1.bar(scenarios, test_accuracies, color=colors, alpha=0.7)
#         ax1.set_title('Test Accuracy Comparison')
#         ax1.set_ylabel('Accuracy')
#         ax1.set_ylim(0, 1)
#         ax1.grid(False)
#         for i, v in enumerate(test_accuracies):
#             ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
#         plt.tight_layout()
#         figures.append(fig1)
#         plt.show()
        
#         # Plot 2: Test F1-Macro
#         fig2, ax2 = plt.subplots(1, 1, figsize=individual_figsize)
#         ax2.bar(scenarios, test_f1_scores, color=colors, alpha=0.7)
#         ax2.set_title('Test F1-Macro Comparison')
#         ax2.set_ylabel('F1-Macro Score')
#         ax2.set_ylim(0, 1)
#         ax2.grid(False)
#         for i, v in enumerate(test_f1_scores):
#             ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
#         plt.tight_layout()
#         figures.append(fig2)
#         plt.show()
        
#         # Plot 3: Test ROC-AUC (if available)
#         if roc_auc_available:
#             fig3, ax3 = plt.subplots(1, 1, figsize=individual_figsize)
#             ax3.bar(scenarios, test_roc_aucs, color=colors, alpha=0.7)
#             ax3.set_title('Test ROC-AUC Comparison')
#             ax3.set_ylabel('ROC-AUC Score')
#             ax3.set_ylim(0, 1.05)
#             ax3.grid(False)
#             for i, v in enumerate(test_roc_aucs):
#                 ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
#             plt.tight_layout()
#             figures.append(fig3)
#             plt.show()
#         else:
#             print("Note: ROC AUC plot skipped due to calculation issues")
        
#         # Plot 4: CV Accuracy with Error Bars
#         fig4, ax4 = plt.subplots(1, 1, figsize=individual_figsize)
#         ax4.bar(scenarios, cv_accuracies, yerr=cv_accuracy_stds, 
#                 color=colors, alpha=0.7, capsize=5)
#         ax4.set_title('Cross-Validation Accuracy (with std)')
#         ax4.set_ylabel('CV Accuracy')
#         ax4.set_ylim(0, 1)
#         ax4.grid(False)
#         for i, (v, std) in enumerate(zip(cv_accuracies, cv_accuracy_stds)):
#             ax4.text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
#         plt.tight_layout()
#         figures.append(fig4)
#         plt.show()
        
#         return figures
    
#     else:
#         # Create subplots - adjust based on ROC AUC availability
#         if roc_auc_available:
#             combined_figsize = (15, 10) if figsize is None else figsize
#             fig, axes = plt.subplots(2, 2, figsize=combined_figsize)
#             plot_roc = True
#         else:
#             combined_figsize = (18, 6) if figsize is None else figsize
#             fig, axes = plt.subplots(1, 3, figsize=combined_figsize)
#             plot_roc = False
#             print("Note: ROC AUC plot skipped due to calculation issues")
        
#         if plot_roc:
#             # Plot 1: Test Accuracy
#             axes[0, 0].bar(scenarios, test_accuracies, color=colors, alpha=0.7)
#             axes[0, 0].set_title('Test Accuracy Comparison')
#             axes[0, 0].set_ylabel('Accuracy')
#             axes[0, 0].set_ylim(0, 1)
#             axes[0, 0].grid(False)
#             for i, v in enumerate(test_accuracies):
#                 axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
#             # Plot 2: Test F1-Macro
#             axes[0, 1].bar(scenarios, test_f1_scores, color=colors, alpha=0.7)
#             axes[0, 1].set_title('Test F1-Macro Comparison')
#             axes[0, 1].set_ylabel('F1-Macro Score')
#             axes[0, 1].set_ylim(0, 1)
#             axes[0, 1].grid(False)
#             for i, v in enumerate(test_f1_scores):
#                 axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
#             # Plot 3: Test ROC-AUC
#             axes[1, 0].bar(scenarios, test_roc_aucs, color=colors, alpha=0.7)
#             axes[1, 0].set_title('Test ROC-AUC Comparison')
#             axes[1, 0].set_ylabel('ROC-AUC Score')
#             axes[1, 0].set_ylim(0, 1.05)
#             axes[1, 0].grid(False)
#             for i, v in enumerate(test_roc_aucs):
#                 axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
#             # Plot 4: CV Accuracy with Error Bars
#             axes[1, 1].bar(scenarios, cv_accuracies, yerr=cv_accuracy_stds, 
#                            color=colors, alpha=0.7, capsize=5)
#             axes[1, 1].set_title('Cross-Validation Accuracy (with std)')
#             axes[1, 1].set_ylabel('CV Accuracy')
#             axes[1, 1].set_ylim(0, 1)
#             axes[1, 1].grid(False)
#             for i, (v, std) in enumerate(zip(cv_accuracies, cv_accuracy_stds)):
#                 axes[1, 1].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
#         else:
#             # Single row layout when ROC AUC is not available
#             # Plot 1: Test Accuracy
#             axes[0].bar(scenarios, test_accuracies, color=colors, alpha=0.7)
#             axes[0].set_title('Test Accuracy Comparison')
#             axes[0].set_ylabel('Accuracy')
#             axes[0].set_ylim(0, 1)
#             axes[0].grid(False)
#             for i, v in enumerate(test_accuracies):
#                 axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
#             # Plot 2: Test F1-Macro
#             axes[1].bar(scenarios, test_f1_scores, color=colors, alpha=0.7)
#             axes[1].set_title('Test F1-Macro Comparison')
#             axes[1].set_ylabel('F1-Macro Score')
#             axes[1].set_ylim(0, 1)
#             axes[1].grid(False)
#             for i, v in enumerate(test_f1_scores):
#                 axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
#             # Plot 3: CV Accuracy with Error Bars
#             axes[2].bar(scenarios, cv_accuracies, yerr=cv_accuracy_stds, 
#                         color=colors, alpha=0.7, capsize=5)
#             axes[2].set_title('Cross-Validation Accuracy (with std)')
#             axes[2].set_ylabel('CV Accuracy')
#             axes[2].set_ylim(0, 1)
#             axes[2].grid(False)
#             for i, (v, std) in enumerate(zip(cv_accuracies, cv_accuracy_stds)):
#                 axes[2].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        
#         plt.tight_layout()
#         plt.show()

# def print_summary_table(results):
#     """
#     Print a summary table of all results.
    
#     Args:
#         results: Results dictionary from run_utility_evaluation
#     """
#     print("\n" + "=" * 80)
#     print("SUMMARY TABLE")
#     print("=" * 80)
    
#     print(f"{'Scenario':<30} {'CV Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Test AUC':<12}")
#     print("-" * 85)
    
#     scenarios = [
#         ('Raw Synthetic → Real', results['raw_synthetic']),
#         ('Refined Synthetic → Real', results['refined_synthetic']),
#         ('Real Train → Real Test', results['real_baseline'])
#     ]
    
#     for name, result in scenarios:
#         cv_acc = f"{result['cv_accuracy_mean']:.4f}±{result['cv_accuracy_std']:.3f}"
#         test_acc = f"{result['test_accuracy']:.4f}"
#         test_f1 = f"{result['test_f1_macro']:.4f}"
        
#         if np.isnan(result['test_roc_auc']):
#             test_auc = "N/A"
#         else:
#             test_auc = f"{result['test_roc_auc']:.4f}"
        
#         print(f"{name:<30} {cv_acc:<12} {test_acc:<12} {test_f1:<12} {test_auc:<12}")
    
#     # Calculate improvement (using accuracy as primary metric)
#     raw_acc = results['raw_synthetic']['test_accuracy']
#     refined_acc = results['refined_synthetic']['test_accuracy']
#     baseline_acc = results['real_baseline']['test_accuracy']
    
#     improvement = ((refined_acc - raw_acc) / raw_acc) * 100 if raw_acc > 0 else 0
#     baseline_gap_raw = ((baseline_acc - raw_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
#     baseline_gap_refined = ((baseline_acc - refined_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
    
#     print("\n" + "=" * 85)
#     print("IMPROVEMENT ANALYSIS")
#     print("=" * 85)
#     print(f"Improvement from C-MAPS refinement: {improvement:+.2f}%")
#     print(f"Gap to baseline (Raw Synthetic): {baseline_gap_raw:.2f}%")
#     print(f"Gap to baseline (Refined Synthetic): {baseline_gap_refined:.2f}%")
#     print(f"Refinement reduces gap by: {baseline_gap_raw - baseline_gap_refined:.2f} percentage points")
    
#     # Additional analysis using F1 scores
#     raw_f1 = results['raw_synthetic']['test_f1_macro']
#     refined_f1 = results['refined_synthetic']['test_f1_macro']
#     baseline_f1 = results['real_baseline']['test_f1_macro']
    
#     f1_improvement = ((refined_f1 - raw_f1) / raw_f1) * 100 if raw_f1 > 0 else 0
#     print(f"\nF1-Macro improvement from refinement: {f1_improvement:+.2f}%")

# def debug_data_types(real_data, raw_synthetic_data, refined_synthetic_data, target_column='cancer_types'):
#     """
#     Debug function to check data types and identify potential issues.
#     """
#     print("=" * 60)
#     print("DATA TYPE DEBUGGING")
#     print("=" * 60)
    
#     datasets = {
#         'Real': real_data,
#         'Raw Synthetic': raw_synthetic_data,
#         'Refined Synthetic': refined_synthetic_data
#     }
    
#     for name, data in datasets.items():
#         print(f"\n{name} Dataset:")
#         print(f"  Shape: {data.shape}")
#         print(f"  Columns: {list(data.columns)}")
#         print(f"  Target column '{target_column}' present: {target_column in data.columns}")
        
#         if target_column in data.columns:
#             print(f"  Target unique values: {data[target_column].nunique()}")
#             print(f"  Target sample values: {data[target_column].unique()[:5]}")
        
#         # Check for categorical columns
#         categorical_cols = []
#         for col in data.columns:
#             if col != target_column and (data[col].dtype == 'object' or data[col].dtype.name == 'category'):
#                 categorical_cols.append(col)
#                 print(f"  Categorical column '{col}': {data[col].unique()[:3]}...")
        
#         print(f"  Total categorical features: {len(categorical_cols)}")
#         print(f"  Data types: {data.dtypes.value_counts().to_dict()}")

"""
Utility Evaluation for C-MAPS Framework
Train on Synthetic, Test on Real approach with Random Forest
Fixed to properly handle cross-validation: Train on Synthetic, Validate on Real
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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
    """
    # Split real validation data into folds
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    cv_accuracy_scores = []
    cv_f1_scores = []
    
    # For each fold, train on all synthetic data and validate on a portion of real data
    for fold_idx, (_, val_indices) in enumerate(cv_strategy.split(X_real_val, y_real_val)):
        # Use the validation indices to get the validation set for this fold
        X_val_fold = X_real_val.iloc[val_indices] if hasattr(X_real_val, 'iloc') else X_real_val[val_indices]
        y_val_fold = y_real_val[val_indices]
        
        # Train on all synthetic data
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        rf.fit(X_synthetic, y_synthetic)
        
        # Validate on real data fold
        y_pred = rf.predict(X_val_fold)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        
        cv_accuracy_scores.append(accuracy)
        cv_f1_scores.append(f1)
        
        print(f"    Fold {fold_idx+1}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return cv_accuracy_scores, cv_f1_scores

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
        Dictionary with evaluation results
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
        
        cv_accuracy_scores, cv_f1_scores = train_synthetic_validate_real_cv(
            X_train, y_train, X_real_for_cv, y_real_for_cv, 
            n_estimators, cv_folds, random_state
        )
    else:
        # Standard CV: Train and validate on same type of data
        print(f"  Performing standard CV: Train and validate on same data type")
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_accuracy_scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='accuracy')
        cv_f1_scores = cross_val_score(rf, X_train, y_train, cv=cv_strategy, scoring='f1_macro')
    
    # Train on full training set for final test evaluation
    rf.fit(X_train, y_train)
    
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
    
    results = {
        'model_name': model_name,
        'cv_accuracy_mean': np.mean(cv_accuracy_scores),
        'cv_accuracy_std': np.std(cv_accuracy_scores),
        'cv_f1_mean': np.mean(cv_f1_scores),
        'cv_f1_std': np.std(cv_f1_scores),
        'test_accuracy': test_accuracy,
        'test_f1_macro': test_f1,
        'test_roc_auc': test_roc_auc,
        'cv_scores': cv_accuracy_scores,
        'cv_f1_scores': cv_f1_scores,
        'classification_report': classification_report(y_test, y_pred),
        'trained_model': rf
    }
    
    return results

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
    print("UTILITY EVALUATION: TRAIN ON SYNTHETIC, TEST ON REAL")
    print("Fixed Cross-Validation: Train on Synthetic, Validate on Real")
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
    
    print("\nPreparing raw synthetic data...")
    X_raw_syn, y_raw_syn_temp, _, _ = prepare_data_for_classification(
        raw_synthetic_data, target_column, feature_encoders
    )
    y_raw_syn = target_encoder.transform(raw_synthetic_data[target_column].astype(str))
    
    print("\nPreparing refined synthetic data...")
    X_refined_syn, y_refined_syn_temp, _, _ = prepare_data_for_classification(
        refined_synthetic_data, target_column, feature_encoders
    )
    y_refined_syn = target_encoder.transform(refined_synthetic_data[target_column].astype(str))
    
    print(f"\nData preparation complete:")
    print(f"Real test data shape: {X_real_test.shape}")
    print(f"Real train data shape: {X_real_train.shape}")
    print(f"Real CV validation data shape: {X_real_cv_val.shape}")
    print(f"Raw synthetic data shape: {X_raw_syn.shape}")
    print(f"Refined synthetic data shape: {X_refined_syn.shape}")
    print(f"Number of encoded categorical features: {len(feature_encoders)}")
    
    # Check class distributions
    print(f"\nClass distributions:")
    print(f"Real test classes: {np.bincount(y_real_test)}")
    print(f"Real train classes: {np.bincount(y_real_train)}")
    print(f"Real CV val classes: {np.bincount(y_real_cv_val)}")
    print(f"Raw synthetic classes: {np.bincount(y_raw_syn)}")
    print(f"Refined synthetic classes: {np.bincount(y_refined_syn)}")
    
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
    print(f"Test Accuracy: {results['real_baseline']['test_accuracy']:.4f}")
    print(f"Test F1-Macro: {results['real_baseline']['test_f1_macro']:.4f}")
    if not np.isnan(results['real_baseline']['test_roc_auc']):
        print(f"Test ROC-AUC: {results['real_baseline']['test_roc_auc']:.4f}")
    
    return results


def plot_evaluation_results(results, separate_figures=False, figsize=None):
    """
    Plot comparison of evaluation results.
    
    Args:
        results: Results dictionary from run_utility_evaluation
        separate_figures: bool, whether to create separate figures for each plot
        figsize: tuple, optional, figure size for individual plots
    """
    # Prepare data for plotting
    scenarios = ['Raw Synthetic\n→ Real', 'Refined Synthetic\n→ Real', 'Real Train\n→ Real Test\n("Oracle")']
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
    
    # Colors for different scenarios
    colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']  # Light red, green, blue
    
    if separate_figures:
        # Create separate figures for each plot
        figures = []
        individual_figsize = (8, 6) if figsize is None else figsize
        
        # Plot 1: Test Accuracy
        fig1, ax1 = plt.subplots(1, 1, figsize=individual_figsize)
        ax1.bar(scenarios, test_accuracies, color=colors, alpha=0.7)
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(False)
        for i, v in enumerate(test_accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        figures.append(fig1)
        plt.show()
        
        # Plot 2: Test F1-Macro
        fig2, ax2 = plt.subplots(1, 1, figsize=individual_figsize)
        ax2.bar(scenarios, test_f1_scores, color=colors, alpha=0.7)
        ax2.set_title('Test F1-Macro Comparison')
        ax2.set_ylabel('F1-Macro Score')
        ax2.set_ylim(0, 1)
        ax2.grid(False)
        for i, v in enumerate(test_f1_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        figures.append(fig2)
        plt.show()
        
        # Plot 3: Test ROC-AUC (if available)
        if roc_auc_available:
            fig3, ax3 = plt.subplots(1, 1, figsize=individual_figsize)
            ax3.bar(scenarios, test_roc_aucs, color=colors, alpha=0.7)
            ax3.set_title('Test ROC-AUC Comparison')
            ax3.set_ylabel('ROC-AUC Score')
            ax3.set_ylim(0, 1.05)
            ax3.grid(False)
            for i, v in enumerate(test_roc_aucs):
                ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            plt.tight_layout()
            figures.append(fig3)
            plt.show()
        else:
            print("Note: ROC AUC plot skipped due to calculation issues")
        
        # Plot 4: CV Accuracy with Error Bars
        fig4, ax4 = plt.subplots(1, 1, figsize=individual_figsize)
        ax4.bar(scenarios, cv_accuracies, yerr=cv_accuracy_stds, 
                color=colors, alpha=0.7, capsize=5)
        ax4.set_title('Cross-Validation Accuracy (with std)')
        ax4.set_ylabel('CV Accuracy')
        ax4.set_ylim(0, 1)
        ax4.grid(False)
        for i, (v, std) in enumerate(zip(cv_accuracies, cv_accuracy_stds)):
            ax4.text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        figures.append(fig4)
        plt.show()
        
        return figures
    
    else:
        # Create subplots - adjust based on ROC AUC availability
        if roc_auc_available:
            combined_figsize = (15, 10) if figsize is None else figsize
            fig, axes = plt.subplots(2, 2, figsize=combined_figsize)
            plot_roc = True
        else:
            combined_figsize = (18, 6) if figsize is None else figsize
            fig, axes = plt.subplots(1, 3, figsize=combined_figsize)
            plot_roc = False
            print("Note: ROC AUC plot skipped due to calculation issues")
        
        if plot_roc:
            # Plot 1: Test Accuracy
            axes[0, 0].bar(scenarios, test_accuracies, color=colors, alpha=0.7)
            axes[0, 0].set_title('Test Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].grid(False)
            for i, v in enumerate(test_accuracies):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot 2: Test F1-Macro
            axes[0, 1].bar(scenarios, test_f1_scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('Test F1-Macro Comparison')
            axes[0, 1].set_ylabel('F1-Macro Score')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(False)
            for i, v in enumerate(test_f1_scores):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot 3: Test ROC-AUC
            axes[1, 0].bar(scenarios, test_roc_aucs, color=colors, alpha=0.7)
            axes[1, 0].set_title('Test ROC-AUC Comparison')
            axes[1, 0].set_ylabel('ROC-AUC Score')
            axes[1, 0].set_ylim(0, 1.05)
            axes[1, 0].grid(False)
            for i, v in enumerate(test_roc_aucs):
                axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot 4: CV Accuracy with Error Bars
            axes[1, 1].bar(scenarios, cv_accuracies, yerr=cv_accuracy_stds, 
                           color=colors, alpha=0.7, capsize=5)
            axes[1, 1].set_title('Cross-Validation Accuracy (with std)\nFixed: Train Synthetic→Validate Real vs Train Real→Validate Real')
            axes[1, 1].set_ylabel('CV Accuracy')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(False)
            for i, (v, std) in enumerate(zip(cv_accuracies, cv_accuracy_stds)):
                axes[1, 1].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        else:
            # Single row layout when ROC AUC is not available
            # Plot 1: Test Accuracy
            axes[0].bar(scenarios, test_accuracies, color=colors, alpha=0.7)
            axes[0].set_title('Test Accuracy Comparison')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_ylim(0, 1)
            axes[0].grid(False)
            for i, v in enumerate(test_accuracies):
                axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot 2: Test F1-Macro
            axes[1].bar(scenarios, test_f1_scores, color=colors, alpha=0.7)
            axes[1].set_title('Test F1-Macro Comparison')
            axes[1].set_ylabel('F1-Macro Score')
            axes[1].set_ylim(0, 1)
            axes[1].grid(False)
            for i, v in enumerate(test_f1_scores):
                axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Plot 3: CV Accuracy with Error Bars
            axes[2].bar(scenarios, cv_accuracies, yerr=cv_accuracy_stds, 
                        color=colors, alpha=0.7, capsize=5)
            axes[2].set_title('Cross-Validation Accuracy (with std)\nFixed: Train Synthetic→Validate Real vs Train Real→Validate Real')
            axes[2].set_ylabel('CV Accuracy')
            axes[2].set_ylim(0, 1)
            axes[2].grid(False)
            for i, (v, std) in enumerate(zip(cv_accuracies, cv_accuracy_stds)):
                axes[2].text(i, v + std + 0.01, f'{v:.3f}±{std:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def print_summary_table(results):
    """
    Print a summary table of all results.
    
    Args:
        results: Results dictionary from run_utility_evaluation
    """
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    print(f"{'Scenario':<30} {'CV Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Test AUC':<12}")
    print("-" * 85)
    
    scenarios = [
        ('Raw Synthetic → Real', results['raw_synthetic']),
        ('Refined Synthetic → Real', results['refined_synthetic']),
        ('Real Train → Real Test', results['real_baseline'])
    ]
    
    for name, result in scenarios:
        cv_acc = f"{result['cv_accuracy_mean']:.4f}±{result['cv_accuracy_std']:.3f}"
        test_acc = f"{result['test_accuracy']:.4f}"
        test_f1 = f"{result['test_f1_macro']:.4f}"
        
        if np.isnan(result['test_roc_auc']):
            test_auc = "N/A"
        else:
            test_auc = f"{result['test_roc_auc']:.4f}"
        
        print(f"{name:<30} {cv_acc:<12} {test_acc:<12} {test_f1:<12} {test_auc:<12}")
    
    # Calculate improvement (using accuracy as primary metric)
    raw_acc = results['raw_synthetic']['test_accuracy']
    refined_acc = results['refined_synthetic']['test_accuracy']
    baseline_acc = results['real_baseline']['test_accuracy']
    
    improvement = ((refined_acc - raw_acc) / raw_acc) * 100 if raw_acc > 0 else 0
    baseline_gap_raw = ((baseline_acc - raw_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
    baseline_gap_refined = ((baseline_acc - refined_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
    
    print("\n" + "=" * 85)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 85)
    print(f"Improvement from C-MAPS refinement: {improvement:+.2f}%")
    print(f"Gap to baseline (Raw Synthetic): {baseline_gap_raw:.2f}%")
    print(f"Gap to baseline (Refined Synthetic): {baseline_gap_refined:.2f}%")
    print(f"Refinement reduces gap by: {baseline_gap_raw - baseline_gap_refined:.2f} percentage points")
    
    # Additional analysis using F1 scores
    raw_f1 = results['raw_synthetic']['test_f1_macro']
    refined_f1 = results['refined_synthetic']['test_f1_macro']
    baseline_f1 = results['real_baseline']['test_f1_macro']
    
    f1_improvement = ((refined_f1 - raw_f1) / raw_f1) * 100 if raw_f1 > 0 else 0
    print(f"\nF1-Macro improvement from refinement: {f1_improvement:+.2f}%")
    
    # CV comparison analysis
    raw_cv = results['raw_synthetic']['cv_accuracy_mean']
    refined_cv = results['refined_synthetic']['cv_accuracy_mean']
    baseline_cv = results['real_baseline']['cv_accuracy_mean']
    
    cv_improvement = ((refined_cv - raw_cv) / raw_cv) * 100 if raw_cv > 0 else 0
    print(f"\nCV accuracy improvement from refinement: {cv_improvement:+.2f}%")
    print("Note: CV for synthetic scenarios uses 'Train on Synthetic, Validate on Real'")

def debug_data_types(real_data, raw_synthetic_data, refined_synthetic_data, target_column='cancer_types'):
    """
    Debug function to check data types and identify potential issues.
    """
    print("=" * 60)
    print("DATA TYPE DEBUGGING")
    print("=" * 60)
    
    datasets = {
        'Real': real_data,
        'Raw Synthetic': raw_synthetic_data,
        'Refined Synthetic': refined_synthetic_data
    }
    
    for name, data in datasets.items():
        print(f"\n{name} Dataset:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Target column '{target_column}' present: {target_column in data.columns}")
        
        if target_column in data.columns:
            print(f"  Target unique values: {data[target_column].nunique()}")
            print(f"  Target sample values: {data[target_column].unique()[:5]}")
        
        # Check for categorical columns
        categorical_cols = []
        for col in data.columns:
            if col != target_column and (data[col].dtype == 'object' or data[col].dtype.name == 'category'):
                categorical_cols.append(col)
                print(f"  Categorical column '{col}': {data[col].unique()[:3]}...")
        
        print(f"  Total categorical features: {len(categorical_cols)}")
        print(f"  Data types: {data.dtypes.value_counts().to_dict()}")