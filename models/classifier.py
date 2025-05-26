"""
Classifier utilities for fidelity estimation in C-MAPS framework.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')


class FidelityClassifierTrainer:
    """
    Trainer for fidelity classifiers used in density ratio estimation.
    """
    
    def __init__(self, 
                 classifier_type: str = 'mlp',
                 calibration_method: str = 'isotonic',
                 random_seed: int = 42,
                 verbose: bool = True):
        """
        Initialize the classifier trainer.
        
        Args:
            classifier_type: Type of classifier ('mlp', 'lr', 'lgbm')
            calibration_method: Calibration method ('isotonic', 'sigmoid')
            random_seed: Random seed for reproducibility
            verbose: Whether to print training progress
        """
        self.classifier_type = classifier_type
        self.calibration_method = calibration_method
        self.random_seed = random_seed
        self.verbose = verbose
        
        self.classifier = None
        self.scaler = None
        self.kappa = None
        self.is_fitted = False
        
    def train(self, 
              real_data: Union[pd.DataFrame, np.ndarray],
              synthetic_data: Union[pd.DataFrame, np.ndarray],
              test_size: float = 0.2,
              mlp_hidden_layer_sizes: Tuple[int, ...] = (100, 50),
              show_calibration_plot: bool = True) -> Tuple[float, float]:
        """
        Train the fidelity classifier.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            test_size: Proportion of data to use for testing
            mlp_hidden_layer_sizes: Hidden layer sizes for MLP
            show_calibration_plot: Whether to show calibration plot
            
        Returns:
            Tuple of (accuracy, roc_auc) on test set
        """
        if self.verbose:
            print(f"Training {self.classifier_type} classifier with {self.calibration_method} calibration")
        
        # Convert to numpy arrays
        X_real = real_data.values if isinstance(real_data, pd.DataFrame) else real_data
        X_synthetic = synthetic_data.values if isinstance(synthetic_data, pd.DataFrame) else synthetic_data
        
        # Ensure 2D arrays
        if X_real.ndim == 1:
            X_real = X_real.reshape(-1, 1)
        if X_synthetic.ndim == 1:
            X_synthetic = X_synthetic.reshape(-1, 1)
        
        # Combine data and create labels (1 for real, 0 for synthetic)
        X = np.vstack((X_real, X_synthetic))
        y = np.concatenate((np.ones(len(X_real)), np.zeros(len(X_synthetic))))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_seed
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create base classifier
        if self.classifier_type == 'mlp':
            base_classifier = MLPClassifier(
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=self.random_seed
            )
        elif self.classifier_type == 'lr':
            base_classifier = LogisticRegression(
                solver='liblinear',
                random_state=self.random_seed,
                max_iter=1000,
                C=1.0
            )
        elif self.classifier_type == 'lgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM not available. Install with: pip install lightgbm")
            base_classifier = lgb.LGBMClassifier(
                random_state=self.random_seed,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                verbose=-1
            )
        else:
            raise ValueError(f"Unsupported classifier_type: {self.classifier_type}")
        
        # Create calibrated classifier
        self.classifier = CalibratedClassifierCV(
            base_classifier,
            method=self.calibration_method,
            cv=5,
            ensemble=True
        )
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_scaled)
        y_prob = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        if self.verbose:
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test ROC AUC: {roc_auc:.4f}")
        
        # Calculate kappa (odds ratio)
        self.kappa = len(X_synthetic) / len(X_real)
        if self.verbose:
            print(f"Odds ratio (kappa): {self.kappa:.4f}")
        
        # Show calibration plot
        if show_calibration_plot and self.verbose:
            self._plot_calibration(X_test_scaled, y_test)
        
        self.is_fitted = True
        return accuracy, roc_auc
    
    def estimate_importance_weights(self, 
                                   data: Union[pd.DataFrame, np.ndarray],
                                   epsilon: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate importance weights for given data.
        
        Args:
            data: Data to estimate weights for
            epsilon: Small constant for numerical stability
            
        Returns:
            Tuple of (importance_weights, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be trained before estimating weights")
        
        # Convert to numpy array
        X = data.values if isinstance(data, pd.DataFrame) else data
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        probs = self.classifier.predict_proba(X_scaled)[:, 1]  # Prob of being real
        
        # Clip probabilities to avoid numerical issues
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        # Calculate importance weights: w(x) = κ * c_φ(x) / (1 - c_φ(x))
        weights = self.kappa * probs / (1 - probs)
        
        return weights, probs
    
    def _plot_calibration(self, X_test: np.ndarray, y_test: np.ndarray):
        """Plot calibration diagram."""
        plt.figure(figsize=(10, 7))
        
        disp = CalibrationDisplay.from_estimator(
            self.classifier,
            X_test,
            y_test,
            n_bins=10,
            name=f"{self.classifier_type}_{self.calibration_method}",
            strategy='uniform'
        )
        
        plt.title(f"Reliability Diagram for {self.classifier_type} with {self.calibration_method} calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_classifier(self):
        """Get the trained classifier."""
        if not self.is_fitted:
            raise ValueError("Classifier must be trained first")
        return self.classifier
    
    def get_scaler(self):
        """Get the fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Classifier must be trained first")
        return self.scaler
    
    def get_kappa(self):
        """Get the calculated kappa (odds ratio)."""
        if not self.is_fitted:
            raise ValueError("Classifier must be trained first")
        return self.kappa