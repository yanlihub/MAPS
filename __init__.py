"""
C-MAPS: Classifier-based Model Agnostic Post-hoc Synthetic Data Refinement

A framework for improving synthetic data quality through fidelity enhancement
and privacy-preserving sampling.
"""

__version__ = "0.1.0"
__author__ = "Yan Li"
__email__ = "yanli@di.ku.dk"

from .core.fidelity import FidelityClassifier
from .core.privacy import IdentifiabilityAnalyzer
from .core.sampling import SamplingEngine
from .core.utility import run_utility_evaluation
from .utils.visualization import visualize_data_distribution
from .utils.metrics import compute_mmd, compute_correlation_metrics

__all__ = [
    "FidelityClassifier",
    "IdentifiabilityAnalyzer", 
    "SamplingEngine",
    "run_utility_evaluation",
    "visualize_data_distribution",
    "compute_mmd",
    "compute_correlation_metrics"
]