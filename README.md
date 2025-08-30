# MAPS: Model Agnostic Post-hoc Synthetic Data Refinement

MAPS is a novel framework for improving the quality and privacy of synthetic data through post-hoc refinement. It combines fidelity enhancement through importance weighting with privacy preservation through identifiability constraints.

## Overview

MAPS addresses critical fidelity-privacy tradeoff in synthetic data:

1. **Fidelity**: Synthetic data often exhibits distributional bias compared to real data
2. **Privacy**: Synthetic samples may be too similar to real samples, causing privacy risks

The framework operates in two main stages:

1. **Privacy Analysis**: Compute identifiability flags based on sample distinctness
2. **Fidelity Classification**: Train a classifier to distinguish real from synthetic data and estimate importance weights

## Quick Start

```
The tutorial.ipynb provide an concrete example of how MAPS works.
```

## 📁 Repository Structure

```
maps/
├── core/                    # Core functionality
│   ├── fidelity.py         # FidelityClassifier
│   ├── privacy.py          # IdentifiabilityAnalyzer
│   ├── sampling.py         # SamplingEngine
│   ├── preprocessing.py    # Data preprocessing
│   └── utility.py          # Utility evaluation
├── models/                  # Model implementations
│   ├── autoencoder.py      # Tabular autoencoder
│   └── classifiers.py      # Classifier training
├── utils/                  # Evaluation utilities
│   ├── CorrAnalyzer.py     # Correlation analysis
│   ├── DistSimTest.py      # Distribution similarity
│   └── ResultsAnalyzer.py  # Results analysis
└── tutorial.ipynb         # Complete tutorial
```

## 📄 License

This project is licensed under the MIT License.
