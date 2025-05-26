# C-MAPS: Classifier-based Model Agnostic Post-hoc Synthetic Data Refinement

C-MAPS is a novel framework for improving the quality and privacy of synthetic data through post-hoc refinement. It combines fidelity enhancement through importance weighting with privacy preservation through identifiability constraints.

## 🚀 Key Features

- **Fidelity Enhancement**: Uses autoencoder embeddings and calibrated classifiers to estimate density ratios and importance weights
- **Privacy Preservation**: Implements identifiability analysis to ensure privacy constraints are met
- **Flexible Sampling**: Supports both standard SIR and privacy-constrained SIR-IC sampling
- **Comprehensive Evaluation**: Built-in metrics for fidelity, privacy, and utility assessment
- **Easy to Use**: Clean, modular API that works with any synthetic data generation method

## 📖 Overview

C-MAPS addresses two critical challenges in synthetic data:

1. **Fidelity**: Synthetic data often exhibits distributional bias compared to real data
2. **Privacy**: Synthetic samples may be too similar to real samples, causing privacy risks

The framework operates in three main stages:

1. **Fidelity Classification**: Train a classifier to distinguish real from synthetic data and estimate importance weights
2. **Privacy Analysis**: Compute identifiability flags based on sample distinctness
3. **Constrained Sampling**: Select high-fidelity samples while respecting privacy constraints

## 🛠️ Installation

### From Source

```bash
git clone https://github.com/yourusername/c-maps.git
cd c-maps
pip install -e .
```

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: LightGBM support
pip install lightgbm

# Development dependencies  
pip install -e ".[dev]"
```

### Check Installation

```python
from c_maps.utils import check_dependencies
check_dependencies()
```

## 🎯 Quick Start

```python
import pandas as pd
from c_maps import FidelityClassifier, IdentifiabilityAnalyzer, SamplingEngine

# Load your data
real_data = pd.read_csv('real_data.csv')
synthetic_data = pd.read_csv('synthetic_data.csv')

# 1. Train fidelity classifier
fidelity_classifier = FidelityClassifier(
    embedding_dim=10,
    classifier_type='mlp',
    random_seed=42
)

accuracy, roc_auc = fidelity_classifier.fit(real_data, synthetic_data)
importance_weights, _ = fidelity_classifier.estimate_importance_weights()

# 2. Compute identifiability flags  
identifiability_analyzer = IdentifiabilityAnalyzer()
identifiability_flags = identifiability_analyzer.fit(real_data, synthetic_data)

# 3. Privacy-preserving sampling
sampling_engine = SamplingEngine(random_seed=42)
refined_data, refined_weights, refined_flags, selected_indices = sampling_engine.sample(
    synthetic_data=synthetic_data,
    importance_weights=importance_weights,
    n_samples=len(real_data),
    use_identifiability_constraint=True,
    identifiability_flags=identifiability_flags,
    epsilon_identifiability=0.05  # Max 5% identifiable samples
)

print(f"Refined {len(refined_data)} samples with {np.mean(refined_flags):.2%} identifiable")
```

## 📁 Repository Structure

```
c_maps/
├── __init__.py              # Main package exports
├── core/                    # Core functionality
│   ├── fidelity.py         # FidelityClassifier
│   ├── privacy.py          # IdentifiabilityAnalyzer  
│   ├── sampling.py         # SamplingEngine
│   └── preprocessing.py    # Data preprocessing
├── models/                  # Model implementations
│   ├── autoencoder.py      # Tabular autoencoder
│   └── classifiers.py     # Classifier training
├── utils/                  # Utilities
│   ├── visualization.py   # Plotting functions
│   ├── metrics.py         # Evaluation metrics
│   └── helpers.py         # Helper functions
└── examples/               # Usage examples
    └── basic_usage.py     # Basic example
```

## 🔧 Advanced Usage

### Custom Autoencoder Configuration

```python
fidelity_classifier = FidelityClassifier(
    embedding_dim=16,
    classifier_type='mlp',
    random_seed=42
)

accuracy, roc_auc = fidelity_classifier.fit(
    real_data=real_data,
    synthetic_data=synthetic_data,
    # Autoencoder parameters
    autoencoder_epochs=200,
    autoencoder_batch_size=512,
    autoencoder_lr=0.001,
    autoencoder_hidden_dims=[256, 128, 64],
    use_synthetic_for_autoencoder=False,
    # Classifier parameters  
    classifier_test_size=0.2,
    mlp_hidden_layer_sizes=(100, 50),
    show_calibration_plot=True
)
```

### Different Classifier Types

```python
# Logistic Regression
fidelity_classifier = FidelityClassifier(classifier_type='lr')

# LightGBM (requires: pip install lightgbm)
fidelity_classifier = FidelityClassifier(classifier_type='lgbm')

# MLP with different calibration
fidelity_classifier = FidelityClassifier(
    classifier_type='mlp',
    calibration_method='sigmoid'  # or 'isotonic'
)
```

### Sampling Options

```python
# Standard SIR (fidelity only)
refined_data, weights, _, indices = sampling_engine.sample(
    synthetic_data=synthetic_data,
    importance_weights=importance_weights,
    n_samples=1000,
    use_identifiability_constraint=False,
    method='weighted'  # or 'top_k'
)

# SIR-IC (fidelity + privacy)
refined_data, weights, flags, indices = sampling_engine.sample(
    synthetic_data=synthetic_data,
    importance_weights=importance_weights,
    n_samples=1000,
    use_identifiability_constraint=True,
    identifiability_flags=identifiability_flags,
    epsilon_identifiability=0.1  # Allow 10% identifiable
)
```

## 📊 Evaluation and Visualization

### Comprehensive Evaluation

```python
from c_maps.utils import comprehensive_evaluation, print_evaluation_summary

evaluation = comprehensive_evaluation(
    real_data=real_data,
    synthetic_data=synthetic_data,
    identifiability_flags=identifiability_flags,
    importance_weights=importance_weights,
    selected_indices=selected_indices,
    epsilon_target=0.05
)

print_evaluation_summary(evaluation)
```

### Visualization

```python
from c_maps.utils import visualize_data_distribution

# Compare data distributions
real_embeddings, synthetic_embeddings = fidelity_classifier.get_embeddings()
selected_embeddings = synthetic_embeddings[selected_indices]

visualize_data_distribution(
    data_list=[real_embeddings, synthetic_embeddings, selected_embeddings],
    names=['Real', 'Original Synthetic', 'Refined'],
    colors=['blue', 'red', 'green'],
    method='pca'  # or 'tsne'
)

# Visualize importance weights
fidelity_classifier.visualize_importance_weights(importance_weights)

# Analyze identifiability
identifiability_analyzer.visualize_identifiability_flags(identifiability_flags)
```

## 🔬 Key Parameters

### FidelityClassifier

- `embedding_dim`: Autoencoder embedding dimension (default: 10)
- `classifier_type`: 'mlp', 'lr', or 'lgbm' (default: 'mlp')
- `calibration_method`: 'isotonic' or 'sigmoid' (default: 'isotonic')
- `autoencoder_epochs`: Training epochs for autoencoder (default: 400)
- `use_synthetic_for_autoencoder`: Include synthetic data in autoencoder training (default: False)

### IdentifiabilityAnalyzer

- Automatically computes feature weights based on entropy
- Uses weighted Euclidean distance for identifiability assessment
- Computes distinctness thresholds for each real sample

### SamplingEngine

- `use_identifiability_constraint`: Enable privacy constraints (default: False)
- `epsilon_identifiability`: Max proportion of identifiable samples (default: 0.05)
- `method`: Sampling method - 'weighted' or 'top_k' (default: 'weighted')

## 📈 Performance Tips

### Memory Usage

```python
from c_maps.utils import estimate_memory_usage, print_memory_estimate

estimates = estimate_memory_usage(real_data, synthetic_data, embedding_dim=10)
print_memory_estimate(estimates)
```

### Large Datasets

- Use smaller `embedding_dim` for large datasets
- Reduce `autoencoder_batch_size` if memory is limited
- Consider sampling synthetic data for classifier training
- Use `method='top_k'` for deterministic sampling

## 🧪 Examples

See `examples/basic_usage.py` for a complete working example with dummy data.

```bash
cd examples
python basic_usage.py
```

## 📚 Citation

If you use C-MAPS in your research, please cite:

```bibtex
@article{cmaps2024,
  title={C-MAPS: Classifier-based Model Agnostic Post-hoc Synthetic Data Refinement},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work is based on the paper "Bias Correction of Learned Generative Models using Likelihood-Free Importance Weighting" and extends it with privacy-preserving constraints.

## 📞 Support

If you have questions or need help:

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/c-maps/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/c-maps/wiki)