# 🔍 MAPS: Model-Agnostic Post-hoc Synthetic Data Refinement

**Balancing Fidelity and Privacy while Preserving the Distribution Properties of Synthetic Data**

Reference implementation for *Generating Better Synthetic Healthcare Data with MAPS:
Model-Agnostic Post-hoc Synthetic Data Refinement Framework*.

## Overview

MAPS is a model-agnostic framework for evaluating and refining synthetic data through a
two-stage approach:

1. **Stage 1 — Identifiability filtering.** Removes every synthetic record that lies closer to
   a real record than that real record's nearest real neighbour, enforcing a
   0-identifiability constraint at the level of individual samples.
2. **Stage 2 — Fidelity enhancement.** Trains a calibrated real-vs-synthetic classifier,
   converts its output into a density ratio (likelihood-free importance weighting), and
   resamples the filtered pool by Sampling-Importance-Resampling.

This comprehensive approach bridges the gap between sample-level quality and
distribution-level fidelity, resulting in synthetic datasets that are both high-quality and
privacy-preserving.

## Key Features

- **Model-agnostic**: works with any synthetic data generation method
- **Post-hoc refinement**: apply after generation, no need to modify existing pipelines
- **Dual-objective filtering**: balance data utility and privacy protection
- **Flexible configuration**: adapt parameters for various use cases (clinical decision
  support, educational applications, rare disease research)
- **Distribution-preserving capabilities**: maintain statistical properties while filtering
  individual samples

## Installation

```bash
pip install -r requirements.txt
```

`synthcity` is only needed to regenerate the synthetic pools; the refinement library itself
depends on numpy, pandas, scikit-learn, scipy and torch.

## Getting started

`notebooks/tutorial.ipynb` walks through the framework end to end on a single dataset:
generate a pool, apply both stages, and evaluate fidelity, utility and privacy.
`examples/basic_usage.py` is the same path as a script.

```python
from maps import FidelityClassifier, IdentifiabilityAnalyzer, SamplingEngine
```

## Repository layout

```
maps/           the library
  core/           the two stages: privacy.py (Stage 1), fidelity.py + sampling.py (Stage 2),
                  plus conformal.py, utility.py, preprocessing.py
  baselines/      competing post-processing methods used for comparison
  models/         autoencoder and classifier backbones
  utils/          correlation, distributional-similarity, privacy and statistics helpers
scripts/        experiment pipeline, numbered in execution order
configs/        one YAML per dataset, plus the filesystem layout
notebooks/      tutorial
examples/       minimal usage script
```

## Running the experiments

**Before anything else, set `data_root` in `configs/paths.yaml` to a directory of your own**
(or export `MAPS_DATA_ROOT`). Pools run to tens of GB, so scratch or project storage is a
better home for them than a home directory.

```bash
python scripts/00_make_splits.py       --dataset tcga
python scripts/01_generate_pool.py     --dataset tcga --model tvae
python scripts/08_select_alpha.py      --dataset tcga --model tvae
python scripts/02_run_maps.py          --dataset tcga --model tvae --variant full
python scripts/03_evaluate.py          --dataset tcga --model tvae --variant full
python scripts/03b_evaluate_privacy.py --dataset tcga --model tvae --variant full
python scripts/04_aggregate.py         --dataset tcga
```

`--model` is one of `tvae`, `ctgan`, `ddpm`, `dgd`, and `--dataset` is `tcga` or `gossis`.

**Ablations.** `scripts/02_run_maps.py --variant` selects the arm — `full`, `stage1_only`,
`stage2_only`, `random` (uniform selection from the same pool), `reverse` (the importance
weights inverted) and `top_k` (deterministic instead of resampled). `--pool-multiplier`
varies the candidate pool size.

**Comparison methods.** `scripts/02b_run_baselines.py --baseline {alaa,wang,qde}` runs the
competing post-processing methods on the same candidate pool, so the comparison isolates the
selection rule.

**Other generators.** `scripts/01b_generate_pool_synthpop.py` uses the R package **synthpop**
and needs R with `synthpop` installed. `scripts/01c_generate_pool_dgd.py` needs the DGD
packages, which are not distributed here — set `DGD_SRC` and `MAPS_DGD_DATA`, see the script
header.

## License

MIT — see `LICENSE`.
