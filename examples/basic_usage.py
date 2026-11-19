#!/usr/bin/env python
"""Minimal MAPS example: both stages on a small built-in dataset.

The steps and argument values follow ``scripts/02_run_maps.py``:

  1. Stage 1 removes every candidate that violates 0-identifiability with respect to the
     real data.
  2. Stage 2 trains a real-vs-synthetic MLP classifier on N real records and N Stage 1
     survivors, calibrates it by sigmoid scaling, and turns its output into importance
     weights for the remaining survivors.
  3. The weights are tempered with an exponent alpha and N records are drawn with
     replacement (Sampling-Importance-Resampling).

The real data is scikit-learn's breast cancer dataset, and the candidate pool comes from
``noisy_resampling`` below, which is not a generative model: it is only there to make the
example self-contained. With a real generator, fit it on the same real data and pass its
output as the pool.

Run from the repository root:

    python examples/basic_usage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maps import FidelityClassifier, IdentifiabilityAnalyzer, SamplingEngine  # noqa: E402
from maps.utils.DistSimTest import compute_jensen_shannon_distance_marginal  # noqa: E402

POOL_MULTIPLIER = 30    # M = 30 N candidates, as in configs/*.yaml
ALPHA = 1.0             # tempering exponent; the paper uses 1.4 (TCGA-metadata) and
                        # 0.8 (GOSSIS-1-eICU-cardiovascular), see configs/*.yaml


def noisy_resampling(real: pd.DataFrame, n: int, target: str, seed: int = 0) -> pd.DataFrame:
    """Stand-in for a generator: real records resampled with Gaussian noise of random scale.

    Records that receive little noise stay close to a real record, which gives Stage 1
    something to remove; the uneven noise distorts the distribution, which gives Stage 2
    something to correct.
    """
    rng = np.random.default_rng(seed)
    pool = real.iloc[rng.integers(0, len(real), size=n)].reset_index(drop=True)
    features = [c for c in real.columns if c != target]
    scale = rng.uniform(0.0, 1.0, size=(n, 1)) * real[features].std().values
    pool[features] = pool[features] + rng.normal(size=(n, len(features))) * scale
    return pool


def identifiability_score(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """Fraction of synthetic records lying closer to a real record than that record's
    nearest real neighbour."""
    flags = IdentifiabilityAnalyzer(verbose=False).fit(real, synthetic)
    return float(np.mean(flags))


def main() -> None:
    real = load_breast_cancer(as_frame=True).frame
    n_target = len(real)
    pool = noisy_resampling(real, POOL_MULTIPLIER * n_target, target="target")

    # Raw comparator: N records drawn uniformly from the pool.
    rng = np.random.default_rng(42)
    raw = pool.iloc[rng.choice(len(pool), size=n_target, replace=False)].reset_index(drop=True)

    # ---- Stage 1: identifiability filtering -----------------------------------
    flags = IdentifiabilityAnalyzer(verbose=False).fit(real, pool)
    filtered = pool.iloc[np.where(flags == 0)[0]].reset_index(drop=True)
    if len(filtered) < 2 * n_target:
        raise RuntimeError("Stage 1 left fewer than 2N records; use a larger pool.")

    # ---- Stage 2: density-ratio weights ---------------------------------------
    # Synthetic records used to train the classifier are held out from resampling,
    # so the classifier never scores its own training data.
    syn_train = filtered.sample(n=n_target, random_state=42)
    syn_rest = filtered.drop(syn_train.index).reset_index(drop=True)
    clf = FidelityClassifier(classifier_type="mlp", calibration_method="sigmoid", verbose=False)
    acc, auc = clf.fit(
        real, syn_train,
        mlp_hidden_layer_sizes=(240, 120),
        classifier_test_size=0.2,
        use_embedding=False,
        show_calibration_plot=False,
    )
    weights, _ = clf.estimate_importance_weights(syn_rest, use_embedding=False)

    # ---- Resampling ------------------------------------------------------------
    engine = SamplingEngine(random_seed=42, verbose=False)
    refined, _, _, _, stats = engine.sample(
        synthetic_data=syn_rest,
        importance_weights=weights,
        n_samples=n_target,
        method="weighted",
        weight_processing="flatten",
        alpha=ALPHA,
        replacement=True,
    )

    # ---- Summary ---------------------------------------------------------------
    def jsd(synthetic):
        return compute_jensen_shannon_distance_marginal(real, synthetic, discretize_numerical=True)

    print(f"Real records (N):          {n_target}")
    print(f"Candidate pool:            {len(pool)} ({POOL_MULTIPLIER}N)")
    print(f"Stage 1 survivors:         {len(filtered)} ({len(filtered) / len(pool):.1%})")
    print(f"Classifier accuracy / AUC: {acc:.3f} / {auc:.3f}")
    print(f"Refined records:           {stats['selected_samples']} "
          f"({stats['n_unique_samples']} unique)")
    print(f"Identifiability score:     raw {identifiability_score(real, raw):.4f}"
          f"  ->  refined {identifiability_score(real, refined):.4f}")
    print(f"Marginal JS distance:      raw {jsd(raw):.4f}  ->  refined {jsd(refined):.4f}")


if __name__ == "__main__":
    main()
