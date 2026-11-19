"""Post-processing by moment matching, after Wang et al. (2023).

Wang et al. reweight an already-generated synthetic dataset by convex optimisation so that a
chosen set of summary statistics matches the real data. The method requires the practitioner
to name the statistics in advance -- the limitation MAPS is designed to avoid -- so we use
the natural default: first moments of the numeric variables and the category frequencies of
the discrete variables.

We solve

    min_w  || Phi^T w - mu_real ||^2      s.t.  w >= 0,  sum(w) = 1

over the pool, then draw the target number of records with probability proportional to w.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ._encoding import encode_pair


def _feature_map(real: pd.DataFrame, pool: pd.DataFrame):
    """Statistics to be matched: numeric values plus one-hot category indicators."""
    cols = list(real.columns)
    numeric = real.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = [c for c in cols if c not in numeric]

    R_num, S_num, _ = encode_pair(real[numeric], pool[numeric]) if numeric else (None, None, None)

    blocks_r, blocks_s = [], []
    if numeric:
        blocks_r.append(R_num)
        blocks_s.append(S_num)
    for c in categorical:
        cats = sorted(set(real[c].astype(str)) | set(pool[c].astype(str)))
        blocks_r.append(pd.get_dummies(real[c].astype(str)).reindex(columns=cats, fill_value=0).to_numpy(float))
        blocks_s.append(pd.get_dummies(pool[c].astype(str)).reindex(columns=cats, fill_value=0).to_numpy(float))
    return np.hstack(blocks_r), np.hstack(blocks_s)


def wang_moment_matching(
    real: pd.DataFrame,
    pool: pd.DataFrame,
    n_target: int,
    random_state: int = 0,
    ridge: float = 1e-6,
    max_pool: Optional[int] = 50000,
) -> pd.DataFrame:
    """Reweight ``pool`` so its summary statistics match ``real``, then resample.

    Args:
        max_pool: cap on the number of candidates handed to the solver. The convex problem
            is dense in the pool size; subsampling keeps it tractable for 30N pools and does
            not favour the method under comparison.
    """
    rng = np.random.default_rng(random_state)
    if max_pool is not None and len(pool) > max_pool:
        sub = rng.choice(len(pool), size=max_pool, replace=False)
        pool_used = pool.iloc[sub].reset_index(drop=True)
    else:
        pool_used = pool.reset_index(drop=True)

    Phi_r, Phi_s = _feature_map(real, pool_used)
    mu = Phi_r.mean(axis=0)

    # Non-negative least squares on Phi_s^T w = mu, with a row pinning sum(w) = 1.
    A = np.vstack([Phi_s.T, np.ones((1, len(pool_used)))])
    b = np.concatenate([mu, [1.0]])
    A = A + ridge * rng.normal(scale=1e-8, size=A.shape)
    w, _ = nnls(A, b)

    if w.sum() <= 0:
        raise RuntimeError("moment matching produced an all-zero weight vector")
    p = w / w.sum()

    idx = rng.choice(len(pool_used), size=n_target, replace=True, p=p)
    return pool_used.iloc[idx].reset_index(drop=True)
