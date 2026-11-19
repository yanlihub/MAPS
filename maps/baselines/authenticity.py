"""Sample-level filtering in the spirit of Alaa et al. (2022).

Alaa et al. propose alpha-Precision, beta-Recall and Authenticity as sample-level audit
metrics for generative models. They are diagnostics rather than a refinement method, so we
use them the way a practitioner naturally would: discard synthetic records that fall outside
the alpha-support of the real data or that fail the authenticity check, then draw the target
number of records uniformly from what survives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ._encoding import encode_pair


def alaa_authenticity_filter(
    real: pd.DataFrame,
    pool: pd.DataFrame,
    n_target: int,
    alpha: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Keep records inside the alpha-support of ``real`` that also pass authenticity.

    Args:
        alpha: fraction of real records defining the support region.

    Returns:
        ``n_target`` records drawn uniformly from the surviving pool.
    """
    R, S, _ = encode_pair(real, pool)

    # alpha-Precision: distance to the real centre must be within the alpha-quantile
    # of the real distances.
    centre = R.mean(axis=0)
    r_rad = np.linalg.norm(R - centre, axis=1)
    radius = np.quantile(r_rad, alpha)
    in_support = np.linalg.norm(S - centre, axis=1) <= radius

    # Authenticity: a synthetic record must not sit closer to a real record than that
    # real record's own nearest real neighbour.
    nn_rr = NearestNeighbors(n_neighbors=2).fit(R)
    d_rr = nn_rr.kneighbors(R)[0][:, 1]
    nn_sr = NearestNeighbors(n_neighbors=1).fit(R)
    d_sr, idx_sr = nn_sr.kneighbors(S)
    authentic = d_sr[:, 0] >= d_rr[idx_sr[:, 0]]

    keep = np.where(in_support & authentic)[0]
    if len(keep) < n_target:
        raise RuntimeError(
            f"Alaa filter left {len(keep)} records, fewer than the {n_target} required"
        )
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(keep, size=n_target, replace=False)
    return pool.iloc[chosen].reset_index(drop=True)
