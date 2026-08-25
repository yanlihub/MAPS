"""Shared numeric encoding for the baselines."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def encode_pair(
    real: pd.DataFrame, synthetic: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Min-max numeric + integer categorical encoding, fitted on ``real`` only."""
    cols = list(real.columns)
    syn = synthetic[cols].copy()
    rl = real.copy()

    numeric = rl.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = [c for c in cols if c not in numeric]

    for c in categorical:
        enc = LabelEncoder().fit(pd.concat([rl[c], syn[c]]).astype(str))
        rl[c] = enc.transform(rl[c].astype(str))
        syn[c] = enc.transform(syn[c].astype(str))

    scaler = MinMaxScaler().fit(rl[cols])
    return scaler.transform(rl[cols]), scaler.transform(syn[cols]), cols
