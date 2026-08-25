"""Utility-driven filtering in the spirit of QDE (Sachdeva et al., 2025).

QDE keeps synthetic records according to their contribution to a specific downstream task
rather than to distributional similarity. We reproduce that idea in its simplest defensible
form: fit the downstream classifier on the real training partition, score every synthetic
record by the probability the classifier assigns to that record's own label, and keep the
highest-scoring records.

Like the original, this optimises a named downstream task and carries no privacy or
calibration mechanism.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ._encoding import encode_pair


def qde_utility_filter(
    real: pd.DataFrame,
    pool: pd.DataFrame,
    n_target: int,
    target: str,
    n_estimators: int = 100,
    random_state: int = 0,
) -> pd.DataFrame:
    """Keep the ``n_target`` synthetic records most consistent with the downstream task."""
    feats = [c for c in real.columns if c != target]
    R, S, _ = encode_pair(real[feats], pool[feats])

    classes = sorted(set(real[target].astype(str)) | set(pool[target].astype(str)))
    code = {c: i for i, c in enumerate(classes)}
    y_real = real[target].astype(str).map(code).to_numpy()
    y_pool = pool[target].astype(str).map(code).to_numpy()

    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    ).fit(R, y_real)

    proba = clf.predict_proba(S)
    col = {c: i for i, c in enumerate(clf.classes_)}
    scores = np.array([
        proba[i, col[y]] if y in col else 0.0 for i, y in enumerate(y_pool)
    ])

    keep = np.argsort(scores)[-n_target:]
    return pool.iloc[keep].reset_index(drop=True)
