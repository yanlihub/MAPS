"""
Split conformal prediction for evaluating uncertainty quantification of synthetic data.

Protocol: train on synthetic, calibrate on synthetic, test on real. This asks whether
synthetic data can supply reliable uncertainty calibration for real-world deployment,
rather than assuming an expensive real calibration set is available.

The feature scaler is fitted on ``d_train`` only, and the real test set is supplied
explicitly as ``d_test``, so no part of the evaluation set reaches the calibration step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class ConformalClassifier:
    """Split conformal prediction with the inverse-probability non-conformity score.

    Given miscoverage level ``alpha``, the prediction set is guaranteed to contain the true
    label with probability at least ``1 - alpha`` under exchangeability of the calibration
    and test data.
    """

    def __init__(self, model, score_function: str = "inverse_probability", alpha: float = 0.05):
        if score_function != "inverse_probability":
            raise ValueError(f"Unknown score function: {score_function}")
        self.model = model
        self.score_function = score_function
        self.alpha = alpha
        self.quantile: Optional[float] = None
        self.n_classes: Optional[int] = None

    def compute_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """s(x, y) = 1 - P_hat(Y = y | X = x).

        A calibration record may carry a label the classifier never saw, which happens
        whenever the synthetic training split misses a class the calibration split contains.
        Degenerate synthetic data makes this common -- the reverse-weighted control is
        designed to produce exactly that. For such a label the model assigns probability 0,
        so the score is 1, the maximum. Treating it that way is the honest reading: the
        classifier cannot cover a class it has never seen, and the conformal threshold should
        widen accordingly rather than the run failing.
        """
        proba = self.model.predict_proba(X)
        self.n_classes = proba.shape[1]
        class_index = {c: i for i, c in enumerate(self.model.classes_)}
        scores = np.ones(len(y), dtype=float)
        known = np.array([v in class_index for v in y])
        if known.any():
            cols = np.array([class_index[v] for v in np.asarray(y)[known]])
            scores[known] = 1.0 - proba[np.where(known)[0], cols]
        self.n_unseen_labels = int((~known).sum())
        return scores

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """Set the conformal threshold from the calibration scores."""
        scores = self.compute_scores(X_cal, y_cal)
        n = len(scores)
        if n == 0:
            raise ValueError("empty calibration set")
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        # With a small calibration set q_level can exceed 1, in which case no finite
        # threshold attains the target coverage and the set degenerates to all classes.
        self.quantile = float(np.quantile(scores, min(q_level, 1.0)))
        return self.quantile

    def predict_sets(self, X: np.ndarray) -> List[List]:
        if self.quantile is None:
            raise RuntimeError("calibrate() must be called before predict_sets()")
        proba = self.model.predict_proba(X)
        labels = self.model.classes_
        keep = (1.0 - proba) <= self.quantile
        return [list(labels[row]) for row in keep]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        sets = self.predict_sets(X)
        sizes = np.array([len(s) for s in sets])
        covered = np.array([y[i] in s for i, s in enumerate(sets)])
        return {
            "n_calibration_labels_unseen_by_model": getattr(self, "n_unseen_labels", 0),
            "coverage": float(covered.mean()),
            "avg_set_size": float(sizes.mean()),
            "median_set_size": float(np.median(sizes)),
            "frac_empty_sets": float((sizes == 0).mean()),
            "frac_full_sets": float((sizes == len(self.model.classes_)).mean()),
            "n_classes": int(len(self.model.classes_)),
            "quantile": self.quantile,
        }


@dataclass
class _Encoded:
    X: np.ndarray
    y: np.ndarray


def _fit_encoders(
    d_train: pd.DataFrame, synthetic: Sequence[pd.DataFrame], target: str
) -> Tuple[MinMaxScaler, Dict[str, LabelEncoder], List[str], List[str]]:
    """Fit the scaler on D_train; build label vocabularies over D_train plus synthetic.

    The scaler carries distributional information and is therefore fitted on D_train only.
    Label encoders are vocabularies rather than fitted statistics, but they too are built
    without consulting D_test so that the isolation boundary is unambiguous.
    """
    numeric = d_train.drop(columns=[target]).select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = [c for c in d_train.columns if c not in numeric and c != target]

    scaler = MinMaxScaler()
    if numeric:
        scaler.fit(d_train[numeric])

    encoders: Dict[str, LabelEncoder] = {}
    for col in categorical + [target]:
        vocab = pd.concat([d_train[col]] + [s[col] for s in synthetic]).astype(str)
        encoders[col] = LabelEncoder().fit(vocab)
    return scaler, encoders, numeric, categorical


def _encode(
    df: pd.DataFrame, scaler, encoders, numeric, categorical, target: str
) -> _Encoded:
    out = df.copy()
    if numeric:
        out[numeric] = scaler.transform(out[numeric])
    for col in categorical:
        known = set(encoders[col].classes_)
        vals = out[col].astype(str).where(lambda s: s.isin(known), encoders[col].classes_[0])
        out[col] = encoders[col].transform(vals)
    known_t = set(encoders[target].classes_)
    mask = out[target].astype(str).isin(known_t)
    out = out.loc[mask]
    y = encoders[target].transform(out[target].astype(str))
    X = out.drop(columns=[target]).to_numpy(dtype=float)
    return _Encoded(X=X, y=y)


def run_conformal_evaluation(
    d_train: pd.DataFrame,
    d_test: pd.DataFrame,
    synthetic_datasets: Dict[str, pd.DataFrame],
    target: str,
    alpha: float = 0.05,
    cal_fraction: float = 0.2,
    n_estimators: int = 100,
    random_state: int = 42,
    include_oracle: bool = True,
) -> Dict:
    """Train on each synthetic dataset, calibrate on synthetic, evaluate on ``d_test``.

    Args:
        d_train: real training partition. Used to fit the scaler and, for the oracle arm,
            to train and calibrate.
        d_test: held-out real partition. The only data used for evaluation.
        synthetic_datasets: mapping of arm name (e.g. "raw", "refined") to synthetic data.
        target: name of the label column.
        alpha: miscoverage level; target coverage is ``1 - alpha``.
        cal_fraction: fraction of each synthetic dataset reserved for calibration.
        include_oracle: also report the real-train / real-calibrate upper bound.

    Returns:
        Mapping of arm name to coverage, average set size and downstream accuracy.
    """
    arms = list(synthetic_datasets.values())
    scaler, encoders, numeric, categorical = _fit_encoders(d_train, arms, target)

    enc_test = _encode(d_test, scaler, encoders, numeric, categorical, target)
    results: Dict[str, Dict] = {}

    for name, syn in synthetic_datasets.items():
        enc = _encode(syn, scaler, encoders, numeric, categorical, target)
        try:
            X_tr, X_cal, y_tr, y_cal = train_test_split(
                enc.X, enc.y, test_size=cal_fraction,
                random_state=random_state, stratify=enc.y,
            )
        except ValueError:
            # Stratification fails when a class has a single member, which happens with
            # mode-collapsed generators. Fall back rather than dropping the arm.
            X_tr, X_cal, y_tr, y_cal = train_test_split(
                enc.X, enc.y, test_size=cal_fraction, random_state=random_state
            )

        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)

        cp = ConformalClassifier(clf, alpha=alpha)
        cp.calibrate(X_cal, y_cal)
        res = cp.evaluate(enc_test.X, enc_test.y)
        res["accuracy"] = float(accuracy_score(enc_test.y, clf.predict(enc_test.X)))
        res["n_train"] = int(len(X_tr))
        res["n_calibration"] = int(len(X_cal))
        results[name] = res

    if include_oracle:
        enc_tr = _encode(d_train, scaler, encoders, numeric, categorical, target)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            enc_tr.X, enc_tr.y, test_size=cal_fraction,
            random_state=random_state, stratify=enc_tr.y,
        )
        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        cp = ConformalClassifier(clf, alpha=alpha)
        cp.calibrate(X_cal, y_cal)
        res = cp.evaluate(enc_test.X, enc_test.y)
        res["accuracy"] = float(accuracy_score(enc_test.y, clf.predict(enc_test.X)))
        res["n_train"] = int(len(X_tr))
        res["n_calibration"] = int(len(X_cal))
        results["oracle"] = res

    results["_config"] = {
        "alpha": alpha, "target_coverage": 1 - alpha,
        "score_function": "inverse_probability",
        "cal_fraction": cal_fraction, "n_estimators": n_estimators,
        "random_state": random_state, "n_test": int(len(enc_test.y)),
        "scaler_fitted_on": "D_train",
    }
    return results
