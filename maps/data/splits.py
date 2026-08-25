"""
Single entry point for splitting real data, plus a runtime guard against test-set leakage.

Every stage of the MAPS pipeline that touches real data must take it from a ``DataSplit``
produced here. ``assert_no_leakage`` is called by ``scripts/03_evaluate.py`` before any
metric is computed, so a protocol violation fails loudly instead of silently inflating results.

Deployment vs. evaluation
-------------------------
When MAPS is *deployed*, Stage 1 should be run against every real record the institution
holds, otherwise the identifiability guarantee is incomplete. When MAPS is *evaluated*,
Stage 1 must use ``D_train`` only, otherwise the test set leaks into refinement. The
distinction is deliberate and is enforced here for the evaluation path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import pandas as pd
from sklearn.model_selection import train_test_split


def _row_fingerprint(row: pd.Series) -> str:
    """Content hash of a single record, independent of row order or index."""
    payload = "\x1f".join(map(str, row.tolist()))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def record_fingerprints(df: pd.DataFrame) -> Set[str]:
    """Content hashes for every record in ``df``.

    Hashing content rather than positional index means the guard still works after a
    reset_index, a reorder, or a round trip through CSV.
    """
    return {_row_fingerprint(r) for _, r in df.iterrows()}


# 17 significant digits is the shortest format guaranteed to round-trip an IEEE double.
# Without it, writing and re-reading a split silently perturbs values in the 16th digit.
FLOAT_FORMAT = "%.17g"


def _file_sha1(path: Path) -> str:
    """SHA-1 of the bytes on disk.

    Hashing the file rather than a re-serialisation of the parsed frame keeps the check
    exact: CSV parsing is not a perfect inverse of CSV writing (float repr and trailing
    separators both differ), so any round-trip-based fingerprint mismatches itself.
    """
    return hashlib.sha1(path.read_bytes()).hexdigest()


def duplicate_fingerprints(df: pd.DataFrame) -> Set[str]:
    """Fingerprints of records that occur more than once within ``df``.

    A dataset containing exact duplicate records cannot be partitioned into
    content-disjoint halves: copies of the same record land on both sides of any split.
    That is a property of the data, not a leak, and the guard needs to tell them apart.
    """
    counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        fp = _row_fingerprint(row)
        counts[fp] = counts.get(fp, 0) + 1
    return {fp for fp, c in counts.items() if c > 1}


@dataclass
class DataSplit:
    """A frozen train/test partition of a real dataset."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    target: str
    train_frac: float
    seed: int
    sha1: Dict[str, str] = field(default_factory=dict)
    """File hashes recorded when the split was written; empty for an unsaved split."""

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def summary(self) -> dict:
        """Descriptive summary. File hashes are added by :meth:`save`."""
        return {
            "dataset": self.name,
            "target": self.target,
            "train_frac": self.train_frac,
            "seed": self.seed,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_total": self.n_train + self.n_test,
            "n_features": self.train.shape[1],
        }

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_csv = out_dir / f"{self.name}_train.csv"
        test_csv = out_dir / f"{self.name}_test.csv"
        self.train.to_csv(train_csv, index=False, float_format=FLOAT_FORMAT)
        self.test.to_csv(test_csv, index=False, float_format=FLOAT_FORMAT)

        info = self.summary()
        info["train_sha1"] = _file_sha1(train_csv)
        info["test_sha1"] = _file_sha1(test_csv)
        with open(out_dir / f"{self.name}_split_info.json", "w") as fh:
            json.dump(info, fh, indent=2)
        return out_dir


def make_split(
    df: pd.DataFrame,
    name: str,
    target: str,
    train_frac: float = 0.6,
    seed: int = 0,
    stratify: bool = True,
) -> DataSplit:
    """Partition ``df`` once into D_train and D_test.

    This is the only place in the codebase permitted to split real data.
    """
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in dataframe")

    strat = df[target] if stratify else None
    train, test = train_test_split(
        df, train_size=train_frac, random_state=seed, stratify=strat
    )
    return DataSplit(
        name=name,
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        target=target,
        train_frac=train_frac,
        seed=seed,
    )


def load_split(out_dir: str | Path, name: str) -> DataSplit:
    """Reload a split previously written by :meth:`DataSplit.save`."""
    out_dir = Path(out_dir)
    with open(out_dir / f"{name}_split_info.json") as fh:
        info = json.load(fh)
    train_csv = out_dir / f"{name}_train.csv"
    test_csv = out_dir / f"{name}_test.csv"
    for path, key in ((train_csv, "train_sha1"), (test_csv, "test_sha1")):
        if key in info and _file_sha1(path) != info[key]:
            raise RuntimeError(
                f"{path.name} does not match the hash recorded when the split was created. "
                "Regenerate the split, or check that nothing has rewritten it."
            )
    return DataSplit(
        name=name,
        train=pd.read_csv(train_csv),
        test=pd.read_csv(test_csv),
        target=info["target"],
        train_frac=info["train_frac"],
        seed=info["seed"],
        sha1={k: info[k] for k in ("train_sha1", "test_sha1") if k in info},
    )


def assert_no_leakage(
    evaluation_set: pd.DataFrame,
    seen_during_refinement: Iterable[pd.DataFrame],
    labels: Optional[Iterable[str]] = None,
    intrinsic_duplicates: Optional[Set[str]] = None,
) -> dict:
    """Fail unless the evaluation set is disjoint from everything MAPS has seen.

    Called by ``scripts/03_evaluate.py`` before any metric is computed.

    Args:
        intrinsic_duplicates: fingerprints of records that occur more than once in the
            source dataset, from :func:`duplicate_fingerprints`. Overlap confined to these
            is a property of the data rather than a leak: exact duplicate records cannot be
            separated by any partition. Such overlap is reported, not raised. Overlap
            outside this set always raises.
    """
    eval_fp = record_fingerprints(evaluation_set)
    frames = list(seen_during_refinement)
    names = list(labels) if labels is not None else [f"source_{i}" for i in range(len(frames))]
    if len(names) != len(frames):
        raise ValueError("labels and seen_during_refinement must be the same length")

    benign = intrinsic_duplicates or set()
    report = {
        "n_evaluation_records": len(eval_fp),
        "n_intrinsic_duplicate_fingerprints": len(benign),
        "sources": {},
    }
    offenders = []
    n_benign_total = 0
    for name, frame in zip(names, frames):
        overlap = eval_fp & record_fingerprints(frame)
        real_leak = overlap - benign
        n_benign_total += len(overlap) - len(real_leak)
        report["sources"][name] = {
            "n_records": len(frame),
            "n_overlapping": len(overlap),
            "n_explained_by_duplicates": len(overlap) - len(real_leak),
            "n_unexplained": len(real_leak),
        }
        if real_leak:
            offenders.append((name, len(real_leak)))

    if offenders:
        detail = ", ".join(f"{n}: {c} records" for n, c in offenders)
        raise RuntimeError(
            "Data isolation violated -- the evaluation set shares records with data used "
            f"during refinement that are not accounted for by duplicates in the source "
            f"dataset ({detail}). Refusing to compute metrics."
        )

    report["status"] = "clean" if n_benign_total == 0 else "clean (intrinsic duplicates only)"
    report["n_overlap_from_intrinsic_duplicates"] = n_benign_total
    return report
