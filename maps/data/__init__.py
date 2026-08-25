"""Data partitioning and leakage-guard utilities for MAPS."""

from .splits import (
    DataSplit,
    make_split,
    load_split,
    record_fingerprints,
    duplicate_fingerprints,
    assert_no_leakage,
)

__all__ = [
    "DataSplit",
    "make_split",
    "load_split",
    "record_fingerprints",
    "duplicate_fingerprints",
    "assert_no_leakage",
]
