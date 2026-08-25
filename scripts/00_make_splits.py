#!/usr/bin/env python
"""Partition a real dataset once into D_train and D_test.

This is the only script that splits real data. Everything downstream loads the partition it
writes, so the train/test boundary is defined in exactly one place.
"""
import argparse

import pandas as pd

from _common import load_config, load_paths
from maps.data.splits import make_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    ds = cfg["dataset"]

    real = pd.read_csv(paths["data_root"] / ds["real_csv"])
    print(f"Loaded {ds['name']}: {real.shape[0]} records, {real.shape[1]} variables")

    split = make_split(
        real,
        name=ds["name"],
        target=ds["target"],
        train_frac=ds["train_frac"],
        seed=ds["split_seed"],
        stratify=ds["stratify"],
    )
    split.save(paths["splits_dir"])

    print(f"D_train: {split.n_train}   D_test: {split.n_test}")
    print("D_test is now held out from generator training, Stage 1, Stage 2 and "
          "weight-tempering selection.")


if __name__ == "__main__":
    main()
