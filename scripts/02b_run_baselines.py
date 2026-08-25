#!/usr/bin/env python
"""Run the comparison baselines on the same pool MAPS uses.

Each baseline receives D_train and the full pool and returns a dataset of size |D_train|,
so the only thing that differs between arms is the selection rule.
"""
import argparse

import pandas as pd

from _common import Timer, load_config, load_paths, write_json
from maps.baselines import (
    alaa_authenticity_filter,
    qde_utility_filter,
    wang_moment_matching,
)
from maps.data.splits import load_split

BASELINES = ("alaa", "wang", "qde")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", required=True, choices=BASELINES)
    ap.add_argument("--n-runs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    split = load_split(paths["splits_dir"], cfg["dataset"]["name"])
    n_runs = args.n_runs or cfg["evaluation"]["n_runs"]
    n_target = split.n_train

    pool = pd.read_csv(
        paths["pools_dir"] / args.dataset / f"{args.model}_{cfg['maps']['pool_multiplier']}x.csv"
    )
    out_dir = paths["results_dir"] / args.dataset / args.model / f"baseline_{args.baseline}"
    out_dir.mkdir(parents=True, exist_ok=True)

    timings = []
    for run in range(n_runs):
        with Timer(f"{args.baseline}:run{run}") as t:
            if args.baseline == "alaa":
                sel = alaa_authenticity_filter(split.train, pool, n_target, random_state=42 + run)
            elif args.baseline == "wang":
                sel = wang_moment_matching(split.train, pool, n_target, random_state=42 + run)
            else:
                sel = qde_utility_filter(
                    split.train, pool, n_target,
                    target=cfg["dataset"]["target"], random_state=42 + run,
                )
        timings.append(t.record)
        sel.to_csv(out_dir / f"refined_run{run}.csv", index=False)
        print(f"  run {run}: {len(sel)} records")

    write_json(
        {"dataset": args.dataset, "model": args.model, "baseline": args.baseline,
         "n_target": n_target, "pool_size": int(len(pool)),
         "real_data_used": "D_train", "timing": timings},
        out_dir / "baseline_meta.json",
    )


if __name__ == "__main__":
    main()
