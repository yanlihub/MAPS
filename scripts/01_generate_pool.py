#!/usr/bin/env python
"""Train a generative model on D_train only and emit a synthetic pool of size M = k*|D_train|.

Note on synthcity
-----------------
``GenericDataLoader(df, train_size=...)`` does NOT hand a training subset to
``Plugin.fit``. The loader stores the full frame; ``train_size`` only affects what the
``.train()`` / ``.test()`` accessors return, and ``Plugin.fit(X)`` fits on all of ``X``
(synthcity's own benchmark harness calls ``generator.fit(X.train())`` explicitly).

This script therefore passes D_train to the loader directly and never sets ``train_size``,
so the generator provably never sees D_test.
"""
import argparse

import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from _common import Timer, load_config, load_paths, write_json
from maps.data.splits import load_split

PLUGIN_KWARGS = {
    "ddpm": {"n_iter": 10000},
    "tvae": {},
    "ctgan": {},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the generator. Pools for seeds other than 0 are written "
                         "with a _seed<N> suffix so they do not overwrite the main pool.")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    split = load_split(paths["splits_dir"], cfg["dataset"]["name"])

    n_train = split.n_train
    multiplier = cfg["maps"]["pool_multiplier"]
    n_pool = multiplier * n_train
    print(f"{args.dataset}/{args.model}: fitting on D_train ({n_train} records), "
          f"generating {n_pool} synthetic records ({multiplier}N)")

    # D_train only. No train_size argument -- see module docstring.
    loader = GenericDataLoader(split.train)

    timing = {}
    # The seed has to reach the plugin. It was previously accepted, recorded in the
    # metadata and then ignored, so nothing about generation was actually reproducible.
    kwargs = dict(PLUGIN_KWARGS.get(args.model, {}))
    kwargs["random_state"] = args.seed
    plugin = Plugins().get(args.model, **kwargs)
    with Timer(f"fit:{args.dataset}:{args.model}") as t:
        plugin.fit(loader)
    timing["fit"] = t.record

    with Timer(f"generate:{args.dataset}:{args.model}") as t:
        pool = plugin.generate(count=n_pool).dataframe()
    timing["generate"] = t.record

    suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    out = paths["pools_dir"] / args.dataset / f"{args.model}_{multiplier}x{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out, index=False)
    print(f"[write] {out}  ({pool.shape[0]} x {pool.shape[1]})")

    write_json(
        {
            "dataset": args.dataset,
            "model": args.model,
            "fitted_on": "D_train",
            "n_train": n_train,
            "n_pool": int(pool.shape[0]),
            "pool_multiplier": multiplier,
            "train_sha1": split.sha1.get("train_sha1"),
            "seed": args.seed,
            "timing": timing,
        },
        paths["results_dir"] / args.dataset / f"{args.model}_pool_meta{suffix}.json",
    )


if __name__ == "__main__":
    main()
