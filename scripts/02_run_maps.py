#!/usr/bin/env python
"""Run MAPS refinement on a synthetic pool.

Real data used here is D_train only. D_test is never loaded.

The ``--variant`` flag selects the refinement configuration, and provides the stage-wise
ablation:

  random         N drawn uniformly from the pool -- the raw baseline. Same pool as every
                 other variant, so any difference cannot be attributed to pool size alone.
  stage1_only    identifiability filtering, then a uniform draw
  stage2_only    importance weighting and SIR on the unfiltered pool
  full           Stage 1 followed by Stage 2 (this is MAPS)
  reverse        Stage 1, then SIR with inverted weights -- negative control
  top_k          Stage 1, then the N highest-weight records instead of SIR
"""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from _common import Timer, load_config, load_paths, write_json
from maps import FidelityClassifier, IdentifiabilityAnalyzer, SamplingEngine
from maps.data.splits import load_split

VARIANTS = ("random", "stage1_only", "stage2_only", "full", "reverse", "top_k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="full", choices=VARIANTS)
    ap.add_argument("--pool-multiplier", type=int, default=None,
                    help="Subsample the pool to this multiple of |D_train| (pool-size ablation)")
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None,
                    help="Override the configured tempering parameter. Use with "
                         "--variant-suffix so the two settings do not overwrite each other.")
    ap.add_argument("--no-replacement", action="store_true",
                    help="Draw the refined set without replacement, so it holds N distinct "
                         "records. This departs from the SIR formulation, which assumes "
                         "replacement; use it with --variant-suffix.")
    ap.add_argument("--variant-suffix", default="",
                    help="Appended to the output directory name.")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    mcfg = dict(cfg["maps"])
    # Per-model tempering parameter, selected inside D_train. Falls back to the dataset
    # default for any model the sweep has not covered.
    by_model = mcfg.get("alpha_by_model") or {}
    if args.model in by_model:
        mcfg["alpha"] = by_model[args.model]
        print(f"Using alpha selected for {args.model}: {mcfg['alpha']}")
    if args.alpha is not None:
        print(f"Overriding alpha: {mcfg['alpha']} -> {args.alpha}")
        mcfg["alpha"] = args.alpha
    if args.no_replacement:
        mcfg["replacement"] = False
        print("Sampling without replacement: the refined set will hold N distinct records.")
    split = load_split(paths["splits_dir"], cfg["dataset"]["name"])
    d_train = split.train                      # D_test is deliberately not loaded
    n_target = split.n_train
    n_runs = args.n_runs or cfg["evaluation"]["n_runs"]

    pool_csv = (paths["pools_dir"] / args.dataset
                / f"{args.model}_{mcfg['pool_multiplier']}x.csv")
    pool = pd.read_csv(pool_csv)
    pool_sha1 = hashlib.sha1(pool_csv.read_bytes()).hexdigest()
    if args.pool_multiplier:
        keep = args.pool_multiplier * n_target
        pool = pool.sample(n=min(keep, len(pool)), random_state=0).reset_index(drop=True)
        print(f"Pool-size ablation: using {len(pool)} records ({args.pool_multiplier}N)")

    timing, meta = {}, {}

    # ---- Stage 1: identifiability filtering against D_train -------------------
    # Stage 1 depends only on D_train and the pool, so every variant that uses it produces
    # the same flags. On GOSSIS the computation is ~16 minutes, and four variants per model
    # need it, so the result is cached against a key covering both inputs. A mismatch in
    # either recomputes rather than reusing.
    #
    # The key hashes the pool file's contents, not just its row count. Regenerating a pool
    # with the same seed and the same record count leaves that count identical while every
    # coordinate can shift, so a count-based key would return flags computed against the
    # superseded file. Records Stage 1 would now reject then survive into the refined set,
    # and the identifiability score that the filtering makes exactly zero comes out
    # marginally above it. Hashing the file costs seconds against a recomputation in minutes.
    if args.variant in ("stage1_only", "full", "reverse", "top_k"):
        cache_key = hashlib.sha1(
            f"{split.sha1.get('train_sha1')}|{args.model}|{len(pool)}|"
            f"{args.pool_multiplier or mcfg['pool_multiplier']}|{pool_sha1}".encode()
        ).hexdigest()[:16]
        cache_dir = paths["results_dir"] / args.dataset / args.model / "_stage1_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_npy = cache_dir / f"flags_{cache_key}.npy"
        cache_meta = cache_dir / f"flags_{cache_key}.json"

        if cache_npy.exists():
            flags = np.load(cache_npy)
            if len(flags) != len(pool):
                print("Stage 1 cache size does not match the pool; recomputing")
                cache_npy.unlink()
            else:
                meta = json.loads(cache_meta.read_text()) if cache_meta.exists() else {}
                timing["stage1"] = meta.get("timing", {"note": "reused from cache"})
                timing["stage1"]["cache_hit"] = True
                print(f"Stage 1 reused from cache ({cache_npy.name})")

        if not cache_npy.exists():
            with Timer(f"stage1:{args.dataset}:{args.model}") as t:
                flags = IdentifiabilityAnalyzer().fit(d_train, pool)
            timing["stage1"] = t.record
            np.save(cache_npy, flags)
            cache_meta.write_text(json.dumps(
                {"dataset": args.dataset, "model": args.model,
                 "train_sha1": split.sha1.get("train_sha1"), "pool_sha1": pool_sha1,
                 "pool_size": int(len(pool)), "n_safe": int((flags == 0).sum()),
                 "timing": t.record}, indent=2, default=str))
        safe = np.where(flags == 0)[0]
        filtered = pool.iloc[safe].reset_index(drop=True)
        meta["stage1_survival_rate"] = float(len(filtered) / len(pool))
        print(f"Stage 1: {len(filtered)}/{len(pool)} records retained "
              f"({meta['stage1_survival_rate']:.1%})")
    else:
        filtered = pool
        meta["stage1_survival_rate"] = 1.0

    if len(filtered) < 2 * n_target:
        # Stage 2 needs N records to train the classifier and at least N more to resample
        # from, so the pool must satisfy  M * survival >= 2N. Report the multiplier that
        # would have worked, since that is the number a practitioner actually needs.
        survival = len(filtered) / len(pool)
        needed = 2 / survival if survival > 0 else float("inf")
        raise RuntimeError(
            f"Filtered pool ({len(filtered)}) is too small for Stage 2, which needs "
            f"{n_target} records to train the classifier plus at least {n_target} to "
            f"resample from ({2 * n_target} in total). "
            f"Stage 1 retained {survival:.1%} of the {len(pool)}-record pool, so at least "
            f"{needed:.1f}N is required here; this run used "
            f"{args.pool_multiplier or mcfg['pool_multiplier']}N. "
            f"This is the pool-size requirement in concrete form: M >= 2N / survival."
        )

    # ---- Stage 2: density-ratio weights ---------------------------------------
    if args.variant in ("stage2_only", "full", "reverse", "top_k"):
        # Synthetic records used to train the classifier are held out from resampling,
        # so the classifier never scores its own training data.
        syn_train = filtered.sample(n=n_target, random_state=42)
        syn_rest = filtered.drop(syn_train.index).reset_index(drop=True)
        ccfg = mcfg["classifier"]
        with Timer(f"stage2:{args.dataset}:{args.model}") as t:
            clf = FidelityClassifier(
                classifier_type="mlp", calibration_method=ccfg["calibration_method"]
            )
            acc, auc = clf.fit(
                d_train, syn_train,
                mlp_hidden_layer_sizes=tuple(ccfg["mlp_hidden_layer_sizes"]),
                classifier_test_size=ccfg["classifier_test_size"],
                use_embedding=ccfg["use_embedding"],
                show_calibration_plot=False,
            )
            weights, _ = clf.estimate_importance_weights(
                syn_rest, use_embedding=ccfg["use_embedding"]
            )
        timing["stage2"] = t.record
        meta.update(classifier_accuracy=float(acc), classifier_roc_auc=float(auc))
    else:
        syn_rest = filtered.reset_index(drop=True)
        weights = np.ones(len(syn_rest))

    method = {"reverse": "reverse_weighted", "top_k": "top_k"}.get(args.variant, "weighted")
    weight_processing = "raw" if args.variant in ("random", "stage1_only") else mcfg["weight_processing"]

    # ---- Resampling ------------------------------------------------------------
    out_dir = (paths["results_dir"] / args.dataset / args.model
               / f"{args.variant}{args.variant_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stats = []
    for run in range(n_runs):
        engine = SamplingEngine(random_seed=42 + run, verbose=False)
        if args.variant in ("random", "stage1_only"):
            rng = np.random.default_rng(42 + run)
            idx = rng.choice(len(syn_rest), size=n_target, replace=False)
            refined = syn_rest.iloc[idx].reset_index(drop=True)
            stats = {"method": "uniform", "selected_samples": len(refined),
                     "n_unique_samples": int(len(np.unique(idx))), "duplicate_rate": 0.0,
                     "pool_size": int(len(syn_rest)), "replacement": False}
        else:
            refined, _, _, _, stats = engine.sample(
                synthetic_data=syn_rest,
                importance_weights=weights,
                n_samples=n_target,
                method=method,
                weight_processing=weight_processing,
                alpha=mcfg["alpha"],
                replacement=mcfg["replacement"],
            )
        refined.to_csv(out_dir / f"refined_run{run}.csv", index=False)
        run_stats.append(stats)
        print(f"  run {run}: {stats['selected_samples']} records, "
              f"{stats['n_unique_samples']} unique")

    write_json(
        {"dataset": args.dataset, "model": args.model, "variant": args.variant,
         "n_target": n_target, "pool_size": int(len(pool)),
         "pool_multiplier": args.pool_multiplier or mcfg["pool_multiplier"],
         "alpha": mcfg["alpha"], "replacement": mcfg["replacement"],
         "real_data_used": "D_train", "train_sha1": split.sha1.get("train_sha1"),
         "pool_sha1": pool_sha1,
         "meta": meta, "timing": timing, "runs": run_stats},
        out_dir / "maps_meta.json",
    )


if __name__ == "__main__":
    main()
