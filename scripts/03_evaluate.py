#!/usr/bin/env python
"""Evaluate refined synthetic data against the held-out D_test.

Before any metric is computed, ``assert_no_leakage`` verifies that D_test shares no record
with D_train or with any refined dataset. A protocol violation aborts the run rather than
silently inflating the numbers.

Fidelity note: distributional fidelity is a property of the synthetic distribution relative
to the real reference distribution, so it is reported against D_train (the reference the
generator and MAPS were fitted to) and, separately, against D_test as an out-of-sample check.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import Timer, load_config, load_paths, write_json
from maps.core.conformal import run_conformal_evaluation
from maps.core.utility import run_clustering_evaluation, run_utility_evaluation
from maps.data.splits import assert_no_leakage, duplicate_fingerprints, load_split
from maps.utils.CorrAnalyzer import (
    compute_similarity_scores_asymmetric,
    identify_column_types,
    mixed_correlation_lower_triangle,
)
from maps.utils.DistSimTest import evaluate_distribution_similarity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--baseline-variant", default="random",
                    help="Variant used as the raw comparator")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    split = load_split(paths["splits_dir"], cfg["dataset"]["name"])
    n_runs = cfg["evaluation"]["n_runs"]
    res_dir = paths["results_dir"] / args.dataset / args.model

    def load_runs(variant):
        return [pd.read_csv(res_dir / variant / f"refined_run{r}.csv") for r in range(n_runs)]

    refined_runs = load_runs(args.variant)
    baseline_runs = load_runs(args.baseline_variant)

    # ---- Data isolation guard --------------------------------------------------
    # A dataset containing exact duplicate records cannot be split into content-disjoint
    # halves; those fingerprints are passed in so the guard can tell them from a real leak.
    source_real = pd.concat([split.train, split.test], ignore_index=True)
    intrinsic = duplicate_fingerprints(source_real)

    guard = assert_no_leakage(
        evaluation_set=split.test,
        seen_during_refinement=[split.train, *refined_runs, *baseline_runs],
        labels=["D_train",
                *[f"{args.variant}_run{i}" for i in range(n_runs)],
                *[f"{args.baseline_variant}_run{i}" for i in range(n_runs)]],
        intrinsic_duplicates=intrinsic,
    )
    print(f"[guard] data isolation: {guard['status']} "
          f"({guard['n_evaluation_records']} evaluation records checked, "
          f"{guard['n_overlap_from_intrinsic_duplicates']} overlap explained by "
          f"duplicate records present in the source dataset)")

    ds = cfg["dataset"]
    results = {"dataset": args.dataset, "model": args.model, "variant": args.variant,
               "isolation_report": guard, "runs": []}

    # Columns that leak the target are dropped per task, not globally: dx_sub leaks
    # dx_class but is a legitimate feature for mortality prediction, and icu_death leaks
    # hospital_death but not the diagnosis. The original evaluation notebooks did these
    # drops inline; they are configuration now so they cannot be lost in a port again.
    excl = ds.get("exclude_for") or {}

    def drop_leaks(frames, task):
        cols = [c for c in (excl.get(task) or []) if c in frames[0].columns]
        if not cols:
            return frames, []
        return [f.drop(columns=cols) for f in frames], cols

    for run in range(n_runs):
        raw_i, ref_i = baseline_runs[run], refined_runs[run]
        entry = {"run": run}

        with Timer(f"fidelity:{run}") as t:
            entry["fidelity_vs_train"] = evaluate_distribution_similarity(
                split.train, raw_i, ref_i, mmd_kernel="rbf", verbose=False
            ).to_dict("records")
            entry["fidelity_vs_test"] = evaluate_distribution_similarity(
                split.test, raw_i, ref_i, mmd_kernel="rbf", verbose=False
            ).to_dict("records")
        entry["fidelity_timing"] = t.record

        # Correlation structure. The normalised Frobenius norm of the difference between
        # the real and synthetic association matrices is the third column of the joint
        # fidelity table; the distribution-similarity routine does not produce it.
        with Timer(f"correlation:{run}") as t:
            def nfn(reference, candidate):
                # compute_similarity_scores_asymmetric takes the difference matrix itself;
                # analyze_correlation_structure_asymmetric returns a summary dictionary and
                # is not an intermediate step.
                num, cat = identify_column_types(reference)
                ref_corr, n_cat, n_num = mixed_correlation_lower_triangle(reference, num, cat)
                syn_corr, _, _ = mixed_correlation_lower_triangle(candidate, num, cat)
                # It indexes .values internally, so the difference must stay a DataFrame.
                scores = compute_similarity_scores_asymmetric(
                    ref_corr - syn_corr, n_cat, n_num)
                return float(scores["frobenius_normalized"])

            entry["correlation"] = {
                "nfn_raw": nfn(split.test, raw_i),
                "nfn_refined": nfn(split.test, ref_i),
                "nfn_raw_vs_train": nfn(split.train, raw_i),
                "nfn_refined_vs_train": nfn(split.train, ref_i),
            }
        entry["correlation_timing"] = t.record

        with Timer(f"utility:{run}") as t:
            (tr_c, te_c, raw_c, ref_c), dropped_c = drop_leaks(
                [split.train, split.test, raw_i, ref_i], "classification")
            entry["classification"] = run_utility_evaluation(
                tr_c, raw_c, ref_c,
                target_column=ds["target"],
                random_state=42 + run,
                real_train_data=tr_c, real_test_data=te_c,
            )
            entry["classification_excluded_columns"] = dropped_c
            if ds.get("secondary_target"):
                (tr_m, te_m, raw_m, ref_m), dropped_m = drop_leaks(
                    [split.train, split.test, raw_i, ref_i], "mortality")
                entry["mortality"] = run_utility_evaluation(
                    tr_m, raw_m, ref_m,
                    target_column=ds["secondary_target"],
                    random_state=42 + run,
                    real_train_data=tr_m, real_test_data=te_m,
                )
                entry["mortality_excluded_columns"] = dropped_m
        entry["utility_timing"] = t.record

        # Feature-importance correlation: Spearman between the importance ranking a model
        # learns from real data and the one it learns from synthetic data. Reported in the
        # paper but never computed as a scalar in the library.
        for block in ("classification", "mortality"):
            if block not in entry:
                continue
            res = entry[block]
            real_imp = res.get("real_baseline", {}).get("feature_importances")
            if real_imp is None:
                continue
            corr = {}
            for arm in ("raw_synthetic", "refined_synthetic"):
                imp = res.get(arm, {}).get("feature_importances")
                if imp is None or len(imp) != len(real_imp):
                    corr[arm.split("_")[0]] = float("nan")
                else:
                    corr[arm.split("_")[0]] = float(spearmanr(real_imp, imp).statistic)
            res["feature_importance_correlation"] = corr

        with Timer(f"clustering:{run}") as t:
            # target_column is deliberately not passed. With it, ARI/AMI are measured
            # against the true class labels; without it, the reference is the clustering
            # that the same algorithm finds on real data, which is what the submitted
            # results report. Passing the labels instead makes k=5 clusters compete with
            # 33 cancer types and drives ARI to near zero for every arm, so the two
            # settings are not comparable. Kept as published.
            entry["clustering"] = run_clustering_evaluation(
                split.train, raw_i, ref_i,
                selected_k=cfg["evaluation"]["clustering_k"],
                exclude_columns=excl.get("clustering") or None,
                real_train_data=split.train, real_test_data=split.test,
            )
        entry["clustering_timing"] = t.record

        with Timer(f"conformal:{run}") as t:
            entry["conformal"] = run_conformal_evaluation(
                d_train=split.train,
                d_test=split.test,
                synthetic_datasets={"raw": raw_i, "refined": ref_i},
                target=ds["target"],
                alpha=cfg["evaluation"].get("cp_alpha", 0.05),
                random_state=42 + run,
                include_oracle=(run == 0),
            )
        entry["conformal_timing"] = t.record

        results["runs"].append(entry)
        cp = entry["conformal"]
        print(f"[run {run}] done  |  CP coverage raw={cp['raw']['coverage']:.3f} "
              f"refined={cp['refined']['coverage']:.3f}  "
              f"set size raw={cp['raw']['avg_set_size']:.2f} "
              f"refined={cp['refined']['avg_set_size']:.2f}")

    def _jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    write_json(_jsonable(results), res_dir / args.variant / "evaluation.json")
    print("NOTE: privacy metrics are produced by 03b_evaluate_privacy.py.", file=sys.stderr)


if __name__ == "__main__":
    main()
