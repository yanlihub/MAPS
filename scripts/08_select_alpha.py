#!/usr/bin/env python
"""Select the weight tempering parameter alpha inside D_train.

Parameter selection is one of the stages that has to stay clear of the test data, so alpha
is chosen without D_test being loaded at any point.

D_train is split into a fitting part and a validation part. MAPS runs on the fitting part
alone, and each candidate alpha is scored by distributional fidelity against the validation
part, which the refinement never sees. D_test is not loaded at any point.

Selection uses fidelity rather than downstream accuracy: alpha controls how sharply the
importance weights concentrate, which is a distributional question, and tuning it against a
downstream metric would bias the utility comparison.
"""
import argparse

import numpy as np
import pandas as pd

from _common import Timer, load_config, load_paths, write_json
from maps import FidelityClassifier, IdentifiabilityAnalyzer, SamplingEngine
from maps.data.splits import load_split, make_split
from maps.utils.DistSimTest import evaluate_distribution_similarity

# Lower is better for each of these.
SCORE_METRICS = ["Jensen-Shannon Distance (Marginal)",
                 "Total Variation Distance (Marginal)",
                 "Hellinger Distance (Marginal)"]


def fidelity_score(reference: pd.DataFrame, candidate: pd.DataFrame) -> float:
    """Mean of the bounded marginal distances, so the score sits on a comparable scale.

    All three are in [0, 1] and lower is better, so an unweighted mean is meaningful.
    Unbounded metrics such as Wasserstein distance are excluded for that reason.
    """
    df = evaluate_distribution_similarity(reference, candidate, candidate, verbose=False)
    vals = []
    for m in SCORE_METRICS:
        row = df[df["Metric"].str.startswith(m)]
        if not row.empty:
            vals.append(float(row.iloc[0]["Raw Synthetic"]))
    if len(vals) != len(SCORE_METRICS):
        raise RuntimeError(f"expected {len(SCORE_METRICS)} score metrics, found {len(vals)}")
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0])
    ap.add_argument("--val-frac", type=float, default=0.25)
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    mcfg = cfg["maps"]
    split = load_split(paths["splits_dir"], cfg["dataset"]["name"])

    # Inner split of D_train only. D_test is never touched.
    inner = make_split(split.train, f"{args.dataset}_inner", cfg["dataset"]["target"],
                       train_frac=1 - args.val_frac, seed=cfg["dataset"]["split_seed"])
    fit_part, val_part = inner.train, inner.test
    n_target = len(val_part)
    print(f"Inner split of D_train: fit {len(fit_part)}, validation {len(val_part)}")

    pool = pd.read_csv(
        paths["pools_dir"] / args.dataset / f"{args.model}_{mcfg['pool_multiplier']}x.csv"
    )

    with Timer("alpha:stage1") as t1:
        flags = IdentifiabilityAnalyzer(verbose=False).fit(fit_part, pool)
    filtered = pool.iloc[np.where(flags == 0)[0]].reset_index(drop=True)
    print(f"Stage 1 kept {len(filtered)}/{len(pool)} records")

    syn_train = filtered.sample(n=min(len(fit_part), len(filtered) // 2), random_state=42)
    syn_rest = filtered.drop(syn_train.index).reset_index(drop=True)

    ccfg = mcfg["classifier"]
    with Timer("alpha:stage2") as t2:
        clf = FidelityClassifier(classifier_type="mlp",
                                 calibration_method=ccfg["calibration_method"], verbose=False)
        clf.fit(fit_part, syn_train,
                mlp_hidden_layer_sizes=tuple(ccfg["mlp_hidden_layer_sizes"]),
                classifier_test_size=ccfg["classifier_test_size"],
                use_embedding=ccfg["use_embedding"], show_calibration_plot=False)
        weights, _ = clf.estimate_importance_weights(syn_rest,
                                                     use_embedding=ccfg["use_embedding"])

    results = []
    for alpha in args.alphas:
        scores = []
        for run in range(3):
            refined, _, _, _, _ = SamplingEngine(random_seed=100 + run, verbose=False).sample(
                synthetic_data=syn_rest, importance_weights=weights, n_samples=n_target,
                method="weighted", weight_processing="flatten", alpha=alpha,
                replacement=mcfg["replacement"],
            )
            scores.append(fidelity_score(val_part, refined))
        mean, sd = float(np.mean(scores)), float(np.std(scores, ddof=1))
        results.append({"alpha": alpha, "score_mean": mean, "score_sd": sd})
        print(f"  alpha={alpha:<5} validation fidelity {mean:.5f} +/- {sd:.5f}")

    best = min(results, key=lambda r: r["score_mean"])
    published = mcfg["alpha"]
    pub_row = next((r for r in results if abs(r["alpha"] - published) < 1e-9), None)

    print(f"\nBest alpha on the held-in validation part: {best['alpha']} "
          f"(score {best['score_mean']:.5f})")
    if pub_row:
        gap = pub_row["score_mean"] - best["score_mean"]
        print(f"Published alpha {published}: score {pub_row['score_mean']:.5f} "
              f"({gap:+.5f} vs best, {100 * gap / max(best['score_mean'], 1e-12):+.1f}%)")

    write_json({"dataset": args.dataset, "model": args.model,
                "selected_on": "validation part of D_train", "d_test_used": False,
                "val_frac": args.val_frac, "published_alpha": published,
                "selected_alpha": best["alpha"], "results": results,
                "timing": {"stage1": t1.record, "stage2": t2.record}},
               paths["results_dir"] / args.dataset / f"{args.model}_alpha_selection.json")


if __name__ == "__main__":
    main()
