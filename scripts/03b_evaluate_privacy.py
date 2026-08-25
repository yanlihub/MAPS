#!/usr/bin/env python
"""Privacy evaluation: identifiability score, membership inference, DOMIAS.

Membership inference needs a membership boundary. The generator and both MAPS stages see
D_train and nothing else, so D_train records are members and D_test records are non-members.

The identifiability score is reported against D_train, since that is the reference Stage 1
enforced the constraint against.
"""
import argparse

import numpy as np
import pandas as pd

from _common import Timer, load_config, load_paths, write_json
from maps import IdentifiabilityAnalyzer
from maps.data.splits import load_split


def identifiability(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """Fraction of real records closer to a synthetic record than to their nearest real one."""
    flags = IdentifiabilityAnalyzer().fit(real, synthetic)
    return float(np.mean(flags))


def membership_inference(d_train, d_test, synthetic, n_iter=10):
    from syntheval.metrics.privacy.metric_MIA_classification import MIAClassifier

    clf = MIAClassifier(
        real_data=d_train, synt_data=synthetic, hout_data=d_test, do_preprocessing=True
    )
    res = clf.evaluate(num_eval_iter=n_iter)
    return {k: float(v) for k, v in res.items() if isinstance(v, (int, float))}


def domias(d_train, d_test, synthetic, synth_val, analyzer, reference_size=1000,
           pca_components=None):
    """DOMIAS, following the procedure in the published notebooks.

    Details that are not optional, each of which silently breaks the metric if missed:

    * every frame is encoded by the identifiability preprocessor first;
    * X_gt, X_train and synth_val_set must all arrive as DataLoaders;
    * synth_val_set is a separate draw from the pool, not the dataset under evaluation;
    * on GOSSIS the encoded table is reduced by PCA first. Its 68 variables include
      collinear ones -- ten binary comorbidity flags among them -- so the covariance matrix
      is singular and `gaussian_kde` refuses to run. The published runs project to 20
      components, fitted once on all frames together so they share a basis.
    """
    from synthcity.metrics.eval_privacy import DomiasMIAKDE, DomiasMIAPrior
    from synthcity.plugins.core.dataloader import GenericDataLoader

    frames = {}
    for key, df in (("gt", d_test), ("train", d_train), ("syn", synthetic), ("val", synth_val)):
        _, processed, *_ = analyzer.preprocessor.fit_transform(d_train, df)
        frames[key] = processed

    if pca_components:
        from sklearn.decomposition import PCA
        import numpy as np

        order = list(frames)
        combined = pd.concat([frames[k] for k in order], axis=0, ignore_index=True)
        pca = PCA(n_components=pca_components)
        transformed = pca.fit_transform(combined.values.astype(np.float64))
        start = 0
        for k in order:
            n = len(frames[k])
            frames[k] = pd.DataFrame(transformed[start:start + n])
            start += n

    loaders = {k: GenericDataLoader(v) for k, v in frames.items()}
    out = {"pca_components": pca_components, "reference_size": reference_size}
    for name, evaluator in (("prior", DomiasMIAPrior()), ("kde", DomiasMIAKDE())):
        try:
            res = evaluator.evaluate(
                X_gt=loaders["gt"], X_syn=loaders["syn"], X_train=loaders["train"],
                synth_val_set=loaders["val"], reference_size=reference_size,
            )
            out[name] = {k: float(v) for k, v in res.items()}
        except Exception as exc:                      # noqa: BLE001
            out[name] = {"error": str(exc)[:300]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--baseline-variant", default="random")
    ap.add_argument("--skip-domias", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    split = load_split(paths["splits_dir"], cfg["dataset"]["name"])
    n_runs = cfg["evaluation"]["n_runs"]
    res_dir = paths["results_dir"] / args.dataset / args.model
    pool = pd.read_csv(
        paths["pools_dir"] / args.dataset
        / f"{args.model}_{cfg['maps']['pool_multiplier']}x.csv"
    )

    results = {"dataset": args.dataset, "model": args.model, "variant": args.variant,
               "membership_boundary": {"members": "D_train", "non_members": "D_test"},
               "runs": []}

    for run in range(n_runs):
        entry = {"run": run}
        for arm, variant in (("raw", args.baseline_variant), ("refined", args.variant)):
            syn = pd.read_csv(res_dir / variant / f"refined_run{run}.csv")
            with Timer(f"privacy:{arm}:{run}") as t:
                analyzer = IdentifiabilityAnalyzer(verbose=False)
                flags = analyzer.fit(split.train, syn)
                arm_res = {
                    "identifiability_score": float(np.mean(flags)),
                    "mia": membership_inference(split.train, split.test, syn),
                }
                if not args.skip_domias:
                    # Independent draw from the pool as the synthetic reference.
                    val = pool.sample(n=len(syn), random_state=1000 + run).reset_index(drop=True)
                    dcfg = cfg.get("privacy", {})
                    arm_res["domias"] = domias(
                        split.train, split.test, syn, val, analyzer,
                        reference_size=dcfg.get("domias_reference_size", 1000),
                        pca_components=dcfg.get("domias_pca_components"))
            arm_res["timing"] = t.record
            entry[arm] = arm_res
        results["runs"].append(entry)
        print(f"[run {run}] IS raw={entry['raw']['identifiability_score']:.4f} "
              f"refined={entry['refined']['identifiability_score']:.4f}")

    write_json(results, res_dir / args.variant / "privacy.json")

    ident = [r["refined"]["identifiability_score"] for r in results["runs"]]
    if max(ident) > 0:
        print(f"WARNING: refined identifiability score is not zero (max {max(ident):.6f}). "
              "Stage 1 should make this exact.")
    else:
        print("Refined identifiability score is 0.0000 across all runs, as Stage 1 enforces.")


if __name__ == "__main__":
    main()
