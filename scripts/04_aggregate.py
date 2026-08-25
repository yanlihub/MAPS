#!/usr/bin/env python
"""Aggregate evaluation runs into the tables reported in the paper."""
import argparse
import json
from pathlib import Path

from _common import load_config, load_paths, write_json
from maps.utils.aggregate import paired_comparison, summarise

# Every metric in the statistical-significance table of the manuscript, with where to find
# it in an evaluation run and whether higher is better. "fidelity" entries are looked up by
# name in the distribution-similarity frame; the rest are dictionary paths.
# (label, source, locator, higher_is_better)
METRICS = [
    # --- marginal and joint fidelity (Table F3 rows 1-9) ---
    ("Kolmogorov-Smirnov",        "fidelity", "Kolmogorov-Smirnov Test (Marginal)",        True),
    ("Chi-square Test",           "fidelity", "Chi-squared Test p-value (Marginal)",       True),
    ("Jensen-Shannon Dist.",      "fidelity", "Jensen-Shannon Distance (Marginal)",        False),
    ("Total Variation Dist.",     "fidelity", "Total Variation Distance (Marginal)",       False),
    ("Hellinger Distance",        "fidelity", "Hellinger Distance (Marginal)",             False),
    ("Inverse KL Divergence",     "fidelity", "Inverse KL Divergence (Marginal)",          True),
    ("WD(Joint distribution)",    "fidelity", "Wasserstein Distance (Joint)",              False),
    ("JSD(Joint distribution)",   "fidelity", "Jensen-Shannon Distance (Joint Categorical)", False),
    ("MMD(Joint distribution)",   "fidelity", "Maximum Mean Discrepancy (Joint, rbf)",     False),
    ("NFN(Correlation)",          "path", ("correlation", "nfn_{arm}_vs_train"),           False),
    # --- clustering and classification (rows 10-15) ---
    ("ARI(K-means)",              "path", ("clustering","clustering_agreement_cv","{arm}_ari","mean"),  True),
    ("AMI(K-means)",              "path", ("clustering","clustering_agreement_cv","{arm}_ami","mean"),  True),
    ("Accuracy(classification)",  "path", ("classification","{arm}_synthetic","test_accuracy"),         True),
    ("F1 Score(classification)",  "path", ("classification","{arm}_synthetic","test_f1_macro"),         True),
    ("ROC-AUC(classification)",   "path", ("classification","{arm}_synthetic","test_roc_auc"),          True),
    ("Feat. Imp. Corr.(classification)", "path", ("classification","feature_importance_correlation","{arm}"), True),
    # --- mortality prediction (rows 16-19) ---
    ("Accuracy(MP)",              "path", ("mortality","{arm}_synthetic","test_accuracy"),   True),
    ("F1 Score(MP)",              "path", ("mortality","{arm}_synthetic","test_f1_macro"),   True),
    ("ROC-AUC(MP)",               "path", ("mortality","{arm}_synthetic","test_roc_auc"),    True),
    ("Feat. Imp. Corr.(MP)",      "path", ("mortality","feature_importance_correlation","{arm}"), True),
    # --- uncertainty quantification (rows 20-21) ---
    # Coverage is tested on distance to the 0.95 target, as in the manuscript, so that
    # over-coverage counts as miscalibration rather than as an improvement.
    ("Coverage",                  "target", (("conformal","{arm}","coverage"), 0.95),  False),
    ("Avg.Set Size",              "path", ("conformal","{arm}","avg_set_size"),      False),
]

# Privacy metrics live in a separate file written by 03b_evaluate_privacy.py.
PRIVACY_METRICS = [
    ("IS(re-identification risk)", "path",   ("{arm}","identifiability_score"),      False),
    ("F1 (MIA)",                   "path",   ("{arm}","mia","MIA macro F1"),          False),
    ("Precision(MIA)",             "path",   ("{arm}","mia","MIA precision"),         False),
    ("Recall(MIA)",                "path",   ("{arm}","mia","MIA recall"),            False),
    # A density attack at 0.5 has no power, so the quantity that matters is the distance
    # from 0.5, not the value. Testing the raw value would score a move from 0.494 to 0.500
    # as a worsening when it is the opposite. Where every value sits on one side of 0.5 the
    # transform is a shift and the p-value is unchanged; it matters when they straddle it,
    # as the published numbers do.
    ("Accuracy(DOMIAS)",           "target", (("{arm}","domias","prior","accuracy"), 0.5), False),
    ("AUCROC(DOMIAS)",             "target", (("{arm}","domias","prior","aucroc"),   0.5), False),
]

TARGET_COVERAGE = 0.95


def _load_json(path):
    """None if the file is mid-write. Jobs keep running while aggregates are inspected."""
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"[skip] {path} is being written")
        return None


def _from_path(run, path, arm):
    node = run
    for key in path:
        node = node[key.format(arm=arm)] if "{arm}" in key else node[key]
    return float(node)


def _from_fidelity(run, metric_name, arm):
    """Pull a value out of the distribution-similarity records for one arm.

    Measured against D_train. Fidelity asks how close the synthetic distribution is to the
    reference distribution, and the reference is the one the generator and Stage 2 were
    fitted to. There is no held-out notion here: both arms are compared to the same
    reference, and a distributional distance has no overfitting to guard against.

    Comparing against D_test instead answers a different question -- whether the alignment
    generalises -- and is kept in the result files under fidelity_vs_test rather than
    reported in the paper's tables.
    """
    col = "Raw Synthetic" if arm == "raw" else "Refined Synthetic"
    for rec in run.get("fidelity_vs_train", []):
        if rec.get("Metric") == metric_name:
            return float(rec[col])
    raise KeyError(metric_name)


def collect(runs, spec, arm):
    """Values for one metric across runs, NaN where a run could not produce it."""
    label, source, locator, _ = spec
    out = []
    for run in runs:
        try:
            if source == "fidelity":
                out.append(_from_fidelity(run, locator, arm))
            elif source == "target":
                # Distance to the metric's target value, so overshooting is not rewarded.
                path, target = locator
                out.append(abs(_from_path(run, path, arm) - target))
            else:
                out.append(_from_path(run, locator, arm))
        except (KeyError, TypeError, ValueError):
            out.append(float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    ap.add_argument("--variants", nargs="*", default=["full"])
    ap.add_argument("--baseline-variant", default="random",
                    help="Arm used as 'raw'. The refined arm comes from --variants.")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    res = paths["results_dir"] / args.dataset
    summary = {"dataset": args.dataset, "target_coverage": TARGET_COVERAGE, "models": {}}

    for model in cfg["models"] + ["synthpop"]:
        for variant in args.variants:
            path = res / model / variant / "evaluation.json"
            if not path.exists():
                continue
            doc = _load_json(path)
            if doc is None:
                continue
            runs = doc["runs"]
            entry = {}
            for spec in METRICS:
                label = spec[0]
                raw = collect(runs, spec, "raw")
                ref = collect(runs, spec, "refined")
                comp = paired_comparison(raw, ref)
                comp["higher_is_better"] = spec[3]
                entry[label] = comp

            # Privacy comes from a separate file and may not exist yet.
            priv_path = res / model / variant / "privacy.json"
            if priv_path.exists() and _load_json(priv_path):
                pdoc = _load_json(priv_path)
                pruns = pdoc["runs"] if pdoc else []
                for label, kind, locator, hib in PRIVACY_METRICS:
                    raw = collect(pruns, (label, kind, locator, hib), "raw")
                    ref = collect(pruns, (label, kind, locator, hib), "refined")
                    comp = paired_comparison(raw, ref)
                    comp["higher_is_better"] = hib
                    entry[label] = comp

            summary["models"].setdefault(model, {})[variant] = entry
            n_sig = sum(1 for v in entry.values() if v.get("significant"))
            print(f"[ok] {model}/{variant}: {len(entry)} metrics, {n_sig} significant at p<0.05")

    write_json(summary, res / "summary.json")

    # Overall count, which the manuscript reports as "40 of 48 tests significant".
    total = sig = 0
    for m in summary["models"].values():
        for v in m.values():
            for comp in v.values():
                if comp.get("n_runs", 0) >= 2:
                    total += 1
                    sig += bool(comp.get("significant"))
    print(f"\nAcross the whole sweep: {sig}/{total} comparisons significant at p<0.05")

    if args.latex:
        print("\n% --- Table F3 body ---")
        for model, variants in summary["models"].items():
            for variant, entry in variants.items():
                print(f"% {model} / {variant}")
                for label, c in entry.items():
                    p = c.get("p_value"); d = c.get("cohens_dz")
                    fp = "---" if p is None or p != p else (f"{p:.4f}" if p >= 1e-4 else "$<$0.0001")
                    fd = "---" if d is None or d != d else (
                        r"$\infty$" if d == float("inf") else f"{d:.4f}")
                    print(f"{label} & {fp} & {fd} \\\\")


if __name__ == "__main__":
    main()
