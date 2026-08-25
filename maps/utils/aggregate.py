"""Aggregate repeated runs into mean +/- sd with paired t-tests and Cohen's d_z.

Matches the analysis reported in the paper: paired t-tests across runs, with the paired
effect size d_z = mean(diff) / sd(diff) where diff = refined - raw.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def paired_comparison(raw: Sequence[float], refined: Sequence[float]) -> Dict:
    """Compare two arms measured on the same runs."""
    raw = np.asarray(raw, dtype=float)
    refined = np.asarray(refined, dtype=float)
    n_total = len(raw)
    ok = ~(np.isnan(raw) | np.isnan(refined))
    n_usable = int(ok.sum())

    # The manuscript marks a cell N/A when the raw arm could not be evaluated, rather than
    # testing on whichever runs happened to succeed. A metric that fails on two of five runs
    # because the generator dropped a minority class would otherwise be tested at a
    # different sample size from every other cell, and the comparison would not mean the
    # same thing. Any missing run makes the cell N/A.
    if n_usable < n_total:
        return {
            "n_runs": n_usable, "n_runs_expected": n_total,
            "raw_mean": float("nan"), "raw_sd": float("nan"),
            "refined_mean": float("nan"), "refined_sd": float("nan"),
            "p_value": float("nan"), "cohens_dz": float("nan"),
            "significant": False, "reportable": False,
            "note": (f"N/A: {n_total - n_usable} of {n_total} runs could not be evaluated, "
                     "so paired testing is not viable"),
        }

    raw, refined = raw[ok], refined[ok]

    out = {
        "reportable": True,
        "n_runs": int(len(raw)),
        "raw_mean": float(np.mean(raw)) if len(raw) else float("nan"),
        "raw_sd": float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0,
        "refined_mean": float(np.mean(refined)) if len(refined) else float("nan"),
        "refined_sd": float(np.std(refined, ddof=1)) if len(refined) > 1 else 0.0,
    }
    if len(raw) < 2:
        out.update(p_value=float("nan"), cohens_dz=float("nan"), significant=False)
        return out

    diff = refined - raw
    sd = np.std(diff, ddof=1)
    # Exact cancellation leaves floating-point residue rather than a true zero
    # (e.g. sd = 6.8e-17 for three identical differences), which would turn into an
    # effect size of order 1e15. Compare against the scale of the data instead.
    scale = max(np.max(np.abs(diff)), np.max(np.abs(raw)), 1.0)
    if sd <= 1e-12 * scale:
        # Identical across every run, e.g. an identifiability score that is 0 by
        # construction. A t-test is undefined; report it rather than emitting inf.
        out.update(
            p_value=0.0 if np.mean(diff) != 0 else 1.0,
            cohens_dz=float("inf") if np.mean(diff) != 0 else 0.0,
            significant=bool(np.mean(diff) != 0),
            note="zero variance across runs",
        )
        return out

    t_stat, p = stats.ttest_rel(refined, raw)
    out.update(
        t_statistic=float(t_stat),
        p_value=float(p),
        cohens_dz=float(np.mean(diff) / sd),
        significant=bool(p < 0.05),
    )
    return out


def summarise(runs: List[Dict], metric_path: Sequence[str]) -> List[float]:
    """Pull ``metric_path`` out of each run dictionary, tolerating missing entries."""
    values = []
    for run in runs:
        node = run
        try:
            for key in metric_path:
                node = node[key]
            values.append(float(node))
        except (KeyError, TypeError, ValueError):
            values.append(float("nan"))
    return values
