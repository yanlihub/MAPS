#!/usr/bin/env python
"""Generate a synthetic pool with synthpop (R), fitted on D_train only.

synthpop is a sequential CART synthesiser with no neural component, so it sits in a
different model family from TVAE, CTGAN, TabDDPM and DGD.
"""
import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _common import Timer, load_config, load_paths, write_json
from maps.data.splits import load_split

R_SCRIPT = Path(__file__).resolve().parent / "R" / "generate_synthpop.R"


def plan_shards(m, n_shards):
    """Split m replicates over at most n_shards processes, as evenly as possible."""
    n = max(1, min(n_shards, m))
    return [m // n + (1 if i < m % n else 0) for i in range(n)]


def concat_shards(shard_csvs, out):
    """Stack the shard pools into the single pool file, and merge their R metadata."""
    metas = []
    with out.open("w") as dst:
        for i, csv in enumerate(shard_csvs):
            with csv.open() as src:
                header = src.readline()
                if i == 0:
                    dst.write(header)
                for line in src:
                    dst.write(line)
            meta = Path(str(csv).replace(".csv", "_meta.json"))
            if meta.exists():
                metas.append(json.loads(meta.read_text()))
                meta.unlink()
            csv.unlink()
    return {"n_pool": sum(x["n_pool"] for x in metas),
            "m": sum(x["m"] for x in metas),
            "method": metas[0]["method"] if metas else None,
            "seeds": [x["seed"] for x in metas],
            "wall_seconds_max": max((x["wall_seconds"] for x in metas), default=0.0),
            "wall_seconds_sum": sum(x["wall_seconds"] for x in metas)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tcga", "gossis"])
    ap.add_argument("--m", type=int, default=None,
                    help="Number of synthetic replicates; pool size is m*|D_train|. "
                         "Defaults to the configured pool multiplier.")
    ap.add_argument("--method", default="cart")
    ap.add_argument("--seed", type=int, default=0)
    # syn(m = k) walks the k replicates one after another in a single R process, and one
    # replicate of TCGA's D_train measured 4.5 hours, so m = 30 in one process is most of a
    # week. The replicates are independent, so split them over concurrent single-replicate
    # processes instead and concatenate; --shards 30 turns that week into an afternoon.
    ap.add_argument("--shards", type=int, default=1,
                    help="Number of concurrent R processes to split the m replicates over.")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    paths = load_paths()
    split = load_split(paths["splits_dir"], cfg["dataset"]["name"])
    m = args.m or cfg["maps"]["pool_multiplier"]

    out = paths["pools_dir"] / args.dataset / f"synthpop_{m}x.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        train_csv = fh.name
    split.train.to_csv(train_csv, index=False)

    print(f"{args.dataset}/synthpop: fitting on D_train ({split.n_train} records), "
          f"m={m} -> {m * split.n_train} synthetic records")

    shards = plan_shards(m, args.shards)

    def run(idx_and_m):
        idx, shard_m = idx_and_m
        shard_out = out.with_name(f"{out.stem}.shard{idx}.csv")
        # Distinct seeds, or every shard would return the same replicate.
        proc = subprocess.run(
            ["Rscript", str(R_SCRIPT), train_csv, str(shard_out), str(shard_m),
             str(args.seed * 1000 + idx), args.method],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"synthpop shard {idx} failed:\n{proc.stderr[-2000:]}")
        print(f"[shard {idx}] m={shard_m}\n{proc.stdout}")
        return shard_out

    with Timer(f"synthpop:{args.dataset}") as t:
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            written = list(pool.map(run, list(enumerate(shards))))
        r_meta = concat_shards(written, out)
    write_json(
        {"dataset": args.dataset, "model": "synthpop", "fitted_on": "D_train",
         "n_train": split.n_train, "pool_multiplier": m, "method": args.method,
         "train_sha1": split.sha1.get("train_sha1"), "shards": len(shards),
         "r_meta": r_meta, "timing": t.record},
        paths["results_dir"] / args.dataset / "synthpop_pool_meta.json",
    )
    Path(train_csv).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
