#!/usr/bin/env python3
"""collect_holdout.py -- rebuild new_slim/holdout_and_sweep.csv from holdout.py
result files, replacing the rows of the cities that were rerun.

Each holdout.py pickle carries a `row` dict; `seed_id` is the training seed
(seed 0 rows carry an empty tag, seeds 1 and 2 carry "seed1"/"seed2", and the
physics sweep carries "wpde<w>").

    python collect_holdout.py --rerun holdout_rerun --cities Amran
"""
from __future__ import annotations

import argparse, glob, os, pickle
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="new_slim/holdout_and_sweep.csv")
    ap.add_argument("--rerun", required=True, help="outdir holdout.py wrote to")
    ap.add_argument("--cities", nargs="+", required=True)
    a = ap.parse_args()

    old = pd.read_csv(a.csv)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.rerun, "*", "*.pkl"))):
        d = pickle.load(open(p, "rb"))
        r = dict(d["row"])
        tag = r.get("tag") or ""
        r["seed_id"] = int(tag[4:]) if tag.startswith("seed") else 0
        r["tag"] = tag if tag else float("nan")
        rows.append(r)
    new = pd.DataFrame(rows)
    assert set(new.city) == set(a.cities), (set(new.city), a.cities)
    keep = old[~(old.city.isin(a.cities) & (old["mode"] == "interleave"))]
    out = pd.concat([keep, new[[c for c in old.columns if c in new.columns]]],
                    ignore_index=True)
    out = out.sort_values(["city", "arm", "seed_id"], kind="stable").reset_index(drop=True)
    out.to_csv(a.csv, index=False)
    print(f"replaced {len(old) - len(keep)} rows with {len(new)} for {a.cities} -> {a.csv}")


if __name__ == "__main__":
    main()
