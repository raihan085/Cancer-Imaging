#!/usr/bin/env python
"""Index a local UPENN-GBM NIfTI release and choose the pilot subset.

    python scripts/select_subset.py --root /data/UPENN-GBM --clinical /data/UPENN-GBM/UPENN-GBM_clinical_info_v2.1.csv \
        --n 25 --out results/pilot_subset.json

Inclusion: baseline scan with T1, T1-Gd, T2, FLAIR; expert-revised segmentation;
DTI and perfusion derivative maps; known IDH/MGMT preferred.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glioma_scenarios.data.upenn import (attach_clinical, index_dataset, read_clinical,  # noqa: E402
                                         select_pilot_subset, summarize)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--clinical", type=Path)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--allow-missing-dti", action="store_true")
    ap.add_argument("--allow-missing-perfusion", action="store_true")
    ap.add_argument("--allow-auto-segm", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/pilot_subset.json"))
    a = ap.parse_args()
    cases = index_dataset(a.root)
    if a.clinical:
        attach_clinical(cases, read_clinical(a.clinical))
    print("dataset:", summarize(cases))
    sel = select_pilot_subset(cases, a.n, require_dti=not a.allow_missing_dti,
                              require_perfusion=not a.allow_missing_perfusion,
                              require_expert_segm=not a.allow_auto_segm)
    rows = [dict(case_id=c.case_id, files={k: str(v) for k, v in c.files.items()},
                 molecular_known=c.molecular_known()) for c in sel]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2))
    print(f"selected {len(rows)} cases -> {a.out}")
    for r in rows:
        print(" ", r["case_id"], r["molecular_known"])


if __name__ == "__main__":
    main()
