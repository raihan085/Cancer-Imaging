#!/usr/bin/env python
"""Treatment-scenario analysis anchored on one UPENN-GBM case.

    python scripts/run_patient_scenarios.py --root /data/UPENN-GBM --case UPENN-GBM-00001_11 \
        --factor 3 --slice-axis 2 --out results/UPENN-GBM-00001_11

Real MRI provides: brain geometry, tissue/motility map (+DTI FA), permeability
proxy (T1-Gd, rCBV) and soft tumour-compartment observations.  Tumour kinetics
constrained by the scan are fitted with uncertainty; drug and immune kinetics are
population priors.  Output: report.md, analysis.json, trajectories.npz.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glioma_scenarios.data.upenn import index_dataset, load_case  # noqa: E402
from glioma_scenarios.pipeline import AnalysisConfig, analyze  # noqa: E402
from glioma_scenarios.preprocess import build_domain  # noqa: E402
from glioma_scenarios.scenarios import ScenarioConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--case", required=True, help="case id, e.g. UPENN-GBM-00001_11")
    ap.add_argument("--factor", type=int, default=3, help="down-sampling factor (mm)")
    ap.add_argument("--slice-axis", type=int, default=None, help="2-D analysis on the largest-core slice")
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--n-adam", type=int, default=150)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--horizon", type=float, default=56.0)
    ap.add_argument("--hmc", action="store_true", help="HMC instead of Laplace draws")
    ap.add_argument("--seeding-sensitivity", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cases = index_dataset(a.root)
    if a.case not in cases:
        sys.exit(f"case {a.case} not found under {a.root} ({len(cases)} cases indexed)")
    case = cases[a.case]
    images = load_case(case)
    dom = build_domain(images, factor=a.factor, slice_axis=a.slice_axis, case_id=case.case_id)
    print(f"domain {dom.shape} spacing {dom.spacing} voxels in brain {int(dom.mask.sum())}")
    cfg = AnalysisConfig(dt=a.dt, n_adam=a.n_adam, n_scenario_samples=a.samples,
                         uncertainty="hmc" if a.hmc else "laplace",
                         scenario=ScenarioConfig(horizon_days=a.horizon),
                         seeding_sensitivity=a.seeding_sensitivity)
    res = analyze(dom, f"Treatment scenarios for {case.case_id}", cfg=cfg, out_dir=a.out)
    print(res["markdown"])


if __name__ == "__main__":
    main()
