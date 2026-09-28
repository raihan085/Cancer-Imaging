#!/usr/bin/env python
"""Synthetic ground-truth experiment (validation track 1).

    python scripts/run_synthetic_experiment.py --out results/synthetic            # phantom anatomy
    python scripts/run_synthetic_experiment.py --root /data/UPENN-GBM --case UPENN-GBM-00001_11 \
        --slice-axis 2 --out results/synthetic_upenn                              # real anatomy

Known parameters -> simulated hidden biology -> synthetic MRI-like soft labels
(with boundary noise) -> inversion -> recovery / calibration / identifiability
report, plus segmentation-robustness sweep and scenario analysis.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glioma_scenarios.domain import phantom_domain  # noqa: E402
from glioma_scenarios.params import ModelParams  # noqa: E402
from glioma_scenarios.pipeline import AnalysisConfig, analyze  # noqa: E402
from glioma_scenarios.report import save_json  # noqa: E402
from glioma_scenarios.scenarios import ScenarioConfig  # noqa: E402
from glioma_scenarios.synthetic import make_synthetic_case, recovery_experiment, segmentation_robustness  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--case")
    ap.add_argument("--slice-axis", type=int, default=2)
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--grid", type=int, default=64, help="phantom grid size")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--noise", type=float, default=0.7, help="segmentation boundary-noise level")
    ap.add_argument("--n-adam", type=int, default=100)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--robustness", action="store_true", help="sweep boundary-noise levels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/synthetic"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    if a.root:
        from glioma_scenarios.data.upenn import index_dataset, load_case
        from glioma_scenarios.preprocess import build_domain
        case = index_dataset(a.root)[a.case]
        dom = build_domain(load_case(case), factor=a.factor, slice_axis=a.slice_axis, case_id=case.case_id)
        center = dom.tumour_core_centroid()
    else:
        dom = phantom_domain((a.grid, a.grid), spacing=160.0 / a.grid)
        center = (0.62 * a.grid, 0.38 * a.grid)

    truth = ModelParams({"D_white": 0.35, "beta_max": 0.9, "mu_p": 0.06, "T_seed": 50.0})
    syn = make_synthetic_case(dom, truth, center, rng, dt=a.dt, boundary_noise=a.noise)
    print("[recovery] fitting synthetic observation")
    rec = recovery_experiment(syn, dt=a.dt, fit_kwargs=dict(n_adam=a.n_adam, n_lbfgs=10))
    rec.pop("laplace")
    save_json(rec, a.out / "recovery.json")
    for k in rec["fitted"]:
        print(f"  {k:10s} truth={rec['truth'][k]:.3g} fitted={rec['fitted'][k]:.3g} z={rec['z_scores'][k]:+.2f}")
    for c in rec["combinations"]:
        print(f"  combo {c['combination']:45s} informative={c['informative']} covered90={c['covered_90']}")
    print("  state rel. L2:", rec["state_rel_l2"])

    if a.robustness:
        rows = segmentation_robustness(dom, truth, center, dt=a.dt, fit_kwargs=dict(n_adam=a.n_adam, n_lbfgs=0))
        save_json(rows, a.out / "segmentation_robustness.json")

    cfg = AnalysisConfig(dt=a.dt, n_adam=a.n_adam, n_scenario_samples=a.samples,
                         scenario=ScenarioConfig(horizon_days=42.0))
    res = analyze(syn["domain"], "Synthetic ground-truth scenario analysis", center_vox=center,
                  cfg=cfg, out_dir=a.out)
    print(res["markdown"])


if __name__ == "__main__":
    main()
