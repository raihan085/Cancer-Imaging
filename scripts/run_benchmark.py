#!/usr/bin/env python
"""Benchmark (validation track 3) on a synthetic ground truth.

Compares
  1. finite-volume forward solver (reference / ground-truth generator)
  2. discrete-loss inversion of a one-density Fisher-KPP model
  3. standard PINN (no renewal condition)
  4. proposed renewal-aware, observation-aware PINN
on hidden-state recovery, parameter recovery and cost.  The PINNs see
observations at the scan (t=0) and at a synthetic follow-up (t=T) under a
drug schedule; the KPP baseline sees the scan only.

    python scripts/run_benchmark.py --grid 40 --iters 1500 --out results/benchmark
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glioma_scenarios.baselines import FisherKPPInversion  # noqa: E402
from glioma_scenarios.domain import phantom_domain  # noqa: E402
from glioma_scenarios.forward import GliomaModel, param_tensors  # noqa: E402
from glioma_scenarios.observation import ObservationModel, synthesize_observation  # noqa: E402
from glioma_scenarios.params import ModelParams  # noqa: E402
from glioma_scenarios.pinn import GliomaPINN, PINNConfig  # noqa: E402
from glioma_scenarios.pk import daily  # noqa: E402
from glioma_scenarios.report import save_json  # noqa: E402


def rel(a, b, m):
    return float(np.linalg.norm(a[m] - b[m]) / max(np.linalg.norm(b[m]), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--followup", type=float, default=14.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/benchmark"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dom = phantom_domain((a.grid, a.grid), spacing=120.0 / a.grid)
    center = (0.62 * a.grid, 0.38 * a.grid)
    truth = ModelParams({"D_white": 0.35, "beta_max": 0.9, "mu_p": 0.06, "T_seed": 45.0})
    sched = daily(270.0, 5, name="5-day course")

    model = GliomaModel(dom, dt=a.dt)
    om = ObservationModel(dom)
    P = param_tensors(truth)
    t0 = time.time()
    with torch.no_grad():
        s0 = model.grow_from_seed(P, center)
        s1 = model.run(s0, P, a.followup, sched)
        pi0 = om.probabilities_from_state(s0, P).numpy()
        pi1 = om.probabilities_from_state(s1, P).numpy()
    fv_seconds = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        model.run(s0, P, a.followup, sched)
    fv_scenario_seconds = time.time() - t0
    o0 = synthesize_observation(pi0, dom.mask, rng, a.noise)["soft"]
    o1 = synthesize_observation(pi1, dom.mask, rng, a.noise)["soft"]
    m = dom.mask
    T0 = {"P": s0.P.numpy(), "q": s0.q.numpy(), "r": s0.r.numpy()}
    T1 = {"P": s1.P.numpy(), "q": s1.q.numpy(), "r": s1.r.numpy()}
    results = {"truth": {k: truth[k] for k in ("D_white", "beta_max", "mu_p", "T_seed")},
               "fv_forward": dict(seconds_seed_plus_followup=fv_seconds, seconds_per_scenario=fv_scenario_seconds)}

    print("[baseline] Fisher-KPP discrete-loss inversion")
    kpp = FisherKPPInversion(dom, o0, center, dt=0.5)
    fit = kpp.fit(n_iter=100)
    u = kpp.density(fit)
    N_true = T0["P"] + T0["q"] + T0["r"]
    results["fisher_kpp"] = dict(fit=fit, total_density_rel_l2_t0=rel(u, N_true, m))

    for label, mode in (("standard_pinn", "none"), ("renewal_pinn", "quadrature")):
        print(f"[{label}] training")
        cfg = PINNConfig(t_end=a.followup, n_iters=a.iters, renewal_mode=mode, seed=a.seed, log_every=250)
        pinn = GliomaPINN(dom, sched, cfg)
        t0 = time.time()
        pinn.fit([(0.0, o0), (a.followup, o1)], verbose=True)
        secs = time.time() - t0
        t0 = time.time()
        p0, p1 = pinn.predict(0.0), pinn.predict(a.followup)
        pred_secs = time.time() - t0
        results[label] = dict(
            fitted=pinn.fitted_values(), train_seconds=secs, predict_seconds=pred_secs,
            rel_l2_t0={k: rel(p0[k], T0[k], m) for k in T0},
            rel_l2_followup={k: rel(p1[k], T1[k], m) for k in T1},
            final_losses=pinn.history[-1],
        )
    save_json(results, a.out / "benchmark.json")
    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":
    main()
