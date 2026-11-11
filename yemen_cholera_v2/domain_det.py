#!/usr/bin/env python3
"""domain_det.py -- the deterministic arm on the full (t, x) domain, run to
convergence.

All 21 reporting governorates at once, one beta(t) shared across every location,
diffusion D1..D4 held at 50 km^2/day, Poisson deviance, order-4 Fourier beta: the
deterministic counterpart of Section 13. The earlier result stopped at
max_nfev; this one is reported only if least_squares reports a real termination
criterion (status > 0).

    python domain_det.py --out new_slim/domain_det.pkl
"""
from __future__ import annotations

import argparse, pickle, time
import numpy as np

import seip, data_io, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_real")
    ap.add_argument("--out", default="new_slim/domain_det.pkl")
    ap.add_argument("--D", type=float, default=50.0)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--max-nfev", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()

    t_all = time.time()
    panel = data_io.load_panel(a.data_dir, verbose=False)
    panel.set_weights("count")
    grid = seip.Grid(dt=0.25, nx=41, x_max=float(np.max(panel.xg)))
    diff = seip.Diffusion(a.D, a.D, a.D, a.D)
    prob = fit.Problem(grid, panel, objective="poisson")
    _, _, _, seed = fit.grid_search(prob, diff, n=11, verbose=True)
    res = fit.fit(prob, diff, M=a.M, seed_beta=seed, max_nfev=a.max_nfev,
                  n_jobs=a.n_jobs, label="deterministic-domain", verbose=True)
    out = dict(arm="det", name="Deterministic", D=a.D, x_max=grid.x_max,
               names=panel.names, xg=panel.xg, gov=panel.gov, t0=panel.t0, t1=panel.t1,
               panel_y=panel.y, yhat=res.yhat, R0=res.R0_t, beta10=res.beta10_t,
               beta20=res.beta20_t, t=grid.t, converged=res.converged,
               status=res.status, nfev=res.nfev, seconds=time.time() - t_all)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print(f"converged={res.converged} status={res.status} nfev={res.nfev} "
          f"{out['seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
