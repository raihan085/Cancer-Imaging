#!/usr/bin/env python3
"""domain_det.py -- the deterministic arm on the full (t, x) domain, run to
convergence.

All 21 reporting governorates at once, one beta(t) shared across every location,
diffusion D1..D4 held at 50 km^2/day, Poisson deviance, order-4 Fourier beta: the
deterministic counterpart of Section 13.

The problem is ill-conditioned (cond ~1e10), and on it least_squares can stop on
its step-size test (status 3) at a point that is not a minimum. So the fit is
restarted from its own solution, with a fresh trust region, until a restart no
longer lowers the objective. It is reported as converged only if the last pass
stopped on a real criterion (status > 0) AND that restart could not improve it.

The grid search puts beta20 on the floor of any box it is given. Started there,
beta20 sits deep in the softplus tail where finite differences cannot move it,
and the optimiser stops at once with beta20 frozen. So the fit is multi-started
from beta20 seeds where the gradient is alive, and the lowest objective is kept.

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
    ap.add_argument("--max-nfev", type=int, default=1000, help="per pass")
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--b20-starts", nargs="+", default=["1e-6", "1e-5", "1e-4"])
    a = ap.parse_args()

    t_all = time.time()
    panel = data_io.load_panel(a.data_dir, verbose=False)
    panel.set_weights("count")
    grid = seip.Grid(dt=0.25, nx=41, x_max=float(np.max(panel.xg)))
    diff = seip.Diffusion(a.D, a.D, a.D, a.D)
    prob = fit.Problem(grid, panel, objective="poisson")
    _, _, _, seed = fit.grid_search(prob, diff, n=11, verbose=True)
    lsq = dict(xtol=a.tol, ftol=a.tol, gtol=a.tol)
    starts, res = [], None
    for b20 in [seed[1]] + [float(v) for v in a.b20_starts]:
        r = fit.fit(prob, diff, M=a.M, seed_beta=(seed[0], max(b20, seed[1])),
                    max_nfev=a.max_nfev, lsq_kw=lsq, n_jobs=a.n_jobs,
                    label=f"deterministic-domain start b20={b20:.0e}", verbose=True)
        starts.append(dict(b20_seed=b20, cost=r.meta["cost"], status=r.status, nfev=r.nfev,
                           env_share=r.meta["env_share"]))
        print(f"  start b20={b20:.0e}: cost {r.meta['cost']:.10e}", flush=True)
        if res is None or r.meta["cost"] < res.meta["cost"]:
            res = r
    history = [dict(pass_=0, cost=res.meta["cost"], status=res.status, nfev=res.nfev)]
    nfev, improved = sum(z["nfev"] for z in starts), True
    for k in range(1, a.max_restarts + 1):
        nxt = fit.fit(prob, diff, M=a.M, theta0=res.theta, max_nfev=a.max_nfev, lsq_kw=lsq,
                      n_jobs=a.n_jobs, label=f"deterministic-domain pass {k}", verbose=True)
        nfev += nxt.nfev
        history.append(dict(pass_=k, cost=nxt.meta["cost"], status=nxt.status, nfev=nxt.nfev))
        gain = (res.meta["cost"] - nxt.meta["cost"]) / res.meta["cost"]
        print(f"  pass {k}: cost {nxt.meta['cost']:.10e}  relative gain {gain:.3e}", flush=True)
        if nxt.meta["cost"] < res.meta["cost"]:
            res = nxt
        if gain < 1e-8 and nxt.converged:
            improved = False
            break
    converged = bool(res.converged and not improved)
    out = dict(arm="det", name="Deterministic", D=a.D, x_max=grid.x_max,
               names=panel.names, xg=panel.xg, gov=panel.gov, t0=panel.t0, t1=panel.t1,
               panel_y=panel.y, yhat=res.yhat, R0=res.R0_t, beta10=res.beta10_t,
               beta20=res.beta20_t, t=grid.t, theta=res.theta, cost=res.meta["cost"],
               converged=converged, status=res.status, nfev=nfev, history=history, starts=starts,
               seconds=time.time() - t_all)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print(f"converged={converged} status={res.status} nfev={nfev} cost={res.meta['cost']:.10e} "
          f"passes={len(history)} {out['seconds']:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
