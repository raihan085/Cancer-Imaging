#!/usr/bin/env python3
"""network_fit.py -- the governorate NETWORK (metapopulation) model.

The (t, x) model of spatial_fit.py places every governorate on one axis,
distance from Sana'a, and couples them by diffusion along it. That collapses
2-D geography: Shabwah sits 2 km from Aden on that axis, Dhamar 4 km from
Amran, Marib 5 km from Al Hudaydah, and on a 20 km grid such pairs share cells
and cannot carry distinct states.

Here each governorate is its own node running the age-structured SEIP system
(no diffusion), and nodes are coupled through who-infects-whom:

    pressure felt at g = sum_h M_gh * (route pressure generated at h)
    M = (1 - m) I + m W,      W_gh  ~  N_h / d_gh^2   (h != g, rows sum to 1)

with d_gh the great-circle distance between governorate centroids (IASC COD
boundaries, already in governorate_coords.csv) and the mixing fraction m
estimated. Transmission is beta_j(t) * k_g: a shared time course with one
multiplicative offset per governorate, ridge-shrunk toward 1.

    python network_fit.py --model network   --out results/spatial/network.pkl
    python network_fit.py --model isolated  --out results/spatial/isolated.pkl

`isolated` fixes m = 0 (independent governorates sharing only the time course
of beta), so network - isolated is the value of the coupling itself.
"""
from __future__ import annotations

import argparse, os, pickle, time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import seip, data_io, fit, baselines
from spatial_fit import SpatialBeta


def haversine_km(lon1, lat1, lon2, lat2, R=6371.0088):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1; dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def gravity_matrix(names, pop, coords_csv):
    cd = pd.read_csv(coords_csv).set_index("governorate")
    lon = cd.loc[names, "lon"].to_numpy(float); lat = cd.loc[names, "lat"].to_numpy(float)
    d = haversine_km(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    W = np.asarray(pop, float)[None, :] / np.maximum(d, 1.0) ** 2
    np.fill_diagonal(W, 0.0)
    return W / W.sum(axis=1, keepdims=True), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="network", choices=["network", "isolated", "network_shared"])
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data_real")
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("exists", a.out); return
    t_all = time.time()
    panel = data_io.load_panel(a.data_dir, verbose=False)
    panel.set_weights("count")
    panel.nb_k = baselines.nb_k_by_governorate(panel)
    ng = panel.ng
    grid = seip.Grid(dt=0.25, nx=ng, x_max=float(np.max(panel.xg)))
    grid.x = np.asarray(panel.xg, float).copy()     # one node per governorate
    D0 = seip.Diffusion(0, 0, 0, 0)
    prob = fit.Problem(grid, panel, objective="nb")
    prob.Wx = np.eye(ng)                             # node g IS governorate g
    W, dist = gravity_matrix(panel.names, panel.pop,
                             os.path.join(a.data_dir, "governorate_coords.csv"))
    sb = SpatialBeta(grid, panel, "shared" if a.model == "network_shared" else "governorate")
    est_m = a.model != "isolated"
    npar = sb.nparam + int(est_m)

    def unpack(theta):
        m = 1.0 / (1.0 + np.exp(-theta[-1])) if est_m else 0.0
        return theta[:sb.nparam], m

    def mixing(m):
        return (1.0 - m) * np.eye(ng) + m * W

    def predict(theta):
        th, m = unpack(theta)
        b1, b2 = sb.eval(th)
        H, Sb, rec = seip.solve_forward(grid, prob.ic, b1, b2, D0, mix=mixing(m),
                                        want_states=True)
        return H, rec, b1, b2, m

    theta0 = np.zeros(npar)
    if est_m:
        theta0[-1] = np.log(0.02 / 0.98)              # start at 2% mixing
    if a.init and os.path.exists(a.init):
        th = pickle.load(open(a.init, "rb"))["theta"]
        n = min(len(th), sb.nparam)
        theta0[:n] = th[:n]
    else:
        _, _, _, seed = fit.grid_search(prob, D0, n=9, verbose=True)
        theta0[:2 * sb.nper] = sb.bm.start(*seed)

    def resid(theta):
        H, _, _, _, _ = predict(theta)
        yhat = prob.A @ (H @ prob.Wx.T).ravel()
        return np.concatenate([prob.data_residual(yhat), sb.penalty(theta[:sb.nparam])])

    sol = least_squares(resid, theta0, method="trf", x_scale="jac", diff_step=1e-4,
                        max_nfev=a.max_nfev, verbose=1)
    H, rec, b1, b2, m = predict(sol.x)
    yhat = prob.A @ H.ravel()
    kap, J1, J2 = seip.ngm_integrals()
    Rg = seip.R0_of_t(b1, b2, rec["S"], kap, J1, J2)
    d = slice(None, None, int(round(1 / grid.dt)))
    rd = prob.data_residual(yhat)
    out = dict(model=a.model, D=0.0, objective="nb", nb_k=panel.nb_k, mixing=float(m),
               theta=sol.x, nparam=npar, success=bool(sol.success), status=int(sol.status),
               nfev=int(sol.nfev), converged=bool(sol.success) and sol.status > 0,
               deviance=float(rd @ rd) * prob.dev_scale ** 2,
               yhat=yhat, y=panel.y, c=panel.dC, gov=panel.gov, t0=panel.t0, t1=panel.t1,
               names=panel.names, xg=panel.xg, pop=panel.pop, t_day=grid.t[d],
               x=grid.x, H_tx=H[d].astype(np.float32), Hg_day=H[d].astype(np.float32),
               beta10_g=b1[d].astype(np.float32), beta20_g=b2[d].astype(np.float32),
               R_g=Rg[d].astype(np.float32), W=W, dist_km=dist,
               seconds=time.time() - t_all)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(out, open(a.out, "wb"))
    e = yhat - panel.y
    print(f"{a.model}: dev {out['deviance']:.4g}  MAE {np.mean(abs(e)):.3e}  "
          f"nRMSE {np.sqrt(np.mean(e**2))/np.sqrt(np.mean(panel.y**2)):.3f}  "
          f"mixing {m:.4f}  converged {out['converged']}  {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
