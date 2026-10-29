#!/usr/bin/env python3
"""spatial_fit.py -- the full (t, x) model with transmission that varies by
location, fitted by the deterministic solver to all 21 governorates at once.

    model "shared"       beta_j(t)            one history for the whole country
    model "smooth"       beta_j(t, x)         + a cubic Legendre surface in x per
                                              route, and a linear-in-time
                                              interaction with x
    model "governorate"  beta_j(t) * k_g      + one log-scale offset per
                                              governorate (common to both routes),
                                              ridge-shrunk toward zero and
                                              interpolated piecewise-linearly in x

In every model g_j(t, x) = F(t) theta_j + s_j(x, t) and
beta_j = softplus(g_j) + eps, so beta stays positive. Diffusion is held and
swept, never estimated:

    python spatial_fit.py --model shared --D 50
    python spatial_fit.py --model governorate --D 50

The DPINN and FF-PINN are not fitted here. A domain PINN carrying a beta(t, x)
network has no trainer in this project yet, and a deterministic fit is the
reference such a network would have to match, so it is the first thing to have.
"""
from __future__ import annotations

import argparse, os, pickle, time
import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import least_squares

import seip, data_io, fit

EPS = 1e-12


def softplus(z):
    return np.logaddexp(0.0, z)


class SpatialBeta:
    def __init__(self, grid, panel, model="shared", M=4, deg=3):
        self.grid, self.model, self.M = grid, model, M
        self.bm = fit.BetaModel(M=M, trend=True)
        self.F = self.bm.design(grid.t)                      # (nt1, nper)
        self.nper = self.bm.nper
        z = 2.0 * grid.x / grid.x[-1] - 1.0
        self.Lx = np.column_stack([legendre.legval(z, np.eye(deg + 1)[k])
                                   for k in range(1, deg + 1)])   # (nx, deg)
        self.deg = deg
        tt = grid.t / seip.T_MAX
        self.G_int = np.column_stack([tt, np.sin(2 * np.pi * tt),
                                      np.cos(2 * np.pi * tt)])     # (nt1, 3)
        order = np.argsort(panel.xg)
        self.xs, self.order = np.asarray(panel.xg)[order], order
        self.ng = panel.ng
        if model == "shared":
            self.n_sp = 0
        elif model == "smooth":
            self.n_sp = 2 * (deg + 3)
        elif model == "governorate":
            self.n_sp = self.ng
        else:
            raise ValueError(model)
        self.nparam = 2 * self.nper + self.n_sp

    def lift(self, vals):
        v = np.asarray(vals, float)[self.order]
        return np.interp(self.grid.x, self.xs, v, left=v[0], right=v[-1])

    def g(self, theta):
        th1, th2 = theta[:self.nper], theta[self.nper:2 * self.nper]
        g1 = (self.F @ th1)[:, None] + np.zeros((1, self.grid.nx))
        g2 = (self.F @ th2)[:, None] + np.zeros((1, self.grid.nx))
        sp = theta[2 * self.nper:]
        if self.model == "smooth":
            k = self.deg + 3
            for j, (gg, c) in enumerate(((g1, sp[:k]), (g2, sp[k:]))):
                gg += (self.Lx @ c[:self.deg])[None, :]
                gg += (self.G_int @ c[self.deg:])[:, None] * self.Lx[None, :, 0]
        elif self.model == "governorate":
            u = self.lift(sp)[None, :]
            g1 += u; g2 += u
        return g1, g2

    def eval(self, theta):
        g1, g2 = self.g(theta)
        return softplus(g1) + EPS, softplus(g2) + EPS

    def penalty(self, theta, lam_smooth=1e-2, lam_ridge=1e-4, lam_sp=1.0):
        b1, b2 = self.eval(theta)
        dt = self.grid.dt
        s = np.sqrt(lam_smooth * dt / seip.T_MAX / self.grid.nx)
        parts = []
        for b in (b1, b2):
            d = np.diff(b, axis=0) / dt / max(b.mean(), 1e-300)
            parts.append(s * d.ravel())
        mask = self.bm.harmonic_mask()
        parts.append(np.sqrt(lam_ridge) * theta[:2 * self.nper][mask])
        if self.n_sp:
            parts.append(np.sqrt(lam_sp) * theta[2 * self.nper:])
        return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="shared", choices=["shared", "smooth", "governorate"])
    ap.add_argument("--D", type=float, default=50.0)
    ap.add_argument("--max-nfev", type=int, default=60)
    ap.add_argument("--init", default="", help="pickle of a 'shared' fit to warm-start")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data_real")
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("exists", a.out); return
    t_all = time.time()
    panel = data_io.load_panel(a.data_dir, verbose=False)
    panel.set_weights("count")
    grid = seip.Grid(dt=0.25, nx=41, x_max=float(np.max(panel.xg)))
    diff = seip.Diffusion(a.D, a.D, a.D, a.D)
    prob = fit.Problem(grid, panel, objective="poisson")
    sb = SpatialBeta(grid, panel, a.model)

    theta0 = np.zeros(sb.nparam)
    if a.init and os.path.exists(a.init):
        th = pickle.load(open(a.init, "rb"))["theta"]
        if len(th) == sb.nparam:
            theta0[:] = th
        else:
            theta0[:2 * sb.nper] = th[:2 * sb.nper]
    else:
        _, _, _, seed = fit.grid_search(prob, diff, n=9, verbose=True)
        theta0[:2 * sb.nper] = sb.bm.start(*seed)

    nfe = [0]

    def resid(theta):
        b1, b2 = sb.eval(theta)
        _, _, yhat, _ = prob.predict(b1, b2, diff)
        nfe[0] += 1
        return np.concatenate([prob.data_residual(yhat), sb.penalty(theta)])

    sol = least_squares(resid, theta0, method="trf", x_scale="jac", diff_step=1e-4,
                        max_nfev=a.max_nfev, verbose=1)
    b1, b2 = sb.eval(sol.x)
    H, Sb, rec = seip.solve_forward(grid, prob.ic, b1, b2, diff, want_states=True)
    Hg = H @ prob.Wx.T
    yhat = prob.A @ Hg.ravel()
    kap, J1, J2 = seip.ngm_integrals()
    Sg = rec["S"] @ prob.Wx.T                                  # (nt1, ng)
    B1g, B2g = b1 @ prob.Wx.T, b2 @ prob.Wx.T
    Rg = seip.R0_of_t(B1g, B2g, Sg, kap, J1, J2)
    d = slice(None, None, int(round(1 / grid.dt)))
    rd = prob.data_residual(yhat)
    out = dict(model=a.model, D=a.D, theta=sol.x, nparam=sb.nparam,
               success=bool(sol.success), status=int(sol.status), nfev=int(sol.nfev),
               converged=bool(sol.success) and sol.status > 0,
               deviance=float(rd @ rd) * prob.dev_scale ** 2,
               yhat=yhat, y=panel.y, c=panel.dC, gov=panel.gov, t0=panel.t0,
               t1=panel.t1, names=panel.names, xg=panel.xg, pop=panel.pop,
               t_day=grid.t[d], x=grid.x, H_tx=H[d].astype(np.float32),
               Hg_day=Hg[d].astype(np.float32), beta10_g=B1g[d].astype(np.float32),
               beta20_g=B2g[d].astype(np.float32), R_g=Rg[d].astype(np.float32),
               seconds=time.time() - t_all, calls=nfe[0])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(out, open(a.out, "wb"))
    e = yhat - panel.y
    print(f"{a.model} D={a.D}: dev {out['deviance']:.4g}  MAE {np.mean(abs(e)):.3e}  "
          f"nRMSE {np.sqrt(np.mean(e**2))/np.sqrt(np.mean(panel.y**2)):.3f}  "
          f"converged {out['converged']}  {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
