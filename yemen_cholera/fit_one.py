#!/usr/bin/env python3
"""fit_one.py -- one (city, arm) fit on the REAL Yemen record, saved on its own.

Granularity is deliberate: each arm is a separate process writing its own
pickle, so a wall-clock limit can never discard a completed fit.  There is no
synthetic twin and no true parameter vector anywhere in this script; every
number it reports is a comparison of a fitted quantity against the OBSERVED
incidence, or between the two architectures.

    python fit_one.py --city "Al Hudaydah" --arm ff
"""
from __future__ import annotations
import argparse, copy, json, os, pickle, sys, time
import numpy as np, pandas as pd, torch
import seip, data_io, fit, pinn

torch.set_num_threads(2)
P = lambda *a: (print(*a, flush=True))


def slug(c): return c.lower().replace(" ", "_").replace("'", "")


def errors(pred, actual, label=""):
    """MAE, MSE, RMSE and two scale-free normalisations.

    MAE and RMSE are reported together because they answer different
    questions on this record: the incidence series is zero-inflated and
    spiky, so RMSE is dominated by the batched-reporting spikes while MAE
    tracks the typical interval.  Quoting one alone hides which it is.
    """
    pred = np.asarray(pred, float).ravel(); actual = np.asarray(actual, float).ravel()
    m = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[m], actual[m]
    e = pred - actual
    mae = float(np.mean(np.abs(e))); mse = float(np.mean(e ** 2))
    rmse = float(np.sqrt(mse))
    rms_a = float(np.sqrt(np.mean(actual ** 2)))
    rng = float(actual.max() - actual.min())
    return {f"{label}MAE": mae, f"{label}MSE": mse, f"{label}RMSE": rmse,
            f"{label}nRMSE_rms": rmse / rms_a if rms_a > 0 else np.nan,
            f"{label}nRMSE_range": rmse / rng if rng > 0 else np.nan,
            f"{label}n": int(pred.size)}


def city_panel(city, data_dir, winsorize=True):
    panel = data_io.load_panel(data_dir, winsorize=winsorize, verbose=False)
    gi = panel.names.index(city)
    m = panel.gov == gi
    p = copy.copy(panel)
    p.gov = np.zeros(int(m.sum()), int)
    for a in ("t0", "t1", "y", "y_raw", "dC", "dC_raw", "w", "w_count", "w_rate"):
        if hasattr(panel, a):
            setattr(p, a, getattr(panel, a)[m].copy())
    p.names = [city]
    for a in ("xg", "pop", "y_first", "cum_first", "t_first"):
        setattr(p, a, np.asarray(getattr(panel, a))[[gi]].copy())
    p.xg = np.array([0.0])
    for a in ("true_zero", "first_hold"):
        v = getattr(panel, a, None)
        if v is not None:
            setattr(p, a, np.asarray(v)[[gi]].copy())
    # ng is a derived property on Panel; it follows from names/xg
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["det", "dpinn", "dpinn_fs", "fdpinn", "ff"])
    ap.add_argument("--data-dir", default="data_real")
    ap.add_argument("--outdir", default="real_fits")
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--pretrain", type=int, default=800)
    ap.add_argument("--lbfgs", type=int, default=120)
    ap.add_argument("--n-col", type=int, default=192)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--w-ren", type=float, default=20.0)
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-ic", type=float, default=50.0)
    ap.add_argument("--w-smooth", type=float, default=3e-2)
    ap.add_argument("--objective", default="poisson", choices=["poisson", "wls"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t_all = time.time()
    grid = seip.Grid(dt=a.dt, nx=41)
    grid1 = seip.Grid(dt=a.dt, nx=1)
    panel = city_panel(a.city, a.data_dir)
    panel.set_weights("count")
    kap, J1, J2 = seip.ngm_integrals()
    nz = int((panel.y > 0).sum())
    P(f"[{a.city} | {a.arm}] rows {len(panel.y)}  nonzero {nz} "
      f"({100*nz/len(panel.y):.1f}%)  pop {panel.pop[0]:,.0f}  "
      f"mean y {panel.y.mean():.4e}  peak y {panel.y.max():.4e}")

    S0, _, _, Iagg0, Pagg0, icopt = pinn.city_initial_condition(panel, 0, grid)
    P(f"  IC  S0={S0:.6f}  I_agg0={Iagg0:.4e}  P_agg0={Pagg0:.4e}  "
      f"tau_I={icopt['tau_I']:.4f} d")

    os.makedirs(os.path.join(a.outdir, slug(a.city)), exist_ok=True)
    out = os.path.join(a.outdir, slug(a.city), f"{a.arm}.pkl")

    if a.arm == "det":
        prob = fit.Problem(grid1, panel, objective=a.objective)
        _, _, _, seed = fit.grid_search(prob, seip.Diffusion(0, 0, 0, 0), n=11,
                                        verbose=False)
        res = fit.fit(prob, seip.Diffusion(0, 0, 0, 0), M=a.M, seed_beta=seed,
                      n_jobs=2, max_nfev=200, label="deterministic", verbose=False)
        _, Sb, rec = seip.solve_forward(grid1, prob.ic, res.beta10_t, res.beta20_t,
                                        seip.Diffusion(0, 0, 0, 0), record_full=True)
        pack = dict(name="Deterministic (Poisson deviance)", arm="det",
                    beta10=res.beta10_t, beta20=res.beta20_t, R0=res.R0_t,
                    S=Sb, I_surface=rec["I_full"][:, :, 0],
                    P_surface=rec["P_full"][:, :, 0], yhat=res.yhat, t=grid1.t,
                    env_share=float(seip.route_share(res.beta10_t, res.beta20_t,
                                                     J2, grid1)[0]),
                    converged=bool(getattr(res, "converged", True)))
    else:
        nice = pinn.ARCH_NAME[a.arm]
        model = pinn.CityPINN(kind=a.arm,
                              width=a.width, depth=a.depth, M=a.M,
                              scaleI=max(Iagg0 / seip.A_MAX, 1e-9),
                              scaleP=max(Pagg0 / seip.B_MAX, 1e-9), seed=a.seed,
                              hard_ic=True, S0=S0,
                              I0=max(Iagg0 / seip.A_MAX, 1e-12),
                              P0=max(Pagg0 / seip.B_MAX, 1e-12))
        data = pinn.CityData(panel, 0, objective=a.objective)
        hist = pinn.train_city(model, data, S0, Iagg0, Pagg0, n_col=a.n_col,
                               epochs=a.epochs, pretrain=a.pretrain, lbfgs=a.lbfgs,
                               log_every=max(a.epochs // 5, 1), w_ren=a.w_ren,
                               w_pde=a.w_pde, w_ic=a.w_ic, w_smooth=a.w_smooth)
        pred = pinn.city_predict(model, grid1.t, grid1.a, grid1.b)
        R0 = pinn.R0_from(model, grid1.t, kap, J1, J2)
        with torch.no_grad():
            yhat = data.average(model.observable(data.eval_times())).numpy()
        pack = dict(name=nice, arm=a.arm, beta10=pred["b10"], beta20=pred["b20"],
                    R0=R0, S=pred["S"], H=pred["H"], I_surface=pred["I_surface"],
                    P_surface=pred["P_surface"], yhat=yhat, hist=hist, t=grid1.t,
                    env_share=float(seip.route_share(pred["b10"], pred["b20"],
                                                     J2, grid1)[0]),
                    n_params=int(sum(p.numel() for p in model.parameters())))

    pack["seconds"] = time.time() - t_all
    pack["city"] = a.city
    row = dict(city=a.city, method=pack["name"], arm=a.arm)
    row.update(errors(pack["yhat"], panel.y, ""))            # against OBSERVED data
    row["R0_mean"] = float(pack["R0"].mean())
    row["R0_min"] = float(pack["R0"].min())
    row["R0_max"] = float(pack["R0"].max())
    row["R0_final"] = float(pack["R0"][-1])
    row["frac_t_R0_gt1"] = float(np.mean(pack["R0"] > 1.0))
    row["env_share"] = pack["env_share"]
    row["seconds"] = pack["seconds"]
    pack["row"] = row
    pack["observed"] = dict(t0=panel.t0, t1=panel.t1, y=panel.y, pop=float(panel.pop[0]))

    with open(out, "wb") as f:
        pickle.dump(pack, f)
    P(f"  {pack['name']}: {pack['seconds']:.0f}s   "
      f"MAE {row['MAE']:.4e}  RMSE {row['RMSE']:.4e}  "
      f"R0 {row['R0_min']:.2f}-{row['R0_max']:.2f} (mean {row['R0_mean']:.2f}, "
      f"{100*row['frac_t_R0_gt1']:.0f}% of days >1)  env share {row['env_share']:.3f}")
    P(f"  -> {out}")


if __name__ == "__main__":
    main()