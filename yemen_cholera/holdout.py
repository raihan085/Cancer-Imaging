#!/usr/bin/env python3
"""holdout.py -- forward out-of-sample validation for one (city, arm).

Every error reported in the main study is training error, and the FF-PINN
carries capacity the other arms do not: its states are free network outputs
tied to the PDE only softly, so it can represent curves the PDE itself cannot
produce.  That is the same property that leaves its R0 unconstrained.  So the
in-sample advantage has to be tested, not assumed to be reconstruction.

Protocol: fit on t <= TRAIN_UNTIL, predict the whole window, score the two
segments separately.  The initial condition is anchored on the first bulletin,
which lies inside the training segment, so no held-out information enters the
fit through the IC.

    python holdout.py --city "Al Hudaydah" --arm ff --train-until 200
"""
from __future__ import annotations
import argparse, copy, os, pickle, time
import numpy as np, torch
import seip, data_io, fit, pinn, fit_one

torch.set_num_threads(2)
P = lambda *a: print(*a, flush=True)

ROW_ATTRS = ("t0", "t1", "y", "y_raw", "dC", "dC_raw", "w", "w_count", "w_rate", "gov")


def time_slice(panel, keep):
    """Rows of a single-location panel selected by a boolean mask."""
    p = copy.copy(panel)
    for a in ROW_ATTRS:
        if hasattr(panel, a) and getattr(panel, a) is not None:
            v = np.asarray(getattr(panel, a))
            if v.shape[:1] == keep.shape:
                setattr(p, a, v[keep].copy())
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["det", "dpinn", "dpinn_fs", "fdpinn", "ff"])
    ap.add_argument("--train-until", type=float, default=200.0)
    ap.add_argument("--mode", default="forward", choices=["forward", "interleave"],
                    help="forward: fit on t<=train_until and predict beyond it, "
                         "which tests FORECASTING and is dominated by beta being "
                         "extrapolated outside its data support. interleave: hold "
                         "out every 4th interval, spread across the whole window, "
                         "so beta interpolates everywhere and the test isolates "
                         "whether the fitted surface generalises to unseen "
                         "observation times.")
    ap.add_argument("--data-dir", default="data_real")
    ap.add_argument("--outdir", default="holdout_fits")
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--pretrain", type=int, default=800)
    ap.add_argument("--lbfgs", type=int, default=120)
    ap.add_argument("--n-col", type=int, default=192)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-ren", type=float, default=20.0)
    ap.add_argument("--w-ic", type=float, default=50.0)
    ap.add_argument("--w-smooth", type=float, default=3e-2)
    ap.add_argument("--objective", default="poisson", choices=["poisson", "wls"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    t_all = time.time()
    grid1 = seip.Grid(dt=a.dt, nx=1)
    full = fit_one.city_panel(a.city, a.data_dir)
    full.set_weights("count")
    if a.mode == "forward":
        TR = full.t1 <= a.train_until
        TE = full.t0 >= a.train_until
        desc = f"train t<={a.train_until:.0f} (forecast test)"
    else:
        idx = np.arange(len(full.y))
        TE = (idx % 4) == 2          # index 0 stays in train: it anchors the IC
        TR = ~TE
        desc = "every 4th interval held out (generalisation test)"
    train = time_slice(full, TR)
    train.set_weights("count")
    P(f"[{a.city} | {a.arm} | {desc}] "
      f"{len(full.y)} intervals -> train {int(TR.sum())}, held out {int(TE.sum())}, "
      f"neither {int((~TR & ~TE).sum())}")

    S0, _, _, Iagg0, Pagg0, icopt = pinn.city_initial_condition(train, 0, grid1)
    P(f"  IC from the training segment: S0={S0:.6f}  I_agg0={Iagg0:.4e}")

    if a.arm == "det":
        prob_tr = fit.Problem(grid1, train, objective=a.objective)
        _, _, _, seed = fit.grid_search(prob_tr, seip.Diffusion(0, 0, 0, 0), n=11,
                                        verbose=False)
        res = fit.fit(prob_tr, seip.Diffusion(0, 0, 0, 0), M=a.M, seed_beta=seed,
                      n_jobs=2, max_nfev=200, label="det-holdout", verbose=False)
        b10, b20, R0 = res.beta10_t, res.beta20_t, res.R0_t
        # score on EVERY interval, with the betas fitted on the training segment
        prob_full = fit.Problem(grid1, full, objective=a.objective)
        _, _, yhat_full, _ = prob_full.predict(b10, b20, seip.Diffusion(0, 0, 0, 0))
        name = "Deterministic (Poisson deviance)"
    else:
        name = pinn.ARCH_NAME[a.arm]
        model = pinn.CityPINN(kind=a.arm,
                              width=64, depth=4, M=a.M,
                              scaleI=max(Iagg0 / seip.A_MAX, 1e-9),
                              scaleP=max(Pagg0 / seip.B_MAX, 1e-9), seed=a.seed,
                              hard_ic=True, S0=S0,
                              I0=max(Iagg0 / seip.A_MAX, 1e-12),
                              P0=max(Pagg0 / seip.B_MAX, 1e-12))
        d_tr = pinn.CityData(train, 0, objective=a.objective)
        pinn.train_city(model, d_tr, S0, Iagg0, Pagg0, n_col=a.n_col,
                        epochs=a.epochs, pretrain=a.pretrain, lbfgs=a.lbfgs,
                        log_every=max(a.epochs // 4, 1), w_ren=a.w_ren,
                        w_pde=a.w_pde, w_ic=a.w_ic, w_smooth=a.w_smooth)
        d_full = pinn.CityData(full, 0, objective=a.objective)
        with torch.no_grad():
            yhat_full = d_full.average(model.observable(d_full.eval_times())).numpy()
        kap, J1, J2 = seip.ngm_integrals()
        pred = pinn.city_predict(model, grid1.t, grid1.a, grid1.b)
        b10, b20 = pred["b10"], pred["b20"]
        R0 = pinn.R0_from(model, grid1.t, kap, J1, J2)

    row = dict(city=a.city, arm=a.arm, method=name, mode=a.mode,
               train_until=a.train_until, w_pde=a.w_pde, tag=a.tag,
               n_train=int(TR.sum()), n_test=int(TE.sum()))
    row.update(fit_one.errors(yhat_full[TR], full.y[TR], "in_"))
    row.update(fit_one.errors(yhat_full[TE], full.y[TE], "out_"))
    row["R0_mean"] = float(R0.mean()); row["R0_max"] = float(R0.max())
    row["R0_min"] = float(R0.min())
    row["env_share"] = float(seip.route_share(b10, b20,
                                              seip.ngm_integrals()[2], grid1)[0])
    row["seconds"] = time.time() - t_all

    os.makedirs(os.path.join(a.outdir, fit_one.slug(a.city)), exist_ok=True)
    out = os.path.join(a.outdir, fit_one.slug(a.city),
                       f"{a.arm}_{a.mode}{('_' + a.tag) if a.tag else ''}.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(row=row, beta10=b10, beta20=b20, R0=R0, t=grid1.t,
                         yhat=yhat_full, TR=TR, TE=TE,
                         observed=dict(t0=full.t0, t1=full.t1, y=full.y)), f)
    P(f"  {name}: {row['seconds']:.0f}s")
    P(f"    in-sample  MAE {row['in_MAE']:.4e}  RMSE {row['in_RMSE']:.4e}  "
      f"(n={row['n_train']})")
    P(f"    HELD OUT   MAE {row['out_MAE']:.4e}  RMSE {row['out_RMSE']:.4e}  "
      f"(n={row['n_test']})")
    P(f"    degradation out/in: MAE x{row['out_MAE']/row['in_MAE']:.2f}  "
      f"RMSE x{row['out_RMSE']/row['in_RMSE']:.2f}   "
      f"R0 {row['R0_min']:.2f}-{row['R0_max']:.2f}")
    P(f"  -> {out}")


if __name__ == "__main__":
    main()