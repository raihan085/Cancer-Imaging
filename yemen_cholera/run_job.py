#!/usr/bin/env python3
"""run_job.py -- one fit, one result file. Every analysis in the extended study
is a set of these jobs, so a crashed or killed job never costs a finished one.

    python run_job.py --city Hajjah --arm ff --seed 3 --fold 2
    python run_job.py --city Hajjah --arm det --variant trunc
    python run_job.py --city Hajjah --arm dpinn --truth lit_share20 --seed 0

arm      det | dpinn | dpinn_fs | fdpinn | ff | spline | sir
fold     -1 fits every interval; 0..3 holds out every 4th interval at that
         offset (rotated 4-fold interleave). Row 0 is never held out: the
         initial condition is anchored on it.
variant  a sensitivity setting (VARIANTS below); 'base' is the main analysis.
truth    fit SYNTHETIC counts generated from a literature-calibrated
         transmission history (TRUTHS below) on the city's real bulletin
         calendar and population, so recovery of beta1, beta2 and R(t) can be
         scored against known values.
"""
from __future__ import annotations

import argparse, copy, os, pickle, time
import numpy as np
import torch

import seip, data_io, fit, pinn, fit_one, baselines

torch.set_num_threads(1)
PINN_ARMS = ("dpinn", "dpinn_fs", "fdpinn", "ff")
D0 = seip.Diffusion(0, 0, 0, 0)

# ---------------------------------------------------------------- variants
# Each entry overrides the defaults below. Problem-level keys go to the IC /
# observation model; the rest to the fit.
DEFAULTS = dict(truncation_correction=False, age_profile="flat",
                observable=seip.OBS_FORCE_OF_INFECTION, objective="nb",
                winsorize=True, M=4, w_ren=20.0, w_pde=1.0)
VARIANTS = {
    "base": {},
    "trunc": dict(truncation_correction=True),     # truncated-occupancy IC
    "age_eq": dict(age_profile="equilibrium"),      # exponential age profile
    "age_ln": dict(age_profile="lognormal"),        # log-normal age profile
    "onset": dict(observable=seip.OBS_INFECTIOUS_ONSET),
    "nowinsor": dict(winsorize=False),
    "wls": dict(objective="wls"),
    "poisson": dict(objective="poisson"),          # the original likelihood
    "M2": dict(M=2),
    "M6": dict(M=6),
    "wren5": dict(w_ren=5.0),
    "wren80": dict(w_ren=80.0),
}

# ---------------------------------------------------------------- truths
# R(t) calibrated to published Yemen estimates for the second wave:
#   * R about 2 at the May 2017 surge (Camacho et al. 2018: median Rt > 2 in
#     13 of 23 governorates; 2.2 [2.1, 2.3] initial-stage estimate from
#     culture-confirmed cases),
#   * stabilising between 1 and 2 from late May, near 1 through the summer,
#   * below 1 by October 2017 (day ~130 here) (Camacho et al. 2018).
# The human/environmental split is NOT reported anywhere for Yemen, so two
# opposite splits are generated; whether a method can tell them apart is the
# identifiability question.
def lit_R(t):
    t = np.asarray(t, float)
    return 0.85 + 1.15 * np.exp(-t / 35.0) + 0.12 * np.exp(-((t - 95.0) / 30.0) ** 2)


TRUTHS = {"lit_share20": 0.20, "lit_share70": 0.70}


def truth_betas(t, share, S_ref=1.0):
    kap, J1, J2 = seip.ngm_integrals()
    tot = lit_R(t) / (S_ref * kap * J1)
    return tot * (1.0 - share), tot * share / (seip.ALPHA0 * J2)


def estimate_nb_k(panel):
    """Negative-binomial size for one governorate, from the rows it is given:
    method of moments around a penalised-spline Poisson fit, which tracks the
    epidemic curve without any SEIP structure, so k measures noise about the
    trend and not misfit of a particular arm."""
    sp = baselines.SplineBaseline().fit(panel.t0, panel.t1, panel.dC, float(panel.pop[0]))
    mu = sp.predict(panel.t0, panel.t1) * float(panel.pop[0]) * (panel.t1 - panel.t0)
    return fit.nb_size(panel.dC, mu)


def synthesize(panel, share, rng, ic_opts, dt=0.25, k=None):
    """Counts from the single-location model under a known beta pair, on this
    panel's own intervals and population: negative binomial with the city's own
    size k (Poisson if k is None), so the synthetic noise matches the record."""
    g1 = seip.Grid(dt=dt, nx=1)
    ic = seip.build_initial_condition(g1, np.array([0.0]), panel.y_first,
                                      panel.cum_first, **ic_opts)
    b10, b20 = truth_betas(g1.t, share, S_ref=float(ic.S0[0]))
    H, Sbar, _ = seip.solve_forward(g1, ic, b10, b20, D0)
    A = data_io.build_interval_operator(g1, panel)
    mean_rate = A @ H[:, 0]
    N = float(panel.pop[0]); Ndt = N * (panel.t1 - panel.t0)
    lam = np.maximum(mean_rate * Ndt, 1e-12)
    c = (rng.poisson(lam) if k is None else
         rng.negative_binomial(k, k / (k + lam))).astype(float)
    p = copy.copy(panel)
    p.dC = c; p.dC_raw = c.copy(); p.y = c / Ndt; p.y_raw = p.y.copy()
    p.w_count = 1.0 / np.sqrt(np.maximum(c, 1.0)); p.w_count /= p.w_count.mean()
    p.w_rate = Ndt / np.sqrt(np.maximum(c, 1.0)); p.w_rate /= p.w_rate.mean()
    p.w = p.w_count.copy()
    R = seip.R0_of_t(b10, b20, Sbar, *seip.ngm_integrals())
    return p, dict(beta10=b10, beta20=b20, R=R, H=H[:, 0], mean_rate=mean_rate,
                   share=share)


def row_slice(panel, keep):
    p = copy.copy(panel)
    for a in ("t0", "t1", "y", "y_raw", "dC", "dC_raw", "w", "w_count", "w_rate", "gov"):
        v = getattr(panel, a, None)
        if v is not None and np.ndim(v) and len(v) == len(keep):
            setattr(p, a, np.asarray(v)[keep].copy())
    return p


def daily(x, dt):
    """Sub-sample a solver-grid series to whole days to keep files small."""
    step = int(round(1.0 / dt))
    return np.asarray(x)[::step].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["det", "spline", "sir", *PINN_ARMS])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--variant", default="base", choices=list(VARIANTS))
    ap.add_argument("--truth", default="", choices=["", *TRUTHS])
    ap.add_argument("--data-dir", default="data_real")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--pretrain", type=int, default=800)
    ap.add_argument("--lbfgs", type=int, default=120)
    ap.add_argument("--n-col", type=int, default=192)
    ap.add_argument("--det-nfev", type=int, default=200)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print("exists:", a.out); return

    cfg = {**DEFAULTS, **VARIANTS[a.variant]}
    ic_opts = dict(truncation_correction=cfg["truncation_correction"],
                   age_profile=cfg["age_profile"])
    t_all = time.time()
    g1 = seip.Grid(dt=a.dt, nx=1)
    full = fit_one.city_panel(a.city, a.data_dir, winsorize=cfg["winsorize"])
    full.set_weights("count")
    truth = None
    if a.truth:
        rng = np.random.default_rng(1000 + a.seed)
        k_real = estimate_nb_k(full) if cfg["objective"] == "nb" else None
        full, truth = synthesize(full, TRUTHS[a.truth], rng, ic_opts, a.dt, k=k_real)
        truth["k"] = k_real

    n = len(full.y)
    TE = np.zeros(n, bool)
    if a.fold >= 0:
        TE = (np.arange(n) % 4) == a.fold
        TE[0] = False
    TR = ~TE
    train = row_slice(full, TR)
    train.set_weights("count")
    N = float(full.pop[0])
    k_nb = None
    if cfg["objective"] == "nb":
        k_nb = estimate_nb_k(train)                 # training rows only
        full.nb_k = np.full(len(full.y), k_nb)
        train.nb_k = np.full(len(train.y), k_nb)
    kap, J1, J2 = seip.ngm_integrals()
    out = dict(city=a.city, arm=a.arm, seed=a.seed, fold=a.fold,
               variant=a.variant, truth_name=a.truth, cfg=cfg,
               t0=full.t0, t1=full.t1, y=full.y, c=full.dC, N=N, TR=TR, TE=TE,
               t_day=g1.t[::int(round(1 / a.dt))], nb_k=k_nb)

    if a.arm == "det":
        np.random.seed(a.seed)
        prob = fit.Problem(g1, train, objective=cfg["objective"],
                           observable=cfg["observable"], **ic_opts)
        _, _, _, seed_b = fit.grid_search(prob, D0, n=11, verbose=False)
        res = fit.fit(prob, D0, M=cfg["M"], seed_beta=seed_b, n_jobs=1,
                      max_nfev=a.det_nfev, label="det", verbose=False)
        b10, b20, R, S = res.beta10_t, res.beta20_t, res.R0_t, res.Sbar
        pf = fit.Problem(g1, full, objective=cfg["objective"],
                         observable=cfg["observable"], **ic_opts)
        _, _, yhat, _ = pf.predict(b10, b20, D0)
        _, _, rec = seip.solve_forward(g1, prob.ic, b10, b20, D0,
                                       observable=cfg["observable"], record_full=True)
        # Delta-method band on log beta and log R, propagated LINEARLY through
        # the Gauss-Newton covariance (inflated by the fitted dispersion, i.e.
        # quasi-Poisson). Sampling theta instead and pushing draws through the
        # softplus is wrong here: the covariance is near-singular along the
        # beta1/beta2 trade-off, and finite draws along it swing R by orders of
        # magnitude although, to first order, R does not move along it at all.
        unc = res.meta["uncertainty"]
        bm = fit.BetaModel(M=cfg["M"])
        X = bm.design(g1.t)
        ev, V = unc["eigenvalues"], unc["eigenvectors"]
        inv = np.where(ev > ev[0] * 1e-14, 1.0 / np.maximum(ev, 1e-300), 0.0)
        cov = unc["sigma2"] * (V * inv) @ V.T
        step = int(round(1 / a.dt))
        th0 = res.theta

        def logs(th):
            c1, c2 = bm.eval(th, g1.t, X)
            r_ = seip.R0_of_t(c1, c2, S, kap, J1, J2)
            return np.log(np.concatenate([c1[::step], c2[::step], r_[::step]]))
        f0 = logs(th0)
        G = np.empty((f0.size, th0.size))
        for j in range(th0.size):
            h = 1e-5 * max(1.0, abs(th0[j]))
            e = np.zeros_like(th0); e[j] = h
            G[:, j] = (logs(th0 + e) - logs(th0 - e)) / (2 * h)
        sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", G, cov, G), 0.0))
        nd = f0.size // 3

        def q(k):
            m_, s_ = f0[k * nd:(k + 1) * nd], sd[k * nd:(k + 1) * nd]
            return np.exp(np.stack([m_ - 1.96 * s_, m_, m_ + 1.96 * s_])).astype(np.float32)
        out.update(band_beta10=q(0), band_beta20=q(1), band_R=q(2),
                   theta=th0, cov=cov,
                   converged=res.converged, cond=unc["cond"],
                   I_surface=rec["I_full"][::4, :, 0].astype(np.float32),
                   n_params=int(bm.nparam))
    elif a.arm in PINN_ARMS:
        S0, _, _, Iagg0, Pagg0, _ = pinn.city_initial_condition(train, 0, g1, **ic_opts)
        model = pinn.CityPINN(kind=a.arm, width=64, depth=4, M=cfg["M"],
                              scaleI=max(Iagg0 / seip.A_MAX, 1e-9),
                              scaleP=max(Pagg0 / seip.B_MAX, 1e-9), seed=a.seed,
                              hard_ic=True, S0=S0,
                              I0=max(Iagg0 / seip.A_MAX, 1e-12),
                              P0=max(Pagg0 / seip.B_MAX, 1e-12))
        data = pinn.CityData(train, 0, objective=cfg["objective"])
        hist = pinn.train_city(model, data, S0, Iagg0, Pagg0, n_col=a.n_col,
                               epochs=a.epochs, pretrain=a.pretrain, lbfgs=a.lbfgs,
                               log_every=max(a.epochs // 4, 1), w_ren=cfg["w_ren"],
                               w_pde=cfg["w_pde"], verbose=False)
        dfull = pinn.CityData(full, 0, objective=cfg["objective"])
        with torch.no_grad():
            yhat = dfull.average(model.observable(dfull.eval_times())).numpy()
        pred = pinn.city_predict(model, g1.t, g1.a, g1.b)
        b10, b20, S = pred["b10"], pred["b20"], pred["S"]
        R = pinn.R0_from(model, g1.t, kap, J1, J2)
        out.update(I_surface=pred["I_surface"][::4].astype(np.float32),
                   loss_final=float(hist[-1]) if hist else np.nan,
                   n_params=int(sum(p.numel() for p in model.parameters())))
    elif a.arm == "spline":
        sp = baselines.SplineBaseline().fit(train.t0, train.t1, train.dC, N, k=k_nb)
        yhat = sp.predict(full.t0, full.t1)
        b10 = b20 = R = S = None
        out.update(rate_day=sp.rate(out["t_day"]).astype(np.float32),
                   edf=sp.edf, pen=sp.pen, n_params=sp.edf)
    else:   # sir
        sb = baselines.SIRBaseline(M=cfg["M"], dt=a.dt).fit(
            train.t0, train.t1, train.dC, N, full.y_first[0], full.cum_first[0], k=k_nb)
        yhat = sb.predict(full.t0, full.t1)
        b10, b20, R, S = sb.b, None, sb.R, sb.S
        out.update(n_params=len(sb.theta))

    if a.fold >= 0 or a.truth or a.seed > 0 or a.variant != "base":
        out.pop("I_surface", None)       # only the main seed-0 fit is plotted
    out["yhat"] = np.asarray(yhat, float)
    for k, v in (("beta10", b10), ("beta20", b20), ("R", R), ("S", S)):
        out[k] = None if v is None else daily(v, a.dt)
    if truth is not None:
        out["truth"] = {k: (daily(v, a.dt) if k in ("beta10", "beta20", "R", "H") else v)
                        for k, v in truth.items()}
    out["seconds"] = time.time() - t_all
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, a.out)
    e_in = np.mean(np.abs(out["yhat"][TR] - full.y[TR]))
    e_out = np.mean(np.abs(out["yhat"][TE] - full.y[TE])) if TE.any() else np.nan
    print(f"{a.city} {a.arm} s{a.seed} f{a.fold} {a.variant} {a.truth}: "
          f"in {e_in:.3e} out {e_out:.3e}  {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
