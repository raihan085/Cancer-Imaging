"""
analysis.py -- tables and figures for the extended Yemen study, built only from
the result files written by run_job.py and spatial_fit.py. Nothing here fits
anything, so the notebook can re-render every number in seconds.

Items, in the order the notebook presents them:
    seeds        10-seed spread of every PINN arm on the full record
    holdout      rotated 4-fold interleaved holdout, all arms and baselines
    bands        beta1(t), beta2(t), R(t) with uncertainty bands
    recovery     literature-calibrated synthetic truth: what comes back
    literature   fitted R(t) against published Yemen estimates
    cori         model-free reference R(t) (Cori / EpiEstim)
    sensitivity  fixed modelling choices
    diagnostics  residual checks
    all cities   deterministic arm and baselines at all 21 governorates
    spatial      shared beta(t) against beta(t, x)
    surfaces     observed against fitted incidence surfaces
"""
from __future__ import annotations

import glob, os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, LinearSegmentedColormap
from scipy import stats

import seip, data_io, epiestim

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.bbox": "tight",
})

from make_jobs import FOCUS, N_SEEDS, N_SEEDS_AUX      # the study design, defined once


def focus_notes():
    """Label each focus governorate by its rank in total reported cases."""
    try:
        c = pd.read_csv("data_real/cases_long.csv")
        rank = c.groupby("governorate").new_cases.sum().rank(ascending=False).astype(int)
        n = len(rank)
        lab = lambda r: ("median" if r == (n + 1) // 2 else "last" if r == n else "") 
        return {g: f"rank {rank[g]}" + (f", {lab(rank[g])}" if lab(rank[g]) else "")
                for g in FOCUS}
    except Exception:
        return {g: "" for g in FOCUS}


FOCUS_NOTE = focus_notes()
PINN = ["dpinn", "dpinn_fs", "fdpinn", "ff"]
ARM = {  # key: (label, colour, linestyle)
    "det": ("Deterministic", "#555555", "-"),
    "sir": ("SIR baseline", "#CC79A7", "-."),
    "spline": ("Spline baseline", "#E69F00", ":"),
    "dpinn": ("DPINN", "#0072B2", "--"),
    "dpinn_fs": ("DPINN-FS", "#56B4E9", "-."),
    "fdpinn": ("F-DPINN", "#009E73", ":"),
    "ff": ("FF-PINN", "#D55E00", "-"),
}
ORDER = ["det", "sir", "spline", "dpinn", "dpinn_fs", "fdpinn", "ff"]
DIVERGE = LinearSegmentedColormap.from_list(
    "bwr_grey", ["#2166AC", "#7FA8CE", "#D9D9D9", "#E8977A", "#B2182B"])
# Published reference points for the second wave (see the notebook's sources).
LIT = dict(
    surge_R=(2.0, "Camacho et al. 2018: median Rt > 2 in 13/23 governorates, May 2017 surge"),
    initial_R=((2.1, 2.3), "2.2 [2.1, 2.3], initial stage, culture-confirmed cases"),
    summer_R=((1.0, 2.0), "Camacho et al. 2018: Rt stabilised between 1 and 2 after late May"),
    october_R=(1.0, "Camacho et al. 2018: Rt below 1 by October 2017"),
    model_R0=(1.344, "compartmental model with limited medical resources: R0 = 1.344"),
    attack_per_10k={"Amran": 902.3, "Al Mahwit": 878.9, "Al Dhale'e": 639.8,
                    "Dhamar": 524.9},
)
DAY0 = pd.Timestamp("2017-05-22")


def slug(c):
    return c.lower().replace(" ", "_").replace("'", "")


def day_of(date):
    return (pd.Timestamp(date) - DAY0).days


# ------------------------------------------------------------------ loading
def load(root="results"):
    """Every run_job.py result, as a list of dicts with a flat index frame."""
    recs = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*.pkl"))):
        if "/spatial/" in p:
            continue
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        d["path"] = p
        recs.append(d)
    idx = pd.DataFrame([dict(city=d["city"], arm=d["arm"], seed=d["seed"],
                             fold=d["fold"], variant=d["variant"],
                             truth=d["truth_name"], i=i) for i, d in enumerate(recs)])
    return recs, idx


def select(recs, idx, **kw):
    m = np.ones(len(idx), bool)
    for k, v in kw.items():
        m &= (idx[k].isin(v) if isinstance(v, (list, tuple, set)) else idx[k] == v).to_numpy()
    return [recs[i] for i in idx.i[m]]


def errs(yhat, y):
    e = np.asarray(yhat) - np.asarray(y)
    rms = np.sqrt(np.mean(np.asarray(y) ** 2))
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                nRMSE=float(np.sqrt(np.mean(e ** 2)) / rms) if rms > 0 else np.nan)


def r_summary(d):
    if d.get("R") is None:
        return dict(R_mean=np.nan, R_min=np.nan, R_max=np.nan, env_share=np.nan)
    R = np.asarray(d["R"], float)
    out = dict(R_mean=float(R.mean()), R_min=float(R.min()), R_max=float(R.max()))
    if d.get("beta20") is not None:
        J2 = seip.ngm_integrals()[2]
        env = seip.ALPHA0 * np.asarray(d["beta20"], float) * J2
        out["env_share"] = float(np.mean(env / (np.asarray(d["beta10"], float) + env)))
    else:
        out["env_share"] = np.nan
    return out


# ------------------------------------------------------------------ 1. seeds
def seed_table(recs, idx, cities=FOCUS):
    rows = []
    for d in select(recs, idx, fold=-1, variant="base", truth="", city=list(cities)):
        r = dict(city=d["city"], arm=d["arm"], seed=d["seed"], **errs(d["yhat"], d["y"]),
                 **r_summary(d), seconds=d["seconds"])
        rows.append(r)
    return pd.DataFrame(rows)


def seed_summary(tab):
    g = tab.groupby(["city", "arm"])
    s = g.agg(n_seeds=("seed", "nunique"), MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
              nRMSE_mean=("nRMSE", "mean"), R_mean=("R_mean", "mean"),
              R_max_med=("R_max", "median"), R_max_lo=("R_max", "min"),
              R_max_hi=("R_max", "max"), env_mean=("env_share", "mean"),
              env_sd=("env_share", "std"))
    s["MAE_cv"] = s.MAE_sd / s.MAE_mean
    return s.reset_index()


def fig_seed_spread(tab, path, cities=FOCUS):
    arms = [a for a in ORDER if a in set(tab.arm)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, col, lab, logy in ((axes[0], "MAE", "in-sample MAE", True),
                               (axes[1], "R_max", "max R(t) over the window", True),
                               (axes[2], "env_share", "environmental share", False)):
        w = 0.8 / len(arms)
        for k, a in enumerate(arms):
            for j, c in enumerate(cities):
                v = tab[(tab.city == c) & (tab.arm == a)][col].dropna().values
                if not len(v):
                    continue
                x = j + (k - (len(arms) - 1) / 2) * w
                ax.scatter(np.full(len(v), x) + np.random.default_rng(j).uniform(-w / 5, w / 5, len(v)),
                           v, s=9, color=ARM[a][1], alpha=0.75,
                           label=ARM[a][0] if j == 0 else None)
                ax.plot([x - w / 2.5, x + w / 2.5], [np.median(v)] * 2, color=ARM[a][1], lw=2)
        ax.set_xticks(range(len(cities))); ax.set_xticklabels(cities, fontsize=8, rotation=10)
        if logy:
            ax.set_yscale("log")
        if col == "R_max":
            ax.axhline(1, color="k", lw=0.8, ls=":")
        ax.set_title(lab, loc="left"); ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5, framealpha=0.9)
    fig.suptitle("Spread across 10 random seeds (each dot is one seed; bar = median)",
                 y=1.02, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------ 2. holdout
def cv_table(recs, idx, cities=None):
    """Pool the 4 folds: every interval except the IC anchor is held out once.
    One row per (city, arm, seed) with the pooled held-out and mean in-sample
    errors. Rows whose 4 folds are not all present are dropped."""
    sel = select(recs, idx, variant="base", truth="")
    by = {}
    for d in sel:
        if d["fold"] < 0 or (cities and d["city"] not in cities):
            continue
        by.setdefault((d["city"], d["arm"], d["seed"]), {})[d["fold"]] = d
    rows = []
    for (c, a, s), F in by.items():
        if len(F) < 4:
            continue
        y = F[0]["y"]
        yh_out = np.full(len(y), np.nan)
        e_in = []
        for f, d in F.items():
            yh_out[d["TE"]] = d["yhat"][d["TE"]]
            e_in.append(errs(d["yhat"][d["TR"]], y[d["TR"]])["MAE"])
        m = np.isfinite(yh_out)
        o = errs(yh_out[m], y[m])
        rows.append(dict(city=c, arm=a, seed=s, out_MAE=o["MAE"], out_RMSE=o["RMSE"],
                         out_nRMSE=o["nRMSE"], in_MAE=float(np.mean(e_in)),
                         degradation=o["MAE"] / float(np.mean(e_in)),
                         y_mean=float(np.mean(y)), n_out=int(m.sum())))
    return pd.DataFrame(rows)


def cv_summary(cv):
    s = cv.groupby(["city", "arm"]).agg(
        n_seeds=("seed", "nunique"), in_MAE=("in_MAE", "mean"),
        out_MAE=("out_MAE", "mean"), out_MAE_sd=("out_MAE", "std"),
        out_MAE_lo=("out_MAE", "min"), out_MAE_hi=("out_MAE", "max"),
        degradation=("degradation", "mean")).reset_index()
    det = s[s.arm == "det"].set_index("city").out_MAE
    s["vs_det"] = s.out_MAE / s.city.map(det)
    return s


def paired_vs_det(cv):
    """Seed-level held-out MAE of each arm against the deterministic arm (which
    has no seed), as a one-sample Wilcoxon test on log ratios, per city."""
    rows = []
    det = cv[cv.arm == "det"].set_index("city").out_MAE
    for (c, a), g in cv[cv.arm.isin(PINN)].groupby(["city", "arm"]):
        if c not in det.index or len(g) < 3:
            continue
        lr = np.log(g.out_MAE.values / det[c])
        p = stats.wilcoxon(lr).pvalue if len(lr) >= 5 and np.any(lr != 0) else np.nan
        rows.append(dict(city=c, arm=a, n=len(lr), ratio_median=float(np.exp(np.median(lr))),
                         frac_seeds_better=float(np.mean(lr < 0)), p_wilcoxon=p))
    return pd.DataFrame(rows)


def fig_cv(cvs, path, cities=FOCUS):
    arms = [a for a in ORDER if a in set(cvs.arm)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3), sharey=True)
    w = 0.8 / len(arms)
    for ax, col, lo, hi, lab in ((axes[0], "in_MAE", None, None, "in-sample MAE (mean of 4 folds)"),
                                 (axes[1], "out_MAE", "out_MAE_lo", "out_MAE_hi",
                                  "held-out MAE, pooled over 4 folds")):
        for k, a in enumerate(arms):
            g = cvs[cvs.arm == a].set_index("city").reindex(cities)
            x = np.arange(len(cities)) + (k - (len(arms) - 1) / 2) * w
            ax.bar(x, g[col].values, width=w * 0.9, color=ARM[a][1], label=ARM[a][0],
                   edgecolor="white", lw=0.5)
            if lo:
                ok = g[lo].notna() & (g.n_seeds > 1)
                ax.errorbar(x[ok.values], g[col].values[ok.values],
                            yerr=[g[col].values[ok.values] - g[lo].values[ok.values],
                                  g[hi].values[ok.values] - g[col].values[ok.values]],
                            fmt="none", ecolor="#222", lw=0.8, capsize=1.5)
        ax.set_yscale("log"); ax.set_xticks(range(len(cities)))
        ax.set_xticklabels([f"{c}\n({FOCUS_NOTE.get(c, '')})" for c in cities], fontsize=7.6)
        ax.set_title(lab, loc="left"); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAE (person$^{-1}$ d$^{-1}$)")
    axes[0].legend(fontsize=7, ncol=2, framealpha=0.9)
    fig.suptitle("Rotated 4-fold interleaved holdout. Whiskers: range over seeds.",
                 y=1.02, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------ 3. bands
def ensemble(recs, idx, city, arm, key, **kw):
    ds = select(recs, idx, city=city, arm=arm, fold=-1, variant=kw.get("variant", "base"),
                truth=kw.get("truth", ""))
    if not ds or ds[0].get(key) is None:
        return None, None
    M = np.array([np.asarray(d[key], float) for d in ds])
    return ds[0]["t_day"], M


def band_of(recs, idx, city, arm, key, **kw):
    """(t, lo, mid, hi, kind). PINNs: 2.5/50/97.5 percentiles over seeds.
    Deterministic: delta-method band from the Gauss-Newton covariance."""
    if arm == "det":
        ds = select(recs, idx, city=city, arm="det", fold=-1,
                    variant=kw.get("variant", "base"), truth=kw.get("truth", ""))
        ds = [d for d in ds if d["seed"] == 0] or ds
        if not ds:
            return None
        d = ds[0]
        b = d.get({"beta10": "band_beta10", "beta20": "band_beta20", "R": "band_R"}[key])
        if b is None:
            return None
        return d["t_day"], b[0], np.asarray(d[key], float), b[2], "delta"
    t, M = ensemble(recs, idx, city, arm, key, **kw)
    if M is None:
        return None
    lo, mid, hi = np.percentile(M, [2.5, 50, 97.5], axis=0)
    return t, lo, mid, hi, f"{len(M)} seeds"


def fig_bands(recs, idx, path, cities=FOCUS, arms=("det", "dpinn", "ff"), cori=None):
    keys = [("beta10", r"$\beta_1(t)$  human route"),
            ("beta20", r"$\beta_2(t)$  environmental route"),
            ("R", r"$\mathcal{R}(t)$")]
    fig, axes = plt.subplots(3, len(cities), figsize=(3.4 * len(cities), 9.0), sharex=True)
    axes = np.atleast_2d(axes)
    for j, c in enumerate(cities):
        for i, (key, lab) in enumerate(keys):
            ax = axes[i, j]
            for a in arms:
                b = band_of(recs, idx, c, a, key)
                if b is None:
                    continue
                t, lo, mid, hi, kind = b
                # draw at most x1000 either side of the curve: a band wider than
                # that is uninformative, and leaving it uncapped (a beta on its
                # positivity floor gives 1e-48..1e40) flattens every other curve
                lo = np.maximum(lo, mid / 1e3); hi = np.minimum(hi, mid * 1e3)
                ax.fill_between(t, lo, hi, color=ARM[a][1], alpha=0.18, lw=0)
                ax.plot(t, mid, color=ARM[a][1], ls=ARM[a][2], lw=1.2,
                        label=f"{ARM[a][0]} ({kind})")
            if key == "R":
                ax.axhline(1, color="k", lw=0.8, ls=":")
                if cori is not None and c in cori:
                    r = cori[c]
                    ax.fill_between(r["t"], r["lo"], r["hi"], color="k", alpha=0.12, lw=0)
                    ax.plot(r["t"], r["mean"], color="k", lw=1.0, label="Cori (model-free)")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"{c}  ({FOCUS_NOTE.get(c, '')})", loc="left", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 2:
                ax.set_xlabel("t (days from 2017-05-22)")
            if i == 2 and j == len(cities) - 1:
                ax.legend(fontsize=6.6, framealpha=0.9, loc="upper right")
    fig.suptitle("Transmission rates and reproduction number with uncertainty bands. "
                 "PINN bands: 95% range over 10 seeds; deterministic: 95% delta-method "
                 "band (negative binomial, dispersion-scaled). Bands drawn at most x1000 around the curve.", y=1.005, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------ 4/5. truth, literature, Cori
def recovery_table(recs, idx):
    rows = []
    for d in select(recs, idx, variant="base", fold=-1):
        if not d["truth_name"] or d.get("R") is None:
            continue
        tr = d["truth"]
        R, Rt = np.asarray(d["R"], float), np.asarray(tr["R"], float)
        J2 = seip.ngm_integrals()[2]
        b1, b2 = np.asarray(d["beta10"], float), np.asarray(d["beta20"], float)
        env = seip.ALPHA0 * b2 * J2
        share = float(np.mean(env / (b1 + env)))
        rel = lambda est, true: float(np.median(np.abs(est - true) / np.abs(true)))
        rows.append(dict(city=d["city"], arm=d["arm"], seed=d["seed"],
                         truth=d["truth_name"], true_share=tr["share"], est_share=share,
                         R_relerr=rel(R, Rt),
                         beta1_relerr=rel(b1, np.asarray(tr["beta10"], float)),
                         beta2_relerr=rel(b2, np.asarray(tr["beta20"], float)),
                         R_true_max=float(Rt.max()), R_est_max=float(R.max()),
                         fit_MAE=errs(d["yhat"], d["y"])["MAE"],
                         truth_MAE=errs(tr["mean_rate"], d["y"])["MAE"]))
    return pd.DataFrame(rows)


def fig_recovery(recs, idx, rec_tab, path, cities=FOCUS):
    truths = sorted(set(rec_tab.truth))
    fig, axes = plt.subplots(len(truths), len(cities) + 1,
                             figsize=(3.2 * (len(cities) + 1), 3.3 * len(truths)))
    axes = np.atleast_2d(axes)
    for i, tr in enumerate(truths):
        for j, c in enumerate(cities):
            ax = axes[i, j]
            ds = select(recs, idx, city=c, truth=tr, fold=-1)
            if not ds:
                ax.set_visible(False); continue
            ax.plot(ds[0]["t_day"], ds[0]["truth"]["R"], color="k", lw=2, label="truth")
            for a in ("det", "dpinn", "ff"):
                for k, d in enumerate([d for d in ds if d["arm"] == a]):
                    ax.plot(d["t_day"], d["R"], color=ARM[a][1], ls=ARM[a][2], lw=0.9,
                            alpha=0.8, label=ARM[a][0] if k == 0 else None)
            ax.axhline(1, color="k", lw=0.6, ls=":"); ax.set_yscale("log")
            ax.set_title(f"{c}: {tr}", loc="left", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(r"$\mathcal{R}(t)$")
            if i == 0 and j == 0:
                ax.legend(fontsize=6.5)
        ax = axes[i, -1]
        g = rec_tab[rec_tab.truth == tr]
        for a in ("det", "dpinn", "ff"):
            v = g[g.arm == a].est_share
            ax.scatter(np.full(len(v), ORDER.index(a)), v, color=ARM[a][1], s=14)
        ax.axhline(g.true_share.iloc[0] if len(g) else np.nan, color="k", lw=1.5,
                   label="true share")
        ax.set_ylim(0, 1); ax.set_xticks([ORDER.index(a) for a in ("det", "dpinn", "ff")])
        ax.set_xticklabels(["Det.", "DPINN", "FF"]); ax.set_title("environmental share", loc="left")
        ax.legend(fontsize=7)
    fig.suptitle("Recovery of a literature-calibrated transmission history from synthetic "
                 "counts on each city's real bulletin calendar", y=1.01, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def cori_all(panel, cities, si=epiestim.SI_PRIMARY):
    out = {}
    for c in cities:
        gi = panel.names.index(c)
        out[c], _ = epiestim.cori_for_panel(panel, gi, si=si)
    out["National"], _ = epiestim.cori_for_panel(panel, None, si=si)
    return out


def cori_agreement(recs, idx, cori, cities=FOCUS, arms=("det", "sir", *PINN)):
    rows = []
    for c in cities:
        r = cori[c]
        ok = np.isfinite(r["mean"])
        for a in arms:
            ds = select(recs, idx, city=c, arm=a, fold=-1, variant="base", truth="")
            ds = [d for d in ds if d.get("R") is not None]
            if not ds:
                continue
            R = np.median([np.asarray(d["R"], float) for d in ds], axis=0)
            Rm = np.interp(r["t"][ok], ds[0]["t_day"], R)
            rows.append(dict(city=c, arm=a, n_seeds=len(ds),
                             mean_abs_diff=float(np.mean(np.abs(Rm - r["mean"][ok]))),
                             log_ratio_median=float(np.median(np.log(Rm / r["mean"][ok]))),
                             frac_in_CrI=float(np.mean((Rm >= r["lo"][ok]) & (Rm <= r["hi"][ok]))),
                             spearman=float(stats.spearmanr(Rm, r["mean"][ok]).statistic)))
    return pd.DataFrame(rows)


def literature_table(recs, idx, cori, cities=FOCUS, arms=("det", "sir", *PINN)):
    """Fitted R in the windows where published estimates exist."""
    win = {"early (d 0-20, late May-June)": (0, 20),
           "summer (d 30-100)": (30, 100),
           "October (d 132-162)": (day_of("2017-10-01"), day_of("2017-10-31"))}
    rows = []
    for c in list(cities) + ["National"]:
        for a in (["cori"] + list(arms)):
            if a == "cori":
                r = cori.get(c)
                if r is None:
                    continue
                t, R = r["t"], r["mean"]
            elif c == "National":
                continue
            else:
                ds = select(recs, idx, city=c, arm=a, fold=-1, variant="base", truth="")
                ds = [d for d in ds if d.get("R") is not None]
                if not ds:
                    continue
                t, R = ds[0]["t_day"], np.median([d["R"] for d in ds], axis=0)
            row = dict(city=c, method="Cori (SI 5 d)" if a == "cori" else ARM[a][0])
            for nm, (lo, hi) in win.items():
                m = (t >= lo) & (t <= hi)
                row[nm] = float(np.nanmean(R[m])) if m.any() else np.nan
            rows.append(row)
    tab = pd.DataFrame(rows)
    return tab


def attack_rates(panel):
    rows = []
    for g, c in enumerate(panel.names):
        m = panel.gov == g
        ar = 1e4 * panel.dC_raw[m].sum() / panel.pop[g]
        rows.append(dict(governorate=c, record_attack_per_10k=ar,
                         WHO_EMRO_2nd_wave_per_10k=LIT["attack_per_10k"].get(c, np.nan)))
    return pd.DataFrame(rows).sort_values("record_attack_per_10k", ascending=False)


def fig_cori(cori, cori_short, path, cities=("National", *FOCUS)):
    fig, axes = plt.subplots(1, len(cities), figsize=(3.0 * len(cities), 3.4), sharey=True)
    for ax, c in zip(axes, cities):
        for r, col, lab in ((cori[c], "k", epiestim.SI_PRIMARY["label"]),
                            (cori_short[c], "#D55E00", epiestim.SI_SHORT["label"])):
            ax.fill_between(r["t"], r["lo"], r["hi"], color=col, alpha=0.15, lw=0)
            ax.plot(r["t"], r["mean"], color=col, lw=1.1, label=lab)
        ax.axhline(1, color="k", lw=0.7, ls=":")
        ax.axhspan(1, 2, color="#009E73", alpha=0.06)
        ax.axvline(day_of("2017-10-01"), color="#888", lw=0.7, ls="--")
        ax.set_title(c, loc="left"); ax.set_xlabel("t (days)")
        ax.set_ylim(0.3, 4); ax.set_yscale("log")
    axes[0].set_ylabel(r"$R_t$ (Cori, 7-day window)")
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Model-free reference R(t). Green band: 1-2, the published range after "
                 "late May 2017; dashed line: 1 October 2017.", y=1.03, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------ 6. sensitivity
def sensitivity_table(recs, idx, cities=FOCUS):
    rows = []
    for d in select(recs, idx, fold=-1, truth="", city=list(cities)):
        if d["arm"] in PINN and d["seed"] != 0:
            continue
        if d["arm"] not in ("det", "dpinn", "ff"):
            continue
        rows.append(dict(city=d["city"], arm=d["arm"], variant=d["variant"],
                         **errs(d["yhat"], d["y"]), **r_summary(d)))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    base = t[t.variant == "base"].set_index(["city", "arm"])
    t["dMAE_%"] = [100 * (r.MAE / base.MAE.get((r.city, r.arm), np.nan) - 1) for r in t.itertuples()]
    t["dRmean_%"] = [100 * (r.R_mean / base.R_mean.get((r.city, r.arm), np.nan) - 1)
                     for r in t.itertuples()]
    return t


# ------------------------------------------------------------------ 7. diagnostics
def residual_diagnostics(d, max_lag=10):
    y, yh, c = np.asarray(d["y"]), np.asarray(d["yhat"]), np.asarray(d["c"])
    Ndt = d["N"] * (d["t1"] - d["t0"])
    lam = np.maximum(yh * Ndt, 1e-9)
    k = d.get("nb_k")
    var = lam if not k else lam + lam ** 2 / k
    pr_pois = (c - lam) / np.sqrt(lam)
    pr = (c - lam) / np.sqrt(var)
    n = len(pr); p = float(d.get("n_params") or 0)
    p_eff = p if 0 < p < n / 2 else 0.0
    phi = float(np.sum(pr ** 2) / max(n - p_eff, 1))
    phi_pois = float(np.sum(pr_pois ** 2) / max(n - p_eff, 1))
    r = pr - pr.mean()
    acf = np.array([np.sum(r[k:] * r[:-k]) / np.sum(r ** 2) for k in range(1, max_lag + 1)])
    Q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, max_lag + 1)))
    p_lb = float(stats.chi2.sf(Q, max_lag))
    # NB size by method of moments: Var = mu + mu^2/k
    excess = np.sum((c - lam) ** 2 - lam)
    k_nb = float(np.sum(lam ** 2) / excess) if excess > 0 else np.inf
    # randomized quantile residuals (Dunn & Smyth) for a Poisson mean lam
    rng = np.random.default_rng(0)
    if not k:
        a = stats.poisson.cdf(c - 1, lam); b = stats.poisson.cdf(c, lam)
    else:
        pp = k / (k + lam)
        a = stats.nbinom.cdf(c - 1, k, pp); b = stats.nbinom.cdf(c, k, pp)
    u = np.clip(a + rng.uniform(size=n) * (b - a), 1e-12, 1 - 1e-12)
    rq = stats.norm.ppf(u)
    return dict(dispersion=phi, dispersion_vs_poisson=phi_pois, nb_k_used=k or np.inf,
                acf1=float(acf[0]), ljung_box_p=p_lb, nb_size=k_nb,
                rqr_sd=float(np.std(rq)), shapiro_p=float(stats.shapiro(rq).pvalue),
                pearson=pr, acf=acf, rqr=rq, n=n)


def diagnostics_table(recs, idx, cities=FOCUS):
    rows, keep = [], {}
    for d in select(recs, idx, fold=-1, variant="base", truth="", seed=0, city=list(cities)):
        dg = residual_diagnostics(d)
        keep[(d["city"], d["arm"])] = dg
        rows.append(dict(city=d["city"], arm=d["arm"],
                         **{k: dg[k] for k in ("nb_k_used", "dispersion", "dispersion_vs_poisson",
                                               "acf1", "ljung_box_p", "nb_size", "rqr_sd",
                                               "shapiro_p")}))
    return pd.DataFrame(rows), keep


def fig_diagnostics(keep, path, cities=FOCUS, arms=("det", "spline", "dpinn", "ff")):
    fig, axes = plt.subplots(3, len(cities), figsize=(3.3 * len(cities), 8.4))
    for j, c in enumerate(cities):
        for a in arms:
            dg = keep.get((c, a))
            if dg is None:
                continue
            col = ARM[a][1]
            axes[0, j].plot(np.arange(dg["n"]), dg["pearson"], color=col, lw=0.7,
                            alpha=0.85, label=ARM[a][0])
            axes[1, j].plot(np.arange(1, len(dg["acf"]) + 1), dg["acf"], "o-", ms=3,
                            color=col, lw=0.8)
            q = np.sort(dg["rqr"]); th = stats.norm.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q))
            axes[2, j].plot(th, q, ".", ms=2.5, color=col)
        n = next((keep[(c, a)]["n"] for a in arms if (c, a) in keep), 100)
        axes[1, j].axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color="k", alpha=0.08)
        axes[2, j].plot([-3.5, 3.5], [-3.5, 3.5], "k:", lw=0.8)
        axes[0, j].set_yscale("symlog", linthresh=3)
        axes[0, j].set_title(c, loc="left")
        axes[0, j].set_xlabel("interval index"); axes[1, j].set_xlabel("lag")
        axes[2, j].set_xlabel("normal quantile")
    axes[0, 0].set_ylabel("Pearson residual"); axes[1, 0].set_ylabel("ACF")
    axes[2, 0].set_ylabel("randomized quantile residual")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Residual diagnostics, full-record fits (seed 0), under the negative-binomial "
                 "variance each fit used. A model that fits has Pearson residuals of order 1, no "
                 "autocorrelation and randomized quantile residuals on the diagonal.",
                 y=1.01, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------ 9. all cities
def fig_all_cities_R(recs, idx, cori, names, path):
    n = len(names); ncol = 7; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.3 * nrow), sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), names):
        r = cori.get(c)
        if r is not None:
            ax.fill_between(r["t"], r["lo"], r["hi"], color="k", alpha=0.12, lw=0)
            ax.plot(r["t"], r["mean"], color="k", lw=0.8)
        for a in ("det", "sir"):
            b = band_of(recs, idx, c, a, "R")
            if b is not None:
                t, lo, mid, hi, _ = b
                ax.fill_between(t, lo, hi, color=ARM[a][1], alpha=0.2, lw=0)
                ax.plot(t, mid, color=ARM[a][1], lw=1.0, ls=ARM[a][2])
            elif a == "sir":
                ds = select(recs, idx, city=c, arm="sir", fold=-1, variant="base", truth="")
                if ds:
                    ax.plot(ds[0]["t_day"], ds[0]["R"], color=ARM[a][1], lw=0.9, ls=ARM[a][2])
        ax.axhline(1, color="k", lw=0.6, ls=":")
        ax.set_yscale("log"); ax.set_ylim(0.2, 6)
        ax.set_title(c, loc="left", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("R(t) at all 21 governorates: deterministic SEIP (grey, 95% band), SIR "
                 "baseline (pink), Cori reference (black, 95% CrI)", y=1.01, fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------ 10/11. spatial and surfaces
def load_spatial(root="results/spatial"):
    S = {}
    for p in sorted(glob.glob(os.path.join(root, "*.pkl"))):
        d = pickle.load(open(p, "rb"))
        S[os.path.basename(p)[:-4]] = d
    return S


def spatial_table(S):
    rows = []
    for k, d in S.items():
        e = errs(d["yhat"], d["y"])
        rows.append(dict(fit=k, model=d["model"], D=d["D"], mixing=d.get("mixing", np.nan),
                         n_params=d["nparam"],
                         deviance=d["deviance"], AIC=d["deviance"] + 2 * d["nparam"],
                         **e, converged=d["converged"], nfev=d["nfev"],
                         R_mean=float(np.mean(d["R_g"])), R_max=float(np.max(d["R_g"])),
                         seconds=d["seconds"]))
    t = pd.DataFrame(rows).sort_values(["D", "model"])
    return t


def fig_spatial_R(S, path, key="network", ref="shared_D50", names=None):
    d, r = S[key], S.get(ref)
    names = names or d["names"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    order = np.argsort(d["xg"])
    for k, g in enumerate(order):
        ax.plot(d["t_day"], d["R_g"][:, g], color=cmap(k / len(order)), lw=0.9,
                label=d["names"][g] if k % 4 == 0 else None)
    if r is not None:
        ax.plot(r["t_day"], np.median(r["R_g"], axis=1), "k--", lw=1.6, label="shared beta(t)")
    ax.axhline(1, color="k", lw=0.6, ls=":"); ax.set_yscale("log")
    ax.set_title(f"(a) R(t) at each governorate, {key} model (colour = distance)",
                 loc="left")
    ax.set_xlabel("t (days)"); ax.legend(fontsize=6.5, ncol=2)
    ax = axes[1]
    for nm, dd, col in (("shared beta(t), 1-D diffusion", r, "k"),
                        ("governorate_D50", S.get("governorate_D50"), "#0072B2"),
                        (key, d, "#D55E00")):
        if dd is None:
            continue
        bias = [np.mean(dd["yhat"][dd["gov"] == g]) / max(np.mean(dd["y"][dd["gov"] == g]), 1e-12)
                for g in range(len(dd["names"]))]
        ax.plot(dd["xg"], bias, "o", color=col, ms=5, label=nm, alpha=0.8)
    ax.axhline(1, color="k", lw=0.7, ls=":"); ax.set_yscale("log")
    for g in range(len(d["names"])):
        ax.annotate(d["names"][g], (d["xg"][g], 1), fontsize=5.5, rotation=90,
                    xytext=(0, 3), textcoords="offset points", color="#666")
    ax.set_xlabel("x (km from Sana'a)"); ax.set_ylabel("fitted mean / observed mean")
    ax.set_title("(b) level bias by governorate (1 = unbiased)", loc="left"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def interval_field(t0, t1, vals, gov, ng, t_day):
    """(days, ng) with each interval's value drawn across its own span."""
    Y = np.full((len(t_day), ng), np.nan)
    for a, b, v, g in zip(t0, t1, vals, gov):
        k0, k1 = int(round(a)), int(round(b))
        Y[k0 + 1:k1 + 1, g] = v
    return Y


def city_fields(recs, idx, panel, arm, names):
    """Assemble per-city full-record fits into a (days, ng) interval field.
    PINN arms use the median over seeds; cities without that arm are NaN."""
    t_day = np.arange(int(seip.T_MAX) + 1)
    Y = np.full((len(t_day), panel.ng), np.nan)
    for g, c in enumerate(panel.names):
        ds = select(recs, idx, city=c, arm=arm, fold=-1, variant="base", truth="")
        if not ds:
            continue
        yh = np.median([d["yhat"] for d in ds], axis=0)
        d = ds[0]
        F = interval_field(d["t0"], d["t1"], yh, np.zeros(len(yh), int), 1, t_day)
        Y[:, g] = F[:, 0]
    return t_day, Y


def fig_surfaces(panel, fields, path, title):
    """Observed against fitted interval fields on (t, governorate ordered by x),
    with signed relative residuals. Nothing interpolated between bulletins."""
    t_day = np.arange(int(seip.T_MAX) + 1)
    obs = interval_field(panel.t0, panel.t1, panel.y, panel.gov, panel.ng, t_day)
    order = np.argsort(panel.xg)
    names = [panel.names[g] for g in order]
    pos = obs[np.isfinite(obs) & (obs > 0)]
    nrm = LogNorm(vmin=max(np.percentile(pos, 1), 1e-8), vmax=np.nanmax(obs))
    n = len(fields) + 1
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 8.2), sharex=True, sharey=True)
    m = axes[0, 0].pcolormesh(t_day, np.arange(panel.ng), np.where(obs > 0, obs, np.nan)[:, order].T,
                              cmap="viridis", norm=nrm, shading="nearest")
    axes[0, 0].set_title("observed", loc="left")
    fig.colorbar(m, ax=axes[0, 0], pad=0.02, fraction=0.05)
    axes[1, 0].axis("off")
    for k, (nm, F) in enumerate(fields.items()):
        ax = axes[0, k + 1]
        m = ax.pcolormesh(t_day, np.arange(panel.ng), np.where(F > 0, F, np.nan)[:, order].T,
                          cmap="viridis", norm=nrm, shading="nearest")
        ax.set_title(nm, loc="left", fontsize=9)
        fig.colorbar(m, ax=ax, pad=0.02, fraction=0.05)
        rel = (F - obs) / np.nanmean(obs, axis=0, keepdims=True)
        v = np.nanpercentile(np.abs(rel), 98) if np.isfinite(rel).any() else 1
        ax = axes[1, k + 1]
        m = ax.pcolormesh(t_day, np.arange(panel.ng), rel[:, order].T, cmap=DIVERGE,
                          norm=TwoSlopeNorm(0, -v, v), shading="nearest")
        mae = np.nanmean(np.abs(F - obs))
        ax.set_title(f"(fitted - observed) / mean   MAE {mae:.2e}", loc="left", fontsize=8)
        fig.colorbar(m, ax=ax, pad=0.02, fraction=0.05)
    axes[0, 0].set_yticks(range(panel.ng)); axes[0, 0].set_yticklabels(names, fontsize=6)
    for ax in axes[1]:
        ax.set_xlabel("t (days from 2017-05-22)")
    fig.suptitle(title, y=1.005, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_surfaces_3d(panel, fields, path):
    """3-D renderings: observed and each fitted field, same axes and colour
    scale. Rows are governorates ordered by distance; the surface is the
    interval field, so steps are bulletin intervals, not interpolation."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    t_day = np.arange(int(seip.T_MAX) + 1)
    obs = interval_field(panel.t0, panel.t1, panel.y, panel.gov, panel.ng, t_day)
    order = np.argsort(panel.xg)
    allF = {"observed": obs, **fields}
    vmax = np.nanmax(obs)
    n = len(allF)
    fig = plt.figure(figsize=(4.6 * min(n, 3), 4.0 * int(np.ceil(n / 3))))
    T, G = np.meshgrid(t_day[::2], np.asarray(panel.xg)[order])
    for k, (nm, F) in enumerate(allF.items()):
        ax = fig.add_subplot(int(np.ceil(n / 3)), min(n, 3), k + 1, projection="3d")
        Z = np.nan_to_num(F[::2][:, order].T, nan=0.0)
        ax.plot_surface(T, G, Z, cmap="viridis", vmin=0, vmax=vmax, linewidth=0,
                        antialiased=True, rcount=60, ccount=120)
        ax.set_zlim(0, vmax); ax.view_init(elev=28, azim=-125)
        ax.set_title(nm, loc="left", fontsize=9)
        ax.set_xlabel("t (days)", fontsize=7); ax.set_ylabel("x (km)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Observed against fitted incidence surfaces I(t, x)", y=1.0, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def spatial_field(d, panel):
    t_day = np.arange(int(seip.T_MAX) + 1)
    return interval_field(d["t0"], d["t1"], d["yhat"], d["gov"], panel.ng, t_day)


def surface_errors(panel, fields, cities=None):
    """Cell-wise error of each fitted field against the observed one, over the
    governorates in `cities` (all by default) so every field is scored on the
    same cells."""
    t_day = np.arange(int(seip.T_MAX) + 1)
    obs = interval_field(panel.t0, panel.t1, panel.y, panel.gov, panel.ng, t_day)
    keep = np.zeros(panel.ng, bool)
    for g, c in enumerate(panel.names):
        keep[g] = cities is None or c in cities
    rows = []
    for nm, F in fields.items():
        m = np.isfinite(F) & np.isfinite(obs) & keep[None, :]
        e = F[m] - obs[m]
        rows.append(dict(field=nm, cells=int(m.sum()), MAE=float(np.mean(np.abs(e))),
                         RMSE=float(np.sqrt(np.mean(e ** 2))),
                         nRMSE=float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(obs[m] ** 2))),
                         corr=float(np.corrcoef(F[m], obs[m])[0, 1])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ mass balance
def mass_balance(recs, idx, panel, cities=FOCUS, arms=("det", *PINN)):
    """Susceptible depletion over the window against the cumulative attack rate.

    Every infection in this model is an observed case (H = S * force of
    infection is what a bulletin reports), so S(0) - S(T) must equal the
    reported attack rate up to recruitment and mortality (~1e-4 here). A
    solver satisfies it by construction; a soft-physics network need not.
    Returns one row per (city, arm) with the seed median and range."""
    rows = []
    for c in cities:
        g = panel.names.index(c)
        attack = float(panel.dC_raw[panel.gov == g].sum() / panel.pop[g])
        for a in arms:
            ds = select(recs, idx, city=c, arm=a, fold=-1, variant="base", truth="")
            ds = [d for d in ds if d.get("S") is not None]
            if not ds:
                continue
            drop = np.array([float(d["S"][0] - d["S"][-1]) for d in ds])
            rows.append(dict(city=c, arm=a, n=len(ds), attack_rate=attack,
                             S_drop_median=float(np.median(drop)),
                             S_drop_min=float(drop.min()), S_drop_max=float(drop.max()),
                             ratio_median=float(np.median(drop) / max(attack, 1e-12))))
    return pd.DataFrame(rows)
