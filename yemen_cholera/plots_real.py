"""
plots_real.py -- publication figures for the real-record DPINN vs FF-PINN study.

Conventions inherited from plots.py: colourblind-safe Okabe-Ito accents, a
perceptually uniform ramp for fields, a grey-midpoint diverging ramp for signed
error, model and data compared on the SAME footing (interval averages over each
reported interval, which is what the loss sees), and nothing interpolated
between bulletins.

There is no synthetic twin here and no true parameter vector. Every error is a
fitted quantity against the OBSERVED incidence; every parameter panel is a
comparison BETWEEN architectures, never against a known answer.
"""
from __future__ import annotations

import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import seip, data_io

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 170, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.bbox": "tight",
})

ARMS = [("det", "Deterministic", "#666666", "-"),
        ("dpinn", "DPINN", "#0072B2", "--"),
        ("ff", "FF-PINN", "#D55E00", "-")]
C_INK, C_MUT = "#222222", "#666666"
FIELD_CMAP = "viridis"
DIVERGE = LinearSegmentedColormap.from_list(
    "bwr_grey", ["#2166AC", "#7FA8CE", "#D9D9D9", "#E8977A", "#B2182B"])


def slug(c): return c.lower().replace(" ", "_").replace("'", "")


def load(cities, root="real_fits"):
    R = {}
    for c in cities:
        R[c] = {}
        for arm, *_ in ARMS:
            p = os.path.join(root, slug(c), f"{arm}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    R[c][arm] = pickle.load(f)
    return R


def _step(ax, t0, t1, y, **kw):
    """Draw each reported interval across its own span. No interpolation."""
    for i in range(len(y)):
        ax.plot([t0[i], t1[i]], [y[i], y[i]], **kw)
        kw.pop("label", None)


# --------------------------------------------------------------- R1
def fig_observed_surface(data_dir, path):
    """The observed incidence record as a (t, x) field, two honest renderings."""
    from scipy.interpolate import griddata
    panel = data_io.load_panel(data_dir, verbose=False)
    xs = np.asarray(panel.xg)[panel.gov]
    tm = 0.5 * (panel.t0 + panel.t1)
    y = panel.y

    fig = plt.figure(figsize=(13.4, 5.3))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ti = np.linspace(tm.min(), tm.max(), 170)
    xi = np.linspace(xs.min(), xs.max(), 170)
    TI, XI = np.meshgrid(ti, xi)
    ZI = griddata((tm, xs), y, (TI, XI), method="linear")
    ax.plot_surface(TI, XI, ZI, cmap=FIELD_CMAP, linewidth=0, antialiased=True,
                    rcount=90, ccount=90)
    ax.set_xlabel("t  (days from 2017-05-22)"); ax.set_ylabel("x  (km from Sana'a)")
    ax.set_zlabel("incidence  (person$^{-1}$d$^{-1}$)")
    ax.set_title("(a)  observed $I(t,x)$, interpolated for display only", loc="left")
    ax.view_init(elev=26, azim=-124)

    ax = fig.add_subplot(1, 2, 2)
    sc = ax.scatter(tm, xs, c=y, s=13, cmap=FIELD_CMAP,
                    norm=LogNorm(vmin=max(y[y > 0].min(), 1e-8), vmax=y.max()),
                    edgecolor="none")
    ax.scatter(tm[y == 0], xs[y == 0], s=13, facecolor="none",
               edgecolor="#b0b0b0", linewidth=0.5, label="reported zero")
    for g, nm in enumerate(panel.names):
        ax.axhline(panel.xg[g], color="#000000", alpha=0.05, linewidth=0.5)
    ax.set_xlabel("t  (days)"); ax.set_ylabel("x  (km from Sana'a)")
    ax.set_title("(b)  every reported bulletin, nothing interpolated", loc="left")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.colorbar(sc, ax=ax, pad=0.02, label="incidence (person$^{-1}$d$^{-1}$)")
    fig.suptitle("Yemen cholera second wave, 21 reporting governorates, "
                 f"{len(y)} bulletin intervals", y=1.005, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return panel


# --------------------------------------------------------------- R2 / R3
def fig_fit_and_error(R, path):
    """Observed against fitted, and the signed residual, for every city."""
    cities = list(R)
    n = len(cities)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.2), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, c in enumerate(cities):
        obs = R[c]["ff"]["observed"]
        t0, t1, y = obs["t0"], obs["t1"], obs["y"]
        ax = axes[0, j]
        _step(ax, t0, t1, y, color=C_INK, linewidth=1.6, label="observed",
              solid_capstyle="butt")
        for arm, nice, col, ls in ARMS:
            if arm not in R[c]:
                continue
            r = R[c][arm]["row"]
            _step(ax, t0, t1, R[c][arm]["yhat"], color=col, linewidth=1.15, ls=ls,
                  label=f"{nice}  MAE {r['MAE']:.2e}", solid_capstyle="butt")
        ax.set_title(c, loc="left")
        ax.set_yscale("symlog", linthresh=1e-6)
        if j == 0:
            ax.set_ylabel("incidence  (person$^{-1}$d$^{-1}$)")
        ax.legend(fontsize=6.6, loc="upper right", framealpha=0.9)

        ax = axes[1, j]
        ax.axhline(0, color=C_MUT, linewidth=0.7)
        for arm, nice, col, ls in ARMS:
            if arm not in R[c]:
                continue
            e = R[c][arm]["yhat"] - y
            r = R[c][arm]["row"]
            _step(ax, t0, t1, e, color=col, linewidth=1.1, ls=ls,
                  label=f"{nice}  RMSE {r['RMSE']:.2e}", solid_capstyle="butt")
        ax.set_xlabel("t  (days)")
        if j == 0:
            ax.set_ylabel("fitted $-$ observed")
        ax.legend(fontsize=6.6, loc="upper right", framealpha=0.9)
    fig.suptitle("Fitted against observed incidence, and the signed residual "
                 "(no reference solution is used)", y=1.003, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------- R4
def fig_beta(R, path):
    cities = list(R)
    n = len(cities)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.6), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, c in enumerate(cities):
        for row, key, lab in ((0, "beta10", r"$\beta_{1}(t)$  human route"),
                              (1, "beta20", r"$\beta_{2}(t)$  environmental route")):
            ax = axes[row, j]
            for arm, nice, col, ls in ARMS:
                if arm not in R[c]:
                    continue
                p = R[c][arm]
                ax.plot(p["t"], p[key], color=col, ls=ls, linewidth=1.2, label=nice)
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(c, loc="left")
            if j == 0:
                ax.set_ylabel(lab)
            if row == 1:
                ax.set_xlabel("t  (days)")
            if row == 0 and j == n - 1:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("Inferred time-dependent transmission rates. The two routes are "
                 "weakly identified: the data constrains their combination, not "
                 "the split.", y=1.004, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------- R5
def fig_R0(R, path):
    cities = list(R)
    n = len(cities)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.9), sharex=True)
    axes = np.atleast_1d(axes)
    for j, c in enumerate(cities):
        ax = axes[j]
        ax.axhline(1.0, color=C_INK, linewidth=0.9, ls=":", zorder=1)
        ax.axhspan(1.0, 5.0, color="#009E73", alpha=0.07, zorder=0)
        for arm, nice, col, ls in ARMS:
            if arm not in R[c]:
                continue
            p = R[c][arm]
            ax.plot(p["t"], p["R0"], color=col, ls=ls, linewidth=1.25,
                    label=f"{nice}  mean {p['row']['R0_mean']:.2f}")
        ax.set_yscale("log"); ax.set_title(c, loc="left")
        ax.set_xlabel("t  (days)")
        if j == 0:
            ax.set_ylabel(r"$\mathcal{R}_0(t)$")
        ax.legend(fontsize=6.8, loc="upper right", framealpha=0.9)
    fig.suptitle(r"Temporal reproduction number. Shaded band is the range "
                 r"reported for cholera ($1\leq\mathcal{R}_0\leq5$); the dotted "
                 r"line is the epidemic threshold.", y=1.01, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------- R6
def fig_error_bars(R, path):
    cities = list(R)
    arms = [a for a in ARMS if any(a[0] in R[c] for c in cities)]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1))
    w = 0.8 / len(arms)
    for ax, metric, lab in ((axes[0], "MAE", "mean absolute error"),
                            (axes[1], "RMSE", "root mean squared error")):
        for k, (arm, nice, col, _) in enumerate(arms):
            v = [R[c][arm]["row"][metric] if arm in R[c] else np.nan for c in cities]
            pos = np.arange(len(cities)) + (k - (len(arms) - 1) / 2) * w
            ax.bar(pos, v, width=w * 0.92, color=col, label=nice,
                   edgecolor="white", linewidth=0.6)
            for x, vv in zip(pos, v):
                if np.isfinite(vv):
                    ax.text(x, vv * 1.07, f"{vv:.1e}", ha="center", fontsize=6.2,
                            rotation=90, color=C_MUT)
        ax.set_yscale("log"); ax.set_xticks(np.arange(len(cities)))
        ax.set_xticklabels(cities, fontsize=8)
        ax.set_ylabel(f"{lab}  (person$^{{-1}}$d$^{{-1}}$)")
        ax.set_title(f"({'ab'[axes.tolist().index(ax)]})  {metric} against the "
                     f"observed record", loc="left")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=7.4, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# --------------------------------------------------------------- R7
def fig_age_surfaces(R, city, path):
    """The age-resolved infectious field each architecture reconstructs, and
    their difference. With no truth available the comparison is between the two
    architectures, and it is labelled as such."""
    have = [a for a in ("det", "dpinn", "ff") if a in R[city]]
    nice = {a: n for a, n, _, _ in ARMS}
    A = seip.A_MAX
    fig, axes = plt.subplots(1, len(have) + 1, figsize=(4.0 * (len(have) + 1), 3.9))
    surfs = {}
    for k, arm in enumerate(have):
        S = np.asarray(R[city][arm]["I_surface"])
        surfs[arm] = S
        t = R[city][arm]["t"]
        a = np.linspace(0, A, S.shape[1])
        m = axes[k].pcolormesh(t, a, S.T, cmap=FIELD_CMAP, shading="auto")
        axes[k].set_title(f"({'abcd'[k]})  {nice[arm]}   $I(t,a)$", loc="left")
        axes[k].set_xlabel("t  (days)")
        if k == 0:
            axes[k].set_ylabel("a  (days since infection)")
        fig.colorbar(m, ax=axes[k], pad=0.02)
    if "dpinn" in surfs and "ff" in surfs:
        d = surfs["ff"] - surfs["dpinn"]
        t = R[city]["ff"]["t"]; a = np.linspace(0, A, d.shape[1])
        v = np.abs(d).max()
        ax = axes[-1]
        m = ax.pcolormesh(t, a, d.T, cmap=DIVERGE, shading="auto",
                          norm=TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v))
        ax.set_title(f"({'abcd'[len(have)]})  FF-PINN $-$ DPINN", loc="left")
        ax.set_xlabel("t  (days)")
        fig.colorbar(m, ax=ax, pad=0.02)
    fig.suptitle(f"{city}: age-resolved infectious field. The right panel is a "
                 "difference between architectures, not an error against a "
                 "known field.", y=1.01, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def summary_table(R):
    import pandas as pd
    rows = []
    for c in R:
        for arm, *_ in ARMS:
            if arm in R[c]:
                rows.append(R[c][arm]["row"])
    return pd.DataFrame(rows)


def main(data_dir="data_real", outdir="figures_real",
         cities=("Al Hudaydah", "Amanat Al Asimah", "Hajjah", "Amran")):
    os.makedirs(outdir, exist_ok=True)
    R = load(cities)
    fig_observed_surface(data_dir, f"{outdir}/R1_observed_surface.png")
    fig_fit_and_error(R, f"{outdir}/R2_fit_and_residual.png")
    fig_beta(R, f"{outdir}/R3_beta.png")
    fig_R0(R, f"{outdir}/R4_reproduction_number.png")
    fig_error_bars(R, f"{outdir}/R5_error_comparison.png")
    fig_age_surfaces(R, "Al Hudaydah", f"{outdir}/R6_age_surfaces.png")
    tab = summary_table(R)
    tab.to_csv(f"{outdir}/real_metrics.csv", index=False)
    cols = ["city", "method", "MAE", "RMSE", "nRMSE_rms", "R0_min", "R0_max",
            "R0_mean", "frac_t_R0_gt1", "env_share", "seconds"]
    import pandas as pd
    pd.set_option("display.width", 220)
    print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return R, tab


if __name__ == "__main__":
    main()