"""Full-domain (t,x) comparison on the real record, diffusion on."""
from __future__ import annotations
import pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm, LinearSegmentedColormap
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 170, "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
DIV = LinearSegmentedColormap.from_list("d", ["#2166AC", "#7FA8CE", "#D9D9D9",
                                              "#E8977A", "#B2182B"])
ARMS = [("det", "Deterministic", "#666666", "-"),
        ("dpinn", "DPINN", "#0072B2", "--"),
        ("dpinn_fs", "DPINN-FS", "#56B4E9", "-."),
        ("fdpinn", "F-DPINN", "#009E73", ":"),
        ("ff", "FF-PINN", "#D55E00", "-")]
NOTE = {"det": " (stopped at max_nfev: NOT converged)"}


def load(root="domain_fits"):
    R = {}
    for a, *_ in ARMS:
        try:
            d = pickle.load(open(f"{root}/domain_{a}.pkl", "rb"))
        except FileNotFoundError:
            continue
        # the slimmed files are flat; the full ones nest the arm under "rec"
        if "rec" not in d:
            d = {**d, "rec": d}
        R[a] = d
    return R


def table(R):
    rows = []
    for a, nice, *_ in ARMS:
        if a not in R:
            continue
        d = R[a]; r = d["rec"]; y = d["panel_y"]; yh = np.asarray(r["yhat"]); e = yh - y
        rows.append(dict(arm=a, method=nice + NOTE.get(a, ""),
                         MAE=np.mean(np.abs(e)), MSE=np.mean(e ** 2),
                         RMSE=np.sqrt(np.mean(e ** 2)),
                         nRMSE=np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(y ** 2)),
                         R0_min=r["R0"].min(), R0_max=r["R0"].max(),
                         R0_mean=r["R0"].mean(), seconds=r["seconds"]))
    return pd.DataFrame(rows)


def fig_domain(R, path):
    """Observed against fitted on the (t,x) plane, at the real bulletin points."""
    have = [a for a, *_ in ARMS if a in R]
    d0 = R[have[0]]
    tm = 0.5 * (d0["t0"] + d0["t1"]); xs = np.asarray(d0["xg"])[d0["gov"]]
    y = d0["panel_y"]
    pos = y[y > 0]
    nrm = LogNorm(vmin=max(pos.min(), 1e-9), vmax=y.max())
    n = len(have) + 1
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7.2), sharex=True, sharey=True)
    sc = axes[0, 0].scatter(tm, xs, c=np.maximum(y, nrm.vmin), s=9, cmap="viridis",
                            norm=nrm, edgecolor="none")
    axes[0, 0].set_title("(a)  observed", loc="left")
    axes[0, 0].set_ylabel("x  (km from Sana'a)")
    fig.colorbar(sc, ax=axes[0, 0], pad=.02)
    axes[1, 0].axis("off")
    for k, a in enumerate(have):
        yh = np.asarray(R[a]["rec"]["yhat"])
        nice = dict((x[0], x[1]) for x in ARMS)[a]
        ax = axes[0, k + 1]
        m = ax.scatter(tm, xs, c=np.maximum(yh, nrm.vmin), s=9, cmap="viridis",
                       norm=nrm, edgecolor="none")
        ax.set_title(f"({'abcdefg'[k+1]})  {nice} fitted", loc="left")
        fig.colorbar(m, ax=ax, pad=.02)
        e = yh - y; v = np.percentile(np.abs(e), 99)
        ax = axes[1, k + 1]
        m = ax.scatter(tm, xs, c=np.clip(e, -v, v), s=9, cmap=DIV,
                       norm=TwoSlopeNorm(0.0, -v, v), edgecolor="none")
        ax.set_title(f"      {nice} residual   MAE {np.mean(np.abs(e)):.2e}", loc="left")
        ax.set_xlabel("t  (days)")
        if k == 0:
            ax.set_ylabel("x  (km from Sana'a)")
        fig.colorbar(m, ax=ax, pad=.02)
    fig.suptitle(f"Full domain, 21 governorates, $D_1..D_4=50$ km$^2$/day, "
                 f"$x\\in[0,{d0['x_max']:.1f}]$ km — one $\\beta(t)$ shared across "
                 f"every location", y=1.003, fontsize=10.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_R0(R, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axhline(1.0, color="#222", lw=.9, ls=":")
    ax.axhspan(1, 5, color="#009E73", alpha=.07, zorder=0)
    for a, nice, col, ls in ARMS:
        if a not in R:
            continue
        r = R[a]["rec"]
        ax.plot(R[a]["t"][:len(r["R0"])] if len(R[a]["t"]) >= len(r["R0"])
                else np.linspace(0, 272, len(r["R0"])), r["R0"],
                color=col, ls=ls, lw=1.3,
                label=f"{nice}  mean {r['R0'].mean():.2f}" + NOTE.get(a, ""))
    ax.set_yscale("log"); ax.set_xlabel("t  (days)")
    ax.set_ylabel(r"$\mathcal{R}_0(t)$")
    ax.set_title("Temporal reproduction number, full domain with diffusion on", loc="left")
    ax.legend(fontsize=7.2, framealpha=.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figures_real", exist_ok=True)
    R = load()
    t = table(R)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    t.to_csv("figures_real/domain_metrics.csv", index=False)
    fig_domain(R, "figures_real/R7_domain_surface.png")
    fig_R0(R, "figures_real/R8_domain_R0.png")
    print("\nwrote figures_real/R7_domain_surface.png, R8_domain_R0.png")