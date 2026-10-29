"""The 2x2 design: in-sample against held-out, and the attributed effects."""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 170, "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})
ARMS = [("det", "Deterministic", "#666666"), ("dpinn", "DPINN", "#0072B2"),
        ("dpinn_fs", "DPINN-FS", "#56B4E9"), ("fdpinn", "F-DPINN", "#009E73"),
        ("ff", "FF-PINN", "#D55E00")]
CITIES = ["Al Hudaydah", "Amanat Al Asimah", "Hajjah", "Amran"]


def tables(csv="holdout_fits/factorial_metrics.csv"):
    d = pd.read_csv(csv)
    d = d[(d["mode"] == "interleave") & (d.tag.isna() | (d.tag == ""))]
    keys = [a for a, *_ in ARMS]
    ins = d.pivot_table(index="city", columns="arm", values="in_MAE").loc[CITIES, keys]
    out = d.pivot_table(index="city", columns="arm", values="out_MAE").loc[CITIES, keys]
    return d, ins, out


def effects(out):
    L = np.log(out[["dpinn", "dpinn_fs", "fdpinn", "ff"]])
    return pd.DataFrame({
        "Fourier states": 0.5 * ((L.fdpinn + L.ff) - (L.dpinn + L.dpinn_fs)),
        "Fourier beta":   0.5 * ((L.dpinn_fs + L.ff) - (L.dpinn + L.fdpinn)),
        "interaction":    0.5 * ((L.ff + L.dpinn) - (L.dpinn_fs + L.fdpinn))})


def fig(ins, out, path):
    E = effects(out)
    fig = plt.figure(figsize=(14.6, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=.28)
    w = 0.16
    for col, (tab, lab) in enumerate(((ins, "in-sample MAE"), (out, "held-out MAE"))):
        ax = fig.add_subplot(gs[0, col])
        for k, (arm, nice, c) in enumerate(ARMS):
            ax.bar(np.arange(len(CITIES)) + (k - 2) * w, tab[arm].values,
                   width=w * .9, color=c, label=nice, edgecolor="white", lw=.5)
        ax.set_yscale("log"); ax.set_xticks(np.arange(len(CITIES)))
        ax.set_xticklabels(CITIES, fontsize=7.6, rotation=12)
        ax.set_ylabel("MAE  (person$^{-1}$d$^{-1}$)")
        ax.set_title(f"({'ab'[col]})  {lab}", loc="left")
        ax.grid(axis="x", visible=False)
        if col == 0:
            ax.legend(fontsize=6.9, ncol=2, framealpha=.9)
    lo = min(ins.values.min(), out.values.min()) * .6
    hi = max(ins.values.max(), out.values.max()) * 1.6
    for a in fig.axes:
        a.set_ylim(lo, hi)

    ax = fig.add_subplot(gs[0, 2])
    m = E.mean(); x = np.arange(3)
    cols = ["#009E73", "#D55E00", "#888888"]
    ax.bar(x, np.exp(m.values), color=cols, edgecolor="white", lw=.6)
    for xi, v in zip(x, np.exp(m.values)):
        ax.text(xi, v * 1.02, f"x{v:.2f}", ha="center", fontsize=8)
    for ci, city in enumerate(CITIES):
        ax.plot(x, np.exp(E.loc[city].values), "o", ms=3.4, color="#222",
                alpha=.65, label="per city" if ci == 0 else None)
    ax.axhline(1.0, color="#222", lw=.9, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(E.columns, fontsize=8)
    ax.set_ylabel("factor on held-out MAE")
    ax.set_title("(c)  attributed effects", loc="left")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=7)
    fig.suptitle("The Fourier-feature state embedding buys the in-sample fit and "
                 "causes the overfitting; the $\\beta$ basis does neither",
                 y=1.03, fontsize=10.5)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figures_real", exist_ok=True)
    d, ins, out = tables()
    pd.set_option("display.width", 220)
    print("IN-SAMPLE MAE\n", ins.to_string(float_format=lambda v: "%.3e" % v))
    print("\nHELD-OUT MAE\n", out.to_string(float_format=lambda v: "%.3e" % v))
    print("\nDEGRADATION out/in\n", (out / ins).to_string(float_format=lambda v: "%.1f" % v))
    E = effects(out)
    print("\nEFFECTS on log held-out MAE\n", E.to_string(float_format=lambda v: "%+.3f" % v))
    print("\nmean factor:", {k: round(float(v), 3) for k, v in np.exp(E.mean()).items()})
    r = d.pivot_table(index="city", columns="arm", values=["R0_min", "R0_max"])
    print("\nR0 range by arm")
    for a, nice, _ in ARMS:
        lo = r[("R0_min", a)]; hi = r[("R0_max", a)]
        print(f"  {nice:14s} " + "  ".join(f"{c[:11]}: {lo[c]:.2f}-{hi[c]:.2f}" for c in CITIES))
    fig(ins, out, "figures_real/R11_factorial.png")
    print("\nwrote figures_real/R11_factorial.png")