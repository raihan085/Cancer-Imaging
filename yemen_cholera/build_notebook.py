#!/usr/bin/env python3
"""Build yemen_cholera_extended.ipynb from the cells below. The narrative cells
are kept here, next to the code they describe, so the notebook can be
regenerated after any rerun."""
import nbformat as nbf
from narrative import TEXT

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(TEXT["title"])
code('''
import os, sys, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.abspath("."))
import numpy as np, pandas as pd
from IPython.display import Image, display
import seip, data_io, epiestim, analysis as A
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
FIG = "figures"; os.makedirs(FIG, exist_ok=True)
fmt = lambda df, d=3: df.style.format(precision=d, thousands=",").hide(axis="index")

# RUN = True re-runs every fit (about a day of CPU on 4 cores). Otherwise the
# result files shipped in results/ are loaded.
RUN = False
if RUN:
    import subprocess
    for lst in ("cheap", "pinn"):
        subprocess.run(f"python make_jobs.py {lst} > jobs_{lst}.txt && ./run_queue.sh jobs_{lst}.txt 4",
                       shell=True, check=True)
    subprocess.run("./spatial_chain.sh", shell=True, check=True)

recs, idx = A.load("results")
S = A.load_spatial("results/spatial")
panel = data_io.load_panel("data_real", verbose=False)
FOCUS = A.FOCUS
print(f"{len(recs)} result files; spatial fits: {sorted(S)}")
''')
md(TEXT["status"])
code('''
P = idx.arm.isin(A.PINN); B = idx.variant == "base"; T = idx.truth == ""
expected = [
    ("deterministic, spline, SIR: 21 governorates x (full + 4 folds)", 315,
     (~P & B & T & (idx.seed == 0)).sum()),
    ("PINN, full record: 5 cities x 4 arms x 10 seeds", 200, (P & B & T & (idx.fold < 0)).sum()),
    ("PINN holdout, DPINN and FF-PINN: 5 x 2 x 10 seeds x 4 folds", 400,
     (idx.arm.isin(["dpinn", "ff"]) & B & T & (idx.fold >= 0)).sum()),
    ("PINN holdout, DPINN-FS and F-DPINN: 5 x 2 x 10 x 4", 400,
     (idx.arm.isin(["dpinn_fs", "fdpinn"]) & B & T & (idx.fold >= 0)).sum()),
    ("recovery, deterministic: 5 x 2 truths x 3", 30, (~P & (idx.truth != "") & idx.city.isin(FOCUS)).sum()),
    ("recovery, DPINN and FF-PINN: 5 x 2 x 2 truths x 3", 60, (P & (idx.truth != "")).sum()),
    ("sensitivity, deterministic: 5 x 8 variants", 40,
     ((idx.arm == "det") & ~B & idx.city.isin(FOCUS)).sum()),
    ("sensitivity, DPINN and FF-PINN: 5 x 2 x 4 variants", 40, (P & ~B).sum()),
    ("spatial fits (shared / smooth / governorate, D sweep)", 7, len(S)),
]
st_ = pd.DataFrame(expected, columns=["block", "planned", "done"])
st_["complete"] = (st_.done >= st_.planned).map({True: "yes", False: "pending"})
display(fmt(st_, 0))
''')

md(TEXT["cities"])
code('''
cases = pd.read_csv("data_real/cases_long.csv")
tot = (cases.groupby("governorate").new_cases.sum().sort_values(ascending=False)
       .rename("total reported cases").reset_index())
tot.insert(0, "rank", np.arange(1, len(tot) + 1))
tot["selected"] = tot.governorate.map(lambda g: A.FOCUS_NOTE.get(g, ""))
display(fmt(tot, 0))
''')

# ---------------------------------------------------------------- 1. seeds
md(TEXT["seeds"])
code('''
seeds = A.seed_table(recs, idx)
ss = A.seed_summary(seeds[seeds.arm.isin(A.PINN + ["det"])])
display(fmt(ss[["city", "arm", "n_seeds", "MAE_mean", "MAE_sd", "MAE_cv", "nRMSE_mean",
                "R_mean", "R_max_med", "R_max_lo", "R_max_hi", "env_mean", "env_sd"]], 4))
if seeds.arm.isin(A.PINN).any():
    A.fig_seed_spread(seeds[seeds.arm.isin(A.PINN)], f"{FIG}/E1_seed_spread.png")
    display(Image(f"{FIG}/E1_seed_spread.png"))
else:
    print("PINN seed runs not finished yet")
''')
md(TEXT["seeds_after"])

# ---------------------------------------------------------------- 2. holdout
md(TEXT["holdout"])
code('''
cv = A.cv_table(recs, idx)
cvs = A.cv_summary(cv)
foc = cvs[cvs.city.isin(FOCUS)].copy()
foc["arm"] = pd.Categorical(foc.arm, A.ORDER); foc = foc.sort_values(["city", "arm"])
display(fmt(foc[["city", "arm", "n_seeds", "in_MAE", "out_MAE", "out_MAE_sd",
                 "degradation", "vs_det"]], 4))
A.fig_cv(foc, f"{FIG}/E2_holdout.png")
display(Image(f"{FIG}/E2_holdout.png"))
pv = A.paired_vs_det(cv[cv.city.isin(FOCUS)])
print("Held-out MAE of each PINN seed relative to the deterministic arm "
      "(ratio < 1 = PINN better; one-sample Wilcoxon on log ratios across seeds)")
display(fmt(pv, 3))
''')
md(TEXT["holdout_after"])

# ---------------------------------------------------------------- 3. bands
md(TEXT["bands"])
code('''
cori = A.cori_all(panel, panel.names)
cori_short = A.cori_all(panel, panel.names, si=epiestim.SI_SHORT)
A.fig_bands(recs, idx, f"{FIG}/E3a_bands_main.png", arms=("det", "dpinn", "ff"), cori=cori)
display(Image(f"{FIG}/E3a_bands_main.png"))
A.fig_bands(recs, idx, f"{FIG}/E3b_bands_factorial.png", arms=("det", "dpinn_fs", "fdpinn"), cori=cori)
display(Image(f"{FIG}/E3b_bands_factorial.png"))
''')
code('''
# Band widths: median over the window of hi/lo, a scale-free width
rows = []
for c in FOCUS:
    for a in ("det", "sir", *A.PINN):
        for key in ("beta10", "beta20", "R"):
            b = A.band_of(recs, idx, c, a, key)
            if b is None:
                continue
            t, lo, mid, hi, kind = b
            rows.append(dict(city=c, arm=a, quantity=key, band=kind,
                             median_hi_over_lo=float(np.median(hi / np.maximum(lo, 1e-300)))))
bw = pd.DataFrame(rows).pivot_table(index=["city", "arm"], columns="quantity",
                                    values="median_hi_over_lo").reset_index()
display(fmt(bw, 2))
''')
md(TEXT["bands_after"])

# ---------------------------------------------------------------- 4. recovery & literature
md(TEXT["recovery"])
code('''
rt = A.recovery_table(recs, idx)
if rt.empty:
    print("recovery runs not finished yet")
else:
  rsum = (rt.groupby(["truth", "arm"])
              .agg(n=("seed", "size"), true_share=("true_share", "first"),
                 est_share_mean=("est_share", "mean"), est_share_sd=("est_share", "std"),
                 R_relerr=("R_relerr", "median"), beta1_relerr=("beta1_relerr", "median"),
                 beta2_relerr=("beta2_relerr", "median"), fit_MAE=("fit_MAE", "mean"),
                 truth_MAE=("truth_MAE", "mean")).reset_index())
  display(fmt(rsum, 3))
  A.fig_recovery(recs, idx, rt, f"{FIG}/E4a_recovery.png")
  display(Image(f"{FIG}/E4a_recovery.png"))
''')
md(TEXT["literature"])
code('''
lt = A.literature_table(recs, idx, cori)
display(fmt(lt, 2))
ar = A.attack_rates(panel)
display(fmt(ar[ar.WHO_EMRO_2nd_wave_per_10k.notna() | ar.governorate.isin(FOCUS)], 1))
''')
md(TEXT["literature_after"])

# ---------------------------------------------------------------- 5. Cori
md(TEXT["cori"])
code('''
A.fig_cori(cori, cori_short, f"{FIG}/E5_cori.png")
display(Image(f"{FIG}/E5_cori.png"))
ag = A.cori_agreement(recs, idx, cori)
display(fmt(ag, 3))
''')
md(TEXT["cori_after"])

# ---------------------------------------------------------------- 6. sensitivity
md(TEXT["sensitivity"])
code('''
st = A.sensitivity_table(recs, idx)
if st.empty or "variant" not in st:
    st = pd.DataFrame(columns=["city", "arm", "variant", "MAE", "dMAE_%", "R_mean", "dRmean_%", "R_min", "R_max", "env_share"])
cols = ["city", "arm", "variant", "MAE", "dMAE_%", "R_mean", "dRmean_%", "R_min", "R_max", "env_share"]
for arm in ("det", "dpinn", "ff"):
    t = st[st.arm == arm]
    if len(t):
        print(f"--- {A.ARM[arm][0]} ---")
        display(fmt(t[cols].sort_values(["city", "variant"]), 3))
spread = (st.groupby(["arm", "variant"]).agg(dMAE_med=("dMAE_%", "median"),
          dR_med=("dRmean_%", "median"), dR_max=("dRmean_%", lambda v: v.abs().max()))
          .reset_index())
display(fmt(spread, 1))
''')
md(TEXT["sensitivity_after"])

# ---------------------------------------------------------------- 7. diagnostics
md(TEXT["diagnostics"])
code('''
dt, keep = A.diagnostics_table(recs, idx, cities=FOCUS)
dt["arm"] = pd.Categorical(dt.arm, A.ORDER)
display(fmt(dt.sort_values(["city", "arm"]), 3))
A.fig_diagnostics(keep, f"{FIG}/E7_diagnostics.png")
display(Image(f"{FIG}/E7_diagnostics.png"))
dall, _ = A.diagnostics_table(recs, idx, cities=panel.names)
print("all 21 governorates, deterministic arm: dispersion median %.1f (range %.1f-%.1f), "
      "NB size median %.1f"
      % tuple(dall[dall.arm == "det"].dispersion.quantile([.5, 0, 1]).tolist()
              + [dall[dall.arm == "det"].nb_size.median()]))
''')
md(TEXT["diagnostics_after"])

# ---------------------------------------------------------------- 8. baselines
md(TEXT["baselines"])
code('''
b = cvs[cvs.arm.isin(["det", "sir", "spline"])]
piv = b.pivot_table(index="city", columns="arm", values="out_MAE")
piv["sir/det"] = piv["sir"] / piv["det"]; piv["spline/det"] = piv["spline"] / piv["det"]
display(piv.reindex([c for c in panel.names if c in piv.index]).style.format("{:.3g}"))
print("median over 21 governorates: SIR/det %.3f, spline/det %.3f; det best at %d, SIR best at %d, "
      "spline best at %d" % (piv["sir/det"].median(), piv["spline/det"].median(),
      (piv[["det", "sir", "spline"]].idxmin(axis=1) == "det").sum(),
      (piv[["det", "sir", "spline"]].idxmin(axis=1) == "sir").sum(),
      (piv[["det", "sir", "spline"]].idxmin(axis=1) == "spline").sum()))
''')
md(TEXT["baselines_after"])

# ---------------------------------------------------------------- 9. all cities
md(TEXT["allcities"])
code('''
A.fig_all_cities_R(recs, idx, cori, list(panel.names[i] for i in np.argsort(panel.xg)),
                   f"{FIG}/E9_all_cities_R.png")
display(Image(f"{FIG}/E9_all_cities_R.png"))
ag21 = A.cori_agreement(recs, idx, cori, cities=panel.names, arms=("det", "sir"))
display(fmt(ag21.groupby("arm")[["mean_abs_diff", "log_ratio_median", "frac_in_CrI", "spearman"]]
            .median().reset_index(), 3))
r21 = []
for c in panel.names:
    for a in ("det", "sir"):
        ds = A.select(recs, idx, city=c, arm=a, fold=-1, variant="base", truth="")
        if ds:
            r21.append(dict(city=c, arm=a, **A.r_summary(ds[0]), **A.errs(ds[0]["yhat"], ds[0]["y"])))
r21 = pd.DataFrame(r21)
display(fmt(r21[r21.arm == "det"][["city", "MAE", "nRMSE", "R_min", "R_mean", "R_max", "env_share"]], 3))
''')
md(TEXT["allcities_after"])

# ---------------------------------------------------------------- 10. spatial
md(TEXT["spatial"])
code('''
if S:
    display(fmt(A.spatial_table(S), 4))
if "governorate_D50" in S:
    A.fig_spatial_R(S, f"{FIG}/E10_spatial_R.png")
    display(Image(f"{FIG}/E10_spatial_R.png"))
    d = S["governorate_D50"]
    k = pd.DataFrame(dict(governorate=d["names"], x_km=d["xg"],
                          offset=d["theta"][-len(d["names"]):],
                          R_mean=d["R_g"].mean(0), R_max=d["R_g"].max(0))).sort_values("x_km")
    display(fmt(k, 3))
''')
md(TEXT["spatial_after"])

# ---------------------------------------------------------------- 11. surfaces
md(TEXT["surfaces"])
code('''
fields = {"Deterministic, per city": A.city_fields(recs, idx, panel, "det", panel.names)[1],
          "Spline baseline, per city": A.city_fields(recs, idx, panel, "spline", panel.names)[1]}
for key, nm in (("shared_D50", "Spatial, shared beta(t)"),
                ("governorate_D50", "Spatial, beta(t, x)")):
    if key in S:
        fields[nm] = A.spatial_field(S[key], panel)
A.fig_surfaces(panel, fields, f"{FIG}/E11a_surfaces_all.png",
               "All 21 governorates: observed against fitted incidence, rows ordered by "
               "distance from Sana'a")
display(Image(f"{FIG}/E11a_surfaces_all.png"))
A.fig_surfaces_3d(panel, fields, f"{FIG}/E11b_surfaces_3d.png")
display(Image(f"{FIG}/E11b_surfaces_3d.png"))
display(fmt(A.surface_errors(panel, fields), 4))
''')
code('''
# the five focus governorates, where every architecture was fitted
ff = {"Deterministic": fields["Deterministic, per city"]}
for a in ("dpinn", "ff", "dpinn_fs", "fdpinn"):
    ff[A.ARM[a][0] + " (seed median)"] = A.city_fields(recs, idx, panel, a, panel.names)[1]
if "Spatial, beta(t, x)" in fields:
    ff["Spatial, beta(t, x)"] = fields["Spatial, beta(t, x)"]
foc_idx = [panel.names.index(c) for c in FOCUS]
mask = np.isin(np.arange(panel.ng), foc_idx)
sub = {k: np.where(mask[None, :], v, np.nan) for k, v in ff.items()}
A.fig_surfaces(panel, sub, f"{FIG}/E11c_surfaces_focus.png",
               "Five focus governorates: observed against every architecture")
display(Image(f"{FIG}/E11c_surfaces_focus.png"))
display(fmt(A.surface_errors(panel, ff, cities=FOCUS), 4))
''')
md(TEXT["surfaces_after"])
md(TEXT["conclusions"])

nb = nbf.v4.new_notebook(cells=C)
nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
nbf.write(nb, "yemen_cholera_extended.ipynb")
print("wrote yemen_cholera_extended.ipynb with", len(C), "cells")
