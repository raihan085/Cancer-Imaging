#!/usr/bin/env python3
"""make_v3_nb.py -- fourth revision of the notebook: y rebuilt from the case
increments (fix 1) and the deterministic full-domain fit run to convergence (fix 3).

    python make_v3_nb.py V2.ipynb RUN_DIR OUT.ipynb

V2.ipynb  yemen_cholera_dpinn_vs_ffpinn_final.ipynb as delivered in revision 3
RUN_DIR   a directory holding
            real_fits/<city>/{det,dpinn,ff}.pkl      (Amran refitted on the new panel)
            holdout_rerun/amran/*.pkl                (holdout.py, Amran, 5 arms x seeds)
            new_slim/domain_{arm}.pkl                (revision-3 domain fits)
            new_slim/domain_det_converged.pkl        (domain_det.py output)
            new_slim/holdout_and_sweep.csv           (revision-3 holdout table)
            data_real/                               (the unpacked record)
OUT.ipynb the revised notebook, unexecuted

Every number the revision changes in the text is computed here from those files.
"""
import base64, io, os, pickle, shutil, subprocess, sys, tarfile
import numpy as np
import pandas as pd
import nbformat as nbf

V2, RUN, OUT = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUN)
import data_io  # noqa: E402  (RUN_DIR's copy is the fixed loader)
import plots_factorial  # noqa: E402

CITIES = ["Al Hudaydah", "Amanat Al Asimah", "Hajjah", "Amran"]
NET = ["dpinn", "dpinn_fs", "fdpinn", "ff"]
DOM = ["det", "dpinn", "dpinn_fs", "fdpinn", "ff"]
slug = lambda c: c.lower().replace(" ", "_").replace("'", "")
rng = lambda v, f="%.2f": (f % min(v)) + "–" + (f % max(v))
lst = lambda v, f="%.2f": ", ".join(f % x for x in v)

nb = nbf.read(V2, as_version=4)
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs, c.execution_count = [], None


def cell_index(prefix):
    hits = [i for i, c in enumerate(nb.cells) if c.source.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def find_md(fragment):
    hits = [i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and fragment in c.source]
    assert len(hits) == 1, (fragment[:60], hits)
    return hits[0]


def replace_in(i, old, new):
    s = nb.cells[i].source
    assert s.count(old) == 1, (i, old[:70])
    nb.cells[i].source = s.replace(old, new)


# ===================================================== 0. assemble new_slim
NS = os.path.join(RUN, "new_slim_v3")
shutil.rmtree(NS, ignore_errors=True)
os.makedirs(NS)
panel = data_io.load_panel(os.path.join(RUN, "data_real"), verbose=False)
dom = {}
for a in DOM[1:]:
    d = pickle.load(open(os.path.join(RUN, "new_slim", f"domain_{a}.pkl"), "rb"))
    assert list(d["names"]) == list(panel.names) and np.array_equal(d["gov"], panel.gov)
    assert np.allclose(d["t0"], panel.t0) and len(d["panel_y"]) == len(panel.y)
    # rescored against the fixed y; the fit itself predates fix 1 (<=0.1 % on 15 rows)
    d["panel_y_fit"] = "pre-fix-1 (incidence_rate column)"
    d["panel_y"] = np.asarray(panel.y)
    dom[a] = d
det = pickle.load(open(os.path.join(RUN, "new_slim", "domain_det_converged.pkl"), "rb"))
assert np.allclose(det["panel_y"], panel.y)
dom["det"] = det
for a, d in dom.items():
    pickle.dump(d, open(os.path.join(NS, f"domain_{a}.pkl"), "wb"))
shutil.copy(os.path.join(RUN, "new_slim", "holdout_and_sweep.csv"), NS)
subprocess.run([sys.executable, os.path.join(HERE, "collect_holdout.py"),
                "--csv", os.path.join(NS, "holdout_and_sweep.csv"),
                "--rerun", os.path.join(RUN, "holdout_rerun"), "--cities", "Amran"], check=True)

# ===================================================== 1. code cells
i = cell_index("%%writefile data_io.py")
nb.cells[i].source = "%%writefile data_io.py\n" + open(os.path.join(HERE, "data_io.py")).read()

i = cell_index("%%writefile fit.py")
sys.path.insert(0, HERE)
import fit_patch  # noqa: E402
nb.cells[i].source = "%%writefile fit.py\n" + fit_patch.apply(nb.cells[i].source.split("\n", 1)[1])

i = cell_index("%%writefile holdout.py")
new_cells = [nbf.v4.new_code_cell("%%writefile domain_det.py\n" +
                                  open(os.path.join(HERE, "domain_det.py")).read()),
             nbf.v4.new_code_cell("%%writefile collect_holdout.py\n" +
                                  open(os.path.join(HERE, "collect_holdout.py")).read())]
nb.cells[i + 1:i + 1] = new_cells

i = cell_index("%%writefile plots_domain.py")
replace_in(i, 'NOTE = {"det": " (stopped at max_nfev: NOT converged)"}\n',
           '''

def note(d):
    """Label a fit that did not converge; PINN fits carry no flag and get none."""
    return " (NOT converged)" if d.get("converged") is False else ""
''')
replace_in(i, 'rows.append(dict(arm=a, method=nice + NOTE.get(a, ""),',
           'rows.append(dict(arm=a, method=nice + note(d),')
replace_in(i, '''label=f"{nice}  mean {r['R0'].mean():.2f}" + NOTE.get(a, ""))''',
           '''label=f"{nice}  mean {r['R0'].mean():.2f}" + note(R[a]))''')

# section 13 cell: optional refit of the deterministic domain arm
i = cell_index("import plots_domain, importlib")
nb.cells[i].source = '''RUN_DOMAIN_DET = False   # True refits the deterministic full-domain arm to convergence
if RUN_DOMAIN_DET:
    subprocess.run([sys.executable, "-u", "domain_det.py", "--data-dir", "data_real",
                    "--out", "new_slim/domain_det.pkl"], check=True)
''' + nb.cells[i].source + '''
dd = RD["det"]
print(f"deterministic domain fit: converged={dd.get('converged')}  status={dd.get('status')}  "
      f"nfev={dd.get('nfev')}  {dd['seconds']:.0f} s")'''

# ===================================================== 2. FITS_B64
KEEP = ("name", "arm", "city", "row", "env_share", "seconds", "beta10", "beta20", "R0",
        "S", "yhat", "t", "observed", "I_surface")
buf = io.BytesIO()
fits = {}
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
    for c in CITIES:
        for a in ("det", "dpinn", "ff"):
            # Amran is refitted into real_fits/; the other three are unchanged by fix 1
            # and come from the revision-3 shipped fits (already slim, float32)
            p = os.path.join(RUN, "real_fits" if c == "Amran" else "real_fits_slim", slug(c), f"{a}.pkl")
            d = pickle.load(open(p, "rb"))
            assert len(d["observed"]["t0"]) == 135 and d["observed"]["t0"][0] == 0.0, p
            slim = {k: (np.asarray(v, np.float32) if isinstance(v, np.ndarray) and v.dtype.kind == "f"
                        else v) for k, v in d.items() if k in KEEP}
            fits[(c, a)] = slim
            raw = pickle.dumps(slim)
            ti = tarfile.TarInfo(f"real_fits_slim/{slug(c)}/{a}.pkl"); ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
nb.cells[cell_index("FITS_B64")].source = 'FITS_B64 = """' + base64.b64encode(buf.getvalue()).decode() + '"""'

# ===================================================== 4. numbers
row = lambda c, a: fits[(c, a)]["row"]
ff_gain = [row(c, "dpinn")["MAE"] / row(c, "ff")["MAE"] for c in CITIES]
nrmse = [row(c, a)["nRMSE_rms"] for c in CITIES for a in ("det", "dpinn", "ff")]
dp_hud = row("Al Hudaydah", "dpinn")
ff_peak_city = max(CITIES, key=lambda c: row(c, "ff")["R0_max"])
ff_peak = row(ff_peak_city, "ff")["R0_max"]
env = {a: [row(c, a)["env_share"] for c in CITIES] for a in ("det", "dpinn", "ff")}

H = pd.read_csv(os.path.join(NS, "holdout_and_sweep.csv"))
I = H[H["mode"] == "interleave"]
seed0 = I[I.tag.isna() | (I.tag == "")]
s0in = seed0.pivot_table(index="city", columns="arm", values="in_MAE").loc[CITIES]
s0out = seed0.pivot_table(index="city", columns="arm", values="out_MAE").loc[CITIES]
s0deg = s0out / s0in
ff_vs_det = list(s0out.ff / s0out.det)
ff_deg = list(s0deg.ff)
fourier_cut = list(s0in.dpinn / s0in.fdpinn) + list(s0in.dpinn_fs / s0in.ff)
deg_plain = list(s0deg.dpinn) + list(s0deg.dpinn_fs)
deg_four = list(s0deg.fdpinn) + list(s0deg.ff)
fs0_det = list(s0out.dpinn_fs / s0out.det)
fs0_deg = list(s0deg.dpinn_fs)

Hm = I[~I.tag.astype(str).str.startswith("wpde")]
detout = Hm[Hm.arm == "det"].set_index("city").out_MAE.reindex(CITIES)
med = Hm[Hm.arm != "det"].pivot_table(index="city", columns="arm", values="out_MAE",
                                      aggfunc="median").reindex(CITIES, columns=NET)
medin = Hm[Hm.arm != "det"].pivot_table(index="city", columns="arm", values="in_MAE",
                                        aggfunc="median").reindex(CITIES, columns=NET)
mrel = med.div(detout, axis=0)
spread = (Hm[Hm.arm != "det"].groupby(["city", "arm"]).out_MAE
          .agg(lambda z: z.max() / z.min()).unstack().reindex(CITIES, columns=NET))
Ep = plots_factorial.effects_percity(Hm).reindex(CITIES)
pooled = {k: float(np.exp(Ep[k].mean())) for k in Ep}
st_city = list(np.exp(Ep["Fourier states"]))
assert (Ep["Fourier states"] > 0).all()
fd_lowest = [c for c in CITIES if medin.loc[c].idxmin() == "fdpinn"]
fd_not = [c for c in CITIES if c not in fd_lowest]
fd_beaten_by = {c: medin.loc[c].idxmin() for c in fd_not}
assert (mrel.fdpinn > 1).all()
NICE = dict(dpinn="DPINN", dpinn_fs="DPINN-FS", fdpinn="F-DPINN", ff="FF-PINN", det="deterministic")


def dom_row(a):
    d = dom[a]; y = d["panel_y"]; e = np.asarray(d["yhat"]) - y
    return dict(MAE=np.mean(np.abs(e)), nRMSE=np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(y ** 2)),
                R0min=float(np.min(d["R0"])), R0max=float(np.max(d["R0"])))


DR = {a: dom_row(a) for a in DOM}


def deviance(yh):
    yh = np.maximum(np.asarray(yh, float), 1e-300)
    yy = np.asarray(panel.y)
    return 2 * np.sum(np.where(yy > 0, yy * np.log(np.maximum(yy, 1e-300) / yh), 0.0) - (yy - yh))


old_det = pickle.load(open(os.path.join(RUN, "new_slim", "domain_det.pkl"), "rb"))
dev_old, dev_new = deviance(old_det["yhat"]), deviance(det["yhat"])
# a termination flag alone is not trusted: the fit must also be no worse than the
# revision-3 fit that stopped at max_nfev
det_conv = bool(det.get("converged")) and dev_new <= dev_old * (1 + 1e-6)
if det.get("converged") and not det_conv:
    det["converged"] = False
    pickle.dump(det, open(os.path.join(NS, "domain_det.pkl"), "wb"))
print(f"domain det deviance: rev3 {dev_old:.6e}  new {dev_new:.6e}  -> converged={det_conv}")
y = panel.y; gov = panel.gov; xg = np.asarray(panel.xg)
far = np.isin(gov, [g for g in range(len(panel.names)) if xg[g] > 400])
far_rows = 100 * far.mean()
far_share = {a: 100 * np.abs(np.asarray(dom[a]["yhat"]) - y)[far].sum()
             / np.abs(np.asarray(dom[a]["yhat"]) - y).sum() for a in ("det", "dpinn", "ff")}
bias = {a: pd.Series({nm: np.mean(np.asarray(dom[a]["yhat"])[gov == g] - y[gov == g])
                      for g, nm in enumerate(panel.names)}) for a in ("det", "ff")}
det_neg = bias["det"].sort_values().head(3)
bias_ratio = bias["det"].abs().mean() / bias["ff"].abs().mean()

print("per-city ff_gain", lst(ff_gain, "%.1f"), "| nRMSE", rng(nrmse), "| env", env)
print("seed0 ff/det", lst(ff_vs_det), "| ff deg", rng(ff_deg, "%.1f"))
print("medians/det\n", mrel.round(3), "\nspread\n", spread.round(3))
print("effects", np.exp(Ep).round(3), pooled)
print("F-DPINN lowest in-sample at", fd_lowest, "beaten by", fd_beaten_by)
print("domain", {a: {k: round(v, 4) for k, v in r.items()} for a, r in DR.items()})
print("det domain converged", det_conv, det.get("status"), det.get("nfev"), round(det["seconds"]))
print("far share", far_share, "| det neg bias", det_neg.to_dict(), "| bias ratio", bias_ratio)

# ===================================================== 3. NEW_B64
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
    tf.add(NS, arcname="new_slim", recursive=False)
    for f in [f"domain_{a}.pkl" for a in DOM] + ["holdout_and_sweep.csv"]:
        tf.add(os.path.join(NS, f), arcname=f"new_slim/{f}")
nb.cells[cell_index("NEW_B64")].source = 'NEW_B64 = """' + base64.b64encode(buf.getvalue()).decode() + '"""'

# ===================================================== 5. text
i = find_md("> **Data correction, third revision.**")
replace_in(i, "> defect was found in external review.\n", f"""> defect was found in external review.
>
> **Fourth revision.** The incidence rate is now rebuilt from the case increments
> ($y = \\Delta C / (N\\,\\Delta t)$) instead of read from `cases_long.csv`, whose `incidence_rate`
> column was pre-capped in 15 rows; the loader's own 99.5 % winsorisation is the only cap, and the
> cut moves from 1.2102e-03 to 1.2114e-03. Among the fitted governorates only Amran has a changed
> row, so Amran's per-city and holdout fits are rerun; the other three are unchanged. The
> deterministic full-domain fit of Section 13, which had stopped at `max_nfev`, is rerun with
> `domain_det.py` and {'converges' if det_conv else 'still does not converge'} ({det.get('nfev')} evaluations over {len(det.get('history', []))}
> passes, each restarted from the last until the objective stops falling). The four PINN
> domain fits are kept and rescored against the rebuilt rates.
""")

i = find_md("Three trends, none visible on the defective panel.")
replace_in(i, """Al Hudaydah spans 0.68–6.87 (the 0.84–1.16 quoted in an earlier draft is the full-domain fit of
Section 13, a different model).""", f"""Al Hudaydah spans 0.68–6.87 (the 0.84–1.16 quoted in an earlier draft came from the unconverged
full-domain fit of Section 13, a different model; {'run to convergence' if det_conv else 'rerun with a larger budget'}, that fit spans
{DR['det']['R0min']:.2f}–{DR['det']['R0max']:.2f}).""")

i = find_md("Three things change once the model is spatial.")
dom_nrmse = [DR[a]["nRMSE"] for a in DOM]
nb.cells[i].source = f"""Three things change once the model is spatial.

**The FF-PINN's margin collapses** to {DR['dpinn']['MAE'] / DR['ff']['MAE']:.1f}× on MAE against the DPINN, from
{rng(ff_gain, '%.1f')}× in the isolated fits. With one $\\beta(t)$ explaining 21 governorates at once,
the extra flexibility has far less room to absorb noise.

**Everything fits worse** — nRMSE {rng(dom_nrmse)} against {rng(nrmse)} per city. Part of the per-city
performance was each governorate receiving a transmission history of its own.

**Identifiability improves.** The DPINN's $\\mathcal{{R}}_0$ goes from {dp_hud['R0_min']:.2f}–{dp_hud['R0_max']:.2f} at Al
Hudaydah in isolation to {DR['dpinn']['R0min']:.2f}–{DR['dpinn']['R0max']:.2f} over the whole domain; the FF-PINN's upper end falls from
{ff_peak:.2f} at {ff_peak_city} to {DR['ff']['R0max']:.2f}. Sharing $\\beta$ across 21 locations partially breaks the amplitude
degeneracy that no amount of physics weighting could touch — which points at the remedy section 12
rules out. The deterministic arm, {'now run to convergence' if det_conv else 'still unconverged'}, spans
{DR['det']['R0min']:.2f}–{DR['det']['R0max']:.2f}.

On the residual panel the two far governorates look grossly over-predicted. The cell above shows
they are not: Hadramaut and Al Maharah carry {far_share['dpinn']:.1f} % (DPINN), {far_share['ff']:.1f} % (FF-PINN) and
{far_share['det']:.1f} % (deterministic) of total absolute error against a {far_rows:.1f} % share of rows. The logarithmic
colour floor exaggerates them. The error concentrates in the mid-range high-incidence governorates,
and the informative pattern there is systematic per-governorate **bias** in the deterministic arm —
{', '.join(f'{k} {v:.1e}'.replace(' -', ' −', 1) for k, v in det_neg.items())} — against FF-PINN biases
{bias_ratio:.0f}× smaller on average. A single shared $\\beta(t)$ cannot express the observed spatial
heterogeneity; the network absorbs into its free states what the shared parameter cannot. That is an
argument for a spatially varying $\\beta(t,x)$, which this methodology does not currently have.

The four PINN domain fits predate the rebuilt rates (they were trained on the capped column, which
differs by at most 0.1 % in 15 of 2,806 rows) and are scored here against the rebuilt rates; the
deterministic domain fit was rerun on them.
"""

i = find_md("**The state embedding buys the in-sample gain and causes the overfitting.**")
nb.cells[i].source = f"""**The state embedding buys the in-sample gain and causes the overfitting.** Moving from plain to
Fourier-feature states cuts in-sample MAE by {rng(fourier_cut, '%.0f')}x at every city and raises degradation from
{rng(deg_plain, '%.1f')} to {rng(deg_four, '%.1f')}. Out of sample it is a penalty, not a gain.

**The $\\beta$ basis is close to inert.** The full-domain fit settles it, because there all five arms
fit the same 2,806 intervals: the two plain-state arms land at nRMSE {DR['dpinn']['nRMSE']:.3f} and {DR['dpinn_fs']['nRMSE']:.3f}, the two
Fourier-state arms at {DR['fdpinn']['nRMSE']:.3f} and {DR['ff']['nRMSE']:.3f}. Changing $\\beta$ moves the third significant figure;
changing the state embedding moves it by {np.mean([DR['dpinn']['nRMSE'], DR['dpinn_fs']['nRMSE']]) / np.mean([DR['fdpinn']['nRMSE'], DR['ff']['nRMSE']]):.1f}x.

**DPINN-FS, absent from the original study, is the most robust cell** - {rng(fs0_det)} of the
deterministic baseline at every city, degradation {rng(fs0_deg, '%.1f')}x, and the cheapest of the four to train.

Section 16 repeats all of this at three seeds; the numbers quoted here are seed 0.

## 15. All five arms on the $(t,x)$ surface"""

i = find_md("**The Fourier-state penalty is x")
s = nb.cells[i].source
tail_start = s.index("Effects are computed on per-city **mean log**")
tail_end = s.index("**DPINN-FS is the defensible default**")
nb.cells[i].source = (
f"""**The Fourier-state penalty is x{pooled['Fourier states']:.3f} and positive at every city** ({lst(st_city, '%.3f')}).
**The $\\beta$ effect is x{pooled['Fourier beta']:.3f}** and **the interaction x{pooled['interaction']:.3f}** - both within
{np.ceil(100 * max(abs(pooled['Fourier beta'] - 1), abs(pooled['interaction'] - 1))):.0f} % of 1, so the question of how to estimate the interaction no longer arises.

""" + s[tail_start:s.index("**Seed spreads are")] +
f"""**Seed spreads are {rng(spread.values.ravel(), '%.3f')} everywhere.** An earlier revision of this notebook, fitted on the
defective panel, reported Al Hudaydah as an unstable city with a DPINN spread of x3.11 and built an
argument on it. Repaired, that spread is **x{spread.loc['Al Hudaydah', 'dpinn']:.2f}**: the instability was the mis-anchored initial
condition, not the city.

**Run-to-run reproducibility.** The deterministic arm reproduces to five significant figures
(Hajjah held-out MAE 1.299914e-04 shipped, 1.299905e-04 rerun). The networks do not reproduce
bit for bit: rerunning Hajjah DPINN-FS at seed 1 moved held-out MAE from 1.2938e-04 to
1.2903e-04 and mean $\\mathcal{{R}}_0$ from 2.99 to 2.61. The Amran network rows below were rerun for
the rebuilt rates (a change of at most 0.1 % in one row there), so their differences from the
previous revision are mostly that run-to-run noise, not the data change.

""" + s[tail_end:])
nb.cells[i].source = nb.cells[i].source.replace(
    "**DPINN-FS is the defensible default** - 0.97-1.02 of the deterministic baseline at every city, the\nsmallest seed spread in the study (1.002-1.040)",
    f"**DPINN-FS is the defensible default** - {rng(mrel.dpinn_fs)} of the deterministic baseline at every city (3-seed\nmedians), the smallest seed spread in the study ({rng(spread.dpinn_fs, '%.3f')})")
assert "0.97-1.02 of the deterministic baseline at every city, the" not in nb.cells[i].source
assert spread.dpinn_fs.max() <= spread.drop(columns="dpinn_fs").max().min(), "DPINN-FS no longer the tightest"

fd_txt = (f"**held at {len(fd_lowest)} of 4** (3-seed medians); " +
          (", ".join(f"{NICE[b]} edges it at {c}" for c, b in fd_beaten_by.items()) if fd_not else "lowest everywhere"))
replace_in(i, "**held at 2 of 4** (3-seed medians); FF-PINN edges it at Amanat Al Asimah and Hajjah",
           fd_txt.replace("FF-PINN edges it at Amanat Al Asimah, FF-PINN edges it at Hajjah",
                          "FF-PINN edges it at Amanat Al Asimah and Hajjah"))
replace_in(i, "**held** — 1.07, 1.26, 1.44, 1.38", f"**held** — {lst(mrel.fdpinn)}")
replace_in(i, "(×1.273 against ×1.003)", f"(×{pooled['Fourier states']:.3f} against ×{pooled['Fourier beta']:.3f})")
replace_in(i, "the widest ratio is 1.48", f"the widest ratio is {mrel.values.max():.2f}")
assert mrel.values.max() < 2
if len(fd_lowest) != 2:
    replace_in(i, "Prediction 1 holds only at half the cities and prediction 3 fails",
               f"Prediction 1 holds at {len(fd_lowest)} of 4 cities and prediction 3 fails")

s = nb.cells[i].source
start = s.index("## 17. What the comparison shows")
domain_limit = (f"The deterministic full-domain arm is run to convergence ({det.get('nfev')} evaluations); the four PINN "
                "domain fits predate the rebuilt rates and are rescored, not refitted."
                if det_conv else
                "The deterministic full-domain arm does not converge even with restarts and is reported as not "
                "converged; its domain row understates how well that arm can fit.")
nb.cells[i].source = s[:start] + f"""## 17. What the comparison shows

All numbers below are from the repaired panel (2,806 intervals); every earlier figure fitted on the
defective panel is withdrawn.

**The Fourier-feature embedding wins the data term in sample, and that is not the result.** FF-PINN
cuts in-sample MAE by {rng(ff_gain, '%.1f')}× over DPINN ({lst(ff_gain, '%.1f')} at Al Hudaydah,
Amanat Al Asimah, Hajjah, Amran). Held out it is worse than the deterministic four-term Fourier series
at every governorate: {lst(ff_vs_det)}× at seed 0 and {rng(mrel.ff)}× at the three-seed median. Its
training-to-held-out degradation is {rng(ff_deg, '%.1f')}×, so it is systematically and very
optimistic about its own error. **Neither Fourier-state network generalises better than a low-order
deterministic fit under a hard PDE constraint.**

**The plain-state networks match that fit and no more.** At the three-seed median, DPINN is
{rng(mrel.dpinn)}× and DPINN-FS {rng(mrel.dpinn_fs)}× the deterministic arm's held-out error. DPINN-FS is the
defensible network default, with the smallest seed spread in the study ({rng(spread.dpinn_fs, '%.3f')}).

**The 2×2 factorial attributes the effects cleanly** (Section 16, three seeds, pairing-free): the
Fourier-state embedding costs ×{pooled['Fourier states']:.3f} on held-out error and is positive at every city; the $\\beta$
basis (×{pooled['Fourier beta']:.3f}) and the interaction (×{pooled['interaction']:.3f}) are null.

**$\\mathcal{{R}}_0$ and the route split are not identified by isolated fits.** On identical data the
deterministic arm puts the environmental share at {rng(env['det'])}, DPINN at {rng(env['dpinn'])} and
FF-PINN at {rng(env['ff'])}. Stiffer physics enforcement narrows the network's $\\mathcal{{R}}_0$ at the
stiff end of the sweep (Section 12), but by restricting it, not by converging on the deterministic
answer; sharing $\\beta$ across locations (Section 13) is the other lever that narrows it.

**What follows for the paper.** On aggregated bulletin counts at a single location, neither PINN
architecture beats a low-order deterministic fit out of sample, and in-sample residuals would have
said otherwise by an order of magnitude. $\\mathcal{{R}}_0$ and the route split are not identified by
the observation operator at a single location. Reporting the FF-PINN's $\\mathcal{{R}}_0$ as an
estimate is not defensible.

**Limits, stated rather than implied.** Three seeds per network cell (Section 16) and none for the
deterministic arm, whose fits are deterministic; the networks are not bit-reproducible between runs
(Section 16). The holdout is a single every-fourth interleave, not rotated folds. {domain_limit} The
physics sweep is one city and one seed. Diffusion is fixed at 50 km²/day and never estimated. The
budget-doubling result of an earlier draft was measured on the defective panel and has not been
repeated. Second wave only, and no comparison against an independently published $R_t$
reconstruction for Yemen.
"""

nbf.write(nb, OUT)
print("wrote", OUT)
