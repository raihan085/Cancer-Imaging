#!/usr/bin/env python3
"""make_final_nb.py -- produce the corrected revision of the user's notebook.

    python make_final_nb.py IN.ipynb RUN_DIR OUT.ipynb

IN.ipynb  the uploaded yemen_cholera_dpinn_vs_ffpinn_2.ipynb
RUN_DIR   a directory where IN.ipynb was executed and where fit_one.py has
          refitted real_fits/<city>/{det,dpinn,ff}.pkl on the repaired panel
OUT.ipynb the corrected notebook (unexecuted)

1. replaces the data_io.py cell with the fixed data_io.py next to this script;
2. repacks FITS_B64 from the refitted per-city fits (the shipped ones predated
   the loader repair: 134 intervals starting at day 5);
3. corrects every sentence that contradicts the notebook's own output, with
   numbers computed here from the refitted fits and the shipped holdout table.
"""
import base64, io, os, pickle, sys, tarfile
import numpy as np
import pandas as pd
import nbformat as nbf

IN, RUN, OUT = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))
CITIES = ["Al Hudaydah", "Amanat Al Asimah", "Hajjah", "Amran"]
slug = lambda c: c.lower().replace(" ", "_").replace("'", "")
nb = nbf.read(IN, as_version=4)


def cell_index(prefix):
    hits = [i for i, c in enumerate(nb.cells) if c.source.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def replace_in(i, old, new):
    s = nb.cells[i].source
    assert s.count(old) == 1, (i, old[:70])
    nb.cells[i].source = s.replace(old, new)


def find_md(fragment):
    hits = [i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and fragment in c.source]
    assert len(hits) == 1, (fragment[:60], hits)
    return hits[0]


# ---------------------------------------------------------------- 1. data_io
i = cell_index("%%writefile data_io.py")
nb.cells[i].source = "%%writefile data_io.py\n" + open(os.path.join(HERE, "data_io.py")).read()

# ---------------------------------------------------------------- 2. fits
KEEP = ("name", "arm", "city", "row", "env_share", "seconds", "beta10", "beta20", "R0",
        "S", "yhat", "t", "observed", "I_surface")
buf = io.BytesIO()
fits = {}
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
    for c in CITIES:
        for a in ("det", "dpinn", "ff"):
            p = os.path.join(RUN, "real_fits", slug(c), f"{a}.pkl")
            d = pickle.load(open(p, "rb"))
            assert len(d["observed"]["t0"]) == 135 and d["observed"]["t0"][0] == 0.0, p
            slim = {k: (np.asarray(v, np.float32) if isinstance(v, np.ndarray) and v.dtype.kind == "f"
                        else v) for k, v in d.items() if k in KEEP}
            fits[(c, a)] = slim
            raw = pickle.dumps(slim)
            ti = tarfile.TarInfo(f"real_fits_slim/{slug(c)}/{a}.pkl"); ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
i = cell_index("FITS_B64")
nb.cells[i].source = 'FITS_B64 = """' + base64.b64encode(buf.getvalue()).decode() + '"""'

# ---------------------------------------------------------------- numbers
row = lambda c, a: fits[(c, a)]["row"]
ff_gain = [row(c, "dpinn")["MAE"] / row(c, "ff")["MAE"] for c in CITIES]
nrmse = [row(c, a)["nRMSE_rms"] for c in CITIES for a in ("det", "dpinn", "ff")]
dp_hud = row("Al Hudaydah", "dpinn")
ff_peak_city = max(CITIES, key=lambda c: row(c, "ff")["R0_max"])
ff_peak = row(ff_peak_city, "ff")["R0_max"]
env = {a: [row(c, a)["env_share"] for c in CITIES] for a in ("det", "dpinn", "ff")}
H = pd.read_csv(os.path.join(RUN, "new_slim/holdout_and_sweep.csv"))
I = H[H["mode"] == "interleave"]
seed0 = I[I.tag.isna() | (I.tag == "")]
det_out = seed0[seed0.arm == "det"].set_index("city").out_MAE
ff0 = seed0[seed0.arm == "ff"].set_index("city")
ff_vs_det = [ff0.out_MAE[c] / det_out[c] for c in CITIES]
ff_deg = [ff0.out_MAE[c] / ff0.in_MAE[c] for c in CITIES]
Hm = I[~I.tag.astype(str).str.startswith("wpde")]
med = Hm.pivot_table(index="city", columns="arm", values="out_MAE", aggfunc="median")
dp_med = [med.dpinn[c] / det_out[c] for c in CITIES]
fs_med = [med.dpinn_fs[c] / det_out[c] for c in CITIES]
rng = lambda v, f="%.2f": (f % min(v)) + "–" + (f % max(v))
lst = lambda v, f="%.2f": ", ".join(f % x for x in v)
print("FF/DPINN in-sample gain", lst(ff_gain), "| per-city nRMSE", rng(nrmse),
      "| DPINN R0 Al Hudaydah %.2f-%.2f" % (dp_hud["R0_min"], dp_hud["R0_max"]),
      "| FF peak R0 %.2f at %s" % (ff_peak, ff_peak_city))

# ---------------------------------------------------------------- 3. text
i = find_md("## 11. Held out")
replace_in(i, "101 training intervals, 33 held out", "101 training intervals, 34 held out")

i = find_md("Three trends, none visible on the defective panel.")
replace_in(i, """And $\\mathcal{R}_0$
**contracts into the admissible band**: 0.89–6.96 at $w_{\\mathrm{pde}}=0.1$, 0.92–2.05 at $10^3$,
against the deterministic arm's own 0.84–1.16. Degradation falls with it, ×4.12 to ×1.76.""",
"""And $\\mathcal{R}_0$
**contracts into the admissible band once the constraint binds**: 0.89–6.96 at
$w_{\\mathrm{pde}}=0.1$, 0.92–2.05 at $10^3$ and 0.92–2.36 at $10^4$. The path is not monotone
(2.57–12.3 at $w_{\\mathrm{pde}}=10$, and a floor of 0.0007 at 100), so the contraction is a property
of the stiff end, not of the sweep. Degradation falls from ×3.73 at $w_{\\mathrm{pde}}=0.1$ to ×1.76
at $10^4$, again not monotonically (×4.12 at 100).""")
replace_in(i, """The honest reading is that a hard-constrained network and the deterministic arm converge
on the same $\\mathcal{R}_0$ — a consistency check both pass, not a reason to prefer the network,
which needs an order of magnitude more compute to reach what the forward solve gives directly.""",
"""The two do **not** converge on the same $\\mathcal{R}_0$: the deterministic arm's own holdout fit at
Al Hudaydah spans 0.68–6.87 (the 0.84–1.16 quoted in an earlier draft is the full-domain fit of
Section 13, a different model). The stiff network's 0.92–2.05 is the narrower of the two, which
shows the penalty restricting $\\mathcal{R}_0$, not two methods agreeing on it.""")
replace_in(i, "21 reporting governorates, 2,785 intervals,", "21 reporting governorates, 2,806 intervals,")

i = find_md("Three things change once the model is spatial.")
replace_in(i, """**The FF-PINN's margin collapses** to 1.6× on MAE against the DPINN, from 5.3–8.5× in the isolated
fits.""", f"""**The FF-PINN's margin collapses** to 1.6× on MAE against the DPINN, from {rng(ff_gain, '%.1f')}× in the
isolated fits.""")
replace_in(i, """**Everything fits worse** — nRMSE 0.45–0.67 against 0.03–0.50 per city.""",
           f"""**Everything fits worse** — nRMSE 0.44–0.67 against {rng(nrmse)} per city.""")
replace_in(i, """The DPINN's $\\mathcal{R}_0$ falls from 3.24–16.31 at Al Hudaydah in
isolation to 0.87–5.37 over the whole domain; the FF-PINN's upper end falls from 49.74 at Hajjah to
11.18.""", f"""The DPINN's $\\mathcal{{R}}_0$ goes from {dp_hud['R0_min']:.2f}–{dp_hud['R0_max']:.2f} at Al
Hudaydah in isolation to 0.79–6.87 over the whole domain; the FF-PINN's upper end falls from
{ff_peak:.2f} at {ff_peak_city} to 10.21.""")
replace_in(i, """Hadramaut and Al Maharah carry 1.5 % (DPINN), 1.7 % (FF-PINN) and 6.4 %""",
           """Hadramaut and Al Maharah carry 1.7 % (DPINN), 1.7 % (FF-PINN) and 6.4 %""")
replace_in(i, """Amran −2.0e-04, Abyan −2.0e-04, Al Mahwit −1.7e-04""",
           """Abyan −2.1e-04, Al Dhale'e −2.0e-04, Amran −1.9e-04""")

i = find_md("**The state embedding buys the in-sample gain and causes the overfitting.**")
replace_in(i, """cuts in-sample MAE by 6-45x at every city and raises degradation from
1.2-2.1 to 8.4-92.6.""", """cuts in-sample MAE by 5-44x at every city and raises degradation from
1.2-2.3 to 8.6-87.7.""")

i = find_md("### The recorded predictions")
replace_in(i, """| 1 | F-DPINN lowest in-sample of the four | **held** at 3 of 4 cities; FF-PINN edges it at Amanat Al Asimah and Hajjah |""",
           """| 1 | F-DPINN lowest in-sample of the four | **held at 2 of 4** (3-seed medians); FF-PINN edges it at Amanat Al Asimah and Hajjah |""")
replace_in(i, """Only prediction 3 fails, and it fails""",
           """Prediction 1 holds only at half the cities and prediction 3 fails, and it fails""")

i = find_md("## 17. What the comparison shows")
start = nb.cells[i].source.index("## 17. What the comparison shows")
nb.cells[i].source = nb.cells[i].source[:start] + f"""## 17. What the comparison shows

All numbers below are from the repaired panel (2,806 intervals); every earlier figure fitted on the
defective panel is withdrawn.

**The Fourier-feature embedding wins the data term in sample, and that is not the result.** FF-PINN
cuts in-sample MAE by {rng(ff_gain, '%.1f')}× over DPINN ({lst(ff_gain, '%.1f')} at Al Hudaydah,
Amanat Al Asimah, Hajjah, Amran). Held out it is worse than the deterministic four-term Fourier series
at every governorate: {lst(ff_vs_det)}× at seed 0 and 1.04–1.48× at the three-seed median. Its
training-to-held-out degradation is {rng(ff_deg, '%.1f')}×, so it is systematically and very
optimistic about its own error. **Neither Fourier-state network generalises better than a low-order
deterministic fit under a hard PDE constraint.**

**The plain-state networks match that fit and no more.** At the three-seed median, DPINN is
{rng(dp_med)}× and DPINN-FS {rng(fs_med)}× the deterministic arm's held-out error. DPINN-FS is the
defensible network default, with the smallest seed spread in the study (1.002–1.040).

**The 2×2 factorial attributes the effects cleanly** (Section 16, three seeds, pairing-free): the
Fourier-state embedding costs ×1.273 on held-out error and is positive at every city; the $\\beta$
basis (×1.003) and the interaction (×1.011) are null.

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
deterministic arm. The holdout is a single every-fourth interleave, not rotated folds. The
deterministic full-domain arm stopped at `max_nfev` and is reported as not converged; its domain row
is a lower bound on that arm's attainable fit. The physics sweep is one city and one seed. Diffusion
is fixed at 50 km²/day and never estimated. The budget-doubling result of an earlier draft was
measured on the defective panel and has not been repeated. Second wave only, and no comparison
against an independently published $R_t$ reconstruction for Yemen.
"""

nbf.write(nb, OUT)
print("wrote", OUT)
