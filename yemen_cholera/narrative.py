"""Narrative cells for yemen_cholera_extended.ipynb. Method text describes the
design; the *_after cells interpret the results and are written from them."""

TEXT = {}
PENDING = ("*Interpretation of this section is added once every run it depends on is "
           "complete (see the run-status table at the top).*")

TEXT["title"] = r"""
# Yemen cholera, second wave: the extended comparison

This notebook follows on from `yemen_cholera_dpinn_vs_ffpinn_fixed.ipynb` and changes three things
in every arm before adding eleven new analyses.

**What changed relative to the original study**

1. **Corrected loader.** The original dropped each governorate's first reporting interval (21
   intervals, 16,598 cases) and anchored the initial condition on the second one.
2. **Negative-binomial likelihood.** A pilot run under the original Poisson deviance showed the
   residual variance to be about 80× Poisson (median over 21 governorates; implied negative-binomial
   size $k\approx 11$). A Poisson deviance reads that reporting noise as signal: it over-weights
   batch-release spikes and makes every interval too narrow. Every arm (deterministic, PINN, spline,
   SIR, spatial) now uses the NB deviance with a governorate-specific size $k$, estimated by method of
   moments around a penalised-spline fit **on the training rows of each fit only**. The Poisson
   objective is kept as the `poisson` sensitivity variant (§6).
3. **Smoothed initial-condition anchor.** The initial stock was set from the first positive
   interval, which in a sparse series is often a batch release (Al Maharah's was 3.5× its own mean).
   It is now the case-weighted rate over the 14 days from that interval.

And one new spatial architecture (§10): a **governorate network** model replacing diffusion along a
single distance-from-Sana'a axis, which places Shabwah 2 km from Aden and Dhamar 4 km from Amran.

**The eleven analyses**

| § | analysis | what it answers |
|---|---|---|
| 1 | 10 random seeds per PINN | are between-architecture differences larger than initialisation noise? |
| 2 | rotated 4-fold holdout | which arm generalises, pooled over every interval rather than one split |
| 3 | bands on $\beta_1(t)$, $\beta_2(t)$, $\mathcal R(t)$ | how well each quantity is determined |
| 4 | recovery of known values | given a literature-calibrated truth, what comes back? and do the real-data fits agree with published Yemen estimates? |
| 5 | outside reference for $\mathcal R(t)$ | a model-free Cori/EpiEstim $R_t$ from the same counts |
| 6 | sensitivity | the fixed modelling choices: initial condition, age profile, observable, winsorisation, objective, Fourier order, coupling weight |
| 7 | residual diagnostics | is the Poisson likelihood adequate? |
| 8 | simple baselines | a penalised spline and a plain SIR model |
| 9 | all 21 governorates | the deterministic arm and both baselines everywhere |
| 10 | location-varying $\beta(t,x)$ | does transmission that varies by governorate fit the spatial model better? |
| 11 | observed against fitted surfaces | the whole $I(t,x)$ field, not just per-city curves |

**Scope of the PINN fits.** Each PINN fit costs about 3–4 CPU-minutes. Following the study design
agreed for this extension, the four PINN architectures were run with 10 seeds at five governorates
chosen by total reported cases: the top three (Al Hudaydah, Hajjah, Dhamar), the median governorate
(Al Bayda, 11th of 21) and the smallest (Hadramaut). The deterministic arm and both baselines were
run at all 21. Every fit is a separate result file (`results/<city>/<arm>_s<seed>_f<fold>_<variant>.pkl`),
so the tables below rebuild from disk in seconds; `RUN = True` reruns everything.

Day 0 is 2017-05-22; the record runs to day 272 (2018-02-18). Throughout, $\mathcal R(t)$ is the
model's next-generation number including depletion of susceptibles,
$\mathcal R(t)=\bar S(t)\,\kappa_{EI}J_1[\beta_1(t)+\alpha_0\beta_2(t)J_2]$, so it is directly
comparable with an effective reproduction number $R_t$.
"""

TEXT["status"] = r"""
### Run status

Each block of fits, how many were planned and how many result files exist. Sections whose block is
still pending show what has finished so far; their written interpretation is added once the block
completes.
"""

TEXT["cities"] = r"""
## 0. The five focus governorates

Ranked by total reported cases over the window. Selected: ranks 1–3, rank 11 (the median of 21) and
rank 21. They span almost three orders of magnitude in burden, which matters: Hadramaut's record is
mostly zeros and single-digit counts, the regime where a Poisson likelihood is least informative.
"""

TEXT["seeds"] = r"""
## 1. Ten seeds per architecture

Every PINN arm was refitted from 10 random initialisations on the full record at each focus
governorate, with the training budget unchanged from the original notebook (800 pre-training steps,
1,000 Adam, 120 L-BFGS). The deterministic arm has no seed: its optimiser is started from a grid
search and is deterministic. `MAE_cv` is the seed-to-seed coefficient of variation of the in-sample
error; `R_max_lo`–`R_max_hi` is the range over seeds of the largest $\mathcal R(t)$ in the window.
"""
TEXT["seeds_after"] = r"""
**Architecture differences are far larger than initialisation noise, so the comparison between
arms is real.** Seed-to-seed spread in log in-sample MAE is 0.02–0.05 within an arm; the spread
between arms is 0.13–0.41, two to twenty times larger, at every governorate.

**In sample, only the Fourier-feature state networks beat the deterministic arm.** Median in-sample
MAE relative to the deterministic arm:

| | DPINN | DPINN-FS | F-DPINN | FF-PINN |
|---|---|---|---|---|
| four larger governorates | 1.00–1.02 | 1.00–1.03 | **0.43–0.72** | **0.62–0.89** |
| Hadramaut | 1.22 | 1.27 | 1.01 | 1.02 |

The plain-state networks reproduce the deterministic fit and no more; the Fourier-feature ones fit
more of the data. Whether that is signal or noise is §2's question, and §7 already gives a warning:
F-DPINN's residuals have dispersion 0.84 (below the noise level the likelihood assumes) and a lag-1
autocorrelation of −0.55, the pattern of a model that follows individual reporting spikes.

**The seed spread of the *reproduction number* is not small.** Within an arm the largest
$\mathcal R(t)$ in the window varies by up to an order of magnitude across seeds (F-DPINN at Al
Bayda: 34–122; DPINN at Al Bayda: 2.0–21) and, at Hadramaut, DPINN-FS ranges from 4.7 to 78,590. Two seeds that fit
the counts equally well can disagree by orders of magnitude on $\mathcal R$: the amplitude
degeneracy of the original study survives both the hard initial condition and the
negative-binomial likelihood.
"""

TEXT["holdout"] = r"""
## 2. Fuller holdout: rotated 4-fold interleave

The original study held out one fixed set (every 4th interval). Here the held-out offset rotates
through 0, 1, 2, 3, so **every interval is predicted exactly once by a model that never saw it**
(the first interval stays in training in every fold because the initial condition is anchored on
it). Errors are pooled over the four folds. Each PINN arm was run with 10 seeds per fold; the
spline, SIR and deterministic arms are deterministic. `vs_det` is held-out MAE relative to the
deterministic arm; `degradation` is held-out over in-sample MAE.
"""
TEXT["holdout_after"] = r"""
**DPINN generalises; FF-PINN does not.** Pooled over the four rotated folds and ten seeds, held-out
MAE relative to the deterministic arm:

| | Al Hudaydah | Hajjah | Dhamar | Al Bayda | Hadramaut |
|---|---|---|---|---|---|
| DPINN | 0.98 | **0.88** | 0.95 | 0.97 | 1.09 |
| FF-PINN | 1.00 | 1.17 | 1.16 | **1.81** | 0.91 |
| SIR baseline | 0.96 | 0.96 | 0.95 | 0.99 | 0.96 |
| spline baseline | 0.98 | 1.01 | 1.00 | 0.99 | 0.99 |

* **DPINN** beats the deterministic arm out of sample at four of five governorates, and it does so
  with **every one of its ten seeds** (Wilcoxon p = 0.002 at each). It also degrades least from
  training to held-out data (1.02–1.08×). The margin is small (2–12 %), and the plain SIR baseline
  achieves most of it (4–5 %), so it is a modest gain of a smooth curve-fitter, not of the age
  structure.
* **FF-PINN** is the best in-sample arm (§1) and the worst out of sample: 1.16–1.81× the
  deterministic arm's held-out error at three of five governorates, with **every seed** worse at
  each of those three (p = 0.002), and a training-to-held-out degradation of 1.8–5.3× at the four
  larger governorates. Its in-sample
  advantage is overfitting to reporting noise, as §7's residuals already suggested. The exception is
  Hadramaut (0.91×), the sparsest record, where every arm's error is tiny in absolute terms.
* Seed-to-seed spread of held-out error is under 1 % for DPINN and up to 30 % for FF-PINN at Al
  Bayda: the Fourier-feature network's out-of-sample behaviour also depends more on initialisation.

DPINN-FS and F-DPINN, the two cells that separate the state embedding from the $\beta$ basis, are
the last block of the queue; their rows complete the table when it finishes.
"""

TEXT["bands"] = r"""
## 3. Uncertainty bands on $\beta_1(t)$, $\beta_2(t)$ and $\mathcal R(t)$

Two kinds of band, labelled in every legend because they are not the same object:

* **PINN arms**: the 2.5–97.5 % range over the 10 seeds. This is **optimisation / initialisation
  uncertainty**: how differently the same architecture answers on the same data. It is not a
  sampling interval.
* **Deterministic arm**: a 95 % **delta-method** band. The Gauss–Newton covariance at the optimum
  of the NB deviance, scaled by the residual dispersion, is
  propagated *linearly* to $\log\beta_1$, $\log\beta_2$ and $\log\mathcal R$. Linear propagation is
  essential here: the covariance is near-singular along the direction that trades $\beta_1$ against
  $\beta_2$, and pushing random draws along it through the softplus link produces spurious
  swings in $\mathcal R$ that the linearisation correctly assigns zero first-order variance.

The black curve in the $\mathcal R$ row is the model-free Cori estimate of §5. The table gives the
median band width as the ratio upper/lower: 1 is a perfectly determined curve.
"""
TEXT["bands_after"] = r"""
**The deterministic arm's $\mathcal R(t)$ is well determined; its $\beta_1$ and $\beta_2$ are not.**
Its delta-method band on $\mathcal R(t)$ has a median width (upper ÷ lower) of 1.15–2.2 at the four
larger governorates. For the two rates separately it is $10^3$ to unbounded for $\beta_2$, and for
$\beta_1$ anywhere from 1.2 to $10^{15}$: $\beta_1$ is only well determined where the fit has pushed
$\beta_2$ onto its positivity floor (Dhamar, Al Bayda), so that one route carries everything. The
data fixes the weighted sum, not the split.

**PINN bands are narrow, and that is not evidence of accuracy.** The seed bands of FF-PINN are the
tightest of all (width 1.2–1.8 for $\mathcal R$, $\beta_1$, $\beta_2$ at the three largest
governorates),
yet §5 shows FF-PINN's $\mathcal R(t)$ sits about 2× above the model-free reference everywhere. Ten
seeds converging on the same answer show the optimiser is repeatable, not that the answer is right;
a seed band measures initialisation spread, not uncertainty about transmission.
"""

TEXT["recovery"] = r"""
## 4. Does the method recover known values?

**4a. A literature-calibrated truth.** No published study gives $\beta_1(t)$ or $\beta_2(t)$ for
Yemen, so recovery is tested on synthetic counts whose generating values are known and whose
$\mathcal R(t)$ is set to the published second-wave picture: about 2 at the May 2017 surge
(Camacho et al. 2018 report a median $R_t>2$ in 13 of 23 governorates; an initial-stage estimate
from culture-confirmed cases is 2.2 [2.1, 2.3]), between 1 and 2 through the summer, and below 1
by October 2017. The route split is unknown in the literature, so two opposite truths are used:
20 % and 70 % environmental. Counts are Poisson draws on each focus governorate's **real** bulletin
calendar and population, generated from the same single-location model that is fitted, so the
truth is exactly attainable and any miss is the method's. Counts are drawn negative-binomial with
each governorate's own estimated $k$, so the synthetic noise is as heavy as the record's. Three
replicate datasets per truth.
"""
TEXT["literature"] = r"""
**4b. Against published estimates.** The fitted $\mathcal R(t)$ on the real record, averaged over
three windows where published estimates exist, next to the model-free Cori estimate, and the
record's own attack rates against those WHO EMRO reported for the second wave. The attack rates
check the data rather than the method: if they did not match, no fitted quantity could.
"""
TEXT["literature_after"] = r"""
**Against published Yemen estimates** (Camacho et al. 2018): $R_t$ above 2 in May 2017, between 1
and 2 after late May, and below 1 by October. The record here opens on 22 May, after the surge. The
model-free national $R_t$ is 1.43 over the first three weeks, 1.00 over the summer and 0.88 in
October, matching all three published statements. The deterministic arm gives 1.04–1.11 early,
1.00–1.02 in summer and 0.88–0.97 in October at the four larger focus governorates: correct in the
summer and October windows, and low in the opening weeks for the reason given in §5.

**The data check passes.** The record's second-wave attack rates are within 2–6 % of those WHO
EMRO reported: Amran 864 vs 902 per 10,000, Al Mahwit 827 vs 879, Al Dhale'e 625 vs 640, Dhamar 505
vs 525.

**Recovery of known values (deterministic arm).** On synthetic counts with literature-calibrated
$\mathcal R(t)$ and the record's own negative-binomial noise, **$\mathcal R(t)$ is recovered**:
median relative error 4–16 % across governorates and both truths. **The route split is not**: with a
true environmental share of 20 % the estimates run from 25 % to 64 %, and with 70 % from 37 % to
87 %. The individual rates are off by 36–154 % ($\beta_1$) and 35–250 % ($\beta_2$). The data
determines how much transmission there is, not which route carries it. One of the 30 synthetic
datasets (Hadramaut, almost all zeros) produced an epidemic blow-up of the kind described in §8.

**Recovery by the PINN arms.** On the same synthetic data, with $\max\mathcal R=2.0$ in truth:

| | median rel. error in $\mathcal R(t)$ | median peak $\mathcal R$ | estimated env. share (truth 20 % / 70 %) |
|---|---|---|---|
| deterministic | 8–10 % | 2.3 | 36 % / 60 % (range 3–98 %) |
| DPINN | 174–311 % | 6.6–7.6 | 55 % / 63 % |
| FF-PINN | 100 % | 3.8–4.0 | **71 % / 71 %** |

FF-PINN returns a 71 % environmental share **whatever the truth is**: the split it reports comes
from the architecture, not from the data. Both networks get $\mathcal R(t)$ wrong in the same way.
At the four larger governorates they drive susceptibles almost to zero (median final $S$
0.000–0.055, where the generating model, and the deterministic fit, keep 0.69–0.76), run
$\mathcal R$ at 3–7 early and then let it collapse to ≈ 0 as $S$ runs out. Since the counts are
still fitted, depleting susceptibles that never appear as cases is a degree of freedom the soft
physics does not close (see the mass-balance check in §12).
"""

TEXT["cori"] = r"""
## 5. An outside reference for $\mathcal R(t)$

The Cori et al. (2013) estimator used by EpiEstim, implemented in `epiestim.py`. It uses only the
counts and a serial interval, nothing from the SEIP model. Bulletin totals are spread uniformly
across the days of their interval; $R_t$ is estimated over trailing 7-day windows with the EpiEstim
default Gamma(1, 5) prior. Two serial intervals: mean 5.0 d (the value used for the 2017 Yemen
record) and mean 3.7 d (reconstructed from the 1853 Danish epidemics), to show how much the
reference itself depends on that choice.

The table scores each arm's $\mathcal R(t)$ (seed median for PINNs) against the Cori curve: mean
absolute difference, median log-ratio (0 = no bias), the fraction of days inside the Cori 95 %
credible interval, and the rank correlation.
"""
TEXT["cori_after"] = r"""
**After the first three weeks, the model's $\mathcal R(t)$ and the model-free reference agree.**
Over day 30–100 the deterministic arm gives 1.00–1.02 where the Cori estimate gives 1.02–1.05 at
the four larger focus governorates, and in October 0.88–0.97 against 0.87–0.96. Over all 21
governorates the median log-ratio between the deterministic arm and Cori is −0.01 (essentially no
bias), the mean absolute difference 0.16. The model is **smoother**: its $\mathcal R(t)$ is an
order-4 Fourier series, so it sits inside the Cori 95 % interval on only about a third of days,
because the Cori curve follows week-to-week reporting noise.

The one systematic gap is **the first three weeks**: Cori gives 1.4–1.8 at the three largest
governorates, the deterministic arm 1.08–1.11. Part of it is the reference itself: with a 3.7-day instead of a
5-day serial interval the national early value falls from 1.43 to 1.25. The rest is the window
start: the record opens on the decline after the May surge, and a smooth $\beta(t)$ anchored on an
initial condition near quasi-equilibrium does not reproduce a short early excess.

Hadramaut (589 cases in total) is too sparse for either estimate: Cori is undefined for most of the
window and the model's band is wide.

**Every PINN arm puts $\mathcal R(t)$ well above the reference.** Median log-ratio against Cori, by
arm over the five focus governorates: deterministic −0.08 to +0.01 and SIR −0.12 to +0.00 (unbiased),
against FF-PINN +0.65 to +1.20, DPINN-FS +0.64 to +1.93, DPINN +0.88 to +1.11 and F-DPINN +1.40 to
+3.95, i.e. reproduction numbers about 2× to 50× too high. They sit inside the Cori 95 % interval on
0–22 % of days. The PINN arms also put 67–100 % of transmission on the environmental route at every
governorate. Since predicted incidence is fine (§1), this is the amplitude degeneracy: the
networks reach the observed counts with too much transmission acting on states that are too small,
and nothing in the soft physics penalises that trade.
"""

TEXT["sensitivity"] = r"""
## 6. Sensitivity to the fixed modelling choices

Each row changes one choice and refits on the full record; `dMAE_%` and `dRmean_%` are changes
relative to the baseline fit for that city and arm.

| variant | change |
|---|---|
| `trunc` | initial stocks from the truncated-system occupancy instead of the untruncated identity (1.575× and 1.582× smaller) |
| `age_eq`, `age_ln` | exponential (equilibrium) or log-normal initial age profile instead of flat |
| `onset` | observe infectious onsets $\int\sigma E\,da$ instead of the force of infection |
| `nowinsor` | no winsorisation of the 0.5 % largest rates |
| `poisson` | the original Poisson deviance instead of negative binomial |
| `wls` | weighted least squares instead of a deviance |
| `M2`, `M6` | Fourier order of $\beta$: 2 or 6 instead of 4 |
| `wren5`, `wren80` | PINN renewal-coupling weight 5 or 80 instead of 20 (PINN arms only, seed 0) |

Diffusion is the remaining fixed choice; it only exists in the spatial model and is swept in §10.
"""
TEXT["sensitivity_after"] = r"""
**The deterministic arm is robust to every fixed choice except the objective.** Across the five
governorates the median change from baseline is within ±1.5 % in MAE and ±2.2 % in mean
$\mathcal R(t)$ for the initial-condition variants (`trunc`, `age_eq`, `age_ln`), the observable
(`onset`), winsorisation, the Fourier order (`M2`, `M6`) and the Poisson likelihood. Only weighted
least squares breaks it (+77 % MAE, and $\mathcal R$ inflated up to 80×), which is why the original
study moved to a deviance. Even where $\mathcal R(t)$ barely moves, the environmental share swings
from 0 to 0.9 between variants at the same governorate: the split is not determined by the data,
so any change in the setup can move it.

**The PINN arms' $\mathcal R(t)$ is set by tuning, not by the data.** With in-sample fit nearly
unchanged:

* the renewal-coupling weight moves FF-PINN's mean $\mathcal R$ from 3.5 to **213** at Al Bayda and
  from 2.5 to 111 at Hajjah (`wren5`), and DPINN's from 2.7 to 22 at Al Bayda (`wren80`);
* reverting to the Poisson likelihood moves DPINN's mean $\mathcal R$ at Al Hudaydah from 3.7 to 28,
  and lets FF-PINN cut its in-sample MAE by 75–88 % at the four larger governorates, i.e. fit the
  reporting noise more tightly (the NB likelihood was right to stop it);
* the Fourier order `M` has no effect on DPINN at all, as it should, since DPINN's $\beta$ is a free
  network and not a series. This is a control: it confirms the variants are applied as intended.
"""

TEXT["diagnostics"] = r"""
## 7. Residual diagnostics

For each full-record fit (seed 0), Pearson residuals under the **negative-binomial** variance the
fit used ($\mu+\mu^2/k$): the dispersion $\hat\phi=\sum r_i^2/(n-p)$ (1 when the variance model is
right), the same quantity under a Poisson variance for comparison (`dispersion_vs_poisson`, the
pilot's ~80), lag-1 autocorrelation and a Ljung–Box test over 10 lags, and randomized quantile
residuals (Dunn & Smyth), standard normal under a correct model (`rqr_sd` ≈ 1, Shapiro p not small).
"""
TEXT["diagnostics_after"] = r"""
**The negative-binomial likelihood fixes most of the mis-specification; what remains is reporting.**
Under a Poisson variance the deterministic arm's residuals are 69× over-dispersed (median over 21
governorates); under the negative-binomial variance actually used, $\hat\phi\approx2.6$ with a
median size $k\approx9.7$. The randomized quantile residuals are still too wide (sd ≈ 1.8 rather than
1) and non-normal, and **lag-1 autocorrelation is negative** (≈ −0.22) with a significant Ljung–Box
test at about half the governorates. A negative lag-1 correlation is the signature of batch
reporting: an interval that over-reports a backlog is followed by one that under-reports. No
transmission model removes that; a reporting model (delay or backlog) or weekly aggregation would.
The spline baseline shows the same pattern, which confirms it belongs to the data, not to the
SEIP structure.

The PINN arms split by state embedding. DPINN and DPINN-FS look like the deterministic arm
(dispersion ≈ 2.7). The Fourier-feature arms have **lower** dispersion (FF-PINN 1.7, F-DPINN 0.84)
and **more negative** lag-1 autocorrelation (−0.43, −0.55): they absorb part of the reporting noise
into the fitted curve, so consecutive residuals alternate in sign even more strongly.
"""

TEXT["baselines"] = r"""
## 8. Simple baselines

* **Spline**: Poisson P-spline regression of incidence on time (24 cubic B-splines, smoothing chosen
  by AIC on the training rows). No epidemic mechanism; the floor any curve-fitter reaches.
* **SIR**: $S'=-\beta(t)SI$, $I'=\beta(t)SI-\gamma I$, $\gamma=1/7$ d, $\beta(t)$ an order-4 Fourier
  series, Poisson deviance. The simplest mechanistic model with a reproduction number.

Both run through the same rotated 4-fold holdout as every other arm. Held-out MAE at all 21
governorates:
"""
TEXT["baselines_after"] = r"""
**The simple baselines are as good as the age-structured model at predicting held-out
bulletins.** Over all 21 governorates the plain SIR model's held-out MAE is a median 0.977× the
deterministic SEIP arm's (range 0.86–1.19), and it is the best of the three at 17 governorates; the
spline, with no epidemic mechanism at all, is 0.989× (0.85–1.11). In-sample to held-out degradation
is 1.08–1.14 for all three. The data term therefore does not reward the age structure, the
environmental route or the renewal condition: on this record they are carried by the model
because the question needs them (§3, §4), not because the counts demand them.

*Note on the SIR baseline.* Its $\beta$ is bounded so that $\mathcal R_0\le5$, the top of the published
range for cholera. Unbounded, it blew up in two holdout folds at Sa'ada (14 % of the population
infected per day near day 230). Under the negative-binomial deviance a zero count costs only
$2k\log(1+\lambda/k)$ however large the prediction $\lambda$ gets, so a model that can explode is
barely penalised for doing so in a sparse stretch of record. The age-structured arms did not do this
on the real record; the PINN arms are checked in §1.
"""

TEXT["allcities"] = r"""
## 9. All 21 governorates

The deterministic SEIP arm (with its delta-method band) and the SIR baseline at every governorate,
against the Cori reference. Panels are ordered by distance from Sana'a.
"""
TEXT["allcities_after"] = r"""
**At every governorate with enough cases, the deterministic arm's $\mathcal R(t)$ hovers around 1
from June onward and drops below 1 in autumn, tracking the model-free reference.** The deterministic
SEIP and SIR curves are close to each other everywhere: given the same counts, the age structure
changes the shape of $\mathcal R(t)$ very little. The delta-method band on $\mathcal R(t)$ is narrow
wherever counts are large and opens up at the sparse governorates (Hadramaut, Al Maharah, Shabwah),
which is the honest outcome. The fitted environmental share, by contrast, swings between
governorates with no geographic pattern, consistent with §4: it is not identified.
"""

TEXT["spatial"] = r"""
## 10. Transmission that varies by location, and a network instead of an axis

All 21 governorates (2,806 intervals) fitted at once by the deterministic solver, NB deviance.

**Original geometry: diffusion along distance from Sana'a**, $x\in[0, 799.4]$ km,
$D_1=\dots=D_4=50$ km²/day:

* `shared_D50`: one $\beta_j(t)$ for the whole country (20 parameters), as in the original §13;
* `governorate_D50`: $\beta_j(t)\cdot k_g$, one offset per governorate common to both routes,
  ridge-shrunk toward 1 (41 parameters); refitted at $D=5$ and $D=250$ because diffusion is the
  one fixed choice §6 cannot test.

**New geometry: a governorate network** (`network_fit.py`). Each governorate is its own node running
the age-structured SEIP system with no diffusion. Susceptibles at $g$ feel the pressure
$\sum_h M_{gh}\,(\text{pressure generated at }h)$ with $M=(1-m)I+mW$,
$W_{gh}\propto N_h/d_{gh}^2$ over great-circle distances between governorate centroids (the IASC
COD boundaries already in the data), and the mixing fraction $m$ estimated:

* `network`: $\beta_j(t)\cdot k_g$ plus estimated $m$ (42 parameters);
* `isolated`: the same with $m=0$, so `network` − `isolated` is the value of the coupling;
* `network_shared`: estimated $m$ but one national $\beta_j(t)$, so `network` − `network_shared` is
  the value of letting transmission differ by governorate.

AIC is deviance + 2 × parameters. In the network models $\mathcal R_g(t)$ uses the local
$\beta_g$ and $S_g$; with small $m$ it is the within-governorate reproduction number.
"""
TEXT["spatial_after"] = r"""
**The network geometry is what fixes the spatial model.** Ranked by AIC (deviance + 2 × parameters):

| model | parameters | AIC | nRMSE |
|---|---|---|---|
| `network` | 42 | **18,190** | **0.551** |
| `isolated` (no mixing) | 41 | 18,985 | 0.559 |
| `governorate_D5` (1-D axis, $D=5$) | 41 | 21,111 | 0.595 |
| `governorate_D50` | 41 | 21,673 | 0.613 |
| `governorate_D250` | 41 | 22,099 | 0.620 |
| `network_shared` (one national $\beta$) | 21 | 23,604 | 0.597 |
| `shared_D50` (original model) | 20 | 26,952 | 0.663 |

* **Geometry.** Moving from the distance-from-Sana'a axis to one node per governorate is the largest
  single gain. Even with no mixing at all (`isolated`), the network form beats every 1-D diffusion
  fit; on the axis, governorates 2–5 km apart share grid cells and blend into each other.
* **Coupling.** Estimated mixing is small: 0.76 % of a governorate's infection pressure comes from
  elsewhere. It is still worth 795 AIC units for one parameter.
* **Diffusion on the axis** is best at the smallest value tried ($D=5$): the data prefers less
  spatial spreading, consistent with the small estimated mixing.
* **Per-governorate scales** are worth 5,400 AIC units (`network` against `network_shared`), yet the
  fitted scales are modest (0.85–1.04). Most of each governorate's level is already carried by its
  own data-anchored initial state; the scales correct the time course around it.

Every governorate's level bias (fitted ÷ observed mean) now lies between 0.76 and 1.32, against
0.31–5.12 for the original shared model; Shabwah goes from 5.1× to 0.99×, Sa'ada from 2.9× to
0.91×, Marib from 2.0× to 0.87×.
"""

TEXT["surfaces"] = r"""
## 11. Observed against fitted surfaces

The observed incidence field $I(t,x)$ against each fitted field. Rows are governorates ordered by
distance from Sana'a; each bulletin interval is drawn across its own span, **nothing is
interpolated between bulletins**, and the model is drawn as its interval average, the quantity it
was fitted to. The lower row is the signed residual relative to each governorate's mean incidence.
The 3-D view renders the same interval fields. The per-city fields stitch 21 independent fits
together; the 1-D diffusion and network fields each come from one model of the whole country.
"""
TEXT["surfaces_after"] = r"""
**Observed against fitted fields** (cell-wise over all 21 governorates):

* The 21 separate per-governorate fits reproduce the observed field best (nRMSE 0.42, correlation
  0.84). They also use 21 separate transmission histories, about 420 parameters.
* Of the whole-country models, the **network model** is best (nRMSE 0.54, correlation 0.73) with 42
  parameters, against 0.65 and 0.56 for the original shared-$\beta$ diffusion model.
* What no model reproduces is the vertical striping of the observed field: synchronised reporting
  releases across governorates (for example around day 130). Those are visible as alternating
  red/blue columns in every residual panel and are the reporting artefact §7 identifies, not
  transmission.
"""

TEXT["massbalance"] = r"""
## 12. A mass-balance check that separates the arms

Every infection in this model is a reported case: the bulletin observable is the force of
infection on susceptibles, $H=S\,[\dots]$. So over the window, the fall in susceptibles
$S(0)-S(T)$ must equal the cumulative reported attack rate, up to recruitment and background
mortality (about $10^{-4}$ here). A forward solver satisfies this by construction. A network that
enforces the $S$ equation only as a penalty need not, and if it does not, susceptibles can leave
without ever being counted, which is exactly the freedom that lets $\mathcal R$ inflate. The table
gives the depletion (seed median and range) against the reported attack rate; a ratio of 1 is
consistent.

**The deterministic arm passes everywhere** (ratio 1.00–1.11). **Every PINN arm fails, in both
directions.** At Hajjah, DPINN-FS, F-DPINN and FF-PINN remove 99.8 % of susceptibles (ratio ≈ 20)
while 5 % of the population was reported infected; at Al Bayda and Al Hudaydah, FF-PINN's
susceptibles barely move (ratio 0.03–0.09) although 4–5 % of the population was infected; and at
Hadramaut DPINN-FS removes 82 % against a 0.07 % attack rate. Across seeds, the depletion at a single
governorate ranges from 0 to 0.99 for most PINN arms. Cases appear without depleting anyone, or
susceptibles vanish without becoming cases. This is the mechanism behind every $\mathcal R$ result
above: $\mathcal R(t)$ multiplies by $S$, and the networks fit the counts with susceptibles that
are not conserved, so nothing anchors it. Checking $S(0)-S(T)$ against the attack rate is a one-line
test any physics-informed fit of this system should report.
"""

TEXT["conclusions"] = "## 13. What the extended analysis shows\n\n" + PENDING
