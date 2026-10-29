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
TEXT["seeds_after"] = PENDING

TEXT["holdout"] = r"""
## 2. Fuller holdout: rotated 4-fold interleave

The original study held out one fixed set (every 4th interval). Here the held-out offset rotates
through 0, 1, 2, 3, so **every interval is predicted exactly once by a model that never saw it**
(the first interval stays in training in every fold because the initial condition is anchored on
it). Errors are pooled over the four folds. Each PINN arm was run with 10 seeds per fold; the
spline, SIR and deterministic arms are deterministic. `vs_det` is held-out MAE relative to the
deterministic arm; `degradation` is held-out over in-sample MAE.
"""
TEXT["holdout_after"] = PENDING

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
TEXT["bands_after"] = PENDING

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
TEXT["literature_after"] = PENDING

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
TEXT["cori_after"] = PENDING

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
TEXT["sensitivity_after"] = PENDING

TEXT["diagnostics"] = r"""
## 7. Residual diagnostics

For each full-record fit (seed 0), Pearson residuals under the **negative-binomial** variance the
fit used ($\mu+\mu^2/k$): the dispersion $\hat\phi=\sum r_i^2/(n-p)$ (1 when the variance model is
right), the same quantity under a Poisson variance for comparison (`dispersion_vs_poisson`, the
pilot's ~80), lag-1 autocorrelation and a Ljung–Box test over 10 lags, and randomized quantile
residuals (Dunn & Smyth), standard normal under a correct model (`rqr_sd` ≈ 1, Shapiro p not small).
"""
TEXT["diagnostics_after"] = PENDING

TEXT["baselines"] = r"""
## 8. Simple baselines

* **Spline**: Poisson P-spline regression of incidence on time (24 cubic B-splines, smoothing chosen
  by AIC on the training rows). No epidemic mechanism; the floor any curve-fitter reaches.
* **SIR**: $S'=-\beta(t)SI$, $I'=\beta(t)SI-\gamma I$, $\gamma=1/7$ d, $\beta(t)$ an order-4 Fourier
  series, Poisson deviance. The simplest mechanistic model with a reproduction number.

Both run through the same rotated 4-fold holdout as every other arm. Held-out MAE at all 21
governorates:
"""
TEXT["baselines_after"] = PENDING

TEXT["allcities"] = r"""
## 9. All 21 governorates

The deterministic SEIP arm (with its delta-method band) and the SIR baseline at every governorate,
against the Cori reference. Panels are ordered by distance from Sana'a.
"""
TEXT["allcities_after"] = PENDING

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
TEXT["spatial_after"] = PENDING

TEXT["surfaces"] = r"""
## 11. Observed against fitted surfaces

The observed incidence field $I(t,x)$ against each fitted field. Rows are governorates ordered by
distance from Sana'a; each bulletin interval is drawn across its own span, **nothing is
interpolated between bulletins**, and the model is drawn as its interval average, the quantity it
was fitted to. The lower row is the signed residual relative to each governorate's mean incidence.
The 3-D view renders the same interval fields. The per-city fields stitch 21 independent fits
together; the 1-D diffusion and network fields each come from one model of the whole country.
"""
TEXT["surfaces_after"] = PENDING

TEXT["conclusions"] = "## 12. What the extended analysis shows\n\n" + PENDING
