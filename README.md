# Cancer-Imaging: MRI-based glioma treatment-scenario framework

`glioma_scenarios` is a research framework that puts a simulated glioma into a patient's real brain
anatomy from MRI and explores treatment scenarios. It keeps two kinds of information apart:

* **patient-specific**, taken from the scan: brain geometry, tissue and motility maps (plus DTI),
  a permeability proxy (T1-Gd and rCBV), and soft tumour-compartment observations;
* **population-informed**, taken from priors: cell-cycle, drug PK/PD and immune kinetics.

> This is **not** a clinical decision system. It never outputs "this patient should receive this
> dose". It reports which treatment comparisons a single scan can support once uncertainty is
> taken into account, and which it cannot.

## Model

Spatial, age- and phenotype-structured tumour in MRI-derived anatomy (`glioma_scenarios/forward.py`):

| field | meaning |
|---|---|
| `p(x,a,t)` | cycling cells, with cell-cycle age `a` |
| `q(x,t)` | quiescent cells |
| `r(x,t)` | necrotic tissue |
| `E(x,t)` | immune activity |
| `C(x,t)` | tissue drug concentration (plasma `C_p(t)` from a one-compartment PK model) |

```
p_t + p_a = div(D_p grad p) - [beta(a) + mu_p + gamma_pq(sigma,C) + kappa_C(a,C) + kappa_E E] p
p(x,0,t)  = 2 s(N) int beta(a) p da + gamma_qp(sigma) q                       (nonlocal renewal)
C_t       = div(D_C grad C) + k_in V(x) C_p(t) - k_out C
kappa_C   = kappa_max C^h/(EC50^h + C^h) w_phase(a)                            (phase-specific kill)
```

The other terms: stress `sigma = N + w_E E`; contact-inhibited division success
`s(N) = clip(1-N, 0, 1)`; immune recruitment and exhaustion; no-flux boundaries at the brain
surface.

Numerics:

* age is advanced with the method of characteristics;
* reactions use exponential competing hazards, which keeps every population non-negative;
* drug PK is integrated exactly over each step;
* diffusion uses a conservative finite-volume scheme with automatic sub-stepping.

Everything is written in PyTorch, so it is differentiable with respect to the parameters.

**MRI is treated as evidence, not ground truth** (`observation.py`). Hidden `(P, Q, R)` produce
categorical probabilities over (background, necrotic, edema, enhancing), with nested logits and a
partial-volume blur. The observed soft labels enter through a tempered cross-entropy that accounts
for spatial correlation between voxels. Immune activity and drug concentration do not enter the
likelihood, because structural MRI does not measure them.

## Package layout

| module | methodology step |
|---|---|
| `data/upenn.py`, `data/download.py` | UPENN-GBM indexing, pilot-subset selection, IDC download |
| `preprocess.py` | Step 1: mask, normalisation, GMM tissue maps, down-sampling to soft labels, motility (+FA), permeability (T1-Gd, rCBV) |
| `domain.py` | patient domain, finite-volume no-flux diffusion, synthetic phantom |
| `params.py` | Steps 2–3: parameters, log-normal priors, provenance tags (`data-informable` / `prior-only` / `observation`) |
| `forward.py`, `pk.py` | forward solver, dosing schedules |
| `observation.py` | observation model, synthetic segmentations and intensities |
| `synthetic.py` | Step 4: synthetic ground truth and recovery experiments |
| `identifiability.py` | Step 5: Fisher-information directions, parameter verdicts, profile likelihood, simulation-based calibration |
| `pinn.py` | Steps 6–7: renewal-aware, observation-aware PINN with piecewise-in-time networks between doses |
| `inference.py` | Step 8: MAP, Laplace (Fisher or Hessian curvature), HMC, on a few kinetic parameters only (not network weights) |
| `scenarios.py` | Steps 9–10: scenario simulation with common random numbers, treatment-equivalence classes |
| `seeding.py` | effective time since seeding, as a conditional posterior with a sensitivity table |
| `baselines.py` | discrete-loss Fisher-KPP inversion baseline |
| `gradcam.py` | Grad-CAM, for coarse localisation only |
| `pipeline.py`, `report.py` | end-to-end analysis, and JSON plus Markdown reports with the "claims not made" section |

## Install and test

```bash
pip install -r requirements.txt      # numpy, scipy, torch, nibabel, pytest (+ idc-index)
python -m pytest -q
```

## Data: UPENN-GBM

The primary real dataset is [UPENN-GBM on TCIA](https://www.cancerimagingarchive.net/collection/upenn-gbm/):
630 de novo glioblastoma patients, see the [data descriptor](https://www.nature.com/articles/s41597-022-01560-7).

* **Recommended input:** the TCIA NIfTI package, or the analysis-ready copy on the
  [Pittsburgh Fiber Data Hub](https://brain.labsolver.org/upenn_gbm.html). It contains:
  * co-registered, skull-stripped T1, T1-Gd, T2 and FLAIR;
  * expert-revised segmentations (1 = necrotic/non-enhancing core, 2 = edema, 4 = enhancing);
  * DTI maps (FA, TR, AD, RD) and DSC maps (ap-rCBV).

  The indexer finds files by name, so any directory layout works.
* **IDC pilot download:** the full collection is over 1 TB, so start with 10–20 patients.
  ```bash
  python scripts/download_upenn_pilot.py --out data/idc --n 15 --dry-run   # size check
  python scripts/download_upenn_pilot.py --out data/idc --n 15 [--convert data/nifti]
  ```
  IDC holds DICOM: the processed `…: Processed_CaPTk` structural series plus DICOM-SEG. The
  expert NIfTI segmentations and the DTI/DSC derivative maps come from the TCIA NIfTI package.

Pick the pilot subset: baseline scans with all four sequences, an expert segmentation, DTI and
perfusion maps, with known IDH/MGMT preferred.

```bash
python scripts/select_subset.py --root /data/UPENN-GBM --clinical /data/UPENN-GBM/UPENN-GBM_clinical_info_v2.1.csv --n 25
```

**How the data are used:**

* Real UPENN-GBM data provide only MRI → brain geometry, tissue maps and tumour-compartment
  observations.
* Latent cycling, quiescent and necrotic states, immune dynamics, drug PK/PD and treatment effects
  are validated on **synthetic** data.
* The public release has no paired longitudinal treatment-exposure, immune or PK measurements, so
  drug and immune parameters stay literature priors and scenario inputs.

## Running

```bash
# Validation track 1 - synthetic recovery on a phantom (or --root/--case for real anatomy)
python scripts/run_synthetic_experiment.py --out results/synthetic [--robustness]

# Validation track 2 - real anatomy anchoring + scenario analysis on one case (2-D slice for speed)
python scripts/run_patient_scenarios.py --root /data/UPENN-GBM --case UPENN-GBM-00001_11 \
    --factor 3 --slice-axis 2 --out results/UPENN-GBM-00001_11 [--hmc] [--seeding-sensitivity]

# Validation track 3 - FV solver vs Fisher-KPP discrete-loss vs standard PINN vs renewal-aware PINN
python scripts/run_benchmark.py --grid 40 --iters 3000 --out results/benchmark
```

Each analysis writes three files:

* `report.md`: provenance, identifiability table, scenario outcomes, treatment-equivalence classes,
  and the claims the analysis does not make;
* `analysis.json`;
* `trajectories.npz`.

### Reading the scenario output

For every pair of schedules, on a chosen outcome (default: mean viable burden over the horizon),
the comparison gets one of three verdicts:

* **distinguishable**: the paired relative difference is beyond the margin (default 10%) with
  probability ≥ 90%;
* **equivalent at available resolution**: the difference is within the margin with probability
  ≥ 90%;
* **not supported**: neither.

Each comparison also reports:

* the probability of the effect's direction, which can be settled even when its size is not;
* how much of the variance comes from prior-only parameters.

A comparison driven mainly by those priors is labelled *conditional on population priors*.

## Status and limitations

* Default parameter values in `params.py` are illustrative, literature-*informed* ranges for a
  temozolomide-like drug and high-grade glioma. Replace them with sourced values before any
  scientific use.
* Motility is isotropic (scalar, FA-modulated). An anisotropic DTI tensor is a planned extension.
* The permeability proxy `V(x)` is static and derived from enhancement and rCBV. It is not a
  measurement of drug delivery.
* The single-scan inverse problem fixes the seed location at the core centroid. The effective time
  since seeding depends on the model and should always come with the sensitivity table
  (`--seeding-sensitivity`).
* The PINN needs thousands of iterations and weight tuning to converge. Short runs only check the
  code paths. No claim is made that it beats the finite-volume solver; benchmarking that is the
  purpose of `run_benchmark.py`.
* Value-of-information / optimal experiment design (which extra measurement would most reduce
  decision uncertainty) is out of scope and left for future work.
