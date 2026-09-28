"""Model parameters, population-informed priors and provenance tags.

Units used throughout the package
---------------------------------
* space: millimetres (mm)
* time: days
* tumour densities (p, q, r): fraction of local carrying capacity (dimensionless)
* immune activity E: dimensionless activity index
* drug concentration (plasma and tissue): micromolar (uM)

Every parameter carries a *provenance* tag.  The tag is used by the reporting
layer so that prior-driven quantities are never presented as if they had been
measured from the patient's scan:

* ``"data-informable"`` -- may be (partly) constrained by the MRI observation;
  whether it actually is must be checked with the identifiability tools.
* ``"prior-only"`` -- a single pre-treatment MRI carries no information about it
  (drug PK/PD, immune kinetics).  It stays a population prior / scenario input.
* ``"observation"`` -- observation-model (imaging) nuisance parameter.

The default values and prior widths below are *illustrative, literature-
informed ranges* (temozolomide-like alkylating agent, high-grade glioma
kinetics).  They must be reviewed and replaced with sourced values before any
scientific use; they are not validated clinical constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    """Specification of one positive model parameter with a log-normal prior."""

    name: str
    default: float
    prior_median: float
    prior_log_sd: float
    provenance: str
    unit: str
    description: str
    lower: float = 0.0
    upper: float = math.inf

    def log_prior(self, log_value: float) -> float:
        z = (log_value - math.log(self.prior_median)) / self.prior_log_sd
        return -0.5 * z * z - math.log(self.prior_log_sd * math.sqrt(2 * math.pi))


def _p(name, default, sd, prov, unit, desc, median=None, lower=0.0, upper=math.inf):
    return ParamSpec(
        name=name,
        default=default,
        prior_median=default if median is None else median,
        prior_log_sd=sd,
        provenance=prov,
        unit=unit,
        description=desc,
        lower=lower,
        upper=upper,
    )


DATA = "data-informable"
PRIOR = "prior-only"
OBS = "observation"

#: Registry of all positive model parameters (log-normal priors).
PARAM_SPECS: Dict[str, ParamSpec] = {
    s.name: s
    for s in [
        # --- tumour motility -------------------------------------------------
        _p("D_white", 0.25, 0.8, DATA, "mm^2/day", "tumour-cell diffusivity in white matter"),
        _p("gray_white_ratio", 0.2, 0.4, PRIOR, "-", "gray/white matter motility ratio"),
        # --- cell cycle ------------------------------------------------------
        _p("a_min", 1.5, 0.3, PRIOR, "day", "minimum cell-cycle age before division is possible"),
        _p("beta_max", 0.7, 0.5, DATA, "1/day", "division hazard for cycle age > a_min"),
        _p("mu_p", 0.08, 0.6, DATA, "1/day", "baseline death rate of cycling cells"),
        # --- phenotype switching / stress ---------------------------------------
        _p("gamma_pq", 0.4, 0.6, DATA, "1/day", "max stress-induced cycling->quiescent rate"),
        _p("gamma_qp", 0.05, 0.7, PRIOR, "1/day", "quiescent->cycling re-entry rate at low stress"),
        _p("sigma_half", 0.7, 0.3, PRIOR, "-", "stress level giving half-maximal quiescence entry"),
        _p("mu_q", 0.15, 0.6, DATA, "1/day", "stress-driven necrosis rate of quiescent cells"),
        _p("lambda_r", 0.01, 0.8, PRIOR, "1/day", "clearance rate of necrotic tissue"),
        # --- immune field ------------------------------------------------------
        _p("D_E", 0.5, 0.8, PRIOR, "mm^2/day", "immune-activity diffusivity"),
        _p("a_E", 0.05, 0.8, PRIOR, "1/day", "tumour-driven immune recruitment rate"),
        _p("k_E", 0.3, 0.6, PRIOR, "-", "tumour burden for half-maximal recruitment"),
        _p("delta_E", 0.1, 0.6, PRIOR, "1/day", "immune decay rate"),
        _p("eps_E", 0.2, 0.8, PRIOR, "1/day", "immune exhaustion rate per unit tumour"),
        _p("kappa_E", 0.1, 0.8, PRIOR, "1/day", "immune kill rate per unit immune activity"),
        _p("w_E", 0.5, 0.6, PRIOR, "-", "contribution of immune pressure to stress"),
        # --- drug pharmacokinetics (temozolomide-like, illustrative) ---------------
        _p("c_per_mg", 0.26, 0.3, PRIOR, "uM/mg", "plasma peak concentration per mg dose"),
        _p("k_el", 9.2, 0.25, PRIOR, "1/day", "plasma elimination rate (t1/2 ~ 1.8 h)"),
        _p("k_in", 4.0, 0.5, PRIOR, "1/day", "plasma->tissue transfer rate at permeability V=1"),
        _p("k_out", 6.0, 0.5, PRIOR, "1/day", "tissue clearance rate"),
        _p("D_C", 5.0, 0.5, PRIOR, "mm^2/day", "drug diffusivity in tissue"),
        # --- drug pharmacodynamics --------------------------------------------------
        _p("kappa_max", 1.5, 0.7, PRIOR, "1/day", "maximum drug kill rate of cycling cells"),
        _p("EC50", 5.0, 0.7, PRIOR, "uM", "tissue concentration for half-maximal kill"),
        _p("hill", 1.5, 0.3, PRIOR, "-", "Hill coefficient of drug effect"),
        _p("phase_center", 1.0, 0.3, PRIOR, "day", "cycle age of peak drug sensitivity (S-phase proxy)"),
        _p("phase_width", 0.4, 0.4, PRIOR, "day", "width of the phase-sensitivity window"),
        _p("phase_base", 0.2, 0.5, PRIOR, "-", "sensitivity outside the sensitive phase (0-1)", upper=1.0),
        _p("q_drug_sens", 0.1, 0.6, PRIOR, "-", "relative drug sensitivity of quiescent cells", upper=1.0),
        _p("gamma_C", 0.2, 0.7, PRIOR, "1/day", "drug-induced arrest (cycling->quiescent) rate"),
        # --- seeding --------------------------------------------------------------------
        _p("T_seed", 60.0, 0.5, DATA, "day", "effective time since seeding (model-dependent)"),
        _p("seed_amp", 0.3, 0.5, PRIOR, "-", "peak density of the initial seed"),
        _p("seed_radius", 2.0, 0.3, PRIOR, "mm", "radius of the initial seed"),
        # --- observation model ------------------------------------------------------------
        _p("obs_th_edema", 0.08, 0.5, OBS, "-", "total density at which FLAIR abnormality appears"),
        _p("obs_th_enh", 0.2, 0.3, OBS, "-", "viable density at which enhancement dominates"),
        _p("obs_th_nec", 0.2, 0.4, OBS, "-", "necrotic density at which necrotic core dominates"),
        _p("obs_slope", 25.0, 0.3, OBS, "-", "logit slope of the observation model"),
        _p("obs_blur_mm", 1.5, 0.4, OBS, "mm", "partial-volume blur (Gaussian sigma)"),
    ]
}


@dataclass
class ModelParams:
    """Concrete parameter values.  Missing entries fall back to the defaults."""

    values: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        unknown = set(self.values) - set(PARAM_SPECS)
        if unknown:
            raise KeyError(f"unknown parameters: {sorted(unknown)}")

    def __getitem__(self, name: str) -> float:
        if name in self.values:
            return self.values[name]
        return PARAM_SPECS[name].default

    def get(self, name: str, default=None):
        return self[name] if name in PARAM_SPECS else default

    def with_updates(self, **updates: float) -> "ModelParams":
        v = dict(self.values)
        v.update(updates)
        return ModelParams(v)

    def as_dict(self) -> Dict[str, float]:
        return {k: self[k] for k in PARAM_SPECS}

    @classmethod
    def defaults(cls) -> "ModelParams":
        return cls({})


def names_by_provenance(provenance: str) -> List[str]:
    return [n for n, s in PARAM_SPECS.items() if s.provenance == provenance]


def sample_prior(
    names: Iterable[str], n: int, rng: Optional[np.random.Generator] = None
) -> Dict[str, np.ndarray]:
    """Draw ``n`` samples of the named parameters from their log-normal priors."""
    rng = np.random.default_rng() if rng is None else rng
    out = {}
    for name in names:
        s = PARAM_SPECS[name]
        x = np.exp(math.log(s.prior_median) + s.prior_log_sd * rng.standard_normal(n))
        out[name] = np.clip(x, s.lower if s.lower > 0 else 0.0, s.upper)
    return out


def log_prior(values: Mapping[str, float]) -> float:
    return sum(PARAM_SPECS[k].log_prior(math.log(v)) for k, v in values.items())


def provenance_table(fitted: Iterable[str] = ()) -> List[dict]:
    """Rows describing where each parameter's value comes from, for reports."""
    fitted = set(fitted)
    rows = []
    for name, s in PARAM_SPECS.items():
        if name in fitted:
            source = "fitted to MRI (check identifiability report)"
        elif s.provenance == PRIOR:
            source = "population prior / scenario input (not measured from this patient)"
        elif s.provenance == OBS:
            source = "observation-model prior"
        else:
            source = "population prior (not fitted in this run)"
        rows.append(
            dict(
                name=name,
                unit=s.unit,
                prior_median=s.prior_median,
                prior_log_sd=s.prior_log_sd,
                provenance=s.provenance,
                source=source,
                description=s.description,
            )
        )
    return rows


__all__ = [
    "ParamSpec",
    "PARAM_SPECS",
    "ModelParams",
    "DATA",
    "PRIOR",
    "OBS",
    "names_by_provenance",
    "sample_prior",
    "log_prior",
    "provenance_table",
]
