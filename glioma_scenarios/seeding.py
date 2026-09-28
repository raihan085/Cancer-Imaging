"""Effective time since seeding -- a model-dependent posterior, not a tumour age.

``p(T_seed | MRI, model assumptions)`` depends on the assumed seed size,
growth/death rates, detection threshold, observation model and priors.  This
module reports it as a broad conditional distribution together with a
sensitivity analysis over those assumptions.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .inference import SeedingInverseProblem
from .params import ModelParams

DEFAULT_ASSUMPTIONS: Dict[str, Dict[str, float]] = {
    "baseline": {},
    "small seed (x0.3)": {"seed_amp": 0.09},
    "large seed (x3)": {"seed_amp": 0.9},
    "sensitive detection (edema threshold x0.5)": {"obs_th_edema": 0.04},
    "insensitive detection (edema threshold x2)": {"obs_th_edema": 0.16},
}


def summarize_samples(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return dict(median=float(np.median(x)), q05=float(np.quantile(x, 0.05)),
                q95=float(np.quantile(x, 0.95)))


def seeding_time_sensitivity(domain, soft_labels, base: Optional[ModelParams] = None,
                             assumptions: Mapping[str, Mapping[str, float]] = DEFAULT_ASSUMPTIONS,
                             fit_names: Sequence[str] = ("D_white", "beta_max", "mu_p", "T_seed"),
                             n_samples: int = 400, rng: Optional[np.random.Generator] = None,
                             fit_kwargs: Optional[dict] = None, **problem_kwargs) -> List[dict]:
    """Re-fit under alternative assumptions and report the T_seed posterior for each."""
    if "T_seed" not in fit_names:
        raise ValueError("T_seed must be fitted")
    rng = np.random.default_rng() if rng is None else rng
    base = base or ModelParams.defaults()
    rows = []
    for label, upd in assumptions.items():
        ip = SeedingInverseProblem(domain, soft_labels, fit_names, base.with_updates(**upd), **problem_kwargs)
        fit = ip.fit_map(**(fit_kwargs or {}))
        post = ip.laplace(fit.theta)
        draws = post.sample(n_samples, rng)["T_seed"]
        rows.append(dict(assumption=label, overrides=dict(upd), T_seed_days=summarize_samples(draws)))
    return rows


def seeding_statement(rows: List[dict]) -> str:
    lo = min(r["T_seed_days"]["q05"] for r in rows)
    hi = max(r["T_seed_days"]["q95"] for r in rows)
    return (f"Effective time since seeding (model-dependent, not an observed tumour age): "
            f"90% intervals across the tested assumptions span {lo:.0f}-{hi:.0f} days.")
