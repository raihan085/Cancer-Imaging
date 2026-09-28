"""Synthetic ground-truth experiments (Step 4 and validation track 1).

1. choose known parameters; 2. simulate with the trusted finite-volume solver;
3. convert hidden biology into MRI-like observations with the observation
model; 4. add noise, boundary errors and segmentation errors; 5. give *only*
the observations to the inverse method; 6. check recovery of the hidden state
and of the parameter combinations that should be recoverable.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .domain import BrainDomain
from .forward import GliomaModel, param_tensors
from .identifiability import classify_parameters, information_directions
from .inference import DEFAULT_FIT, SeedingInverseProblem
from .observation import ObservationModel, synthesize_intensities, synthesize_observation
from .params import ModelParams


def make_synthetic_case(domain: BrainDomain, truth: ModelParams, center_vox: Sequence[float],
                        rng: np.random.Generator, dt: float = 0.25, boundary_noise: float = 0.7,
                        corr_vox: float = 1.5, label_flip: float = 0.0, v0: float = 0.3,
                        with_intensities: bool = False) -> Dict:
    model = GliomaModel(domain, dt=dt)
    P = param_tensors(truth)
    with torch.no_grad():
        state = model.grow_from_seed(P, center_vox)
        obs_model = ObservationModel(domain)
        pi = obs_model.probabilities_from_state(state, P).numpy()
    obs = synthesize_observation(pi, domain.mask, rng, boundary_noise, corr_vox, label_flip)
    # permeability proxy as the pipeline would derive it from enhancement
    perm = np.where(domain.mask, v0 + (1 - v0) * obs["soft"][3], 0.0)
    dom = dc_replace(domain, soft_labels=obs["soft"], permeability=perm,
                     meta={**domain.meta, "synthetic": True})
    out = dict(domain=dom, truth=truth, state=state.to_numpy(), pi=pi, obs=obs, center=tuple(center_vox))
    if with_intensities:
        out["images"] = synthesize_intensities(pi, domain, rng)
    return out


def _rel_l2(a, b, mask):
    a, b = a[mask], b[mask]
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def recovery_experiment(case: Dict, fit_names: Sequence[str] = DEFAULT_FIT, dt: float = 0.25,
                        fit_kwargs: Optional[dict] = None, soft: bool = True) -> Dict:
    dom = case["domain"]
    L = case["obs"]["soft"] if soft else case["obs"]["onehot"]
    ip = SeedingInverseProblem(dom, L, fit_names, center_vox=case["center"], dt=dt)
    fit = ip.fit_map(**(fit_kwargs or {}))
    post = ip.laplace(fit.theta)
    F = ip.fisher_information(fit.theta)
    truth = case["truth"]
    theta_true = np.log([truth[k] for k in fit_names])
    sd = np.sqrt(np.diag(post.cov))
    z = (fit.theta - theta_true) / sd
    with torch.no_grad():
        st = ip.state(fit.theta).to_numpy()
    m = dom.mask
    ts = case["state"]
    dirs = information_directions(F, list(fit_names))
    combo = []
    for d in dirs:
        v = np.array([d["loadings"][k] for k in fit_names])
        err = float(v @ (fit.theta - theta_true))
        sdv = float(np.sqrt(v @ post.cov @ v))
        combo.append(dict(combination=d["combination"], informative=d["informative"],
                          info_ratio=d["info_ratio"], error=err, posterior_sd=sdv,
                          covered_90=bool(abs(err) <= 1.645 * sdv)))
    return dict(
        fitted={k: float(np.exp(v)) for k, v in zip(fit_names, fit.theta)},
        truth={k: float(truth[k]) for k in fit_names},
        z_scores={k: float(v) for k, v in zip(fit_names, z)},
        covered_90={k: bool(abs(v) <= 1.645) for k, v in zip(fit_names, z)},
        posterior=post.summary(),
        correlation=post.correlation().tolist(),
        identifiability=classify_parameters(F, list(fit_names), include_unfitted=False),
        combinations=combo,
        state_rel_l2=dict(P=_rel_l2(st["P"], ts["P"], m), q=_rel_l2(st["q"], ts["q"], m),
                          r=_rel_l2(st["r"], ts["r"], m)),
        fit_seconds=fit.seconds,
        laplace=post,
        theta_map=fit.theta,
    )


def segmentation_robustness(domain: BrainDomain, truth: ModelParams, center_vox, levels=(0.0, 0.5, 1.0, 1.5),
                            seed: int = 0, dt: float = 0.5, fit_kwargs: Optional[dict] = None,
                            fit_names: Sequence[str] = DEFAULT_FIT) -> list:
    """Recovery as a function of boundary-noise level of the synthetic segmentation."""
    rows = []
    for lvl in levels:
        rng = np.random.default_rng(seed)
        case = make_synthetic_case(domain, truth, center_vox, rng, dt=dt, boundary_noise=lvl)
        r = recovery_experiment(case, fit_names, dt=dt, fit_kwargs=fit_kwargs)
        rows.append(dict(boundary_noise=lvl, state_rel_l2=r["state_rel_l2"], z_scores=r["z_scores"],
                         combinations=[(c["combination"], c["covered_90"]) for c in r["combinations"]
                                       if c["informative"]]))
    return rows
