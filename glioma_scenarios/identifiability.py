"""Identifiability analysis (Step 5).

Question: if two different parameter choices produce almost the same MRI
observation, can the data tell them apart?

Tools
-----
* :func:`information_directions` -- prior-whitened Fisher-information
  eigen-analysis.  Each eigenvector is a combination of log-parameters (i.e. a
  power-law product such as ``beta_max^0.7 * mu_p^-0.6``); its eigenvalue is
  the data-to-prior information ratio along that direction.
* :func:`classify_parameters` -- reports each parameter as *identifiable*,
  *identifiable only in combination* or *not identifiable*.
* :func:`profile_likelihood` -- profile of the negative log-likelihood.
* :func:`simulation_based_calibration` -- rank statistics of true values
  among posterior draws across repeated synthetic experiments.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .params import PARAM_SPECS, PRIOR

#: posterior/prior variance ratio 1/(1+lambda); lambda=3 -> variance reduced 4x
INFORMATIVE_RATIO = 3.0


def information_directions(F: np.ndarray, names: Sequence[str],
                           threshold: float = INFORMATIVE_RATIO) -> List[dict]:
    S = np.diag([PARAM_SPECS[n].prior_log_sd for n in names])
    Fw = S @ F @ S
    lam, V = np.linalg.eigh(0.5 * (Fw + Fw.T))
    order = np.argsort(lam)[::-1]
    out = []
    for j in order:
        v = V[:, j]
        v = v / np.abs(v).max()
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        expr = " * ".join(f"{n}^{c:+.2f}" for n, c in zip(names, v) if abs(c) >= 0.1)
        out.append(dict(
            info_ratio=float(lam[j]),
            posterior_to_prior_var=float(1.0 / (1.0 + max(lam[j], 0.0))),
            informative=bool(lam[j] >= threshold),
            loadings={n: float(c) for n, c in zip(names, v)},
            combination=expr,
        ))
    return out


def classify_parameters(F: np.ndarray, names: Sequence[str], contraction_threshold: float = 0.75,
                        loading_threshold: float = 0.3, threshold: float = INFORMATIVE_RATIO,
                        include_unfitted: bool = True) -> List[dict]:
    """Per-parameter identifiability verdicts from the Fisher information."""
    dirs = information_directions(F, names, threshold)
    S = np.diag([PARAM_SPECS[n].prior_log_sd for n in names])
    post = np.linalg.inv(np.linalg.inv(S @ S) + F)
    rows = []
    for i, n in enumerate(names):
        contraction = 1.0 - post[i, i] / S[i, i] ** 2
        in_combo = [d["combination"] for d in dirs
                    if d["informative"] and abs(d["loadings"][n]) >= loading_threshold]
        if contraction >= contraction_threshold:
            verdict = "identifiable"
        elif in_combo:
            verdict = "identifiable only in combination"
        else:
            verdict = "not identifiable (prior-dominated)"
        rows.append(dict(name=n, contraction=float(contraction), verdict=verdict, combinations=in_combo))
    if include_unfitted:
        for n, s in PARAM_SPECS.items():
            if n in names or s.provenance != PRIOR:
                continue
            rows.append(dict(name=n, contraction=0.0,
                             verdict="not identifiable from a single pre-treatment MRI (population prior)",
                             combinations=[]))
    return rows


def profile_likelihood(problem, name: str, grid: Sequence[float], theta_map: np.ndarray,
                       n_adam: int = 30, lr: float = 0.05) -> Dict[str, np.ndarray]:
    """Profile of the negative log-likelihood of ``problem`` (a SeedingInverseProblem).

    ``grid`` holds parameter values (natural scale).  For each value the other
    fitted log-parameters are re-optimised (MAP including priors) and the
    *likelihood* part is reported.  A profile whose rise stays below 1.92
    (chi^2_1 95% / 2) over the grid is flagged ``flat`` -> practically
    non-identifiable.
    """
    i = problem.names.index(name)
    free = [j for j in range(len(theta_map)) if j != i]
    prof = []
    for val in grid:
        th = torch.as_tensor(theta_map, dtype=torch.float64).clone()
        th[i] = float(np.log(val))
        sub = th[free].clone().requires_grad_(True)
        opt = torch.optim.Adam([sub], lr=lr)
        for _ in range(n_adam):
            opt.zero_grad()
            full = th.clone()
            full[free] = sub
            problem.neg_log_post(full).backward()
            opt.step()
        full = th.clone()
        full[free] = sub.detach()
        with torch.no_grad():
            ll, _ = problem.terms(full)
        prof.append(-float(ll))
    prof = np.array(prof)
    rise = prof - prof.min()
    return dict(grid=np.asarray(grid, float), neg_log_lik=prof, rise=rise,
                flat=bool(rise.max() < 1.92))


def simulation_based_calibration(simulate_and_fit: Callable[[np.random.Generator], tuple],
                                 n_reps: int = 20, n_post: int = 99,
                                 rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """Simulation-based calibration.

    ``simulate_and_fit(rng)`` must draw true log-parameters from the prior,
    simulate an observation, fit it and return ``(theta_true, posterior)``
    where ``posterior`` has a ``sample(n, rng)`` method returning a dict of
    natural-scale arrays keyed by parameter name (e.g. LaplacePosterior).
    Ranks of the true values among posterior draws should be uniform on
    ``0..n_post`` when the posterior is calibrated.
    """
    rng = np.random.default_rng() if rng is None else rng
    ranks = []
    names = None
    for _ in range(n_reps):
        theta_true, post = simulate_and_fit(rng)
        draws = post.sample(n_post, rng)
        names = post.names
        ranks.append([int((np.log(draws[n]) < theta_true[k]).sum()) for k, n in enumerate(names)])
    ranks = np.array(ranks)
    # chi-square uniformity test with 4 bins per parameter
    bins = np.linspace(0, n_post + 1, 5)
    chi = {}
    for k, n in enumerate(names):
        h, _ = np.histogram(ranks[:, k], bins=bins)
        e = len(ranks) / 4
        chi[n] = float(((h - e) ** 2 / e).sum())
    return dict(ranks=ranks, names=np.array(names), chi2_4bins=chi)
