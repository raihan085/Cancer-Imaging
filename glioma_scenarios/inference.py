"""Inverse inference on a single scan: MAP fit, Laplace approximation and HMC.

The inverse problem fits a *small* set of log-parameters theta (by default the
data-informable tumour kinetics and the effective time since seeding) so that
untreated growth from a seed, pushed through the observation model, explains
the observed soft compartment labels:

    -log p(theta | L) = -w_obs sum_x sum_k L_k log pi_k(x; theta) - log p(theta) + const

All other parameters stay at their population values.  Uncertainty is placed
on theta only -- not on neural-network weights -- via a Laplace approximation
(fast, local) or Hamiltonian Monte Carlo (more complete, more expensive).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .domain import BrainDomain
from .forward import GliomaModel, TumorState, param_tensors
from .observation import ObservationModel
from .params import PARAM_SPECS, ModelParams

DEFAULT_FIT = ("D_white", "beta_max", "mu_p", "T_seed")


@dataclass
class LaplacePosterior:
    names: List[str]
    mean: np.ndarray  # log-space MAP
    cov: np.ndarray  # log-space covariance
    hessian_post: np.ndarray
    hessian_lik: np.ndarray

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng() if rng is None else rng
        z = rng.multivariate_normal(self.mean, self.cov, size=n)
        return {k: np.exp(z[:, i]) for i, k in enumerate(self.names)}

    def summary(self) -> List[dict]:
        sd = np.sqrt(np.diag(self.cov))
        rows = []
        for i, k in enumerate(self.names):
            s = PARAM_SPECS[k]
            rows.append(dict(
                name=k,
                map=float(np.exp(self.mean[i])),
                ci90=[float(np.exp(self.mean[i] - 1.645 * sd[i])), float(np.exp(self.mean[i] + 1.645 * sd[i]))],
                log_sd_posterior=float(sd[i]),
                log_sd_prior=s.prior_log_sd,
                contraction=float(1 - sd[i] ** 2 / s.prior_log_sd ** 2),
            ))
        return rows

    def correlation(self) -> np.ndarray:
        sd = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(sd, sd)


@dataclass
class FitResult:
    theta: np.ndarray
    names: List[str]
    history: List[float] = field(default_factory=list)
    seconds: float = 0.0

    def params(self, base: ModelParams) -> ModelParams:
        return base.with_updates(**{k: float(np.exp(v)) for k, v in zip(self.names, self.theta)})


class SeedingInverseProblem:
    """Single-scan inverse problem based on the finite-volume forward solver."""

    def __init__(self, domain: BrainDomain, soft_labels: Optional[np.ndarray] = None,
                 fit_names: Sequence[str] = DEFAULT_FIT, base_params: Optional[ModelParams] = None,
                 center_vox: Optional[Sequence[float]] = None, dt: float = 0.25,
                 obs_weight: Optional[float] = None, max_prior_sd: float = 4.0):
        self.domain = domain
        self.names = list(fit_names)
        self.base = base_params or ModelParams.defaults()
        self.model = GliomaModel(domain, dt=dt)
        self.obs = ObservationModel(domain, obs_weight=obs_weight)
        L = soft_labels if soft_labels is not None else domain.soft_labels
        if L is None:
            raise ValueError("soft labels are required")
        self.L = torch.as_tensor(L, dtype=torch.float64)
        self.center = tuple(center_vox) if center_vox is not None else domain.tumour_core_centroid()
        self.mu0 = np.array([math.log(PARAM_SPECS[k].prior_median) for k in self.names])
        self.sd0 = np.array([PARAM_SPECS[k].prior_log_sd for k in self.names])
        self.lo = self.mu0 - max_prior_sd * self.sd0
        self.hi = self.mu0 + max_prior_sd * self.sd0

    # ---- core ------------------------------------------------------------------------
    def _tensors(self, theta: torch.Tensor):
        theta = torch.maximum(torch.minimum(theta, torch.as_tensor(self.hi)), torch.as_tensor(self.lo))
        return param_tensors(self.base, {k: torch.exp(theta[i]) for i, k in enumerate(self.names)})

    def state(self, theta) -> TumorState:
        theta = torch.as_tensor(theta, dtype=torch.float64)
        P = self._tensors(theta)
        return self.model.grow_from_seed(P, self.center)

    def terms(self, theta: torch.Tensor):
        P = self._tensors(theta)
        s = self.model.grow_from_seed(P, self.center)
        pi = self.obs.probabilities_from_state(s, P)
        ll = self.obs.log_likelihood(pi, self.L)
        mu0, sd0 = torch.as_tensor(self.mu0), torch.as_tensor(self.sd0)
        lp = -0.5 * (((theta - mu0) / sd0) ** 2).sum()
        return ll, lp

    def neg_log_post(self, theta: torch.Tensor) -> torch.Tensor:
        ll, lp = self.terms(theta)
        return -(ll + lp)

    def value_and_grad(self, theta: np.ndarray, likelihood_only: bool = False):
        th = torch.as_tensor(theta, dtype=torch.float64).clone().requires_grad_(True)
        ll, lp = self.terms(th)
        f = -ll if likelihood_only else -(ll + lp)
        (g,) = torch.autograd.grad(f, th)
        return float(f), g.numpy().copy()

    # ---- MAP -----------------------------------------------------------------------------
    def fit_map(self, theta0: Optional[np.ndarray] = None, n_adam: int = 150, lr: float = 0.05,
                n_lbfgs: int = 20, verbose: bool = False) -> FitResult:
        t0 = time.time()
        th = torch.as_tensor(self.mu0 if theta0 is None else theta0, dtype=torch.float64).clone()
        th.requires_grad_(True)
        hist = []
        opt = torch.optim.Adam([th], lr=lr)
        for i in range(n_adam):
            opt.zero_grad()
            f = self.neg_log_post(th)
            f.backward()
            opt.step()
            with torch.no_grad():
                th.clamp_(torch.as_tensor(self.lo), torch.as_tensor(self.hi))
            hist.append(float(f.detach()))
            if verbose and i % 10 == 0:
                print(f"[MAP adam {i}] -logpost={float(f.detach()):.3f} theta={np.exp(th.detach().numpy())}")
        if n_lbfgs:
            opt = torch.optim.LBFGS([th], lr=0.5, max_iter=n_lbfgs, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                f = self.neg_log_post(th)
                f.backward()
                return f

            try:
                opt.step(closure)
                hist.append(float(self.neg_log_post(th.detach())))
            except RuntimeError:  # pragma: no cover - line search failures are non-fatal
                pass
        theta = np.clip(th.detach().numpy(), self.lo, self.hi)
        return FitResult(theta, self.names, hist, time.time() - t0)

    # ---- curvature / Laplace --------------------------------------------------------------
    def hessian(self, theta: np.ndarray, eps: float = 1e-3, likelihood_only: bool = False) -> np.ndarray:
        """Central finite differences of the autograd gradient (2k backward passes)."""
        k = len(theta)
        H = np.zeros((k, k))
        for i in range(k):
            e = np.zeros(k)
            e[i] = eps
            _, gp = self.value_and_grad(theta + e, likelihood_only)
            _, gm = self.value_and_grad(theta - e, likelihood_only)
            H[i] = (gp - gm) / (2 * eps)
        return 0.5 * (H + H.T)

    def probabilities(self, theta) -> torch.Tensor:
        with torch.no_grad():
            th = torch.as_tensor(theta, dtype=torch.float64)
            P = self._tensors(th)
            s = self.model.grow_from_seed(P, self.center)
            return self.obs.probabilities_from_state(s, P)

    def jacobian_probs(self, theta: np.ndarray, eps: float = 1e-2) -> np.ndarray:
        """d pi / d theta by central differences, shape (k, 4, *grid)."""
        J = []
        for i in range(len(theta)):
            e = np.zeros(len(theta))
            e[i] = eps
            J.append(((self.probabilities(theta + e) - self.probabilities(theta - e)) / (2 * eps)).numpy())
        return np.stack(J)

    def fisher_information(self, theta: np.ndarray, eps: float = 1e-2) -> np.ndarray:
        """Expected Fisher information of the tempered categorical likelihood.

        F = w_obs sum_x sum_k (d pi_k / d theta)(d pi_k / d theta)^T / pi_k  -- always PSD.
        """
        pi = self.probabilities(theta).numpy()
        J = self.jacobian_probs(theta, eps)
        m = self.domain.mask
        Jm = J[..., m]  # (k, 4, nvox)
        w = 1.0 / np.clip(pi[..., m], 1e-6, None)
        F = np.einsum("icv,jcv,cv->ij", Jm, Jm, w)
        return self.obs.obs_weight * 0.5 * (F + F.T)

    def laplace(self, theta_map: np.ndarray, eps: float = 1e-2, curvature: str = "fisher") -> LaplacePosterior:
        """Gaussian approximation around the MAP in log-parameter space.

        ``curvature="fisher"`` (default) uses the expected Fisher information,
        which is positive semi-definite even when the optimiser has not fully
        converged; ``"hessian"`` uses finite differences of the exact gradient.
        """
        if curvature == "fisher":
            H_lik = self.fisher_information(theta_map, eps)
        else:
            H_lik = self.hessian(theta_map, eps, likelihood_only=True)
        H_prior = np.diag(1.0 / self.sd0 ** 2)
        H_post = _make_pd(H_lik + H_prior, floor=1e-6)
        cov = np.linalg.inv(H_post)
        return LaplacePosterior(self.names, np.asarray(theta_map, float), cov, H_post, H_lik)

    # ---- HMC -------------------------------------------------------------------------------
    def hmc(self, theta0: np.ndarray, n_samples: int = 200, n_warmup: int = 50, step_size: float = 0.1,
            n_leapfrog: int = 8, mass_diag: Optional[np.ndarray] = None,
            rng: Optional[np.random.Generator] = None, verbose: bool = False) -> Dict[str, np.ndarray]:
        """Plain HMC with a diagonal mass matrix (e.g. inverse Laplace variances).

        Step size is adapted during warm-up towards ~0.7 acceptance.
        """
        rng = np.random.default_rng() if rng is None else rng
        M = np.ones(len(theta0)) if mass_diag is None else np.asarray(mass_diag, float)
        th = np.asarray(theta0, float).copy()
        U, g = self.value_and_grad(th)
        samples, acc = [], 0
        for it in range(n_warmup + n_samples):
            p0 = rng.standard_normal(len(th)) * np.sqrt(M)
            q, p, gq = th.copy(), p0.copy(), g.copy()
            p = p - 0.5 * step_size * gq
            for j in range(n_leapfrog):
                q = q + step_size * p / M
                Uq, gq = self.value_and_grad(q)
                if j < n_leapfrog - 1:
                    p = p - step_size * gq
            p = p - 0.5 * step_size * gq
            H0 = U + 0.5 * np.sum(p0 ** 2 / M)
            H1 = Uq + 0.5 * np.sum(p ** 2 / M)
            a = math.exp(min(0.0, H0 - H1)) if np.isfinite(H1) else 0.0
            if rng.random() < a:
                th, U, g = q, Uq, gq
                if it >= n_warmup:
                    acc += 1
            if it < n_warmup:
                step_size *= math.exp(0.1 * (a - 0.7))
            else:
                samples.append(th.copy())
            if verbose and it % 10 == 0:
                print(f"[HMC {it}] U={U:.3f} eps={step_size:.3g} a={a:.2f}")
        S = np.array(samples)
        out = {k: np.exp(S[:, i]) for i, k in enumerate(self.names)}
        out["_acceptance"] = np.array(acc / max(n_samples, 1))
        out["_step_size"] = np.array(step_size)
        return out


def _make_pd(H: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    w = np.maximum(w, floor)
    return (V * w) @ V.T
