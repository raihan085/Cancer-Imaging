"""Baseline: discrete-loss inversion of a one-density Fisher-KPP tumour model.

    u_t = div(D_white m(x) grad u) + rho u (1 - u)

on the same domain and finite-volume operator, seeded at the core centroid and
grown for T days.  Because a single density cannot distinguish necrotic from
enhancing tissue, the observation model uses a 3-class version: background,
edema (u above a low threshold) and core (u above a high threshold; necrotic +
enhancing merged).  Parameters (D_white, rho, T) are fitted by MAP with
autograd through the discrete solver ("discrete-loss" inversion).
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .domain import BrainDomain, FVDiffusion
from .observation import ObservationModel, gaussian_blur

PRIORS = {  # (median, log_sd)
    "D_white": (0.25, 0.8),
    "rho": (0.1, 0.7),
    "T": (60.0, 0.5),
}


class FisherKPPInversion:
    def __init__(self, domain: BrainDomain, soft_labels: np.ndarray, center_vox: Optional[Sequence[float]] = None,
                 dt: float = 0.5, th_edema: float = 0.1, th_core: float = 0.6, slope: float = 25.0,
                 blur_mm: float = 1.5, seed_amp: float = 0.3, seed_radius: float = 2.0):
        self.domain = domain
        self.dt = dt
        self.op = FVDiffusion(domain.mobility, domain.mask, domain.spacing)
        self.mask = torch.as_tensor(domain.mask.astype(float))
        L = torch.as_tensor(soft_labels, dtype=torch.float64)
        self.L3 = torch.stack([L[0], L[2], L[1] + L[3]])  # bg, edema, core
        self.obs = ObservationModel(domain)
        self.center = tuple(center_vox) if center_vox is not None else domain.tumour_core_centroid()
        self.th_e, self.th_c, self.slope, self.blur = th_edema, th_core, slope, blur_mm
        grids = np.meshgrid(*[np.arange(n) * h for n, h in zip(domain.shape, domain.spacing)], indexing="ij")
        d2 = sum((g - c * h) ** 2 for g, c, h in zip(grids, self.center, domain.spacing))
        self.u0 = torch.as_tensor(seed_amp * np.exp(-d2 / (2 * seed_radius ** 2)) * domain.mask)
        self.names = list(PRIORS)
        self.mu0 = np.array([math.log(PRIORS[k][0]) for k in self.names])
        self.sd0 = np.array([PRIORS[k][1] for k in self.names])

    def simulate(self, D, rho, T) -> torch.Tensor:
        steps = float(T.detach()) / self.dt
        n = int(math.floor(steps))
        u = self.u0
        for _ in range(n):
            u = self._step(u, D, rho)
        u_next = self._step(u, D, rho)
        w = T / self.dt - n
        return u + w * (u_next - u)

    def _step(self, u, D, rho):
        u = u + self.dt * rho * u * (1 - u)
        u = self.op.step(u, D, self.dt)
        return u * self.mask

    def probabilities(self, u):
        s = self.slope
        z_e = s * (u - self.th_e)
        z_c = z_e + s * (u - self.th_c)
        pi = torch.softmax(torch.stack([torch.zeros_like(u), z_e, z_c]), 0)
        pi = gaussian_blur(pi, [self.blur / h for h in self.domain.spacing])
        return pi / pi.sum(0, keepdim=True)

    def neg_log_post(self, theta):
        D, rho, T = torch.exp(theta)
        pi = self.probabilities(self.simulate(D, rho, T))
        ll = (self.L3 * torch.log(pi.clamp_min(1e-9))).sum(0)
        ll = self.obs.obs_weight * (ll * self.mask).sum()
        lp = -0.5 * (((theta - torch.as_tensor(self.mu0)) / torch.as_tensor(self.sd0)) ** 2).sum()
        return -(ll + lp)

    def fit(self, n_iter: int = 100, lr: float = 0.05) -> Dict[str, float]:
        t0 = time.time()
        th = torch.as_tensor(self.mu0.copy(), dtype=torch.float64).requires_grad_(True)
        opt = torch.optim.Adam([th], lr=lr)
        lo = torch.as_tensor(self.mu0 - 4 * self.sd0)
        hi = torch.as_tensor(self.mu0 + 4 * self.sd0)
        for _ in range(n_iter):
            opt.zero_grad()
            self.neg_log_post(th).backward()
            opt.step()
            with torch.no_grad():
                th.clamp_(lo, hi)
        vals = {k: float(v) for k, v in zip(self.names, torch.exp(th.detach()))}
        vals["seconds"] = time.time() - t0
        vals["neg_log_post"] = float(self.neg_log_post(th.detach()))
        return vals

    @torch.no_grad()
    def density(self, values: Dict[str, float]) -> np.ndarray:
        t = lambda k: torch.tensor(values[k], dtype=torch.float64)  # noqa: E731
        return self.simulate(t("D_white"), t("rho"), t("T")).numpy()
