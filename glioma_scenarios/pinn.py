"""Renewal-aware, observation-aware physics-informed neural network (Steps 6-7).

Hidden fields are represented by neural networks

    p(x, a, t)   (cycling cells, density per unit cycle age)
    q, r, E, C (x, t)

and trained jointly with a small set of log-parameters so that they satisfy

* the PDE residuals of the model in :mod:`glioma_scenarios.forward`
  (continuous form, incl. heterogeneous no-flux diffusion),
* the **nonlocal renewal condition**
      p(x, 0, t) = 2 s(N) int beta(a) p(x, a, t) da + gamma_qp(sigma) q
  where the age integrals are evaluated during training with Gauss-Legendre
  quadrature (the same quadrature feeds P = int p da and the phase-dependent
  drug-kill integral into the coupling terms),
* no-flux boundary conditions on the brain surface,
* a probabilistic observation loss (soft-label cross-entropy against MRI
  compartment probabilities at the observation times),
* log-normal priors on the fitted parameters.

Dosing events make the drug field's time derivative jump, so time is split
into segments between dose times; each segment has its own networks and a
continuity loss ties neighbouring segments together at the event times.

``renewal_mode="none"`` gives the *standard PINN* baseline (no renewal
condition); ``integral_mode="mc"`` replaces quadrature with a few Monte-Carlo
age samples.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from .domain import BrainDomain
from .observation import ObservationModel, compartment_logits
from .params import PARAM_SPECS, ModelParams
from .pk import DosingSchedule

Tensor = torch.Tensor


@dataclass
class PINNConfig:
    t_end: float = 28.0
    a_max: float = 6.0
    hidden: int = 64
    depth: int = 4
    n_colloc: int = 1024
    n_boundary: int = 128
    n_obs_voxels: int = 1024
    n_quad: int = 12
    n_mc: int = 2
    renewal_mode: str = "quadrature"  # or "none"
    integral_mode: str = "gauss"  # or "mc"
    fit_names: Tuple[str, ...] = ("beta_max", "mu_p", "D_white")
    n_iters: int = 2000
    lr: float = 2e-3
    param_lr: float = 1e-2
    weights: Dict[str, float] = field(default_factory=lambda: dict(
        pde=1.0, renewal=1.0, bc=0.1, obs=1.0, cont=1.0, prior=1e-3, ic=10.0))
    seed: int = 0
    log_every: int = 100


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden, depth, out_scale=1.0):
        super().__init__()
        layers = []
        d = n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)
        self.out_scale = out_scale

    def forward(self, x):
        return F.softplus(self.net(x)) * self.out_scale


def plasma_torch(schedule: Optional[DosingSchedule], t: Tensor, c_per_mg: Tensor, k_el: Tensor) -> Tensor:
    out = torch.zeros_like(t)
    if schedule is None:
        return out
    for td, m in schedule.doses:
        dt = t - td
        out = out + torch.where(dt >= 0, m * c_per_mg * torch.exp(-k_el * dt.clamp_min(0)), torch.zeros_like(t))
    return out


class GridField:
    """Bilinear/trilinear lookup of a voxel field at physical coordinates."""

    def __init__(self, arr: np.ndarray, domain: BrainDomain, dtype=torch.float32):
        self.ndim = domain.ndim
        self.ext = torch.tensor([(n - 1) * h for n, h in zip(domain.shape, domain.spacing)], dtype=dtype)
        self.t = torch.as_tensor(arr, dtype=dtype)[None, None]

    def __call__(self, x: Tensor) -> Tensor:
        g = 2 * x.detach() / self.ext - 1  # align_corners=True convention
        g = g.flip(-1)  # grid_sample expects (W, H[, D]) order
        shape = (1, -1) + (1,) * (self.ndim - 1) + (self.ndim,)
        v = F.grid_sample(self.t, g.reshape(shape), mode="bilinear", align_corners=True, padding_mode="border")
        return v.reshape(-1)


def _grad(y: Tensor, x: Tensor) -> Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class GliomaPINN(nn.Module):
    def __init__(self, domain: BrainDomain, schedule: Optional[DosingSchedule] = None,
                 config: Optional[PINNConfig] = None, base_params: Optional[ModelParams] = None,
                 dtype=torch.float32):
        super().__init__()
        self.cfg = cfg = config or PINNConfig()
        torch.manual_seed(cfg.seed)
        self.domain, self.schedule, self.dtype = domain, schedule, dtype
        self.base = base_params or ModelParams.defaults()
        d = domain.ndim
        self.ext = torch.tensor([(n - 1) * h for n, h in zip(domain.shape, domain.spacing)], dtype=dtype)
        times = [t for t in (schedule.dose_times if schedule else []) if 0 < t < cfg.t_end]
        self.bounds = [0.0] + times + [cfg.t_end]
        K = len(self.bounds) - 1
        self.net_p = nn.ModuleList([MLP(d + 2, 1, cfg.hidden, cfg.depth, 0.3) for _ in range(K)])
        self.net_f = nn.ModuleList([MLP(d + 1, 4, cfg.hidden, cfg.depth, 0.3) for _ in range(K)])
        # learnable log-parameters
        self.fit_names = list(cfg.fit_names)
        self.log_theta = nn.Parameter(torch.tensor(
            [math.log(self.base[k]) for k in self.fit_names], dtype=dtype))
        self.fixed = {k: torch.tensor(self.base[k], dtype=dtype) for k in PARAM_SPECS}
        # anatomy lookups
        m_s = ndimage.gaussian_filter(domain.mobility, 0.7)
        grads = np.gradient(m_s, *domain.spacing) if d > 1 else [np.gradient(m_s, domain.spacing[0])]
        self.mob = GridField(m_s, domain)
        self.mob_grad = [GridField(g, domain) for g in grads]
        self.perm = GridField(domain.permeability, domain)
        # quadrature over age
        xg, wg = np.polynomial.legendre.leggauss(cfg.n_quad)
        self.quad_a = torch.tensor((xg + 1) / 2 * cfg.a_max, dtype=dtype)
        self.quad_w = torch.tensor(wg / 2 * cfg.a_max, dtype=dtype)
        # collocation / boundary pools
        self._inside = np.argwhere(domain.mask).astype(np.float64) * np.array(domain.spacing)
        self._boundary, self._normals = self._boundary_points()
        self.obs_model = ObservationModel(domain)
        self.history: List[dict] = []

    # ---- parameters ------------------------------------------------------------------------
    def params(self) -> Dict[str, Tensor]:
        P = dict(self.fixed)
        for i, k in enumerate(self.fit_names):
            P[k] = torch.exp(self.log_theta[i])
        return P

    def fitted_values(self) -> Dict[str, float]:
        return {k: float(torch.exp(v)) for k, v in zip(self.fit_names, self.log_theta.detach())}

    # ---- network evaluation -----------------------------------------------------------------
    def _norm(self, x, t, a=None):
        cols = [2 * x / self.ext - 1, (2 * t / self.cfg.t_end - 1)[:, None]]
        if a is not None:
            cols.append((2 * a / self.cfg.a_max - 1)[:, None])
        return torch.cat(cols, dim=1)

    def _segment(self, t: Tensor) -> Tensor:
        b = torch.tensor(self.bounds[1:-1], dtype=t.dtype)
        return torch.bucketize(t.detach(), b, right=True)

    def _route(self, nets, inp, seg, n_out):
        out = inp.new_zeros(inp.shape[0], n_out)
        for k in range(len(nets)):
            idx = (seg == k).nonzero(as_tuple=True)[0]
            if len(idx):
                out = out.index_put((idx,), nets[k](inp[idx]))
        return out

    def p(self, x, t, a, seg=None):
        seg = self._segment(t) if seg is None else seg
        return self._route(self.net_p, self._norm(x, t, a), seg, 1)[:, 0]

    def fields(self, x, t, seg=None):
        seg = self._segment(t) if seg is None else seg
        out = self._route(self.net_f, self._norm(x, t), seg, 4)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]  # q, r, E, C

    def age_integrals(self, x, t, P: Dict[str, Tensor], seg=None, drug=None):
        """Return (P_tot, int beta p da, int w_phase p da) at (x, t)."""
        n = x.shape[0]
        if self.cfg.integral_mode == "mc":
            m = self.cfg.n_mc
            a = torch.rand(n * m, dtype=x.dtype) * self.cfg.a_max
            w = torch.full((m,), self.cfg.a_max / m, dtype=x.dtype)
        else:
            m = len(self.quad_a)
            a = self.quad_a.repeat(n)
            w = self.quad_w
        xs = x.repeat_interleave(m, 0)
        ts = t.repeat_interleave(m, 0)
        sg = seg.repeat_interleave(m, 0) if seg is not None else None
        pv = self.p(xs, ts, a, sg).reshape(n, m)
        a = a.reshape(n, m)
        beta = self.beta(a, P)
        wph = self.phase(a, P)
        return (pv * w).sum(1), (pv * beta * w).sum(1), (pv * wph * w).sum(1)

    # ---- model functions ---------------------------------------------------------------------
    @staticmethod
    def beta(a, P):
        return P["beta_max"] * torch.sigmoid((a - P["a_min"]) / 0.1)

    @staticmethod
    def phase(a, P):
        base = P["phase_base"].clamp(0, 1)
        return base + (1 - base) * torch.exp(-((a - P["phase_center"]) ** 2) / (2 * P["phase_width"] ** 2))

    def _div_D_grad(self, u, x, coef, heterogeneous: bool):
        g = _grad(u, x)
        lap = 0
        for i in range(x.shape[1]):
            lap = lap + _grad(g[:, i], x)[:, i]
        if not heterogeneous:
            return coef * lap
        m = self.mob(x)
        dm = torch.stack([f(x) for f in self.mob_grad], 1)
        return coef * (m * lap + (dm * g).sum(1))

    # ---- sampling -------------------------------------------------------------------------------
    def _boundary_points(self):
        dom = self.domain
        m = dom.mask
        er = ndimage.binary_erosion(m)
        edge = m & ~er
        sm = ndimage.gaussian_filter(m.astype(float), 1.0)
        grads = np.gradient(sm, *dom.spacing)
        pts = np.argwhere(edge)
        nrm = np.stack([-g[edge] for g in grads], 1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        return pts * np.array(dom.spacing), nrm

    def _sample_x(self, n, pool, jitter=True):
        idx = np.random.randint(0, len(pool), n)
        x = pool[idx]
        if jitter:
            x = x + (np.random.rand(*x.shape) - 0.5) * np.array(self.domain.spacing)
            x = np.clip(x, 0, self.ext.numpy())
        return torch.tensor(x, dtype=self.dtype), idx

    # ---- losses -----------------------------------------------------------------------------------
    def pde_residuals(self, n: int) -> Dict[str, Tensor]:
        P = self.params()
        x, _ = self._sample_x(n, self._inside)
        x.requires_grad_(True)
        t = (torch.rand(n, dtype=self.dtype) * self.cfg.t_end).requires_grad_(True)
        a = (torch.rand(n, dtype=self.dtype) * self.cfg.a_max).requires_grad_(True)
        seg = self._segment(t)
        q, r, E, C = self.fields(x, t, seg)
        Ptot, Bint, Wint = self.age_integrals(x, t, P, seg)
        N = Ptot + q + r
        sigma = N + P["w_E"] * E
        s4 = sigma.clamp_min(0) ** 4
        stress = s4 / (P["sigma_half"] ** 4 + s4)
        Ch = C.clamp_min(0) ** P["hill"]
        drug = Ch / (P["EC50"] ** P["hill"] + Ch)
        success = (1 - N).clamp(0, 1)

        pv = self.p(x, t, a, seg)
        L = (self.beta(a, P) + P["mu_p"] + P["gamma_pq"] * stress + P["gamma_C"] * drug
             + P["kappa_max"] * drug * self.phase(a, P) + P["kappa_E"] * E)
        res_p = _grad(pv, t) + _grad(pv, a) - self._div_D_grad(pv, x, P["D_white"], True) + L * pv

        re_rate = P["gamma_qp"] * (1 - stress)
        die_q = P["mu_q"] * stress + P["kappa_E"] * E + P["q_drug_sens"] * P["kappa_max"] * drug
        res_q = (_grad(q, t) - self._div_D_grad(q, x, P["D_white"], True)
                 - (P["gamma_pq"] * stress + P["gamma_C"] * drug) * Ptot - (1 - success) * Bint
                 + (re_rate + die_q) * q)
        res_r = (_grad(r, t) - (P["mu_p"] + P["kappa_E"] * E) * Ptot - P["kappa_max"] * drug * Wint
                 - die_q * q + P["lambda_r"] * r)
        T = Ptot + q
        res_E = (_grad(E, t) - self._div_D_grad(E, x, P["D_E"], False)
                 - P["a_E"] * T / (P["k_E"] + T) + (P["delta_E"] + P["eps_E"] * T) * E)
        Cp = plasma_torch(self.schedule, t, P["c_per_mg"], P["k_el"])
        res_C = (_grad(C, t) - self._div_D_grad(C, x, P["D_C"], False)
                 - P["k_in"] * self.perm(x) * Cp + P["k_out"] * C)
        out = dict(p=res_p, q=res_q, r=res_r, E=res_E, C=res_C / 10.0)
        if self.cfg.renewal_mode == "quadrature":
            p0 = self.p(x, t, torch.zeros_like(t), seg)
            out["renewal"] = p0 - (2 * success * Bint + re_rate * q)
        return out

    def boundary_loss(self, n: int) -> Tensor:
        if len(self._boundary) == 0:
            return torch.zeros((), dtype=self.dtype)
        x, idx = self._sample_x(n, self._boundary, jitter=False)
        nrm = torch.tensor(self._normals[idx], dtype=self.dtype)
        x.requires_grad_(True)
        t = torch.rand(n, dtype=self.dtype) * self.cfg.t_end
        a = torch.rand(n, dtype=self.dtype) * self.cfg.a_max
        seg = self._segment(t)
        q, _, E, C = self.fields(x, t, seg)
        pv = self.p(x, t, a, seg)
        tot = 0
        for u in (pv, q, E, C):
            tot = tot + ((_grad(u, x) * nrm).sum(1) ** 2).mean()
        return tot

    def continuity_loss(self, n: int) -> Tensor:
        if len(self.bounds) <= 2:
            return torch.zeros((), dtype=self.dtype)
        tot = 0
        for k, tb in enumerate(self.bounds[1:-1]):
            x, _ = self._sample_x(n, self._inside)
            t = torch.full((n,), tb, dtype=self.dtype)
            a = torch.rand(n, dtype=self.dtype) * self.cfg.a_max
            left = torch.full((n,), k, dtype=torch.long)
            right = left + 1
            fl = torch.stack(self.fields(x, t, left), 1)
            fr = torch.stack(self.fields(x, t, right), 1)
            pl, pr = self.p(x, t, a, left), self.p(x, t, a, right)
            tot = tot + ((fl - fr) ** 2).mean() + ((pl - pr) ** 2).mean()
        return tot

    def observation_loss(self, observations: Sequence[Tuple[float, np.ndarray]], n: int) -> Tensor:
        P = self.params()
        tot = 0
        for t_obs, L in observations:
            x, idx = self._sample_x(n, self._inside, jitter=False)
            vox = np.argwhere(self.domain.mask)[idx]
            Lt = torch.tensor(L[(slice(None),) + tuple(vox.T)], dtype=self.dtype)  # (4, n)
            t = torch.full((n,), float(t_obs), dtype=self.dtype)
            seg = self._segment(t)
            q, r, _, _ = self.fields(x, t, seg)
            Ptot, _, _ = self.age_integrals(x, t, P, seg)
            logits = compartment_logits(Ptot, q, r, P)
            logpi = torch.log_softmax(logits, 0)
            tot = tot - (Lt * logpi).sum(0).mean()
        return tot

    def initial_loss(self, init: Dict[str, np.ndarray], n: int) -> Tensor:
        """Optional: match a known state at t=0 (e.g. synthetic-truth ablations)."""
        x, idx = self._sample_x(n, self._inside, jitter=False)
        vox = tuple(np.argwhere(self.domain.mask)[idx].T)
        t = torch.zeros(n, dtype=self.dtype)
        seg = self._segment(t)
        q, r, E, C = self.fields(x, t, seg)
        P = self.params()
        Ptot, _, _ = self.age_integrals(x, t, P, seg)
        tot = 0
        for name, pred in (("P", Ptot), ("q", q), ("r", r), ("E", E), ("C", C)):
            if name in init:
                tot = tot + ((pred - torch.tensor(init[name][vox], dtype=self.dtype)) ** 2).mean()
        return tot

    def prior_loss(self) -> Tensor:
        mu = torch.tensor([math.log(PARAM_SPECS[k].prior_median) for k in self.fit_names], dtype=self.dtype)
        sd = torch.tensor([PARAM_SPECS[k].prior_log_sd for k in self.fit_names], dtype=self.dtype)
        return 0.5 * (((self.log_theta - mu) / sd) ** 2).sum()

    # ---- training ------------------------------------------------------------------------------------
    def fit(self, observations: Sequence[Tuple[float, np.ndarray]],
            initial_state: Optional[Dict[str, np.ndarray]] = None, verbose: bool = False) -> List[dict]:
        cfg, w = self.cfg, self.cfg.weights
        np.random.seed(cfg.seed)
        net_params = [p for n, p in self.named_parameters() if n != "log_theta"]
        opt = torch.optim.Adam([{"params": net_params, "lr": cfg.lr},
                                {"params": [self.log_theta], "lr": cfg.param_lr}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.n_iters)
        t0 = time.time()
        for it in range(cfg.n_iters):
            opt.zero_grad()
            res = self.pde_residuals(cfg.n_colloc)
            l_pde = sum((v ** 2).mean() for k, v in res.items() if k != "renewal")
            l_ren = (res["renewal"] ** 2).mean() if "renewal" in res else torch.zeros(())
            l_bc = self.boundary_loss(cfg.n_boundary)
            l_obs = self.observation_loss(observations, cfg.n_obs_voxels) if observations else torch.zeros(())
            l_cont = self.continuity_loss(cfg.n_boundary)
            l_prior = self.prior_loss()
            l_ic = self.initial_loss(initial_state, cfg.n_obs_voxels) if initial_state else torch.zeros(())
            loss = (w["pde"] * l_pde + w["renewal"] * l_ren + w["bc"] * l_bc + w["obs"] * l_obs
                    + w["cont"] * l_cont + w["prior"] * l_prior + w["ic"] * l_ic)
            loss.backward()
            opt.step()
            sched.step()
            if it % cfg.log_every == 0 or it == cfg.n_iters - 1:
                f = lambda v: float(v.detach()) if torch.is_tensor(v) else float(v)  # noqa: E731
                rec = dict(it=it, loss=f(loss), pde=f(l_pde), renewal=f(l_ren), bc=f(l_bc), obs=f(l_obs),
                           cont=f(l_cont), ic=f(l_ic), seconds=time.time() - t0, **self.fitted_values())
                self.history.append(rec)
                if verbose:
                    print(rec)
        return self.history

    # ---- prediction ----------------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, t: float, batch: int = 4096) -> Dict[str, np.ndarray]:
        dom = self.domain
        vox = np.argwhere(dom.mask)
        out = {k: np.zeros(dom.shape) for k in ("P", "q", "r", "E", "C")}
        P = self.params()
        for s in range(0, len(vox), batch):
            v = vox[s:s + batch]
            x = torch.tensor(v * np.array(dom.spacing), dtype=self.dtype)
            tt = torch.full((len(v),), float(t), dtype=self.dtype)
            seg = self._segment(tt)
            q, r, E, C = self.fields(x, tt, seg)
            Ptot, _, _ = self.age_integrals(x, tt, P, seg)
            for k, val in (("P", Ptot), ("q", q), ("r", r), ("E", E), ("C", C)):
                out[k][tuple(v.T)] = val.numpy()
        return out
