"""Differentiable finite-volume forward solver for the spatial, age- and
phenotype-structured glioma model.

State variables on the voxel grid (``x``) and cell-cycle-age grid (``a``):

* ``p[k, x]`` -- cycling cells whose cycle age lies in bin k (bin content, so
  the total cycling burden is ``P(x) = sum_k p[k, x]``; the continuous density
  is ``p / da``)
* ``q[x]``    -- quiescent cells
* ``r[x]``    -- necrotic tissue
* ``E[x]``    -- immune activity
* ``C[x]``    -- tissue drug concentration (uM); ``Cp`` is the plasma level

Governing equations (continuous form)::

    p_t + p_a = div(D_p grad p) - [beta(a) + mu_p + gamma_pq(sigma, C)
                                   + kappa_C(a, C) + kappa_E E] p
    p(x, 0, t) = 2 s(N) int beta(a) p da + gamma_qp(sigma) q          (renewal)
    q_t = div(D_p grad q) + gamma_pq P + (1 - s(N)) int beta p da
          - [gamma_qp(sigma) + mu_q(sigma) + kappa_E E + kappa_Cq(C)] q
    r_t = (mu_p + kappa_C + kappa_E E) P + (mu_q + kappa_E E + kappa_Cq) q - lambda_r r
    E_t = div(D_E grad E) + a_E T/(k_E + T) - (delta_E + eps_E T) E,   T = P + q
    C_t = div(D_C grad C) + k_in V(x) Cp(t) - k_out C

with N = P + q + r, stress sigma = N + w_E E, contact-inhibited division
success s(N) = clip(1 - N, 0, 1) (daughters of a blocked division become one
quiescent cell) and no-flux boundary conditions on the brain surface.

Numerics: age advection uses the method of characteristics (``da == dt``),
reactions use exponential competing-hazard integration (positivity
preserving), the linear drug PK is integrated exactly over each step, and
diffusion uses a conservative explicit finite-volume scheme with automatic
sub-stepping (Lie splitting).  Everything is written in PyTorch so gradients
with respect to parameters are available for MAP / Laplace / HMC inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .domain import BrainDomain, FVDiffusion
from .params import PARAM_SPECS, ModelParams
from .pk import DosingSchedule

Tensor = torch.Tensor


@dataclass
class TumorState:
    p: Tensor
    q: Tensor
    r: Tensor
    E: Tensor
    C: Tensor
    Cp: Tensor
    t: float = 0.0

    @property
    def P(self) -> Tensor:
        return self.p.sum(0)

    @property
    def N(self) -> Tensor:
        return self.P + self.q + self.r

    def detach(self) -> "TumorState":
        return TumorState(*(getattr(self, k).detach() for k in ("p", "q", "r", "E", "C", "Cp")), t=self.t)

    def lerp(self, other: "TumorState", w) -> "TumorState":
        f = lambda a, b: a + w * (b - a)  # noqa: E731
        return TumorState(
            f(self.p, other.p), f(self.q, other.q), f(self.r, other.r), f(self.E, other.E),
            f(self.C, other.C), f(self.Cp, other.Cp),
            t=self.t + float(w.detach() if torch.is_tensor(w) else w) * (other.t - self.t),
        )

    def to_numpy(self) -> Dict[str, np.ndarray]:
        d = {k: getattr(self, k).detach().cpu().numpy() for k in ("q", "r", "E", "C")}
        d["P"] = self.P.detach().cpu().numpy()
        d["p"] = self.p.detach().cpu().numpy()
        d["t"] = self.t
        return d


def param_tensors(params: ModelParams, overrides: Optional[Mapping[str, Tensor]] = None,
                  dtype=torch.float64, device="cpu") -> Dict[str, Tensor]:
    """Convert parameters to tensors; ``overrides`` (e.g. tensors requiring grad) win."""
    out = {k: torch.tensor(params[k], dtype=dtype, device=device) for k in PARAM_SPECS}
    if overrides:
        out.update(overrides)
    return out


class GliomaModel:
    """Forward solver bound to a :class:`BrainDomain`."""

    def __init__(self, domain: BrainDomain, dt: float = 0.25, a_max: float = 6.0,
                 dtype=torch.float64, device="cpu"):
        self.domain = domain
        self.dt = float(dt)
        self.da = self.dt  # characteristics: age advances with time
        self.n_age = max(3, int(math.ceil(a_max / self.da)))
        self.dtype, self.device = dtype, device
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype, device=device)  # noqa: E731
        self.mask = t(domain.mask.astype(float))
        self.perm = t(domain.permeability)
        self.ages = t((np.arange(self.n_age) + 0.5) * self.da).reshape((-1,) + (1,) * domain.ndim)
        self.op_tumour = FVDiffusion(domain.mobility, domain.mask, domain.spacing, dtype, device)
        self.op_free = FVDiffusion(domain.mask.astype(float), domain.mask, domain.spacing, dtype, device)
        self.coords = [t(g) for g in np.meshgrid(
            *[np.arange(n) * h for n, h in zip(domain.shape, domain.spacing)], indexing="ij")]

    # -- helpers ---------------------------------------------------------------
    def zeros(self) -> Tensor:
        return torch.zeros(self.domain.shape, dtype=self.dtype, device=self.device)

    def division_hazard(self, P: Mapping[str, Tensor]) -> Tensor:
        return P["beta_max"] * torch.sigmoid((self.ages - P["a_min"]) / 0.1)

    def phase_weight(self, P: Mapping[str, Tensor]) -> Tensor:
        base = P["phase_base"].clamp(0, 1)
        bump = torch.exp(-((self.ages - P["phase_center"]) ** 2) / (2 * P["phase_width"] ** 2))
        return base + (1 - base) * bump

    def seed_state(self, P: Mapping[str, Tensor], center_vox: Sequence[float]) -> TumorState:
        """Gaussian seed of cycling cells centred at ``center_vox`` (voxel coordinates)."""
        c_mm = [c * h for c, h in zip(center_vox, self.domain.spacing)]
        d2 = sum((g - c) ** 2 for g, c in zip(self.coords, c_mm))
        profile = P["seed_amp"] * torch.exp(-d2 / (2 * P["seed_radius"] ** 2)) * self.mask
        # asynchronous population: spread over ages below the division threshold
        w = torch.sigmoid((P["a_min"] - self.ages) / 0.25)
        w = w / w.sum()
        p = w * profile
        z = self.zeros()
        return TumorState(p, z.clone(), z.clone(), z.clone(), z.clone(),
                          torch.zeros((), dtype=self.dtype, device=self.device), 0.0)

    # -- one time step -----------------------------------------------------------
    def step(self, s: TumorState, P: Mapping[str, Tensor], dose_mg: float = 0.0) -> TumorState:
        dt = self.dt
        p, q, r, E, C, Cp = s.p, s.q, s.r, s.E, s.C, s.Cp

        # ---- drug: exact linear PK over the step --------------------------------
        if dose_mg:
            Cp = Cp + dose_mg * P["c_per_mg"]
        kel, kin, kout = P["k_el"], P["k_in"], P["k_out"]
        e_out, e_el = torch.exp(-kout * dt), torch.exp(-kel * dt)
        denom = kout - kel
        denom = torch.where(denom.abs() < 1e-6, torch.full_like(denom, 1e-6), denom)
        src = kin * self.perm * Cp
        C_new = C * e_out + src * (e_el - e_out) / denom
        C_avg = (C * (1 - e_out) / kout + src / denom * ((1 - e_el) / kel - (1 - e_out) / kout)) / dt
        Cp_new = Cp * e_el

        # ---- tumour reactions (competing hazards) ------------------------------------
        Ptot = p.sum(0)
        N = Ptot + q + r
        sigma = N + P["w_E"] * E
        s4 = sigma.clamp_min(0) ** 4
        stress = s4 / (P["sigma_half"] ** 4 + s4)
        Ch = C_avg.clamp_min(0) ** P["hill"]
        drug = Ch / (P["EC50"] ** P["hill"] + Ch)

        beta = self.division_hazard(P)
        kill_C = P["kappa_max"] * drug * self.phase_weight(P)
        L_dead = P["mu_p"] + kill_C + P["kappa_E"] * E
        L_q = P["gamma_pq"] * stress + P["gamma_C"] * drug
        H = (beta + L_dead + L_q).clamp_min(1e-12)
        out = p * (1 - torch.exp(-H * dt))
        div_ = (out * beta / H).sum(0)
        to_r = (out * L_dead / H).sum(0)
        to_q = (out * L_q / H).sum(0)
        p_stay = p - out

        success = (1 - N).clamp(0, 1)
        newborn = 2 * success * div_
        arrested = (1 - success) * div_

        re_rate = P["gamma_qp"] * (1 - stress)
        die_rate = P["mu_q"] * stress + P["kappa_E"] * E + P["q_drug_sens"] * P["kappa_max"] * drug
        Hq = (re_rate + die_rate).clamp_min(1e-12)
        outq = q * (1 - torch.exp(-Hq * dt))
        reenter = outq * re_rate / Hq

        q_new = q - outq + to_q + arrested
        r_new = r * torch.exp(-P["lambda_r"] * dt) + to_r + outq * die_rate / Hq

        # ---- age advection by one bin (characteristics) + renewal ------------------
        p_new = torch.cat([(newborn + reenter).unsqueeze(0), p_stay[:-2],
                           (p_stay[-2] + p_stay[-1]).unsqueeze(0)], dim=0)

        # ---- immune activity --------------------------------------------------------
        T = Ptot + q
        E_new = E * torch.exp(-(P["delta_E"] + P["eps_E"] * T) * dt) + P["a_E"] * T / (P["k_E"] + T) * dt

        # ---- diffusion (no-flux) ------------------------------------------------------
        p_new = self.op_tumour.step(p_new, P["D_white"], dt)
        q_new = self.op_tumour.step(q_new, P["D_white"], dt)
        E_new = self.op_free.step(E_new, P["D_E"], dt)
        C_new = self.op_free.step(C_new, P["D_C"], dt)

        m = self.mask
        return TumorState(p_new * m, q_new * m, r_new * m, E_new * m, C_new * m, Cp_new, s.t + dt)

    # -- drivers --------------------------------------------------------------------
    def run(self, state: TumorState, P: Mapping[str, Tensor], duration: float,
            schedule: Optional[DosingSchedule] = None,
            callback: Optional[Callable[[TumorState], None]] = None,
            record_every: float = 1.0) -> TumorState:
        """Advance ``state`` by ``duration`` days under ``schedule``.

        ``schedule`` times are relative to ``state.t``' origin (the scan, t=0).
        ``callback`` is invoked on the initial state and every ``record_every`` days.
        """
        n = int(round(duration / self.dt))
        every = max(1, int(round(record_every / self.dt)))
        if callback:
            callback(state)
        for i in range(n):
            t0 = state.t
            dose = schedule.doses_in(t0 - 1e-9, t0 + self.dt - 1e-9) if schedule else 0.0
            state = self.step(state, P, dose)
            if callback and (i + 1) % every == 0:
                callback(state)
        return state

    def grow_from_seed(self, P: Mapping[str, Tensor], center_vox: Sequence[float],
                       T_seed: Optional[Tensor] = None) -> TumorState:
        """Untreated growth for the effective time since seeding.

        The returned state is linearly interpolated between the two bracketing
        time steps so it is differentiable with respect to ``T_seed``.  Its time
        is reset to 0 (the scan time).
        """
        T = P["T_seed"] if T_seed is None else T_seed
        steps = float(T.detach()) / self.dt
        n = int(math.floor(steps))
        s = self.seed_state(P, center_vox)
        for _ in range(n):
            s = self.step(s, P)
        s_next = self.step(s, P)
        w = T / self.dt - n
        out = s.lerp(s_next, w)
        return replace(out, t=0.0)


def summarize_state(state: TumorState, domain: BrainDomain) -> Dict[str, float]:
    """Voxel-integrated burdens (in carrying-capacity x mL units)."""
    vv = domain.voxel_volume_ml
    return dict(
        t=state.t,
        cycling=float(state.P.sum()) * vv,
        quiescent=float(state.q.sum()) * vv,
        necrotic=float(state.r.sum()) * vv,
        viable=float((state.P + state.q).sum()) * vv,
        total=float(state.N.sum()) * vv,
        immune=float(state.E.sum()) * vv,
        drug_tissue_mean=float(state.C.sum() / max(domain.mask.sum(), 1)),
    )


__all__ = ["TumorState", "GliomaModel", "param_tensors", "summarize_state"]
