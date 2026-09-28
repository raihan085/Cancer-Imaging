"""Patient-specific computational brain domain and finite-volume diffusion.

A :class:`BrainDomain` holds everything the solvers need to know about the
anatomy on a regular voxel grid (2-D slice or 3-D volume):

* ``mask``          -- computational brain domain (no-flux outside it)
* ``mobility``      -- relative tumour-cell motility m(x) in [0, 1]
                       (1 = white matter, ``gray_white_ratio`` = gray matter,
                       0 = CSF); D_p(x) = D_white * m(x)
* ``permeability``  -- V(x) in [0, 1], MRI-derived proxy for how easily drug
                       enters tissue (contrast enhancement / rCBV)
* ``tissue``        -- optional soft tissue probabilities (CSF, GM, WM)
* ``soft_labels``   -- optional soft tumour-compartment probabilities
                       (background, necrotic core, edema, enhancing)

The anatomy is the *patient-specific* part of the framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

#: Order of soft-label classes used throughout the package.
LABEL_CLASSES = ("background", "necrotic", "edema", "enhancing")


@dataclass
class BrainDomain:
    mask: np.ndarray
    spacing: Tuple[float, ...]
    mobility: np.ndarray
    permeability: np.ndarray
    tissue: Optional[np.ndarray] = None  # (3, *grid): CSF, GM, WM
    soft_labels: Optional[np.ndarray] = None  # (4, *grid) in LABEL_CLASSES order
    affine: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.mask = np.asarray(self.mask, dtype=bool)
        self.spacing = tuple(float(s) for s in self.spacing)
        if len(self.spacing) != self.mask.ndim:
            raise ValueError("spacing must have one entry per grid axis")
        for name in ("mobility", "permeability"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != self.mask.shape:
                raise ValueError(f"{name} shape {arr.shape} != mask shape {self.mask.shape}")
            setattr(self, name, np.where(self.mask, arr, 0.0))

    @property
    def ndim(self) -> int:
        return self.mask.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.mask.shape

    @property
    def voxel_volume_ml(self) -> float:
        """Voxel volume in millilitres (for 2-D grids: area x 1 mm slab)."""
        return float(np.prod(self.spacing)) / 1000.0

    def centroid_of(self, weights: np.ndarray) -> Tuple[float, ...]:
        """Centroid (voxel index coordinates) of a non-negative weight map."""
        w = np.where(self.mask, weights, 0.0)
        tot = w.sum()
        if tot <= 0:
            raise ValueError("empty weight map")
        grids = np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij")
        return tuple(float((g * w).sum() / tot) for g in grids)

    def tumour_core_centroid(self) -> Tuple[float, ...]:
        if self.soft_labels is None:
            raise ValueError("domain has no soft labels")
        core = self.soft_labels[1] + self.soft_labels[3]
        if core.sum() <= 0:
            core = 1.0 - self.soft_labels[0]
        return self.centroid_of(core)


# ---------------------------------------------------------------------------
# finite-volume diffusion with no-flux boundaries
# ---------------------------------------------------------------------------


def _pad_axis(x: torch.Tensor, dim: int, before: int, after: int) -> torch.Tensor:
    parts = []
    if before:
        s = list(x.shape)
        s[dim] = before
        parts.append(x.new_zeros(s))
    parts.append(x)
    if after:
        s = list(x.shape)
        s[dim] = after
        parts.append(x.new_zeros(s))
    return torch.cat(parts, dim=dim)


class FVDiffusion:
    """Conservative finite-volume operator  u -> div(m(x) grad u).

    Face conductances use the harmonic mean of the cell mobilities and are set
    to zero on faces that touch a voxel outside the mask, which implements the
    no-flux (homogeneous Neumann) boundary condition on the brain surface: the
    discrete operator conserves total mass exactly.

    The operator acts on the trailing ``ndim`` axes, so leading batch axes (for
    example the cell-cycle-age axis) are supported.
    """

    def __init__(self, mobility: np.ndarray, mask: np.ndarray, spacing: Sequence[float],
                 dtype=torch.float64, device="cpu"):
        mask = np.asarray(mask, dtype=bool)
        m = np.where(mask, np.asarray(mobility, dtype=np.float64), 0.0)
        self.ndim = mask.ndim
        self.shape = mask.shape
        self.conductance = []
        row_sum = np.zeros(mask.shape)
        for ax in range(self.ndim):
            n = mask.shape[ax]
            a = np.take(m, range(0, n - 1), axis=ax)
            b = np.take(m, range(1, n), axis=ax)
            ma = np.take(mask, range(0, n - 1), axis=ax)
            mb = np.take(mask, range(1, n), axis=ax)
            harm = np.where(ma & mb, 2 * a * b / np.maximum(a + b, 1e-12), 0.0)
            T = harm / spacing[ax] ** 2
            self.conductance.append(torch.as_tensor(T, dtype=dtype, device=device))
            pad_lo = [(0, 0)] * self.ndim
            pad_hi = [(0, 0)] * self.ndim
            pad_lo[ax] = (1, 0)
            pad_hi[ax] = (0, 1)
            row_sum += np.pad(T, pad_lo) + np.pad(T, pad_hi)
        self.max_rate = float(row_sum.max()) if row_sum.size else 0.0
        self.mask = torch.as_tensor(mask, device=device)

    def apply(self, u: torch.Tensor, coef=1.0) -> torch.Tensor:
        lead = u.ndim - self.ndim
        out = torch.zeros_like(u)
        for ax, T in enumerate(self.conductance):
            d = lead + ax
            n = u.shape[d]
            flux = (u.narrow(d, 1, n - 1) - u.narrow(d, 0, n - 1)) * T
            out = out + _pad_axis(flux, d, 0, 1) - _pad_axis(flux, d, 1, 0)
        return out * coef

    def n_substeps(self, coef: float, dt: float, safety: float = 0.8) -> int:
        """Number of explicit sub-steps needed for stability."""
        return max(1, int(math.ceil(coef * dt * self.max_rate / safety)))

    def step(self, u: torch.Tensor, coef, dt: float) -> torch.Tensor:
        """Advance u_t = coef * div(m grad u) by ``dt`` with explicit sub-steps."""
        c = float(coef.detach()) if torch.is_tensor(coef) else float(coef)
        if c == 0.0:
            return u
        n = self.n_substeps(c, dt)
        h = dt / n
        for _ in range(n):
            u = u + h * self.apply(u, coef)
        return u


# ---------------------------------------------------------------------------
# synthetic anatomy (for tests and synthetic ground-truth experiments)
# ---------------------------------------------------------------------------


def phantom_domain(shape: Tuple[int, ...] = (64, 64), spacing: float = 2.0,
                   gray_white_ratio: float = 0.2, seed: int = 0) -> BrainDomain:
    """Ellipsoidal 'brain' with a white-matter core, gray-matter rim and ventricles.

    Only meant for method development and tests -- not a real anatomy.
    """
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    radii = [0.9, 0.75, 0.7][: len(shape)]
    r = np.sqrt(sum((g / a) ** 2 for g, a in zip(grids, radii)))
    brain = r < 1.0
    wm = r < 0.72
    vent = np.zeros(shape, bool)
    for sgn in (-1, 1):
        c = [0.0] * len(shape)
        c[1] = 0.18 * sgn
        rv = np.sqrt(sum(((g - ci) / a) ** 2 for g, ci, a in zip(grids, c, [0.25, 0.08, 0.1])))
        vent |= rv < 1.0
    wm_p = np.clip(1.2 - r / 0.72, 0, 1) * brain
    wm_p = np.clip(wm_p + 0.05 * rng.standard_normal(shape), 0, 1)
    csf_p = vent.astype(float)
    gm_p = np.clip(1 - wm_p - csf_p, 0, 1) * brain
    tissue = np.stack([csf_p, gm_p, np.where(vent, 0, wm_p)])
    tissue = tissue / np.maximum(tissue.sum(0, keepdims=True), 1e-9)
    mask = brain & ~vent
    mobility = tissue[2] + gray_white_ratio * tissue[1]
    perm = np.full(shape, 0.3)
    return BrainDomain(
        mask=mask,
        spacing=(spacing,) * len(shape),
        mobility=mobility,
        permeability=perm,
        tissue=tissue * mask,
        meta=dict(source="phantom", note="synthetic anatomy, not patient data"),
    )


def grid_coordinates(domain: BrainDomain) -> np.ndarray:
    """Physical coordinates (mm) of voxel centres, shape (ndim, *grid)."""
    return np.stack(
        np.meshgrid(*[np.arange(n) * h for n, h in zip(domain.shape, domain.spacing)], indexing="ij")
    )
