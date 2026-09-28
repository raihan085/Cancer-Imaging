"""Probabilistic observation model: hidden tumour biology -> MRI evidence.

MRI segmentations are *not* treated as direct measurements of cycling,
quiescent or necrotic cells.  Instead the hidden fields generate a categorical
distribution over the MRI-visible compartments at every voxel::

    (P, Q, R; theta_obs)  ->  pi(x) = (pi_bg, pi_necrotic, pi_edema, pi_enhancing)

using nested logits

    z_bg  = 0
    z_ed  = k log(N / th_edema)               N = P + Q + R  (FLAIR abnormality)
    z_enh = z_ed + s (P + Q - th_enh)         dense viable tumour -> enhancement
    z_nec = z_ed + s (R - th_nec)             necrotic tissue -> necrotic core

followed by a Gaussian partial-volume blur.  The observed data are soft
segmentation probabilities ``L(x)`` (e.g. from downsampled expert labels or a
segmentation ensemble) and enter through a tempered soft-label cross-entropy

    log p(L | hidden) = w_obs * sum_x sum_k L_k(x) log pi_k(x).

``w_obs < 1`` accounts for the strong spatial correlation between voxels (they
are not independent observations); treating them as independent produces
over-confident posteriors.  Immune activity E and drug concentration C do not
enter the default likelihood: structural MRI does not measure them.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .domain import LABEL_CLASSES, BrainDomain

Tensor = torch.Tensor

#: Default correlation length (mm) used to temper the likelihood.
DEFAULT_CORRELATION_MM = 4.0


def compartment_logits(Ptot: Tensor, Q: Tensor, R: Tensor, P: Mapping[str, Tensor]) -> Tensor:
    s = P["obs_slope"]
    N = Ptot + Q + R
    # log-density logit: healthy tissue (N -> 0) maps to ~0 abnormality probability
    z_ed = P["obs_log_slope"] * (torch.log(N.clamp_min(0) + 1e-6) - torch.log(P["obs_th_edema"]))
    z_enh = z_ed + s * (Ptot + Q - P["obs_th_enh"])
    z_nec = z_ed + s * (R - P["obs_th_nec"])
    return torch.stack([torch.zeros_like(z_ed), z_nec, z_ed, z_enh])


def gaussian_blur(x: Tensor, sigma_vox: Sequence[float]) -> Tensor:
    """Separable Gaussian blur over the trailing len(sigma_vox) axes (zero padding)."""
    nd = len(sigma_vox)
    for ax, sg in enumerate(sigma_vox):
        if sg <= 0.05:
            continue
        rad = max(1, int(math.ceil(3 * sg)))
        k = torch.arange(-rad, rad + 1, dtype=x.dtype, device=x.device)
        k = torch.exp(-0.5 * (k / sg) ** 2)
        k = (k / k.sum()).view(1, 1, -1)
        d = x.ndim - nd + ax
        xt = x.movedim(d, -1)
        shp = xt.shape
        y = F.conv1d(xt.reshape(-1, 1, shp[-1]), k, padding=rad)
        x = y.reshape(shp).movedim(-1, d)
    return x


class ObservationModel:
    def __init__(self, domain: BrainDomain, correlation_mm: float = DEFAULT_CORRELATION_MM,
                 obs_weight: Optional[float] = None, dtype=torch.float64, device="cpu"):
        self.domain = domain
        self.mask = torch.as_tensor(domain.mask, device=device)
        self.maskf = self.mask.to(dtype)
        if obs_weight is None:
            vox_per_cell = np.prod([max(1.0, correlation_mm / h) for h in domain.spacing])
            obs_weight = 1.0 / float(vox_per_cell)
        self.obs_weight = float(obs_weight)
        self.dtype, self.device = dtype, device

    def probabilities(self, Ptot: Tensor, Q: Tensor, R: Tensor, P: Mapping[str, Tensor],
                      blur: bool = True) -> Tensor:
        pi = torch.softmax(compartment_logits(Ptot, Q, R, P), dim=0)
        if blur:
            sig_mm = float(P["obs_blur_mm"])
            pi = gaussian_blur(pi, [sig_mm / h for h in self.domain.spacing])
        # outside the brain everything is background
        bg = torch.zeros_like(pi)
        bg[0] = 1.0
        pi = torch.where(self.mask, pi, bg)
        return pi / pi.sum(0, keepdim=True)

    def probabilities_from_state(self, state, P, blur=True) -> Tensor:
        return self.probabilities(state.P, state.q, state.r, P, blur)

    def log_likelihood(self, pi: Tensor, soft_labels: Tensor) -> Tensor:
        ll = (soft_labels * torch.log(pi.clamp_min(1e-9))).sum(0)
        return self.obs_weight * (ll * self.maskf).sum()

    def visible_volume_ml(self, pi: Tensor) -> Dict[str, float]:
        vv = self.domain.voxel_volume_ml
        v = {c: float((pi[i] * self.maskf).sum()) * vv for i, c in enumerate(LABEL_CLASSES)}
        v["abnormal"] = v["necrotic"] + v["edema"] + v["enhancing"]
        v["core"] = v["necrotic"] + v["enhancing"]
        return v


# ---------------------------------------------------------------------------
# label utilities
# ---------------------------------------------------------------------------


def labels_to_onehot(seg: np.ndarray) -> np.ndarray:
    """Map BraTS/UPENN-GBM labels to one-hot (background, necrotic, edema, enhancing).

    UPENN-GBM uses 1 = necrotic/non-enhancing core, 2 = peritumoral edema /
    infiltrated tissue, 4 = enhancing tumour (label 3 is accepted as enhancing
    for BraTS-2023-style files).
    """
    seg = np.asarray(seg).astype(np.int64)
    out = np.zeros((4,) + seg.shape, dtype=np.float32)
    out[1] = seg == 1
    out[2] = seg == 2
    out[3] = (seg == 4) | (seg == 3)
    out[0] = 1 - out[1:].sum(0)
    return out


def soft_labels_from_hard(seg: np.ndarray, smooth_vox: float = 0.0) -> np.ndarray:
    oh = torch.as_tensor(labels_to_onehot(seg), dtype=torch.float64)
    if smooth_vox > 0:
        oh = gaussian_blur(oh, [smooth_vox] * seg.ndim)
    oh = oh / oh.sum(0, keepdim=True).clamp_min(1e-12)
    return oh.numpy()


# ---------------------------------------------------------------------------
# synthetic observations (for ground-truth experiments)
# ---------------------------------------------------------------------------


def smooth_noise(shape: Tuple[int, ...], corr_vox: float, rng: np.random.Generator) -> np.ndarray:
    z = torch.as_tensor(rng.standard_normal(shape))
    z = gaussian_blur(z, [corr_vox] * len(shape))
    z = z / z.std().clamp_min(1e-12)
    return z.numpy()


def synthesize_observation(pi: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                           boundary_noise: float = 1.0, corr_vox: float = 2.0,
                           label_flip: float = 0.0, temperature: float = 1.0) -> Dict[str, np.ndarray]:
    """Turn model compartment probabilities into a realistic 'segmentation'.

    * spatially correlated logit noise -> shifted / blurred boundaries,
      emulating inter-rater and segmentation-model errors;
    * optional random label flips (``label_flip`` fraction of abnormal voxels);
    * returns both soft labels and hard (argmax) labels.
    """
    logp = np.log(np.clip(pi, 1e-9, 1))
    noise = np.stack([smooth_noise(mask.shape, corr_vox, rng) for _ in range(pi.shape[0])])
    z = (logp + boundary_noise * noise) / temperature
    z = z - z.max(0, keepdims=True)
    soft = np.exp(z)
    soft /= soft.sum(0, keepdims=True)
    hard = soft.argmax(0)
    if label_flip > 0:
        abn = (hard > 0) & mask
        flip = abn & (rng.random(mask.shape) < label_flip)
        hard = np.where(flip, rng.integers(1, 4, size=mask.shape), hard)
    hard = np.where(mask, hard, 0)
    soft[:, ~mask] = 0
    soft[0, ~mask] = 1
    return dict(soft=soft, hard=hard, onehot=np.eye(4)[hard].transpose((-1,) + tuple(range(mask.ndim))))


#: Mean intensity (arbitrary z-units) per compartment and sequence for the
#: optional synthetic multi-sequence intensity model.  Rows: sequences,
#: columns: (background tissue, necrotic, edema, enhancing).
SEQUENCE_MEANS = {
    "T1": (0.0, -1.2, -0.6, -0.3),
    "T1GD": (0.0, -0.8, -0.4, 2.0),
    "T2": (0.0, 1.8, 1.4, 0.8),
    "FLAIR": (0.0, 0.6, 2.0, 1.0),
}


def synthesize_intensities(pi: np.ndarray, domain: BrainDomain, rng: np.random.Generator,
                           noise_sd: float = 0.15) -> Dict[str, np.ndarray]:
    """Simple linear-mixing intensity model (advanced-extension placeholder).

    Healthy-tissue contrast comes from the tissue map (CSF/GM/WM) when present.
    """
    tissue_contrast = {"T1": (-1.5, 0.0, 0.8), "T1GD": (-1.5, 0.0, 0.8),
                       "T2": (2.0, 0.3, -0.5), "FLAIR": (-1.0, 0.3, -0.2)}
    out = {}
    for seq, means in SEQUENCE_MEANS.items():
        base = np.zeros(domain.shape)
        if domain.tissue is not None:
            base = np.tensordot(np.array(tissue_contrast[seq]), domain.tissue, axes=1)
        img = pi[0] * base + sum(pi[k] * means[k] for k in range(1, 4))
        img = img + noise_sd * rng.standard_normal(domain.shape)
        out[seq] = np.where(domain.mask, img, 0.0)
    return out
