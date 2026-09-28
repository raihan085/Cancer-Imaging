"""Image preprocessing: MRI (+DTI, +perfusion, +segmentation) -> BrainDomain.

Pipeline (Step 1 of the methodology):

1. brain mask (UPENN-GBM images are already skull-stripped and co-registered)
2. intensity normalisation (z-score inside the brain)
3. crop to the brain bounding box
4. soft tissue segmentation (CSF / GM / WM) with a small Gaussian mixture
5. block-average down-sampling to the simulation resolution; averaging the
   one-hot tumour labels yields partial-volume *soft* compartment
   probabilities
6. motility map m(x) from tissue probabilities, modulated by DTI FA
7. permeability proxy V(x) from T1-Gd enhancement and relative CBV

Scalar (isotropic) motility is used; a full anisotropic tensor built from the
DTI eigen-decomposition is a natural extension but is not implemented here.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .domain import BrainDomain
from .observation import labels_to_onehot


def brain_mask(images: Dict[str, np.ndarray]) -> np.ndarray:
    ref = [images[k] for k in ("T1", "T1GD", "T2", "FLAIR") if k in images]
    if not ref:
        raise ValueError("no structural image to derive a brain mask from")
    m = np.zeros(ref[0].shape, bool)
    for r in ref:
        m |= r > 0
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def zscore(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = img[mask]
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = np.clip(v, lo, hi)
    out = (np.clip(img, lo, hi) - v.mean()) / max(v.std(), 1e-6)
    return np.where(mask, out, 0.0).astype(np.float32)


def bounding_box(mask: np.ndarray, margin: int = 2) -> Tuple[slice, ...]:
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def block_mean(x: np.ndarray, f: int, spatial_axes: int = None) -> np.ndarray:
    """Average non-overlapping f^d blocks over the trailing ``spatial_axes`` axes."""
    if f == 1:
        return x.astype(np.float64)
    nd = x.ndim if spatial_axes is None else spatial_axes
    lead = x.shape[: x.ndim - nd]
    sp = x.shape[x.ndim - nd:]
    pad = [(0, 0)] * len(lead) + [(0, (-n) % f) for n in sp]
    x = np.pad(x.astype(np.float64), pad, mode="edge")
    sp = x.shape[x.ndim - nd:]
    shp = list(lead)
    for n in sp:
        shp += [n // f, f]
    x = x.reshape(shp)
    axes = tuple(len(lead) + 2 * i + 1 for i in range(nd))
    return x.mean(axis=axes)


def gmm_tissue(features: np.ndarray, mask: np.ndarray, n_iter: int = 50, seed: int = 0) -> np.ndarray:
    """3-class diagonal GMM on (T1z[, T2z]) inside the mask -> (CSF, GM, WM) probabilities.

    Classes are ordered by mean T1 intensity (CSF darkest, WM brightest on T1).
    """
    X = features[:, mask].T.astype(np.float64)  # (n, f)
    rng = np.random.default_rng(seed)
    if len(X) > 200_000:
        Xfit = X[rng.choice(len(X), 200_000, replace=False)]
    else:
        Xfit = X
    q = np.quantile(Xfit[:, 0], [0.15, 0.5, 0.85])
    mu = np.stack([np.r_[qi, np.zeros(X.shape[1] - 1)] for qi in q])
    if X.shape[1] > 1:
        for k, qi in enumerate(q):
            sel = np.abs(Xfit[:, 0] - qi) < 0.3
            if sel.any():
                mu[k] = Xfit[sel].mean(0)
    var = np.ones_like(mu) * Xfit.var(0)
    w = np.ones(3) / 3

    def resp(Z):
        ll = -0.5 * (((Z[:, None, :] - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(-1) + np.log(w)
        ll -= ll.max(1, keepdims=True)
        r = np.exp(ll)
        return r / r.sum(1, keepdims=True)

    for _ in range(n_iter):
        r = resp(Xfit)
        nk = r.sum(0) + 1e-9
        w = nk / nk.sum()
        mu = (r.T @ Xfit) / nk[:, None]
        var = (r.T @ (Xfit ** 2)) / nk[:, None] - mu ** 2 + 1e-4
    order = np.argsort(mu[:, 0])
    r = resp(X)[:, order]
    out = np.zeros((3,) + mask.shape)
    out[:, mask] = r.T
    return out


def permeability_proxy(t1: np.ndarray, t1gd: np.ndarray, mask: np.ndarray,
                       rcbv: Optional[np.ndarray] = None, v0: float = 0.3) -> np.ndarray:
    """V(x) in [v0, 1]: intact-BBB baseline v0 plus enhancement / rCBV excess.

    Contrast enhancement reflects BBB disruption and vascular leakage; it is a
    rough proxy for drug entry, not a measurement of drug delivery.
    """
    enh = np.clip(t1gd - t1, 0, None)
    enh = enh / max(np.percentile(enh[mask], 99.5), 1e-6)
    score = np.clip(enh, 0, 1)
    if rcbv is not None:
        rv = np.clip(rcbv, 0, None)
        ref = np.percentile(rv[mask & (rv > 0)], 50) if (mask & (rv > 0)).any() else 1.0
        rv = np.clip((rv / max(ref, 1e-6) - 1.0) / 3.0, 0, 1)
        score = 0.7 * score + 0.3 * rv
    return np.where(mask, v0 + (1 - v0) * score, 0.0)


def motility_map(tissue: np.ndarray, gray_white_ratio: float = 0.2,
                 fa: Optional[np.ndarray] = None) -> np.ndarray:
    csf, gm, wm = tissue
    wm_term = wm
    if fa is not None:
        fa_n = np.clip(fa / 0.5, 0, 1)
        wm_term = wm * (0.5 + 0.5 * fa_n)
    return np.clip(wm_term + gray_white_ratio * gm, 0, 1)


def build_domain(images: Dict[str, np.ndarray], factor: int = 3, gray_white_ratio: float = 0.2,
                 slice_axis: Optional[int] = None, seg_key: str = "segm", v0: float = 0.3,
                 case_id: str = "") -> BrainDomain:
    """Build a simulation-ready :class:`BrainDomain` from one loaded case.

    ``factor``      -- integer down-sampling factor (1 mm -> ``factor`` mm)
    ``slice_axis``  -- if given, return a 2-D domain on the slice (along this
                       axis) with the largest tumour core, for fast experiments
    """
    zooms = np.asarray(images.get("zooms", (1.0, 1.0, 1.0)), dtype=float)
    mask = brain_mask(images)
    seg = images.get(seg_key, images.get("segm_auto"))
    if seg is not None:
        mask |= seg > 0
    bb = bounding_box(mask)
    crop = lambda a: a[bb]  # noqa: E731
    mask = crop(mask)
    z = {k: zscore(crop(images[k]), mask) for k in ("T1", "T1GD", "T2", "FLAIR") if k in images}
    feats = np.stack([z["T1"]] + ([z["T2"]] if "T2" in z else []))
    tissue = gmm_tissue(feats, mask)
    onehot = labels_to_onehot(crop(seg)) if seg is not None else None
    if onehot is not None:
        tumour = onehot[0] < 0.5
        # tissue class is not identifiable from intensities inside the lesion
        tissue[:, tumour] = np.array([0.0, 0.5, 0.5])[:, None]
    fa = crop(images["DTI_FA"]) if "DTI_FA" in images else None
    rcbv = crop(images["DSC_rCBV"]) if "DSC_rCBV" in images else None
    perm = permeability_proxy(z["T1"], z["T1GD"], mask, rcbv, v0=v0) if "T1GD" in z else np.where(mask, v0, 0.0)

    # ---- down-sample ---------------------------------------------------------------
    maskf = block_mean(mask.astype(float), factor)
    tissue_d = block_mean(tissue, factor, 3)
    perm_d = block_mean(perm, factor)
    fa_d = block_mean(fa, factor) if fa is not None else None
    soft = block_mean(onehot, factor, 3) if onehot is not None else None
    mask_d = maskf > 0.5
    if soft is not None:
        mask_d |= soft[0] < 0.5
        s = soft.sum(0, keepdims=True)
        soft = np.where(s > 0, soft / np.maximum(s, 1e-9), 0)
    norm = np.maximum(tissue_d.sum(0, keepdims=True), 1e-9)
    tissue_d = tissue_d / norm
    # ventricles / CSF are outside the tumour-cell domain (no flux)
    csf = tissue_d[0] > 0.6
    if soft is not None:
        csf &= soft[0] > 0.9
    mask_d &= ~csf
    mob = motility_map(tissue_d, gray_white_ratio, fa_d)
    spacing = tuple(float(h) * factor for h in zooms)

    affine = images.get("affine")
    if affine is not None:
        affine = np.array(affine, dtype=float)
        start = np.array([s.start for s in bb], dtype=float)
        affine = affine.copy()
        affine[:3, 3] = affine[:3, :3] @ start + affine[:3, 3]
        affine[:3, :3] = affine[:3, :3] * factor

    if slice_axis is not None:
        core = soft[1] + soft[3] if soft is not None else perm_d
        areas = core.sum(axis=tuple(a for a in range(3) if a != slice_axis))
        k = int(np.argmax(areas))
        take = lambda a, lead=0: np.take(a, k, axis=slice_axis + lead)  # noqa: E731
        mask_d, mob, perm_d = take(mask_d), take(mob), take(perm_d)
        tissue_d = take(tissue_d, 1)
        soft = take(soft, 1) if soft is not None else None
        spacing = tuple(h for i, h in enumerate(spacing) if i != slice_axis)
        affine = None

    if soft is not None:
        soft = np.where(mask_d, soft, np.eye(4)[0][(slice(None),) + (None,) * mask_d.ndim])
    return BrainDomain(
        mask=mask_d,
        spacing=spacing,
        mobility=mob,
        permeability=perm_d,
        tissue=tissue_d * mask_d,
        soft_labels=soft,
        affine=affine,
        meta=dict(
            source="UPENN-GBM" if case_id else "images",
            case_id=case_id,
            factor=factor,
            slice_axis=slice_axis,
            has_dti=fa is not None,
            has_perfusion=rcbv is not None,
            provenance=dict(
                anatomy="patient-specific (MRI)",
                motility="MRI tissue map (+DTI FA) x population D_white prior",
                permeability="T1-Gd enhancement (+rCBV) proxy; not measured drug delivery",
                observations="soft labels from expert segmentation (partial volume)",
                kinetics="population priors",
            ),
        ),
    )


def perturb_segmentation(seg: np.ndarray, rng: np.random.Generator, max_iter: int = 2) -> np.ndarray:
    """Random dilation/erosion of each tumour compartment (segmentation-uncertainty experiments)."""
    out = np.zeros_like(seg)
    for lab in (2, 1, 4, 3):  # edema first so core labels overwrite it
        m = seg == lab
        if not m.any():
            continue
        k = int(rng.integers(-max_iter, max_iter + 1))
        if k > 0:
            m = ndimage.binary_dilation(m, iterations=k)
        elif k < 0:
            m = ndimage.binary_erosion(m, iterations=-k)
        out[m] = lab
    return out


def segmentation_ensemble(segs: Sequence[np.ndarray], factor: int = 1) -> np.ndarray:
    """Average one-hot label maps from several segmentations -> soft labels."""
    oh = np.mean([labels_to_onehot(s) for s in segs], axis=0)
    return block_mean(oh, factor, oh.ndim - 1) if factor > 1 else oh
