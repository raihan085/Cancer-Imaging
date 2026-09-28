"""Grad-CAM for coarse tumour localisation and explainability only.

Grad-CAM heat-maps are too coarse to define compartment boundaries for the
mechanistic model.  The intended workflow is:

1. Grad-CAM -> coarse localisation / sanity check of an image classifier;
2. a dedicated segmentation model -> boundaries and soft compartment
   probabilities;
3. those soft probabilities -> the observation model.

:func:`gradcam` works with any 2-D or 3-D convolutional PyTorch classifier.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model: torch.nn.Module, layer: torch.nn.Module, image: torch.Tensor,
            target: Optional[int] = None) -> np.ndarray:
    """Return a Grad-CAM map in [0, 1] with the spatial shape of ``image``.

    ``image`` has shape (1, C, H, W) or (1, C, D, H, W).
    """
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        image = image.clone().requires_grad_(True)
        logits = model(image)
        idx = int(logits.argmax(1)) if target is None else target
        model.zero_grad()
        logits[0, idx].backward()
        a, g = acts["a"], grads["g"]
        w = g.mean(dim=tuple(range(2, g.ndim)), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        mode = "bilinear" if image.ndim == 4 else "trilinear"
        cam = F.interpolate(cam, size=image.shape[2:], mode=mode, align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-12)
        return cam.detach().numpy()
    finally:
        h1.remove()
        h2.remove()


def coarse_roi(cam: np.ndarray, quantile: float = 0.9, margin: int = 4) -> Tuple[slice, ...]:
    """Bounding box of the top-(1-quantile) Grad-CAM region, for cropping only."""
    m = cam >= np.quantile(cam, quantile)
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, cam.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
