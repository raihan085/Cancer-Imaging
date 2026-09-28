import numpy as np
import torch

from glioma_scenarios.baselines import FisherKPPInversion
from glioma_scenarios.data.upenn import index_dataset, load_case, read_clinical, attach_clinical, select_pilot_subset
from glioma_scenarios.gradcam import coarse_roi, gradcam
from glioma_scenarios.pinn import GliomaPINN, PINNConfig
from glioma_scenarios.pk import daily
from glioma_scenarios.preprocess import block_mean, build_domain, perturb_segmentation


def _write_fake_case(root, cid, with_dti=True):
    import nibabel as nib
    rng = np.random.default_rng(0)
    shape = (30, 30, 20)
    g = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    r = np.sqrt(sum(x ** 2 for x in g))
    brain = r < 0.9
    t1 = np.where(brain, 1.0 + 0.5 * (r < 0.6) + 0.05 * rng.standard_normal(shape), 0)
    tum = np.sqrt((g[0] - 0.3) ** 2 + g[1] ** 2 + g[2] ** 2)
    seg = np.zeros(shape, np.int16)
    seg[(tum < 0.4) & brain] = 2
    seg[(tum < 0.25) & brain] = 4
    seg[(tum < 0.12) & brain] = 1
    t1gd = t1 + 2.0 * (seg == 4)
    d = root / "images_structural" / cid
    d.mkdir(parents=True)
    aff = np.eye(4)
    for k, img in dict(T1=t1, T1GD=t1gd, T2=np.where(brain, 2 - t1, 0), FLAIR=np.where(brain, 1 + (seg > 0), 0)).items():
        nib.save(nib.Nifti1Image(img.astype(np.float32), aff), str(d / f"{cid}_{k}.nii.gz"))
    (root / "images_segm").mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(seg, aff), str(root / "images_segm" / f"{cid}_segm.nii.gz"))
    if with_dti:
        dd = root / "images_DTI" / cid
        dd.mkdir(parents=True)
        nib.save(nib.Nifti1Image((0.4 * (r < 0.6)).astype(np.float32), aff), str(dd / f"{cid}_DTI_FA.nii.gz"))
        dp = root / "images_DSC" / cid
        dp.mkdir(parents=True)
        nib.save(nib.Nifti1Image((1 + 3 * (seg == 4)).astype(np.float32), aff), str(dp / f"{cid}_DSC_ap-rCBV.nii.gz"))


def test_upenn_index_select_and_build(tmp_path):
    _write_fake_case(tmp_path, "UPENN-GBM-00001_11")
    _write_fake_case(tmp_path, "UPENN-GBM-00002_11", with_dti=False)
    (tmp_path / "clin.csv").write_text("ID,IDH1,MGMT\nUPENN-GBM-00001,Wildtype,Methylated\nUPENN-GBM-00002,NA,\n")
    cases = index_dataset(tmp_path)
    assert set(cases) == {"UPENN-GBM-00001_11", "UPENN-GBM-00002_11"}
    c1 = cases["UPENN-GBM-00001_11"]
    assert c1.has_structural and c1.has_expert_segm and c1.has_dti and c1.has_perfusion
    attach_clinical(cases, read_clinical(tmp_path / "clin.csv"))
    assert c1.molecular_known() == {"IDH": True, "MGMT": True}
    sel = select_pilot_subset(cases, 5)
    assert [c.case_id for c in sel] == ["UPENN-GBM-00001_11"]
    imgs = load_case(c1)
    dom = build_domain(imgs, factor=2, case_id=c1.case_id)
    assert dom.ndim == 3 and dom.soft_labels.shape == (4,) + dom.shape
    assert np.allclose(dom.soft_labels.sum(0), 1)
    assert dom.permeability[dom.soft_labels[3] > 0.5].mean() > dom.permeability[dom.mask & (dom.soft_labels[0] > 0.99)].mean()
    dom2 = build_domain(imgs, factor=2, slice_axis=2)
    assert dom2.ndim == 2 and dom2.soft_labels.shape[0] == 4
    seg2 = perturb_segmentation(imgs["segm"], np.random.default_rng(0))
    assert set(np.unique(seg2)) <= {0, 1, 2, 4}


def test_block_mean():
    x = np.arange(16.0).reshape(4, 4)
    assert block_mean(x, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]


def test_pinn_quadrature_and_training(small_domain):
    pinn = GliomaPINN(small_domain, daily(100.0, 2), PINNConfig(t_end=4.0, n_iters=8, n_colloc=64,
                                                               n_obs_voxels=64, hidden=16, depth=2,
                                                               n_boundary=32, log_every=1))
    # Gauss-Legendre over [0, a_max] integrates polynomials exactly
    f = pinn.quad_a ** 3
    assert abs(float((f * pinn.quad_w).sum()) - pinn.cfg.a_max ** 4 / 4) < 1e-2
    assert len(pinn.bounds) == 3  # dose at t=1 splits [0, 4]
    L = small_domain.soft_labels if small_domain.soft_labels is not None else \
        np.eye(4)[np.zeros(small_domain.shape, int)].transpose(2, 0, 1)
    hist = pinn.fit([(0.0, L)])
    assert all(np.isfinite(h["loss"]) for h in hist)
    assert hist[-1]["renewal"] >= 0
    pred = pinn.predict(1.0)
    assert pred["P"].shape == small_domain.shape and np.all(pred["P"] >= 0)
    base = GliomaPINN(small_domain, None, PINNConfig(t_end=2.0, n_iters=2, renewal_mode="none",
                                                     integral_mode="mc", n_colloc=32, hidden=8, depth=1))
    h = base.fit([])
    assert h[-1]["renewal"] == 0.0


def test_kpp_baseline(small_domain):
    L = np.eye(4)[np.zeros(small_domain.shape, int)].transpose(2, 0, 1).astype(float)
    kpp = FisherKPPInversion(small_domain, L, (20, 12))
    fit = kpp.fit(n_iter=3)
    assert set(fit) >= {"D_white", "rho", "T"}
    assert kpp.density(fit).shape == small_domain.shape


def test_gradcam():
    net = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(),
                              torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2))
    img = torch.zeros(1, 1, 16, 16)
    img[..., 4:8, 4:8] = 1
    cam = gradcam(net, net[0], img, target=0)
    assert cam.shape == (16, 16) and cam.min() >= 0 and cam.max() <= 1
    roi = coarse_roi(cam)
    assert len(roi) == 2
