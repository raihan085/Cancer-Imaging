import numpy as np
import torch

from glioma_scenarios.domain import FVDiffusion, phantom_domain
from glioma_scenarios.forward import GliomaModel, param_tensors, summarize_state
from glioma_scenarios.params import ModelParams
from glioma_scenarios.pk import daily


def test_diffusion_conserves_mass_and_no_flux(small_domain):
    d = small_domain
    op = FVDiffusion(d.mobility, d.mask, d.spacing)
    u = torch.zeros(d.shape, dtype=torch.float64)
    idx = tuple(int(c) for c in np.argwhere(d.mask)[len(np.argwhere(d.mask)) // 2])
    u[idx] = 1.0
    v = op.step(u, 5.0, 10.0)
    assert abs(float(v.sum()) - 1.0) < 1e-10
    assert float(v[~torch.as_tensor(d.mask)].abs().max()) == 0.0
    assert float(v.min()) >= -1e-12


def test_diffusion_3d_and_batch_axes():
    d = phantom_domain((12, 12, 10), spacing=3.0)
    op = FVDiffusion(d.mobility, d.mask, d.spacing)
    u = torch.rand((4,) + d.shape, dtype=torch.float64) * torch.as_tensor(d.mask)
    v = op.step(u, 1.0, 1.0)
    assert torch.allclose(v.sum(dim=(1, 2, 3)), u.sum(dim=(1, 2, 3)))


def test_forward_positive_and_growing(small_domain):
    m = GliomaModel(small_domain, dt=0.5)
    P = param_tensors(ModelParams({"T_seed": 20.0}))
    s = m.grow_from_seed(P, (20, 12))
    for f in (s.p, s.q, s.r, s.E, s.C):
        assert float(f.min()) >= -1e-12
    s0 = m.seed_state(P, (20, 12))
    assert float(s.N.sum()) > float(s0.N.sum())


def test_age_advection_without_division_or_death(small_domain):
    """With every rate switched off, cycle ages simply advance and mass is conserved."""
    zero = dict(beta_max=1e-12, mu_p=1e-12, gamma_pq=1e-12, gamma_C=1e-12, a_E=1e-12,
                kappa_E=1e-12, gamma_qp=1e-12, mu_q=1e-12, D_white=1e-12)
    m = GliomaModel(small_domain, dt=0.5, a_max=4.0)
    P = param_tensors(ModelParams(zero))
    s = m.seed_state(P, (16, 16))
    total0 = float(s.p.sum())
    s1 = m.step(s, P)
    assert abs(float(s1.p.sum()) - total0) < 1e-9
    assert torch.allclose(s1.p[1:-1], s.p[:-2])
    assert float(s1.p[0].abs().max()) < 1e-9


def test_renewal_doubles_dividing_cells(small_domain):
    """Low density, no losses: every division adds exactly one cell (2 daughters - 1 mother)."""
    no_loss = dict(mu_p=1e-12, gamma_pq=1e-12, gamma_C=1e-12, a_E=1e-12, kappa_E=1e-12,
                   D_white=1e-12, seed_amp=1e-3)
    m = GliomaModel(small_domain, dt=0.25)
    P = param_tensors(ModelParams(no_loss))
    s = m.seed_state(P, (16, 16))
    s = m.run(s, P, 3.0)  # let cells reach division age
    s1 = m.step(s, P)
    beta = m.division_hazard(P)
    H = beta + P["mu_p"] + P["gamma_pq"] * 0  # dominant hazard
    divided = (s.p * (1 - torch.exp(-H * m.dt)) * beta / H).sum()
    N = s.N
    gain = float(s1.P.sum() + s1.q.sum() - s.P.sum() - s.q.sum())
    assert abs(gain - float(divided)) / float(divided) < 0.02
    assert float(N.max()) < 0.01


def test_drug_reduces_cycling_burden(small_domain):
    m = GliomaModel(small_domain, dt=0.25)
    P = param_tensors(ModelParams({"T_seed": 30.0, "kappa_max": 5.0}))
    s0 = m.grow_from_seed(P, (20, 12))
    with torch.no_grad():
        a = m.run(s0, P, 7.0)
        b = m.run(s0, P, 7.0, daily(270.0, 5))
    assert float(b.P.sum()) < float(a.P.sum())
    assert summarize_state(b, small_domain)["drug_tissue_mean"] >= 0


def test_gradients_wrt_parameters(small_domain):
    m = GliomaModel(small_domain, dt=0.5)
    D = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    T = torch.tensor(15.3, dtype=torch.float64, requires_grad=True)
    P = param_tensors(ModelParams.defaults(), {"D_white": D, "T_seed": T})
    s = m.grow_from_seed(P, (20, 12))
    (s.N ** 2).sum().backward()
    assert D.grad is not None and torch.isfinite(D.grad) and float(D.grad) != 0
    assert T.grad is not None and float(T.grad) != 0
