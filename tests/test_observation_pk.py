import numpy as np
import torch

from glioma_scenarios.forward import GliomaModel, param_tensors
from glioma_scenarios.observation import (ObservationModel, labels_to_onehot, soft_labels_from_hard,
                                          synthesize_intensities, synthesize_observation)
from glioma_scenarios.params import ModelParams
from glioma_scenarios.pk import cycles, daily, plasma_concentration, standard_scenarios


def test_label_mapping_upenn_convention():
    seg = np.array([[0, 1, 2, 4, 3]])
    oh = labels_to_onehot(seg)
    assert oh.shape == (4, 1, 5)
    assert oh[:, 0, 0].tolist() == [1, 0, 0, 0]
    assert oh[:, 0, 1].tolist() == [0, 1, 0, 0]
    assert oh[:, 0, 2].tolist() == [0, 0, 1, 0]
    assert oh[:, 0, 3].tolist() == [0, 0, 0, 1]
    assert oh[:, 0, 4].tolist() == [0, 0, 0, 1]
    soft = soft_labels_from_hard(np.pad(seg, 3), smooth_vox=1.0)
    assert np.allclose(soft.sum(0), 1)


def test_observation_probabilities(small_domain):
    m = GliomaModel(small_domain, dt=0.5)
    P = param_tensors(ModelParams({"T_seed": 40.0}))
    s = m.grow_from_seed(P, (20, 12))
    om = ObservationModel(small_domain)
    pi = om.probabilities_from_state(s, P)
    assert torch.allclose(pi.sum(0), torch.ones_like(pi[0]))
    assert float(pi[0][~torch.as_tensor(small_domain.mask)].min()) == 1.0
    vol = om.visible_volume_ml(pi)
    assert vol["abnormal"] > 0
    # likelihood is higher for the generating probabilities than for a shifted tumour
    s2 = m.grow_from_seed(P, (12, 20))
    pi2 = om.probabilities_from_state(s2, P)
    assert float(om.log_likelihood(pi, pi)) > float(om.log_likelihood(pi2, pi))
    obs = synthesize_observation(pi.numpy(), small_domain.mask, np.random.default_rng(0))
    assert obs["hard"].shape == small_domain.shape
    assert np.allclose(obs["soft"].sum(0), 1)
    imgs = synthesize_intensities(pi.numpy(), small_domain, np.random.default_rng(0))
    assert set(imgs) == {"T1", "T1GD", "T2", "FLAIR"}


def test_pk_schedules():
    s = daily(100.0, 5)
    assert s.total_dose == 500.0 and s.doses_in(0, 2) == 200.0
    c = cycles(150.0, 5, 28, 2)
    assert len(c.doses) == 10 and c.doses[5][0] == 28
    cp = plasma_concentration(daily(100.0, 1), np.array([-1.0, 0.0, 1.0]), 0.26, 9.2)
    assert cp[0] == 0 and abs(cp[1] - 26.0) < 1e-9 and cp[2] < 0.01
    sc = standard_scenarios()
    names = [x.name for x in sc]
    assert "no drug" in names and len(set(names)) == len(names)
    pulse = [x for x in sc if x.name == "high exposure 5d"][0]
    metro = [x for x in sc if x.name.startswith("metronomic")][0]
    assert abs(pulse.total_dose - metro.total_dose) < 1e-6
