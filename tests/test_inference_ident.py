import numpy as np

from glioma_scenarios.identifiability import classify_parameters, information_directions, profile_likelihood
from glioma_scenarios.inference import SeedingInverseProblem
from glioma_scenarios.params import ModelParams
from glioma_scenarios.synthetic import make_synthetic_case


def _case(domain):
    truth = ModelParams({"D_white": 0.35, "T_seed": 35.0})
    return make_synthetic_case(domain, truth, (20, 12), np.random.default_rng(1), dt=0.5, boundary_noise=0.3)


def test_map_laplace_fisher(small_domain):
    case = _case(small_domain)
    ip = SeedingInverseProblem(case["domain"], case["obs"]["soft"], ("D_white", "T_seed"),
                               center_vox=case["center"], dt=0.5)
    fit = ip.fit_map(n_adam=15, lr=0.1, n_lbfgs=0)
    assert fit.history[-1] < fit.history[0]
    post = ip.laplace(fit.theta)
    assert np.all(np.linalg.eigvalsh(post.cov) > 0)
    draws = post.sample(10, np.random.default_rng(0))
    assert draws["T_seed"].shape == (10,)
    F = ip.fisher_information(fit.theta)
    assert np.all(np.linalg.eigvalsh(F) > -1e-8)
    prof = profile_likelihood(ip, "D_white", [0.2, 0.35, 0.6], fit.theta, n_adam=3)
    assert prof["neg_log_lik"].shape == (3,)


def test_identifiability_classification():
    names = ["D_white", "beta_max", "mu_p"]
    # beta_max and mu_p enter only through their difference in log space
    v = np.array([0.0, 1.0, -1.0])
    F = np.diag([400.0, 0.0, 0.0]) + 400.0 * np.outer(v, v)
    rows = {r["name"]: r for r in classify_parameters(F, names, include_unfitted=True)}
    assert rows["D_white"]["verdict"] == "identifiable"
    assert rows["beta_max"]["verdict"] == "identifiable only in combination"
    assert rows["mu_p"]["verdict"] == "identifiable only in combination"
    assert "prior" in rows["kappa_max"]["verdict"]
    dirs = information_directions(F, names)
    assert sum(d["informative"] for d in dirs) == 2
    assert not dirs[-1]["informative"]
