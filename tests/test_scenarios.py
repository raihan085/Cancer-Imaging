import numpy as np

from glioma_scenarios.pk import daily, no_drug
from glioma_scenarios.scenarios import (ScenarioConfig, ScenarioResults, compare, equivalence_classes,
                                        outcome_summary, run_scenarios)


def _fake(values, rng):
    n = len(next(iter(values.values())))
    scheds = [no_drug("A"), daily(1, 1, name="B"), daily(2, 1, name="C")]
    outcomes = {k: {"auc_viable": v, "final_viable": v, "final_cycling": v, "final_total": v,
                    "final_core_ml": v, "final_abnormal_ml": v, "ttp_days": v,
                    "ttp_censored": np.zeros(n, bool)} for k, v in values.items()}
    samples = {"kappa_max": np.exp(rng.standard_normal(n)), "D_white": np.exp(rng.standard_normal(n))}
    return ScenarioResults(scheds, outcomes, {}, samples, ["kappa_max"], ["D_white"], ScenarioConfig())


def test_equivalence_verdicts():
    rng = np.random.default_rng(0)
    base = np.exp(0.1 * rng.standard_normal(400)) + 1.0
    res = _fake({"A": base, "B": base * 1.01, "C": base * 0.5}, rng)
    assert compare(res, "A", "B")["verdict"] == "equivalent"
    c = compare(res, "A", "C")
    assert c["verdict"] == "distinguishable" and c["better"] == "C"
    noisy = base * np.exp(0.5 * rng.standard_normal(400))
    assert compare(_fake({"A": base, "B": noisy, "C": base}, rng), "A", "B")["verdict"] == "not supported"
    eq = equivalence_classes(res)
    assert ["A", "B"] in eq["classes"] and ["C"] in eq["classes"]
    assert eq["classes"][0] == ["C"]  # the distinguishably better class ranks first


def test_prior_driven_share_detected():
    rng = np.random.default_rng(1)
    n = 300
    base = np.ones(n)
    res = _fake({"A": base, "B": base, "C": base}, rng)
    k = res.samples["kappa_max"]
    res.outcomes["B"]["auc_viable"] = base * np.exp(-0.5 * np.log(k))
    c = compare(res, "A", "B")
    assert c["prior_driven_share"] > 0.9 and c["conditional_on_priors"]


def test_run_scenarios_small(small_domain):
    rng = np.random.default_rng(0)
    fs = {"D_white": np.full(3, 0.3), "T_seed": np.full(3, 30.0)}
    res = run_scenarios(small_domain, [no_drug(), daily(270.0, 3, name="drug")], fs,
                        center_vox=(20, 12), config=ScenarioConfig(horizon_days=6, dt=0.5), rng=rng)
    rows = outcome_summary(res)
    assert len(rows) == 2 and res.outcomes["drug"]["auc_viable"].shape == (3,)
    assert res.trajectories["drug"]["viable"].shape == (3, 7)
