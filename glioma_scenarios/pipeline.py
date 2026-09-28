"""End-to-end analysis: fit -> uncertainty -> identifiability -> scenarios -> report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .domain import BrainDomain
from .identifiability import classify_parameters
from .inference import DEFAULT_FIT, SeedingInverseProblem
from .params import ModelParams, provenance_table
from .pk import DosingSchedule, standard_scenarios
from .report import markdown_report, save_json
from .scenarios import ScenarioConfig, equivalence_classes, outcome_summary, run_scenarios
from .seeding import seeding_statement, seeding_time_sensitivity


@dataclass
class AnalysisConfig:
    fit_names: Sequence[str] = DEFAULT_FIT
    dt: float = 0.25
    n_adam: int = 150
    lr: float = 0.05
    n_lbfgs: int = 20
    uncertainty: str = "laplace"  # or "hmc"
    hmc_samples: int = 100
    hmc_warmup: int = 30
    n_scenario_samples: int = 32
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    metric: str = "auc_viable"
    seeding_sensitivity: bool = False
    seed: int = 0


def analyze(domain: BrainDomain, title: str, schedules: Optional[List[DosingSchedule]] = None,
            soft_labels: Optional[np.ndarray] = None, center_vox=None,
            base_params: Optional[ModelParams] = None, cfg: Optional[AnalysisConfig] = None,
            out_dir: Optional[Path] = None, verbose: bool = True) -> Dict:
    cfg = cfg or AnalysisConfig()
    rng = np.random.default_rng(cfg.seed)
    schedules = schedules or standard_scenarios()
    base = base_params or ModelParams.defaults()
    names = list(cfg.fit_names)
    ip = SeedingInverseProblem(domain, soft_labels, names, base, center_vox, dt=cfg.dt)
    if verbose:
        print(f"[fit] MAP over {names} (obs weight {ip.obs.obs_weight:.3f})")
    fit = ip.fit_map(n_adam=cfg.n_adam, lr=cfg.lr, n_lbfgs=cfg.n_lbfgs, verbose=verbose)
    post = ip.laplace(fit.theta)
    F = ip.fisher_information(fit.theta)
    ident = classify_parameters(F, names)
    if cfg.uncertainty == "hmc":
        if verbose:
            print("[uncertainty] HMC")
        draws = ip.hmc(fit.theta, cfg.hmc_samples, cfg.hmc_warmup,
                       mass_diag=1.0 / np.diag(post.cov), rng=rng, verbose=verbose)
    else:
        draws = post.sample(cfg.n_scenario_samples, rng)
    if verbose:
        print("[scenarios] simulating", [s.name for s in schedules])
    scfg = cfg.scenario
    scfg.dt = cfg.dt
    res = run_scenarios(domain, schedules, draws, cfg.n_scenario_samples, base_params=base,
                        center_vox=ip.center, config=scfg, rng=rng, verbose=verbose)
    eq = equivalence_classes(res, cfg.metric)
    outcomes = outcome_summary(res)
    seeding_rows = seeding_text = None
    if cfg.seeding_sensitivity and "T_seed" in names:
        seeding_rows = seeding_time_sensitivity(domain, ip.L.numpy(), base, fit_names=names, rng=rng,
                                                fit_kwargs=dict(n_adam=cfg.n_adam, lr=cfg.lr, n_lbfgs=0),
                                                center_vox=ip.center, dt=cfg.dt)
        seeding_text = seeding_statement(seeding_rows)
    provenance = dict(domain.meta.get("provenance", {}))
    provenance.setdefault("anatomy", domain.meta.get("source", "domain"))
    provenance["fitted parameters"] = ", ".join(names) + " (log-normal priors, Laplace/HMC uncertainty)"
    provenance["drug & immune parameters"] = "sampled from population priors (scenario inputs)"
    md = markdown_report(title, provenance, ident, outcomes, eq, seeding_rows, seeding_text)
    result = dict(
        title=title,
        fit=dict(names=names, map={k: float(np.exp(v)) for k, v in zip(names, fit.theta)},
                 seconds=fit.seconds, history=fit.history[-5:]),
        posterior=post.summary(),
        posterior_correlation=post.correlation(),
        fisher_information=F,
        identifiability=ident,
        outcomes=outcomes,
        equivalence=eq,
        seeding=seeding_rows,
        schedules=[s.to_dict() for s in schedules],
        parameter_provenance=provenance_table(names),
        meta=domain.meta,
    )
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(result, out_dir / "analysis.json")
        (out_dir / "report.md").write_text(md)
        np.savez_compressed(out_dir / "trajectories.npz",
                            **{f"{s}__{k}": v for s, d in res.trajectories.items() for k, v in d.items()})
    result["markdown"] = md
    return result
