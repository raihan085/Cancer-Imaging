"""Treatment-scenario simulation and treatment-equivalence classes (Steps 9-10).

For every posterior draw of the fitted (data-informed) parameters, combined
with a draw of the prior-only parameters (drug PK/PD, immune kinetics), the
scan-time tumour state is reconstructed and every schedule is simulated from
that *same* state (common random numbers), so paired differences between
schedules isolate the effect of the schedule.

Pairwise comparisons are classified as

* ``distinguishable``          -- P(relative difference beyond margin, same sign) >= level
* ``equivalent``               -- P(|relative difference| < margin) >= level
                                  ("equivalent at available resolution")
* ``not supported``            -- neither; the data + priors cannot settle it

and each comparison reports how much of its uncertainty is driven by
prior-only parameters, so conclusions that rest on population assumptions are
labelled as such.  The output is never an individual prescription.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .domain import BrainDomain
from .forward import GliomaModel, param_tensors
from .observation import ObservationModel
from .params import PARAM_SPECS, PRIOR, ModelParams, sample_prior
from .pk import DosingSchedule

#: Prior-only parameters sampled in scenario analysis by default.
SCENARIO_PRIOR_PARAMS = (
    "kappa_max", "EC50", "hill", "phase_center", "phase_width", "k_in", "k_out", "k_el",
    "c_per_mg", "gamma_C", "q_drug_sens", "a_E", "kappa_E", "delta_E", "eps_E",
)

METRICS = ("auc_viable", "final_cycling", "final_viable", "final_total", "final_core_ml", "final_abnormal_ml",
           "ttp_days")


@dataclass
class ScenarioConfig:
    horizon_days: float = 56.0
    record_every: float = 1.0
    #: model-defined radiographic progression: relative increase of core volume
    progression_increase: float = 0.4
    margin: float = 0.10
    level: float = 0.9
    dt: float = 0.25


@dataclass
class ScenarioResults:
    schedules: List[DosingSchedule]
    outcomes: Dict[str, Dict[str, np.ndarray]]
    trajectories: Dict[str, Dict[str, np.ndarray]]
    samples: Dict[str, np.ndarray]
    prior_names: List[str]
    fitted_names: List[str]
    config: ScenarioConfig
    seconds: float = 0.0
    baseline: Dict[str, np.ndarray] = field(default_factory=dict)


def run_scenarios(domain: BrainDomain, schedules: Sequence[DosingSchedule],
                  fitted_samples: Mapping[str, np.ndarray], n_samples: Optional[int] = None,
                  prior_names: Sequence[str] = SCENARIO_PRIOR_PARAMS,
                  base_params: Optional[ModelParams] = None, center_vox: Optional[Sequence[float]] = None,
                  config: Optional[ScenarioConfig] = None, rng: Optional[np.random.Generator] = None,
                  verbose: bool = False) -> ScenarioResults:
    cfg = config or ScenarioConfig()
    rng = np.random.default_rng() if rng is None else rng
    base = base_params or ModelParams.defaults()
    fitted_names = [k for k in fitted_samples if not k.startswith("_")]
    n_avail = min(len(np.atleast_1d(fitted_samples[k])) for k in fitted_names) if fitted_names else n_samples
    n = min(n_samples or n_avail, n_avail) if fitted_names else (n_samples or 1)
    prior = sample_prior(prior_names, n, rng)
    center = tuple(center_vox) if center_vox is not None else domain.tumour_core_centroid()

    model = GliomaModel(domain, dt=cfg.dt)
    obs = ObservationModel(domain)
    n_rec = int(round(cfg.horizon_days / cfg.record_every)) + 1
    traj = {s.name: {k: np.zeros((n, n_rec)) for k in ("cycling", "viable", "total", "core_ml", "abnormal_ml")}
            for s in schedules}
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            vals = {k: float(np.atleast_1d(fitted_samples[k])[i]) for k in fitted_names}
            vals.update({k: float(v[i]) for k, v in prior.items()})
            params = base.with_updates(**vals)
            P = param_tensors(params)
            s0 = model.grow_from_seed(P, center)
            for sch in schedules:
                rec = traj[sch.name]
                j = [0]

                def cb(st, rec=rec, j=j, P=P):
                    if j[0] >= n_rec:
                        return
                    vv = domain.voxel_volume_ml
                    pi = obs.probabilities_from_state(st, P)
                    vol = obs.visible_volume_ml(pi)
                    rec["cycling"][i, j[0]] = float(st.P.sum()) * vv
                    rec["viable"][i, j[0]] = float((st.P + st.q).sum()) * vv
                    rec["total"][i, j[0]] = float(st.N.sum()) * vv
                    rec["core_ml"][i, j[0]] = vol["core"]
                    rec["abnormal_ml"][i, j[0]] = vol["abnormal"]
                    j[0] += 1

                model.run(s0, P, cfg.horizon_days, sch, callback=cb, record_every=cfg.record_every)
            if verbose:
                print(f"scenario sample {i + 1}/{n} ({time.time() - t0:.1f}s)")

    times = np.arange(n_rec) * cfg.record_every
    outcomes = {}
    for sch in schedules:
        rec = traj[sch.name]
        core0 = rec["core_ml"][:, :1]
        prog = rec["core_ml"] >= (1 + cfg.progression_increase) * np.maximum(core0, 1e-9)
        first = np.where(prog.any(1), prog.argmax(1), n_rec - 1)
        outcomes[sch.name] = dict(
            auc_viable=rec["viable"].mean(1),
            final_cycling=rec["cycling"][:, -1],
            final_viable=rec["viable"][:, -1],
            final_total=rec["total"][:, -1],
            final_core_ml=rec["core_ml"][:, -1],
            final_abnormal_ml=rec["abnormal_ml"][:, -1],
            ttp_days=times[first],
            ttp_censored=~prog.any(1),
        )
        rec["times"] = times
    samples = {**{k: np.atleast_1d(fitted_samples[k])[:n] for k in fitted_names}, **prior}
    return ScenarioResults(list(schedules), outcomes, traj, samples, list(prior_names), fitted_names,
                           cfg, time.time() - t0)


# ---------------------------------------------------------------------------
# comparisons and equivalence classes
# ---------------------------------------------------------------------------


def _prior_share(d: np.ndarray, res: ScenarioResults) -> float:
    """Fraction of Var(d) explained (linear R^2) by prior-only parameters."""
    if len(d) < 5 or np.var(d) < 1e-14:
        return 0.0

    def r2(names):
        if not names:
            return 0.0
        X = np.column_stack([np.log(res.samples[k]) for k in names] + [np.ones(len(d))])
        coef, *_ = np.linalg.lstsq(X, d, rcond=None)
        return float(max(0.0, 1 - np.var(d - X @ coef) / np.var(d)))

    prior_names = [k for k in res.prior_names if k in res.samples]
    fitted = [k for k in res.fitted_names if k in res.samples]
    r_all = r2(prior_names + fitted)
    r_fit = r2(fitted)
    return float(np.clip(r_all - r_fit, 0.0, 1.0))


def compare(res: ScenarioResults, a: str, b: str, metric: str = "auc_viable",
            margin: Optional[float] = None, level: Optional[float] = None) -> dict:
    """Classify schedule ``a`` vs ``b`` on ``metric`` (lower burden / longer TTP is better)."""
    margin = res.config.margin if margin is None else margin
    level = res.config.level if level is None else level
    x, y = res.outcomes[a][metric], res.outcomes[b][metric]
    lm = math.log1p(margin)
    d = np.log(np.maximum(x, 1e-12) / np.maximum(y, 1e-12))
    if metric == "ttp_days":
        d = -d  # longer time to progression is better
    # d > 0  => a is worse than b
    p_a_better = float(np.mean(d < -lm))
    p_b_better = float(np.mean(d > lm))
    p_equiv = float(np.mean(np.abs(d) <= lm))
    if p_a_better >= level:
        verdict, better = "distinguishable", a
    elif p_b_better >= level:
        verdict, better = "distinguishable", b
    elif p_equiv >= level:
        verdict, better = "equivalent", None
    else:
        verdict, better = "not supported", None
    share = _prior_share(d, res)
    # direction only (any effect size): useful when the sign is settled but the magnitude is not
    p_dir_a = float(np.mean(d < 0))
    direction = a if p_dir_a >= level else (b if 1 - p_dir_a >= level and np.any(d != 0) else None)
    statement = _statement(verdict, a, b, better, metric, share)
    if verdict == "not supported" and direction is not None:
        statement += (f" The direction favours '{direction}' in >= {level:.0%} of draws, but the size of the "
                      f"difference is not resolved relative to the {margin:.0%} margin.")
    return dict(
        a=a, b=b, metric=metric, verdict=verdict, better=better, direction=direction,
        p_a_better=p_a_better, p_b_better=p_b_better, p_equivalent=p_equiv, p_direction_a=p_dir_a,
        median_log_ratio=float(np.median(d)),
        ci90_log_ratio=[float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))],
        prior_driven_share=share,
        conditional_on_priors=bool(share > 0.5),
        statement=statement,
    )


def _statement(verdict, a, b, better, metric, share) -> str:
    cond = " (driven mainly by population drug/immune priors)" if share > 0.5 else ""
    if verdict == "distinguishable":
        worse = b if better == a else a
        return (f"Under the stated assumptions, '{better}' is likely better than '{worse}' "
                f"on {metric}{cond}.")
    if verdict == "equivalent":
        return f"The scan and priors cannot distinguish '{a}' from '{b}' on {metric} (equivalent at available resolution)."
    return (f"No reliable comparison of '{a}' vs '{b}' on {metric}: the relevant parameters are not "
            f"constrained enough{cond}.")


def equivalence_classes(res: ScenarioResults, metric: str = "auc_viable", **kw) -> dict:
    """Pairwise verdicts plus classes of mutually equivalent schedules.

    Equivalence is not transitive, so a schedule joins a class only if it is
    equivalent to *every* member (greedy clique construction).
    """
    names = [s.name for s in res.schedules]
    pairs = [compare(res, a, b, metric, **kw) for i, a in enumerate(names) for b in names[i + 1:]]
    equiv = {(p["a"], p["b"]) for p in pairs if p["verdict"] == "equivalent"}
    equiv |= {(b, a) for a, b in equiv}
    classes: List[List[str]] = []
    for n in names:
        for c in classes:
            if all((n, m) in equiv for m in c):
                c.append(n)
                break
        else:
            classes.append([n])
    # count how many schedules each one is distinguishably better than
    wins = {n: sum(1 for p in pairs if p["better"] == n) for n in names}
    classes.sort(key=lambda c: -max(wins[n] for n in c))
    return dict(metric=metric, pairs=pairs, classes=classes, wins=wins)


def outcome_summary(res: ScenarioResults) -> List[dict]:
    rows = []
    for s in res.schedules:
        o = res.outcomes[s.name]
        row = dict(schedule=s.name, total_dose_mg=s.total_dose)
        for m in METRICS:
            v = o[m]
            row[m] = dict(median=float(np.median(v)), q05=float(np.quantile(v, 0.05)),
                          q95=float(np.quantile(v, 0.95)))
        row["fraction_progressed"] = float(np.mean(~o["ttp_censored"]))
        rows.append(row)
    return rows
