"""Dosing schedules and a one-compartment plasma pharmacokinetic model.

Doses are oral/IV boluses; plasma concentration after a dose of ``m`` mg is
``m * c_per_mg * exp(-k_el * (t - t_dose))``.  Tissue concentration is solved on
the brain grid by :mod:`glioma_scenarios.forward` using

    dC/dt = div(D_C grad C) + k_in V(x) C_plasma(t) - k_out C.

This is deliberately simple: no blood-brain-barrier transporter kinetics,
protein binding or tissue binding.  PK parameters are population priors, not
patient measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class DosingSchedule:
    """A named list of (day, dose_mg) events, days relative to the scan."""

    name: str
    doses: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def total_dose(self) -> float:
        return float(sum(m for _, m in self.doses))

    @property
    def dose_times(self) -> List[float]:
        return sorted({float(t) for t, m in self.doses if m > 0})

    def doses_in(self, t0: float, t1: float) -> float:
        """Total dose (mg) with t0 <= t < t1."""
        return float(sum(m for t, m in self.doses if t0 <= t < t1))

    def to_dict(self):
        return dict(name=self.name, doses=[list(d) for d in self.doses], total_dose_mg=self.total_dose)


def no_drug(name: str = "no drug") -> DosingSchedule:
    return DosingSchedule(name, [])


def daily(dose_mg: float, days: int, start: float = 0.0, name: str = None,
          every: float = 1.0) -> DosingSchedule:
    """``days`` doses of ``dose_mg`` spaced ``every`` days apart."""
    name = name or f"{dose_mg:g} mg x {days}"
    return DosingSchedule(name, [(start + i * every, dose_mg) for i in range(days)])


def cycles(dose_mg: float, on_days: int, cycle_len: int, n_cycles: int, start: float = 0.0,
           name: str = None) -> DosingSchedule:
    """Pulsed cycles, e.g. temozolomide-like 5 days on / 23 days off."""
    name = name or f"{dose_mg:g} mg {on_days}/{cycle_len}d x{n_cycles}"
    doses = [(start + c * cycle_len + d, dose_mg) for c in range(n_cycles) for d in range(on_days)]
    return DosingSchedule(name, doses)


def plasma_concentration(schedule: DosingSchedule, t: np.ndarray, c_per_mg: float,
                         k_el: float) -> np.ndarray:
    """Analytic plasma concentration (uM) at times ``t`` (days)."""
    t = np.asarray(t, dtype=float)
    c = np.zeros_like(t)
    for td, m in schedule.doses:
        dt = t - td
        c += np.where(dt >= 0, m * c_per_mg * np.exp(-k_el * np.clip(dt, 0, None)), 0.0)
    return c


def standard_scenarios(dose_mg: float = 270.0) -> List[DosingSchedule]:
    """A default set of broad schedule contrasts used in the scenario analysis.

    * no drug
    * low vs high exposure (same timing)
    * 7-day vs 14-day duration
    * same total dose: 5-day pulse vs daily metronomic over 28 days
    """
    low = dose_mg * 0.5
    total = dose_mg * 5
    return [
        no_drug(),
        daily(low, 5, name="low exposure 5d"),
        daily(dose_mg, 5, name="high exposure 5d"),
        daily(low, 7, name="7-day course"),
        daily(low, 14, name="14-day course"),
        daily(total / 28.0, 28, name="metronomic 28d (same total as 5d pulse)"),
    ]


def dose_event_times(schedules: Sequence[DosingSchedule]) -> List[float]:
    ts = set()
    for s in schedules:
        ts.update(s.dose_times)
    return sorted(ts)
