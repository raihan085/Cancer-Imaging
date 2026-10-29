"""
data_io.py -- Load, clean and winsorize the Yemen governorate bulletin panel,
build the observation design, and generate a synthetic twin that matches the
record's SAMPLING DESIGN rather than only its size.

Expects the project's own files in --data-dir:
    cases_long.csv         cleaned, name-harmonized, spatially-referenced panel
    governorate_coords.csv governorate -> great-circle distance from Sana'a (km)
    raw_population.csv     governorate -> population

Column names differ between exports, so COLUMN_HINTS does fuzzy detection and
the loader REPORTS what it matched. Override via COLUMN_OVERRIDE.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

import seip

COLUMN_OVERRIDE = {
    "governorate": None, "date": None, "cumulative": None, "new_cases": None,
    "incidence_rate": None, "interval_days": None, "x_km": None, "population": None,
}

COLUMN_HINTS = {
    "governorate": ["governorate", "gov", "admin1", "region", "name"],
    "date": ["date", "report_date", "day"],
    "cumulative": ["cum", "cumulative", "total_cases", "cases_cum"],
    "new_cases": ["new_cases", "new", "delta_cases", "d_cases", "cases_new", "increment"],
    "incidence_rate": ["incidence_rate", "percapita_incidence", "incidence", "per_capita"],
    "interval_days": ["interval_days", "interval", "dt_days", "gap"],
    "x_km": ["distance", "x_km", "dist_km", "x", "km"],
    "population": ["population", "pop", "n_g", "total"],
}

WINSOR_Q = 0.995
# The record's window opens here (problem statement Sec. 1). Used so that a
# cases_long.csv carrying the first wave or the 2018 resurgence does not silently
# shift the time origin.
WINDOW_START = "2017-05-22"


def _match(cols, key):
    if COLUMN_OVERRIDE.get(key):
        return COLUMN_OVERRIDE[key]
    low = {c.lower().replace(" ", "_"): c for c in cols}
    for hint in COLUMN_HINTS[key]:
        for lc, orig in low.items():
            if lc == hint:
                return orig
    for hint in COLUMN_HINTS[key]:
        for lc, orig in low.items():
            if hint in lc:
                return orig
    return None


@dataclass
class Panel:
    """The fitting design. One row per (governorate, bulletin interval)."""
    gov: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    y: np.ndarray                # per-capita incidence after winsorization
    y_raw: np.ndarray
    dC: np.ndarray               # raw case increment after winsorization
    dC_raw: np.ndarray
    w: np.ndarray                # weight in force
    w_count: np.ndarray          # 1/sqrt(max(dC,1))        -- as specified
    w_rate: np.ndarray           # N dt / sqrt(max(dC,1))   -- rate-consistent
    names: list
    xg: np.ndarray
    pop: np.ndarray
    y_first: np.ndarray
    cum_first: np.ndarray
    t_first: np.ndarray
    winsor_cut: float
    n_winsorized: int
    true_zero: np.ndarray = None      # (ng,) structural zero seed
    first_hold: np.ndarray = None     # (ng,) first bulletin was a reporting hold
    provenance: dict = field(default_factory=dict)
    synthetic: bool = False
    weight_kind: str = "count"

    @property
    def ng(self):
        return len(self.names)

    @property
    def n_obs(self):
        return len(self.y)

    def set_weights(self, kind="count", yhat=None):
        """'count'  1/sqrt(max(dC,1)) -- the weight the methodology specifies.
        'rate'   N dt / sqrt(max(dC,1)) -- what a per-capita RATE residual calls
                 for, since Var(y) = dC/(N dt)^2 for a Poisson count.
        'model'  1/sqrt(max(yhat N dt,1)) scaled to a rate -- iteratively
                 reweighted, using the FITTED mean rather than the realization.

        'count' and 'rate' both put the observed count in the denominator, so a
        row that happened to fluctuate low is up-weighted and an observed zero is
        up-weighted maximally. That is the classic downward bias of
        Poisson-weighted least squares and it acts in the SAME direction as the
        initial-condition truncation mismatch. 'model' removes it: see
        fit.fit(..., irls=True).
        """
        if kind == "count":
            w = self.w_count
        elif kind == "rate":
            w = self.w_rate
        elif kind == "none":
            w = np.ones_like(self.w_count)
        elif kind == "model":
            if yhat is None:
                raise ValueError("kind='model' needs yhat")
            Ndt = self.pop[self.gov] * (self.t1 - self.t0)
            w = Ndt / np.sqrt(np.maximum(np.asarray(yhat, float) * Ndt, 1.0))
        else:
            raise ValueError(kind)
        self.w = w / w.mean()
        self.weight_kind = kind
        return self

    def zero_weight_share(self):
        """Fraction of total squared weight carried by exact-zero rows. On the
        record this is the dominant part of the objective; a synthetic twin that
        under-represents it does not test the thing being relied on."""
        z = self.y_raw == 0
        return float(np.sum(self.w[z] ** 2) / np.sum(self.w ** 2))

    def describe(self):
        iv = self.t1 - self.t0
        days = np.unique(np.round(np.concatenate([self.t0, self.t1]), 6))
        return (f"{self.n_obs} fitting rows, {self.ng} governorates, "
                f"{len(days)} distinct bulletin days, "
                f"{int((self.y_raw == 0).sum())} exact zeros "
                f"({100*np.mean(self.y_raw == 0):.1f}%), "
                f"interval days {iv.min():.3g}-{iv.max():.3g}, "
                f"winsor {self.winsor_cut:.4g} ({self.n_winsorized} rows), "
                f"zero-row weight share {100*self.zero_weight_share():.1f}%")


def load_panel(data_dir="data", winsorize=True, t_max=seip.T_MAX,
               window_start=WINDOW_START, verbose=True) -> Panel:
    cases_p = os.path.join(data_dir, "cases_long.csv")
    coords_p = os.path.join(data_dir, "governorate_coords.csv")
    pop_p = os.path.join(data_dir, "raw_population.csv")
    if not os.path.exists(cases_p):
        raise FileNotFoundError(
            f"{cases_p} not found. Put cases_long.csv, governorate_coords.csv and "
            f"raw_population.csv in {data_dir}/, or run with --synthetic.")

    df = pd.read_csv(cases_p)
    c = {k: _match(df.columns, k) for k in COLUMN_HINTS}
    if verbose:
        print("cases_long.csv columns matched:", {k: v for k, v in c.items() if v})
    if not c["governorate"] or not c["date"]:
        raise ValueError(f"no governorate/date column among {list(df.columns)}; "
                         "set COLUMN_OVERRIDE in data_io.py")
    df = df.rename(columns={v: k for k, v in c.items() if v})
    df["governorate"] = df["governorate"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"])

    if os.path.exists(coords_p):
        cd = pd.read_csv(coords_p)
        cd = cd.rename(columns={_match(cd.columns, "governorate"): "governorate",
                                _match(cd.columns, "x_km"): "x_km"})
        cd["governorate"] = cd["governorate"].astype(str).str.strip()
        coords = dict(zip(cd["governorate"], cd["x_km"].astype(float)))
    elif "x_km" in df.columns:
        coords = df.groupby("governorate")["x_km"].first().to_dict()
    else:
        raise ValueError("no governorate_coords.csv and no distance column")

    if "population" in df.columns:
        pops = df.groupby("governorate")["population"].first().to_dict()
    elif os.path.exists(pop_p):
        pp = pd.read_csv(pop_p)
        pp = pp.rename(columns={_match(pp.columns, "governorate"): "governorate",
                                _match(pp.columns, "population"): "population"})
        pp["governorate"] = pp["governorate"].astype(str).str.strip()
        pops = dict(zip(pp["governorate"], pp["population"].astype(float)))
    else:
        raise ValueError("no population column and no raw_population.csv")

    prov = {}
    missing = sorted(set(df["governorate"]) - set(coords) | set(df["governorate"]) - set(pops))
    if missing:
        prov["dropped_governorates"] = missing
        if verbose:
            print(f"  dropped {len(missing)} governorate(s) lacking a coordinate or "
                  f"population: {', '.join(missing)}")
    df = df[df["governorate"].isin(coords) & df["governorate"].isin(pops)].copy()

    # Explicit time origin. df.date.min() would silently shift the window if the
    # file carries the first wave or the 2018 resurgence.
    origin = pd.Timestamp(window_start) if window_start else df["date"].min()
    prov["window_start"] = str(origin.date())
    n_before = len(df)
    df = df[(df["date"] >= origin) &
            (df["date"] <= origin + pd.Timedelta(days=t_max))]
    prov["rows_outside_window"] = int(n_before - len(df))
    if prov["rows_outside_window"] and verbose:
        print(f"  {prov['rows_outside_window']} row(s) outside the "
              f"{t_max:g}-day window from {prov['window_start']} dropped")
    df["elapsed"] = (df["date"] - origin).dt.days.astype(float)
    df = df.sort_values(["governorate", "elapsed"])
    # Each row of an increment panel is a whole interval ending at `elapsed`.
    # _assemble differences consecutive rows, so without an opening snapshot it
    # treats the first row as a baseline and silently drops that governorate's
    # first interval (21 intervals, 16,598 cases on the record). Prepend the
    # snapshot at the start of the first interval so every row is fitted.
    if {"cumulative", "new_cases", "interval_days"} <= set(df.columns):
        first = df.groupby("governorate", as_index=False).head(1).copy()
        first["elapsed"] = first["elapsed"] - first["interval_days"]
        first["date"] = first["date"] - pd.to_timedelta(first["interval_days"], unit="D")
        first["cumulative"] = first["cumulative"] - first["new_cases"]
        first[["new_cases", "incidence_rate"]] = np.nan
        df = (pd.concat([first, df], ignore_index=True)
                .sort_values(["governorate", "elapsed"], kind="stable"))
    return _assemble(df, coords, pops, winsorize, False, prov, verbose)


def _assemble(df, coords, pops, winsorize, synthetic, prov=None, verbose=False):
    prov = dict(prov or {})
    names = sorted(df["governorate"].unique(), key=lambda g: coords[g])
    gidx = {g: i for i, g in enumerate(names)}
    xg = np.array([coords[g] for g in names], float)
    pop = np.array([pops[g] for g in names], float)

    rows = []
    y_first = np.zeros(len(names)); cum_first = np.zeros(len(names))
    t_first = np.zeros(len(names))
    first_hold = np.zeros(len(names), bool); true_zero = np.zeros(len(names), bool)
    n_dupe = n_monotone = 0
    have_cum = "cumulative" in df.columns

    for g, sub in df.groupby("governorate"):
        i = gidx[g]; N = pops[g]
        sub = sub.sort_values("elapsed")
        # Merge duplicate-date rows instead of dropping their case mass. For a
        # cumulative series the last report on a date wins; for increments they sum.
        if sub["elapsed"].duplicated().any():
            n_dupe += int(sub["elapsed"].duplicated().sum())
            agg = {"cumulative": "max"} if have_cum else {"new_cases": "sum"}
            keep = [c for c in sub.columns if c in agg]
            sub = sub.groupby("elapsed", as_index=False).agg({k: agg[k] for k in keep})
            sub = sub.sort_values("elapsed")
        el = sub["elapsed"].to_numpy(float)
        if have_cum:
            cum = sub["cumulative"].to_numpy(float)
            fixed = np.maximum.accumulate(cum)
            n_monotone += int((fixed != cum).sum())
            cum = fixed
            dC = np.diff(cum)
        elif "new_cases" in sub.columns:
            nc = sub["new_cases"].to_numpy(float)
            dC = nc[1:]
            # cum_first must be the cumulative burden BEFORE the first interval.
            # Rebuilding it as a cumsum from zero would force R(0)=0 everywhere.
            cum = np.concatenate([[np.nan], np.cumsum(nc[1:])])
        else:
            raise ValueError("need a cumulative or new-case column")
        dt_ = np.diff(el)
        keep = dt_ > 0
        if "incidence_rate" in sub.columns and len(sub) == len(el):
            y = sub["incidence_rate"].to_numpy(float)[1:]
        else:
            y = np.where(keep, dC / (N * np.where(keep, dt_, 1.0)), 0.0)
        for k in np.flatnonzero(keep):
            rows.append((i, el[k], el[k + 1], y[k], max(dC[k], 0.0)))
        # The IC anchor must be the first interval that actually OBSERVES
        # transmission. Taking literally the first interval conflates two
        # different things: a governorate with genuinely zero initial stock, and
        # one whose first bulletin happened to report no change because reporting
        # is lumpy. Both give y=0, but only the first is a structural zero. Using
        # the first strictly positive interval removes the spurious ones; a
        # governorate with no positive interval at all is a true zero.
        pos = np.flatnonzero(keep & (y > 0))
        anyk = np.flatnonzero(keep)
        if len(pos):
            k0 = pos[0]
            y_first[i] = float(y[k0])
            first_hold[i] = bool(k0 != anyk[0])
        elif len(anyk):
            k0 = anyk[0]
            y_first[i] = 0.0
            true_zero[i] = True
        else:
            continue
        cum_first[i] = 0.0 if not np.isfinite(cum[k0]) else cum[k0] / N
        t_first[i] = el[k0]

    if n_dupe:
        prov["duplicate_date_rows_merged"] = n_dupe
        if verbose:
            print(f"  merged {n_dupe} duplicate-date row(s)")
    if n_monotone:
        prov["nonmonotone_cum_rows_repaired"] = n_monotone
        if verbose:
            print(f"  repaired {n_monotone} non-monotone cumulative value(s) by "
                  f"running maximum (these would otherwise give negative increments)")
    if first_hold.any():
        prov["ic_anchor_skipped_leading_zero"] = [names[i] for i in
                                                  np.flatnonzero(first_hold)]
        if verbose:
            print("  IC anchored on the first POSITIVE interval for %d governorate(s) "
                  "whose first bulletin reported no change: %s"
                  % (int(first_hold.sum()),
                     ", ".join(prov["ic_anchor_skipped_leading_zero"])))
    if true_zero.any():
        prov["structural_zero_seed"] = [names[i] for i in np.flatnonzero(true_zero)]
        if verbose:
            print("  structural zero seed (no positive interval anywhere): %s"
                  % ", ".join(prov["structural_zero_seed"]))
    if not have_cum:
        prov["cum_first_unavailable"] = True
        warnings.warn("no cumulative column: the removed fraction at t=0 cannot be "
                      "reconstructed and is taken as 0 at every governorate.")

    rows = np.array(rows, float)
    gov = rows[:, 0].astype(int); t0 = rows[:, 1]; t1 = rows[:, 2]
    y_raw = np.maximum(rows[:, 3], 0.0); dC_raw = rows[:, 4]

    cut = float(np.quantile(y_raw, WINSOR_Q))
    if winsorize:
        y = np.minimum(y_raw, cut)
        n_w = int((y_raw > cut).sum())
        dC = np.minimum(dC_raw, cut * pop[gov] * (t1 - t0))
    else:
        y = y_raw.copy(); dC = dC_raw.copy(); n_w = 0

    w_count = 1.0 / np.sqrt(np.maximum(dC, 1.0)); w_count /= w_count.mean()
    w_rate = pop[gov] * (t1 - t0) / np.sqrt(np.maximum(dC, 1.0)); w_rate /= w_rate.mean()

    p = Panel(gov, t0, t1, y, y_raw, dC, dC_raw, w_count.copy(), w_count, w_rate,
              names, xg, pop, y_first, cum_first, t_first, cut, n_w,
              true_zero=true_zero, first_hold=first_hold,
              provenance=prov, synthetic=synthetic)
    return p


# --------------------------------------------------------------------------
# Observation design
# --------------------------------------------------------------------------
def build_interval_operator(grid: seip.Grid, panel: Panel):
    """Sparse (n_obs, (nt+1)*ng) matrix giving the interval AVERAGE of H over each
    reported interval -- what y_{g,k} = dC/(N dt) actually is."""
    from scipy.sparse import csr_matrix
    nt1 = grid.nt + 1; ng = panel.ng
    rows, cols, vals = [], [], []
    for r in range(panel.n_obs):
        k0 = int(np.floor(panel.t0[r] / grid.dt + 1e-9))
        k1 = int(np.ceil(panel.t1[r] / grid.dt - 1e-9))
        k0 = max(0, min(k0, grid.nt)); k1 = max(k0 + 1, min(k1, grid.nt))
        n = k1 - k0 + 1
        wt = np.full(n, grid.dt); wt[0] *= 0.5; wt[-1] *= 0.5
        wt /= wt.sum()
        g = panel.gov[r]
        for j, k in enumerate(range(k0, k1 + 1)):
            rows.append(r); cols.append(k * ng + g); vals.append(wt[j])
    return csr_matrix((vals, (rows, cols)), shape=(panel.n_obs, nt1 * ng))


def build_space_sampler(grid: seip.Grid, panel: Panel):
    """(ng, nx) linear interpolation from the x grid to governorate coordinates."""
    if grid.nx == 1:
        # Single location: without this branch the two interpolation writes land
        # on the same element and the second zeroes the first, giving an all-zero
        # sampler -- a silent failure in which a fit returns its seed with an
        # exactly zero gradient.
        return np.ones((panel.ng, 1))
    W = np.zeros((panel.ng, grid.nx))
    for i, xv in enumerate(panel.xg):
        xv = min(max(xv, grid.x[0]), grid.x[-1])
        j = min(int(np.floor(xv / grid.dx)), grid.nx - 2)
        frac = (xv - grid.x[j]) / grid.dx
        W[i, j] = 1.0 - frac
        W[i, j + 1] = frac
    return W


# --------------------------------------------------------------------------
# Synthetic twin
# --------------------------------------------------------------------------
# Governorate name, great-circle distance from Sana'a (km), population.
SYNTH_GOVS = [
    ("Amanat Al Asimah", 0.0, 2_950_000), ("Sana'a", 24.0, 1_240_000),
    ("Amran", 46.0, 1_090_000),           ("Dhamar", 72.0, 1_900_000),
    ("Al Mahwit", 88.0, 710_000),         ("Raymah", 106.0, 560_000),
    ("Hajjah", 121.0, 2_120_000),         ("Ibb", 137.0, 2_880_000),
    ("Al Hudaydah", 151.0, 3_190_000),    ("Al Jawf", 162.0, 570_000),
    ("Marib", 168.0, 330_000),            ("Al Bayda", 176.0, 740_000),
    ("Sa'ada", 199.0, 980_000),           ("Taizz", 204.0, 3_120_000),
    ("Al Dhale'e", 217.0, 660_000),       ("Lahj", 281.0, 980_000),
    ("Aden", 302.0, 940_000),             ("Abyan", 351.0, 570_000),
    ("Shabwah", 423.0, 650_000),          ("Hadramaut", 601.0, 1_420_000),
    ("Al Maharah", 795.396633, 150_000),
]
# Gaps between consecutive NATIONAL bulletin dates, with the record's weights.
SYNTH_GAPS = np.array([1, 2, 3, 4, 5, 7, 11, 14, 21])
SYNTH_GAP_P = np.array([0.5200, 0.3400, 0.0504, 0.0308, 0.0196, 0.0168, 0.0098, 0.0070, 0.0056])


def synthetic_truth(t, a10=0.030, a20=1.8e-5, tau=40.0, floor=0.49):
    """Declining transmission: R0 about 1.84 at t=0 falling to about 0.89, giving
    an epidemic with the record's shape -- roughly a 14-fold rise over the first
    six weeks, a peak, and a long non-zero tail."""
    shape = floor + (1.0 - floor) * np.exp(-np.asarray(t, float) / tau)
    return a10 * shape, a20 * shape


def make_synthetic_panel(grid: seip.Grid, truth=synthetic_truth, diff=None,
                         seed=11, t_max=seip.T_MAX, scheme="strang",
                         seed_scale=8.0e-5, decay_km=120.0, p_hold=0.18,
                         batch_dates=3, verbose=True, **ic_opts):
    """Generate a panel from a forward solve with a KNOWN transmission pair,
    sampled on the record's own sampling design.

    The cadence is SHARED across governorates, not drawn independently per
    governorate. That is what the record shows: 2,806 rows across only 136
    distinct bulletin days means essentially every governorate reports on
    essentially every national bulletin date. An independent-cadence twin
    produces ~272 distinct days and about a quarter of the record's exact-zero
    fraction, which matters because under the specified weight an exact zero
    carries the MAXIMUM possible weight -- so the zero rows dominate the
    objective, and a twin that under-represents them does not exercise the part
    of the objective being relied on.

    Reporting is also LUMPY, which is where most of the record's zeros come
    from. With probability `p_hold` a governorate reports no change at a
    bulletin and the cases accumulate into a backlog released at its next
    reporting bulletin. On `batch_dates` national dates every governorate
    releases simultaneously, reproducing the synchronized single-day spikes
    the record documents -- so the twin exercises the winsorization rule
    rather than assuming it.
    """
    rng = np.random.default_rng(seed)
    diff = diff or seip.Diffusion()
    names = [q[0] for q in SYNTH_GOVS]
    xg = np.array([q[1] for q in SYNTH_GOVS], float)
    pop = np.array([q[2] for q in SYNTH_GOVS], float)

    seed_rate = seed_scale * np.exp(-xg / decay_km)
    seed_rate[-1] = 0.0          # Al Maharah: zero initial stock, as in the record
    seed_rate[-2] = 0.0          # Hadramaut: likewise
    cum0 = 3.0e-4 * np.exp(-xg / decay_km)

    ic = seip.build_initial_condition(grid, xg, seed_rate, cum0, **ic_opts)
    b10, b20 = truth(grid.t)
    H, Sbar, _ = seip.solve_forward(grid, ic, b10, b20, diff, scheme=scheme)

    Wx = build_space_sampler(grid, type("P", (), {"ng": len(names), "xg": xg})())
    Hg = H @ Wx.T                                   # (nt+1, ng)

    # one national bulletin calendar, shared by every governorate
    dates = [0.0]
    while dates[-1] < t_max - 1:
        dates.append(min(dates[-1] + float(rng.choice(SYNTH_GAPS, p=SYNTH_GAP_P)), t_max))
    dates = np.array(sorted(set(dates)))

    # national dates on which every governorate releases its backlog at once
    forced = set(rng.choice(np.arange(1, len(dates)),
                            size=min(batch_dates, len(dates) - 1),
                            replace=False).tolist())
    recs = []
    for i, name in enumerate(names):
        cum = cum0[i] * pop[i]
        backlog = 0.0
        recs.append((name, dates[0], cum))
        for k in range(len(dates) - 1):
            t0, t1 = dates[k], dates[k + 1]
            k0 = int(round(t0 / grid.dt)); k1 = int(round(t1 / grid.dt))
            lam = max(Hg[k0:k1 + 1, i].mean() * pop[i] * (t1 - t0), 0.0)
            backlog += float(rng.poisson(lam)) if lam < 1e8 else lam
            if (k + 1) in forced or k == len(dates) - 2 or rng.random() >= p_hold:
                cum += backlog
                backlog = 0.0
            recs.append((name, t1, cum))

    df = pd.DataFrame(recs, columns=["governorate", "elapsed", "cumulative"])
    panel = _assemble(df, dict(zip(names, xg)), dict(zip(names, pop)),
                      winsorize=True, synthetic=True,
                      prov=dict(window_start="synthetic", shared_cadence=True))
    truth_d = {"beta10_t": b10, "beta20_t": b20, "diff": diff, "Sbar": Sbar,
               "seed_rate": seed_rate, "cum0": cum0, "scheme": scheme,
               "ic_opts": dict(ic_opts), "p_hold": p_hold,
               "batch_dates": sorted(forced), "decay_km": decay_km}
    if verbose:
        print("synthetic:", panel.describe())
        print("  record for comparison: 2806 rows, 21 governorates, 136 days, "
              "622 exact zeros (22.2%)")
    return panel, truth_d


# --------------------------------------------------------------------------
# Single-city synthetic twin
# --------------------------------------------------------------------------
# Three governorates of the second wave, by whole-epidemic suspected-case count
# (Camacho et al. 2018, Table 1) and by second-wave relative risk.
MAJOR_CITIES = {
    #  name                 x_km        population   cases     why
    "Al Hudaydah":      dict(x=151.0, pop=3_190_000, cases=163_533,
                             note="highest case count in the epidemic"),
    "Amanat Al Asimah": dict(x=0.0,   pop=2_950_000, cases=105_693,
                             note="the capital; singled out early in the second wave"),
    "Hajjah":           dict(x=121.0, pop=2_120_000, cases=123_658,
                             note="2nd highest count; among the top second-wave relative risks"),
    "Taizz":            dict(x=204.0, pop=3_120_000, cases=77_506,
                             note="among the top second-wave relative risks"),
    "Amran":            dict(x=46.0,  pop=1_090_000, cases=104_710,
                             note="highest attack rate after Al Mahwit"),
}


def city_truth(t, a10, a20, tau, floor):
    shape = floor + (1.0 - floor) * np.exp(-np.asarray(t, float) / tau)
    return a10 * shape, a20 * shape


def prefactors_from(R0_0, share):
    """Solve for (beta_10, beta_20) from a reproduction number at t=0 and an
    environmental share. Specifying the truth this way rather than by the two
    prefactors keeps the epidemic's SIZE and its ROUTE SPLIT independent -- the
    two things the comparison has to separate -- and the prefactors themselves
    are not comparable to each other anyway, since their kernels have different
    norms."""
    kap, J1, J2 = seip.ngm_integrals()
    tot = R0_0 / (kap * J1)
    return tot * (1.0 - share), tot * share / (seip.ALPHA0 * J2)


# Three trajectories differing in SIZE, SHAPE and ROUTE SPLIT, so a method that
# does well on one is not simply reproducing a shape common to all three. Each
# is tuned to the record's own profile: a 4-9% attack rate (Camacho et al. report
# 4.75-8.45% for the worst-hit governorates), a peak near day 60-70, and a
# roughly 8-19x rise over the first six weeks.
CITY_TRUTH = {
    "Al Hudaydah":      dict(R0_0=1.63, share=0.63, tau=55.0, floor=0.50),
    "Amanat Al Asimah": dict(R0_0=1.58, share=0.36, tau=48.0, floor=0.52),
    "Hajjah":           dict(R0_0=1.70, share=0.67, tau=55.0, floor=0.48),
    "Taizz":            dict(R0_0=1.55, share=0.50, tau=50.0, floor=0.51),
    "Amran":            dict(R0_0=1.66, share=0.45, tau=52.0, floor=0.50),
}


def city_beta(t, city):
    p = CITY_TRUTH[city]
    a10, a20 = prefactors_from(p["R0_0"], p["share"])
    return city_truth(t, a10, a20, p["tau"], p["floor"])


def make_city_panel(city, grid, seed=7, seed_rate=3.0e-5, cum0=3.0e-4,
                    p_hold=0.0, batch_dates=0, t_max=seip.T_MAX, verbose=True,
                    **ic_opts):
    """Generate one governorate's record from the SINGLE-LOCATION model (D = 0).

    The national twin is generated with diffusion on, so a city model fitted at
    D = 0 cannot reach the generating transmission pair -- the reduction itself
    is misspecified, and any recovery error then mixes architecture error with
    reduction error. Generating each city from the same reduction it will be
    fitted with makes the truth exactly attainable, so a comparison between two
    architectures measures the architectures.

    Sampling follows the record's bulletin calendar and the governorate's real
    population, with Poisson counts.

    p_hold DEFAULTS TO ZERO here, unlike the national twin. Batch reporting --
    a governorate reporting no change for several bulletins and then releasing
    the backlog -- is what produces most of the record's exact zeros, and at the
    national scale it must be reproduced. At a SINGLE governorate it destroys the
    likelihood: measured on this city at p_hold = 0.18, the generating truth
    itself scores a mean Poisson deviance of 1575 per observation, while a
    deterministic fit reaches 9.8 by distorting beta to chase the batch releases,
    returning a reproduction number peaking near 9.9. Any comparison run on such
    a panel measures which method chases reporting artifacts hardest, not which
    recovers transmission. With p_hold = 0 the truth scores 0.91 -- the value a
    well-specified Poisson model should score -- and the problem is well posed.
    Set p_hold > 0 deliberately, as a reporting-artifact sensitivity, not as the
    baseline.
    """
    if city not in MAJOR_CITIES:
        raise KeyError(f"{city!r} not in MAJOR_CITIES: {list(MAJOR_CITIES)}")
    meta = MAJOR_CITIES[city]
    N = float(meta["pop"])
    rng = np.random.default_rng(seed)

    g1 = seip.Grid(dt=grid.dt, nx=1)
    ic = seip.build_initial_condition(g1, np.array([0.0]), np.array([seed_rate]),
                                      np.array([cum0]), **ic_opts)
    b10, b20 = city_beta(g1.t, city)
    H, Sbar, _ = seip.solve_forward(g1, ic, b10, b20, seip.Diffusion(0, 0, 0, 0))
    Hc = H[:, 0]

    dates = [0.0]
    while dates[-1] < t_max - 1:
        dates.append(min(dates[-1] + float(rng.choice(SYNTH_GAPS, p=SYNTH_GAP_P)), t_max))
    dates = np.array(sorted(set(dates)))
    forced = set(rng.choice(np.arange(1, len(dates)),
                            size=min(batch_dates, len(dates) - 1), replace=False).tolist())

    recs = []
    cum = cum0 * N
    backlog = 0.0
    recs.append((city, dates[0], cum))
    for k in range(len(dates) - 1):
        t0, t1 = dates[k], dates[k + 1]
        k0, k1 = int(round(t0 / g1.dt)), int(round(t1 / g1.dt))
        lam = max(Hc[k0:k1 + 1].mean() * N * (t1 - t0), 0.0)
        backlog += float(rng.poisson(lam)) if lam < 1e8 else lam
        if (k + 1) in forced or k == len(dates) - 2 or rng.random() >= p_hold:
            cum += backlog; backlog = 0.0
        recs.append((city, t1, cum))

    df = pd.DataFrame(recs, columns=["governorate", "elapsed", "cumulative"])
    panel = _assemble(df, {city: meta["x"]}, {city: N}, winsorize=True,
                      synthetic=True, prov=dict(window_start="synthetic city",
                                                city=city, reduction="D=0"))
    truth = dict(beta10_t=b10, beta20_t=b20, Sbar=Sbar, H=Hc, grid=g1,
                 seed_rate=seed_rate, cum0=cum0, city=city,
                 params=CITY_TRUTH[city], ic_opts=dict(ic_opts))
    if verbose:
        print(f"{city}: {panel.describe()}")
    return panel, truth