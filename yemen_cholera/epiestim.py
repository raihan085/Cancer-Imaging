"""
epiestim.py -- model-free reference reproduction number (Cori et al. 2013).

The outside reference for R(t). It uses nothing from the SEIP model: only the
reported case counts and a serial-interval distribution taken from the
literature, which is exactly why it is a useful check on the model-based R(t).

Bulletin counts arrive as interval totals over 1-21 day gaps. They are spread
uniformly over the days of each interval to give a daily series. That is the
least-assumption disaggregation available; it smooths within an interval and so
damps, but does not shift, R(t) on the scale of the 7-day estimation window.

    I_t ~ Poisson(R_t * Lambda_t),  Lambda_t = sum_s w_s I_{t-s}
    R_t | data ~ Gamma(a + sum_window I,  1 / (1/b + sum_window Lambda))

with the EpiEstim default prior a=1, b=5 (mean 5, sd 5).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# Serial-interval scenarios. The primary one matches the shifted-gamma, mean
# 5-day serial interval used for the 2017 Yemen record; the alternative is the
# 3.7-day estimate reconstructed from the 1853 Danish epidemics.
SI_PRIMARY = dict(mean=5.0, sd=2.5, label="SI mean 5.0 d (Yemen 2017 analyses)")
SI_SHORT = dict(mean=3.7, sd=1.8, label="SI mean 3.7 d (Danish 1853 epidemics)")


def daily_counts(t0, t1, cases, t_max):
    """Spread each interval's count evenly over its days -> (days, counts)."""
    n = int(np.ceil(t_max)) + 1
    out = np.zeros(n)
    for a, b, c in zip(t0, t1, cases):
        a, b = int(round(a)), int(round(b))
        if b <= a:
            continue
        out[a + 1:b + 1] += c / (b - a)       # cases reported in (a, b]
    return np.arange(n, dtype=float), out


def si_weights(mean, sd, n):
    """Discretised gamma serial interval on days 1..n-1 (w_0 = 0)."""
    k = (mean / sd) ** 2; th = sd ** 2 / mean
    cdf = stats.gamma.cdf(np.arange(n + 1) - 0.5, k, scale=th)
    w = np.diff(cdf)[:n]
    w[0] = 0.0
    return w / w.sum()


def cori(days, inc, si=SI_PRIMARY, window=7, a_prior=1.0, b_prior=5.0,
         start_day=None):
    """Posterior mean and 95% CrI of R_t over trailing windows of `window` days.

    Returns dict(t, mean, lo, hi, n_cases). Windows with fewer than 12 cases in
    total are returned as NaN: the posterior there is the prior, not the data.
    """
    n = len(inc)
    w = si_weights(si["mean"], si["sd"], n)
    lam = np.array([np.dot(w[1:k + 1], inc[k - 1::-1][:k]) if k > 0 else 0.0
                    for k in range(n)])
    first = int(np.flatnonzero(inc > 0)[0]) if (inc > 0).any() else 0
    t0 = max(first + window, int(start_day or 0), window)
    T, M, L, H, NC = [], [], [], [], []
    for e in range(t0, n):
        s = slice(e - window + 1, e + 1)
        shape = a_prior + inc[s].sum()
        rate = 1.0 / b_prior + lam[s].sum()
        T.append(days[e])
        NC.append(inc[s].sum())
        if inc[s].sum() < 12 or lam[s].sum() <= 0:
            M.append(np.nan); L.append(np.nan); H.append(np.nan)
            continue
        g = stats.gamma(shape, scale=1.0 / rate)
        M.append(g.mean()); L.append(g.ppf(0.025)); H.append(g.ppf(0.975))
    return dict(t=np.array(T), mean=np.array(M), lo=np.array(L), hi=np.array(H),
                n_cases=np.array(NC), si=si, window=window)


def cori_for_panel(panel, gi=None, t_max=272.0, **kw):
    """Cori R(t) for one governorate (gi) or the national total (gi=None)."""
    m = np.ones(panel.n_obs, bool) if gi is None else panel.gov == gi
    days, inc = daily_counts(panel.t0[m], panel.t1[m], panel.dC_raw[m], t_max)
    return cori(days, inc, **kw), (days, inc)
