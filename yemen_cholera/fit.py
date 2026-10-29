"""
fit.py -- Transmission-only estimation.

Only beta_10(t) and beta_20(t) are released. Everything else in seip.py is held;
D1..D4 are supplied and swept, never estimated here.

    g_j(t)  = c0 [+ cL t/T] + sum_m [cs_m sin(2 pi m t/T) + cc_m cos(2 pi m t/T)]
    beta_j0 = softplus(g_j) + eps

M is bounded above by the sampling: a harmonic of index m has period T/m and
cannot be resolved from samples more than half its period apart, so
M <= T / (2 * max interval).

DECLARED DEVIATIONS from the written methodology, all deliberate:
  * the smoothness penalty is applied to the RELATIVE derivative by default
    (smooth_scale="relative"). The absolute form (1/T)int[b10'^2 + b20'^2] is
    scale dependent, and with b10 ~ 1e-2 against b20 ~ 1e-5 it regularizes b10
    about a million times harder in relative terms. smooth_scale="absolute"
    gives the specified form.
  * the ridge spares the constant and trend coefficients and penalizes only the
    harmonics, so it shrinks structure rather than level.
  * age quadrature is composite trapezoid on the solver grid rather than
    Gauss-Legendre split at h's breakpoints. Benign at dt <= 0.25 -- both
    breakpoints land on nodes and h is continuous at both -- and
    seip.ngm_integrals(grid, report=True) quantifies it.
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import least_squares

import seip
import data_io

EPS_BETA = 1e-12
FLOOR_TOL = 1e-9          # below this beta is on the positivity floor


def softplus(z):
    return np.logaddexp(0.0, z)


def inv_softplus(v):
    v = np.asarray(v, float)
    return np.where(v > 20.0, v, np.log(np.expm1(np.minimum(v, 20.0))))


def max_fourier_order(panel: data_io.Panel, T=seip.T_MAX):
    """Nyquist ceiling on M from the worst-served governorate and from the pooled
    bulletin calendar."""
    gap_local = float((panel.t1 - panel.t0).max())
    days = np.unique(np.round(np.concatenate([panel.t0, panel.t1]), 6))
    gap_pooled = float(np.diff(days).max()) if len(days) > 1 else gap_local
    return T / (2.0 * gap_local), T / (2.0 * gap_pooled)


@dataclass
class BetaModel:
    """M=0 with trend=True is a constant PLUS a linear trend. trend=False, M=0 is
    the strict constant."""
    M: int = 0
    T: float = seip.T_MAX
    trend: bool = True

    @property
    def nper(self):
        return 1 + int(self.trend) + 2 * self.M

    @property
    def nparam(self):
        return 2 * self.nper

    def names(self):
        base = ["const"] + (["trend"] if self.trend else [])
        for m in range(1, self.M + 1):
            base += ["sin%d" % m, "cos%d" % m]
        return ["b10:" + n for n in base] + ["b20:" + n for n in base]

    def design(self, t):
        t = np.asarray(t, float)
        cols = [np.ones_like(t)]
        if self.trend:
            cols.append(t / self.T)
        for m in range(1, self.M + 1):
            cols.append(np.sin(2 * np.pi * m * t / self.T))
            cols.append(np.cos(2 * np.pi * m * t / self.T))
        return np.column_stack(cols)

    def eval(self, theta, t, X=None):
        X = self.design(t) if X is None else X
        return (softplus(X @ theta[:self.nper]) + EPS_BETA,
                softplus(X @ theta[self.nper:]) + EPS_BETA)

    def start(self, b10, b20):
        th = np.zeros(self.nparam)
        th[0] = float(inv_softplus(b10))
        th[self.nper] = float(inv_softplus(b20))
        return th

    def harmonic_mask(self):
        k = 1 + int(self.trend)
        m = np.zeros(self.nparam, bool)
        m[k:self.nper] = True
        m[self.nper + k:] = True
        return m


@dataclass
class FitResult:
    label: str
    diff: seip.Diffusion
    theta: np.ndarray
    beta10_t: np.ndarray
    beta20_t: np.ndarray
    R0_t: np.ndarray
    Sbar: np.ndarray
    H: np.ndarray
    Hg: np.ndarray
    yhat: np.ndarray
    resid_w: np.ndarray
    wrmse: float
    rmse: float
    rel_mae: float
    nfev: int
    success: bool
    status: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def converged(self):
        """status > 0 means a real termination criterion fired. status == 0 is
        'max_nfev reached', which is not convergence and must not be tabulated or
        plotted as if it were."""
        return bool(self.success) and self.status > 0

    @property
    def saturation(self):
        """Fraction of the window on which each prefactor sits on the positivity
        floor. Anything above a few per cent means the reported curve is the
        constraint, not an estimate, and the Jacobian column there is ~0 -- so
        success=True says nothing about that parameter."""
        return self.meta["sat_b10"], self.meta["sat_b20"]


class Problem:
    """Everything fixed across a fit: grid, panel, IC, operators, normalization."""

    def __init__(self, grid: seip.Grid, panel: data_io.Panel,
                 truncation_correction=False, age_profile="flat",
                 p_from_model=False, row_mask=None, scheme="strang",
                 observable=seip.OBS_FORCE_OF_INFECTION, yscale=None,
                 objective="wls"):
        """row_mask selects which rows enter the DATA TERM. The PDE still runs
        over the whole domain, so a per-governorate fit under live diffusion is a
        different object from one in isolation.

        yscale may be supplied so several Problems sharing a panel normalize
        identically. Without it a masked Problem puts its objective -- and any
        gradient taken from it -- in its own units, which makes cross-location
        comparison meaningless.
        """
        self.grid = grid
        self.panel = panel
        self.scheme = scheme
        self.observable = observable
        self.row_mask = (np.ones(panel.n_obs, bool) if row_mask is None
                         else np.asarray(row_mask, bool))
        self.ic = seip.build_initial_condition(
            grid, panel.xg, panel.y_first, panel.cum_first,
            truncation_correction=truncation_correction,
            age_profile=age_profile, p_from_model=p_from_model)
        self.Wx = data_io.build_space_sampler(grid, panel)
        self.A = data_io.build_interval_operator(grid, panel)
        # Exact next-generation integrals, not this grid's trapezoid values: they
        # define R0 and the published constants are the exact ones.
        self.kappa, self.J1, self.J2 = seip.ngm_integrals()
        m = self.row_mask
        self.yscale = (float(np.sqrt(np.mean((panel.w[m] * panel.y[m]) ** 2)))
                       if yscale is None else float(yscale))
        # With yscale taken from this same mask, sse_null is identically m.sum():
        # a row count, not a measured quantity. sse_floor is the informative one.
        self.sse_null = float(np.sum((panel.w[m] * panel.y[m] / self.yscale) ** 2))
        var_y = panel.dC[m] / np.maximum(
            (panel.pop[panel.gov[m]] * (panel.t1[m] - panel.t0[m])) ** 2, 1e-300)
        self.sse_floor = float(np.sum(panel.w[m] ** 2 * var_y) / self.yscale ** 2)
        self.objective = objective
        # Scale the deviance so the two objectives sit on comparable footing:
        # sqrt(n) makes a well-specified Poisson fit land near 1 per residual.
        self.dev_scale = float(np.sqrt(max(int(m.sum()), 1)))
        self.ic_opts = dict(truncation_correction=truncation_correction,
                            age_profile=age_profile, p_from_model=p_from_model)

    def predict(self, beta10_t, beta20_t, diff: seip.Diffusion):
        H, Sbar, _ = seip.solve_forward(self.grid, self.ic, beta10_t, beta20_t,
                                        diff, scheme=self.scheme,
                                        observable=self.observable)
        Hg = H @ self.Wx.T
        return H, Hg, self.A @ Hg.ravel(), Sbar

    def data_residual(self, yhat):
        """objective='wls'      w (yhat - y) / yscale, the specified form.
        objective='poisson'  the Poisson DEVIANCE residual on COUNTS,

            d = sign(c - lam) sqrt( 2 [ c log(c/lam) - (c - lam) ] ),
            lam = yhat N dt,   with c log(c/lam) := 0 at c = 0 so d = sqrt(2 lam).

        The deviance is the right object here and weighted least squares is not.
        1/sqrt(max(c,1)) estimates the variance from the realization, so a row
        that fluctuated low is up-weighted and an exact zero maximally; on a panel
        with the record's 22% exact-zero fraction the zero rows carry ~82% of the
        objective, and what 82% of the objective asks for is 'predict zero'. The
        deviance has no such pathology: a zero row contributes 2*lam, which
        penalizes over-prediction without rewarding collapse.
        """
        m = self.row_mask
        if self.objective == "wls":
            return self.panel.w[m] * (yhat[m] - self.panel.y[m]) / self.yscale
        Ndt = self.panel.pop[self.panel.gov[m]] * (self.panel.t1[m] - self.panel.t0[m])
        lam = np.maximum(np.asarray(yhat)[m] * Ndt, 1e-10)
        c = self.panel.dC[m]
        term = np.where(c > 0, c * np.log(np.maximum(c, 1e-300) / lam), 0.0)
        d2 = 2.0 * (term - (c - lam))
        return np.sign(c - lam) * np.sqrt(np.maximum(d2, 0.0)) / self.dev_scale

    def residuals(self, theta, bm, diff, X, lam_smooth, lam_ridge,
                  smooth_scale="relative"):
        b10, b20 = bm.eval(theta, self.grid.t, X)
        _, _, yhat, _ = self.predict(b10, b20, diff)
        parts = [self.data_residual(yhat)]
        if lam_smooth > 0 and bm.nper > 1:
            dt = self.grid.dt
            d10, d20 = np.diff(b10) / dt, np.diff(b20) / dt
            if smooth_scale == "relative":
                d10 = d10 / max(b10.mean(), 1e-300)
                d20 = d20 / max(b20.mean(), 1e-300)
            s = np.sqrt(lam_smooth * dt / seip.T_MAX)
            parts += [s * d10, s * d20]
        if lam_ridge > 0 and bm.M > 0:
            parts.append(np.sqrt(lam_ridge) * theta[bm.harmonic_mask()])
        return np.concatenate(parts)

    def penalty_share(self, theta, bm, diff, X, lam_smooth, lam_ridge, smooth_scale):
        """Fraction of the objective contributed by the penalties. The
        methodology calls the smoothness term 'a prior with content'; if this is
        1e-5 the prior has no content at these settings and should be reported as
        inert rather than as doing work."""
        nd = int(self.row_mask.sum())
        r = self.residuals(theta, bm, diff, X, lam_smooth, lam_ridge, smooth_scale)
        tot = float(r @ r)
        return float(r[nd:] @ r[nd:]) / max(tot, 1e-300)


class _Residual:
    """Module level so it pickles for scipy's parallel Jacobian."""

    def __init__(self, prob, bm, diff, X, lam_smooth, lam_ridge, smooth_scale):
        self.prob, self.bm, self.diff, self.X = prob, bm, diff, X
        self.lam_smooth, self.lam_ridge = lam_smooth, lam_ridge
        self.smooth_scale = smooth_scale

    def __call__(self, theta):
        return self.prob.residuals(theta, self.bm, self.diff, self.X,
                                   self.lam_smooth, self.lam_ridge,
                                   self.smooth_scale)


# --------------------------------------------------------------------------
# Coarse search: seeds the optimizer AND is the route-split diagnostic
# --------------------------------------------------------------------------
def grid_search(prob: Problem, diff: seip.Diffusion,
                b10_range=(1e-5, 3e-1), b20_range=(1e-7, 1e-3), n=13, verbose=True):
    """Returns (b10s, b20s, sse, best). Warns on a boundary optimum: that means
    the box chose the seed, not the data."""
    b10s = np.geomspace(*b10_range, n)
    b20s = np.geomspace(*b20_range, n)
    sse = np.full((n, n), np.nan)
    nt1 = prob.grid.nt + 1
    m = prob.row_mask
    for i, a in enumerate(b10s):
        for j, b in enumerate(b20s):
            _, _, yh, _ = prob.predict(np.full(nt1, a), np.full(nt1, b), diff)
            r = prob.data_residual(yh)          # honours prob.objective
            sse[i, j] = float(r @ r)
    k = np.unravel_index(np.nanargmin(sse), sse.shape)
    edge = [nm for nm, idx in (("beta10", k[0]), ("beta20", k[1]))
            if idx in (0, n - 1)]
    if verbose:
        print("  grid search best: beta10=%.4g beta20=%.4g SSE=%.6g"
              % (b10s[k[0]], b20s[k[1]], sse[k]))
        if edge:
            print("    WARNING: optimum on the box edge in %s -- widen the range; "
                  "the seed is set by the box, not the data" % ", ".join(edge))
    return b10s, b20s, sse, (b10s[k[0]], b20s[k[1]])


# --------------------------------------------------------------------------
# Uncertainty from the Jacobian at the solution
# --------------------------------------------------------------------------
def parameter_uncertainty(sol, bm: BetaModel, n_data: int):
    """J^T J at the solution IS the Gauss-Newton Hessian the methodology asks
    for. scipy returns it for free; earlier versions discarded it.

    The standard errors assume independent, correctly scaled residuals, which the
    error surface shows is false, so they are a LOWER bound on the true spread.
    The eigenvalue structure -- in particular the near-null direction -- is the
    part that does not depend on that assumption.
    """
    J = np.asarray(sol.jac, float)
    H = J.T @ J
    ev, evec = np.linalg.eigh(H)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    dof = max(n_data - bm.nparam, 1)
    s2 = 2.0 * float(sol.cost) / dof
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = s2 * np.linalg.pinv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    cond = float(ev[0] / ev[-1]) if ev[-1] > 0 else np.inf
    return dict(eigenvalues=ev, eigenvectors=evec, cond=cond, se=se, sigma2=s2,
                names=bm.names(), null_direction=evec[:, -1],
                null_eigenvalue=float(ev[-1]))


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------
def fit(prob: Problem, diff: seip.Diffusion, M=0, lam_smooth=1e-2, lam_ridge=1e-4,
        seed_beta=None, label=None, smooth_scale="relative", max_nfev=400,
        trend=True, n_jobs=1, irls=0, verbose=True):
    """irls > 0 runs that many extra passes with MODEL-based weights
    1/sqrt(max(yhat N dt, 1)) rebuilt from the previous fit.

    Why it matters: the specified weight 1/sqrt(observed count) uses the
    realization as its own variance estimate, so a row that fluctuated low is
    up-weighted and an exact zero maximally so. On a panel matching the record's
    22% exact-zero fraction the zero rows carry about 82% of the objective. That
    is the downward bias of Poisson-weighted least squares, and it pushes in the
    SAME direction as the initial-condition truncation mismatch.
    """
    bm = BetaModel(M=M, trend=trend)
    X = bm.design(prob.grid.t)
    if seed_beta is None:
        _, _, _, seed_beta = grid_search(prob, diff, verbose=verbose)
    theta0 = bm.start(*seed_beta)

    kw = {}
    if n_jobs and n_jobs != 1:
        kw["workers"] = (os.cpu_count() if n_jobs < 0 else n_jobs)

    passes = [prob.panel.weight_kind]
    sol = None
    for pi in range(1 + max(0, irls)):
        fun = _Residual(prob, bm, diff, X, lam_smooth, lam_ridge, smooth_scale)
        sol = least_squares(fun, theta0, method="trf", x_scale="jac",
                            diff_step=1e-4, max_nfev=max_nfev, verbose=0, **kw)
        theta0 = sol.x
        if pi < max(0, irls):
            b10, b20 = bm.eval(sol.x, prob.grid.t, X)
            _, _, yh, _ = prob.predict(b10, b20, diff)
            prob.panel.set_weights("model", yhat=np.maximum(yh, 0.0))
            passes.append("model")
            m = prob.row_mask
            prob.yscale = float(np.sqrt(np.mean(
                (prob.panel.w[m] * prob.panel.y[m]) ** 2)))

    b10, b20 = bm.eval(sol.x, prob.grid.t, X)
    H, Hg, yhat, Sbar = prob.predict(b10, b20, diff)
    R0 = seip.R0_of_t(b10, b20, Sbar, prob.kappa, prob.J1, prob.J2)
    share, ratio, share_t = seip.route_share(b10, b20, prob.J2, prob.grid)

    m = prob.row_mask
    w, y, yh = prob.panel.w[m], prob.panel.y[m], yhat[m]
    wrmse = float(np.sqrt(np.mean((w * (yh - y)) ** 2)))
    rmse = float(np.sqrt(np.mean((yh - y) ** 2)))
    denom = np.maximum(y, np.quantile(y[y > 0], 0.10) if (y > 0).any() else 1.0)
    rel_mae = float(np.mean(np.abs(yh - y) / denom))
    rd = prob.data_residual(yhat)
    sse = float(rd @ rd)
    # 'explained' is built from the WLS null and Poisson-noise-floor sums of
    # squares. Those are not commensurate with a deviance, so it is reported only
    # under the WLS objective and left undefined otherwise rather than printed as
    # a large negative number.
    if prob.objective == "wls":
        rn = prob.panel.w[m] * prob.panel.y[m] / prob.yscale
        sse_wls = float(np.sum((prob.panel.w[m] * (yh - y) / prob.yscale) ** 2))
        span = prob.sse_null - prob.sse_floor
        explained = ((prob.sse_null - sse_wls) / span
                     if span > 1e-9 * max(prob.sse_null, 1e-300) else float("nan"))
    else:
        explained = float("nan")

    res = FitResult(
        label=label or diff.label(), diff=diff, theta=sol.x,
        beta10_t=b10, beta20_t=b20, R0_t=R0, Sbar=Sbar, H=H, Hg=Hg, yhat=yhat,
        resid_w=w * (yh - y), wrmse=wrmse, rmse=rmse, rel_mae=rel_mae,
        nfev=int(sol.nfev), success=bool(sol.success), status=int(sol.status),
        meta=dict(M=M, trend=trend, explained=explained, sse=sse,
                  sse_null=prob.sse_null, sse_floor=prob.sse_floor,
                  env_share=share, route_ratio=ratio, env_share_t=share_t,
                  sat_b10=float(np.mean(b10 <= FLOOR_TOL)),
                  sat_b20=float(np.mean(b20 <= FLOOR_TOL)),
                  weight_kind=prob.panel.weight_kind, weight_passes=passes,
                  lam_smooth=lam_smooth, lam_ridge=lam_ridge, seed=seed_beta,
                  cost=float(sol.cost), scheme=prob.scheme,
                  observable=prob.observable, objective=prob.objective,
                  penalty_share=prob.penalty_share(sol.x, bm, diff, X, lam_smooth,
                                                   lam_ridge, smooth_scale),
                  uncertainty=parameter_uncertainty(sol, bm, int(m.sum())),
                  **prob.ic_opts))
    if verbose:
        flag = "" if res.converged else "   [NOT CONVERGED: max_nfev]"
        sat = res.meta["sat_b10"]
        satf = ("   [b10 on the floor %.0f%% of the window]" % (100 * sat)
                if sat > 0.02 else "")
        print("  [%s] M=%d%s wRMSE=%.4g relMAE=%.3f expl=%.3f envshare=%.3f "
              "R0 %.2f-%.2f cond=%.1e nfev=%d%s%s"
              % (res.label, M, "+trend" if trend else "", wrmse, rel_mae,
                 explained, share, R0.min(), R0.max(),
                 res.meta["uncertainty"]["cond"], sol.nfev, flag, satf))
    return res


def sweep_diffusion(prob: Problem, diff_list, M=0, seed_beta=None, verbose=True, **kw):
    """Refit the transmission pair under each diffusion setting. D is held and
    swept, never estimated, as the staged argument requires."""
    out = []
    for d in diff_list:
        if verbose:
            print("fitting %s ..." % d.label())
        out.append(fit(prob, d, M=M, seed_beta=seed_beta, label=d.label(),
                       verbose=verbose, **kw))
    return out


# --------------------------------------------------------------------------
# Held-out validation
# --------------------------------------------------------------------------
def blocked_time_folds(panel: data_io.Panel, k=5):
    """Contiguous blocks of bulletin time. A random row split would leak:
    neighbouring intervals at one governorate are nearly the same observation."""
    edges = np.quantile(panel.t1, np.linspace(0, 1, k + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    return [(panel.t1 > edges[i]) & (panel.t1 <= edges[i + 1]) for i in range(k)]


PROB_KEYS = ("truncation_correction", "age_profile", "p_from_model",
             "scheme", "observable", "objective")


def blocked_validation(grid, panel, diff, M=0, k=5, seed_beta=None, verbose=True,
                       **kw):
    """Fit on all but one contiguous time block; score on the held-out block."""
    rows = []
    for i, test in enumerate(blocked_time_folds(panel, k)):
        train = ~test
        prob = Problem(grid, panel, row_mask=train,
                       **{z: kw[z] for z in PROB_KEYS if z in kw})
        fk = {z: v for z, v in kw.items() if z not in PROB_KEYS}
        r = fit(prob, diff, M=M, seed_beta=seed_beta, label="fold%d" % i,
                verbose=False, **fk)
        w, y, yh = panel.w[test], panel.y[test], r.yhat[test]
        rows.append(dict(fold=i, n_train=int(train.sum()), n_test=int(test.sum()),
                         t_lo=float(panel.t1[test].min()),
                         t_hi=float(panel.t1[test].max()),
                         train_wrmse=r.wrmse,
                         test_wrmse=float(np.sqrt(np.mean((w * (yh - y)) ** 2))),
                         R0_mean=float(r.R0_t.mean()),
                         beta10_mean=float(r.beta10_t.mean()),
                         env_share=r.meta["env_share"], converged=r.converged))
        if verbose:
            q = rows[-1]
            print("  fold %d: test days %.0f-%.0f  train wRMSE %.3g  "
                  "test wRMSE %.3g  R0bar %.3f"
                  % (i, q["t_lo"], q["t_hi"], q["train_wrmse"], q["test_wrmse"],
                     q["R0_mean"]))
    return rows


def multistart(prob, diff, seeds, M=0, verbose=True, **kw):
    """Refit from several seeds. The objective is non-convex and the route-split
    direction is near flat, so the SPREAD across seeds is a reported quantity,
    not a diagnostic to pass."""
    out = []
    for i, sb in enumerate(seeds):
        r = fit(prob, diff, M=M, seed_beta=sb, label="seed%d" % i,
                verbose=False, **kw)
        out.append(r)
        if verbose:
            print("  seed %d from (%.3g, %.3g): b10bar=%.4g b20bar=%.4g "
                  "R0bar=%.4f wRMSE=%.4g%s"
                  % (i, sb[0], sb[1], r.beta10_t.mean(), r.beta20_t.mean(),
                     r.R0_t.mean(), r.wrmse,
                     "" if r.converged else "  [not converged]"))
    return out


# --------------------------------------------------------------------------
# Fields for plotting
# --------------------------------------------------------------------------
def observed_field(panel: data_io.Panel, grid: seip.Grid):
    """(nt+1, ng) observed per-capita incidence, NaN where no bulletin interval
    covers that time. Each interval is drawn across its own span -- exactly what
    a bulletin states. Nothing is interpolated between bulletins; gaps occur only
    before a governorate's first report."""
    Y = np.full((grid.nt + 1, panel.ng), np.nan)
    for r in range(panel.n_obs):
        k0 = int(np.floor(panel.t0[r] / grid.dt + 1e-9))
        k1 = int(np.ceil(panel.t1[r] / grid.dt - 1e-9))
        Y[k0:k1 + 1, panel.gov[r]] = panel.y[r]
    return Y


def model_interval_field(panel: data_io.Panel, grid: seip.Grid, res: FitResult):
    """The model on the SAME footing as the observations: each reported interval
    filled with the model's interval AVERAGE, which is the quantity actually
    minimized. Comparing the instantaneous operator against a 21-day interval
    average compares two different objects."""
    Yh = np.full((grid.nt + 1, panel.ng), np.nan)
    for r in range(panel.n_obs):
        k0 = int(np.floor(panel.t0[r] / grid.dt + 1e-9))
        k1 = int(np.ceil(panel.t1[r] / grid.dt - 1e-9))
        Yh[k0:k1 + 1, panel.gov[r]] = res.yhat[r]
    return Yh


def national_curve(panel: data_io.Panel):
    """National daily incidence from the ROWS, not from a filled field: for each
    bulletin interval, total cases divided by total population and interval
    length. Returns (t_mid, cases_per_day, rate). No gap filling anywhere."""
    order = np.argsort(0.5 * (panel.t0 + panel.t1))
    tm = 0.5 * (panel.t0 + panel.t1)[order]
    cases = panel.dC[order]
    dt_ = (panel.t1 - panel.t0)[order]
    uniq = np.unique(np.round(tm, 6))
    cpd = np.array([cases[np.round(tm, 6) == u].sum()
                    / max(dt_[np.round(tm, 6) == u].mean(), 1e-9) for u in uniq])
    rate = cpd / panel.pop.sum()
    return uniq, cpd, rate