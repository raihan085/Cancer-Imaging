"""
seip.py -- Age-structured SEIP transport-diffusion model for the Yemen
second-wave cholera record.

    dS/dt            = D1 S_xx - S[int b1(t,a) I da + int b2(t,b) P db] + Lam - mu1 S
    dE/dt + dE/da    = D2 E_xx - (mu2 + sigma(a)) E
    dI/dt + dI/da    = D3 I_xx - (mu3 + gamma + delta) I
    dP/dt + dP/db    = D4 P_xx - mu4 P

    E(t,0,x) = S(t,x) [ int b1 I da + int b2 P db ]      <- the unknowns enter HERE
    I(t,0,x) = int sigma(a) E(t,a,x) da
    P(t,0,x) = int alpha(a) I(t,a,x) da

Homogeneous Neumann in x at both ends; no terminal condition in a or b -- the age
derivatives are outward transport, so a cohort reaching the truncation age leaves
the tracked population.

NUMERICS.  da = db = dt, so age advection is the exact characteristic shift and
introduces no numerical diffusion in age.  The time integrator is Strang
splitting about the reaction-transport operator,

    exp(dt/2 A) exp(dt B) exp(dt/2 A),      A = diffusion,  B = reaction+transport,

with the renewal inflows evaluated at the midpoint by a predictor-corrector, so
the scheme is second order in dt.  A first-order Lie variant is retained under
`scheme="lie"` because the earlier results in this project were produced with it
and a like-for-like comparison needs it.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Fixed constants. None of these are ever fitted.
# --------------------------------------------------------------------------
MU1 = MU2 = MU3 = 1.0 / (70.0 * 365.0)   # background mortality, /day
LAMBDA_REC = MU1                          # recruitment, population fraction /day
SIGMA0 = 1.0 / 1.4                        # latency progression, /day
GAMMA = 1.0 / 7.0                         # recovery, /day
DELTA = 0.001                             # disease-induced mortality, /day
MU4 = 1.0 / 30.0                          # environmental pathogen decay, /day
ALPHA0 = 2.0                              # shedding amplitude
A_MAX = 7.0                               # infection-age truncation, days
B_MAX = 30.0                              # pathogen-age truncation, days
X_MAX_FULL = 1051.806                     # full domain, km (Socotra)
X_MAX_OBSERVED = 795.396633               # farthest REPORTING governorate (Al Maharah)
T_MAX = 272.0                             # window, days (22 May 2017 - 18 Feb 2018)

R_I = MU3 + GAMMA + DELTA                 # infectious removal rate, /day
R_E = MU2 + SIGMA0                        # exposed removal rate, /day

# Residence times.  TRUNC is the occupancy the model actually holds at constant
# incidence; NAIVE is what the untruncated residence-time identity assumes.
#
# These ratios are NOT both 1/(1-e^-1).  mu4*B_MAX = 1 exactly, so the
# environmental ratio is 1.581977.  R_I*A_MAX = 1.0072740, because gamma = 1/7 is
# joined by delta and mu3, so the infectious ratio is 1.575332.  Earlier project
# text gave 1.58 for both and "about 58% more mass"; the infectious figure is
# 57.5%.
TAU_I_TRUNC = (1.0 - np.exp(-R_I * A_MAX)) / R_I      # 4.411419 d
TAU_I_NAIVE = 1.0 / R_I                               # 6.949450 d
TAU_P_TRUNC = (1.0 - np.exp(-MU4 * B_MAX)) / MU4      # 18.963617 d
TAU_P_NAIVE = 1.0 / MU4                               # 30.000000 d
RATIO_I = TAU_I_NAIVE / TAU_I_TRUNC                   # 1.575332
RATIO_P = TAU_P_NAIVE / TAU_P_TRUNC                   # 1.581977

# Exact kernel integrals (closed form, cross-checked against adaptive quadrature).
# int_0^7 a(7-a) e^{-a/12} da.  An earlier version of this file quoted 43.012,
# which was this module's own trapezoid value at dt=0.25, not the integral.
INT_F_DA = 43.0688907175
INT_H_DB = 1078.5


def f_kernel(a):
    """Infection-age transmission/shedding shape. Zero at 0 and A_MAX, peak at a=3."""
    a = np.asarray(a, dtype=float)
    return a * (A_MAX - a) * np.exp(-a / 12.0)


def h_kernel(b):
    """Pathogen-age infectivity: 700 on [0,1], 1399-699b on (1,2], 1 on (2,30]."""
    b = np.asarray(b, dtype=float)
    out = np.ones_like(b)
    out = np.where(b <= 1.0, 700.0, out)
    mid = (b > 1.0) & (b <= 2.0)
    out = np.where(mid, 1399.0 - 699.0 * b, out)
    return out


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
@dataclass
class Grid:
    """da = db = dt is required, not a convenience: it makes the age shift exact.

    Quadrature in a and b is composite trapezoid on this same grid, which is a
    documented substitution for the Gauss-Legendre rule the methodology specifies.
    It is benign here -- at dt <= 0.25 both breakpoints of h land on grid nodes and
    h is continuous at both -- and the convergence of the next-generation
    integrals is reported by ngm_integrals(..., report=True).
    """
    dt: float = 0.25
    nx: int = 41
    x_max: float = X_MAX_FULL

    na: int = field(init=False); nb: int = field(init=False); nt: int = field(init=False)
    a: np.ndarray = field(init=False); b: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False); t: np.ndarray = field(init=False)
    dx: float = field(init=False)
    wa: np.ndarray = field(init=False); wb: np.ndarray = field(init=False)
    wf: np.ndarray = field(init=False); wh: np.ndarray = field(init=False)
    wsig: np.ndarray = field(init=False); walpha: np.ndarray = field(init=False)

    def __post_init__(self):
        for name, L in (("A_MAX", A_MAX), ("B_MAX", B_MAX), ("T_MAX", T_MAX)):
            n = L / self.dt
            if abs(n - round(n)) > 1e-9:
                raise ValueError(f"dt={self.dt} does not divide {name}={L}; "
                                 "da=db=dt is required for the exact age shift")
        self.na = int(round(A_MAX / self.dt))
        self.nb = int(round(B_MAX / self.dt))
        self.nt = int(round(T_MAX / self.dt))
        self.a = np.linspace(0.0, A_MAX, self.na + 1)
        self.b = np.linspace(0.0, B_MAX, self.nb + 1)
        self.t = np.linspace(0.0, T_MAX, self.nt + 1)
        if self.nx == 1:
            # Single location: the spatial operator is switched off entirely, so
            # dx is never used. Only legal with D1=..=D4=0.
            self.x = np.array([0.0]); self.dx = np.inf
        else:
            self.x = np.linspace(0.0, self.x_max, self.nx)
            self.dx = self.x[1] - self.x[0]

        self.wa = np.full(self.na + 1, self.dt); self.wa[0] *= 0.5; self.wa[-1] *= 0.5
        self.wb = np.full(self.nb + 1, self.dt); self.wb[0] *= 0.5; self.wb[-1] *= 0.5
        self.wf = self.wa * f_kernel(self.a)
        self.wh = self.wb * h_kernel(self.b)
        self.wsig = self.wa * SIGMA0
        self.walpha = self.wa * ALPHA0 * f_kernel(self.a)

    def diffusion_number(self, diff):
        """D dt / dx^2, reported for orientation only. The diffusion sub-flow is
        applied by its exact spectral propagator, so there is NO CFL restriction
        and this number does not bound the step."""
        if self.nx == 1:
            return 0.0
        return max(diff.as_tuple()) * self.dt / self.dx ** 2


def ngm_integrals(grid: Grid = None, report=False):
    """kappa_EI, J1, J2.  Published: 0.993209, 27.477214, 1040.759087.

    With grid=None these are computed by adaptive quadrature and are exact to
    ~1e-10. Passing a Grid returns that grid's trapezoid values instead, which is
    what a fit on that grid actually uses.
    """
    if grid is None:
        from scipy.integrate import quad
        kw = dict(epsabs=1e-12, epsrel=1e-12, limit=200)
        kappa = quad(lambda a: SIGMA0 * np.exp(-R_E * a), 0, A_MAX, **kw)[0]
        j1 = quad(lambda a: f_kernel(a) * np.exp(-R_I * a), 0, A_MAX, **kw)[0]
        j2 = sum(quad(lambda b: h_kernel(b) * np.exp(-MU4 * b), lo, hi, **kw)[0]
                 for lo, hi in ((0, 1), (1, 2), (2, B_MAX)))
    else:
        kappa = float(np.sum(grid.wa * SIGMA0 * np.exp(-R_E * grid.a)))
        j1 = float(np.sum(grid.wa * f_kernel(grid.a) * np.exp(-R_I * grid.a)))
        j2 = float(np.sum(grid.wb * h_kernel(grid.b) * np.exp(-MU4 * grid.b)))
    if report:
        pub = (0.993209, 27.477214, 1040.759087)
        for nm, got, want in zip(("kappa_EI", "J1", "J2"), (kappa, j1, j2), pub):
            print(f"  {nm:<9s} {got:>14.6f}  published {want:>12.6f}  "
                  f"rel {abs(got-want)/want:.2e}")
    return kappa, j1, j2


# --------------------------------------------------------------------------
# Diffusion
# --------------------------------------------------------------------------
def laplacian_x(u, dx, stencil="fv"):
    """Second difference along the LAST axis with zero-flux ends.

    stencil='fv'  finite-volume / cell-centred: no flux across the two outer
                  faces, so the end rows are (u1-u0) and (u_{N-2}-u_{N-1}).
                  SYMMETRIC and MASS-CONSERVING (columns sum to zero), and
                  diagonalised exactly by the DCT-II.
    stencil='ghost'  node-centred mirror, end rows 2(u1-u0). This is what earlier
                  versions of this project used. It is neither symmetric nor
                  conservative -- total mass drifts under pure diffusion -- and is
                  kept only so an earlier result can be reproduced on request.
    """
    lap = np.empty_like(u)
    lap[..., 1:-1] = u[..., 2:] - 2.0 * u[..., 1:-1] + u[..., :-2]
    c = 1.0 if stencil == "fv" else 2.0
    lap[..., 0] = c * (u[..., 1] - u[..., 0])
    lap[..., -1] = c * (u[..., -2] - u[..., -1])
    return lap / (dx * dx)


def diffusion_eigenvalues(nx, dx):
    """Eigenvalues of the finite-volume Neumann Laplacian, whose eigenvectors are
    the DCT-II basis: lambda_j = -4 sin^2(pi j / 2N) / dx^2."""
    j = np.arange(nx)
    return -4.0 * np.sin(np.pi * j / (2.0 * nx)) ** 2 / (dx * dx)


def diffuse_exact(u, factor, axis=-1):
    """Apply exp(tau D L) exactly, where `factor` = exp(tau D lambda_j).

    Spectral, so this is the exact solution operator of the semi-discrete
    diffusion problem rather than an approximation of it: there is no splitting
    error from this sub-flow and no CFL restriction at all. Mass is conserved to
    machine precision because lambda_0 = 0.
    """
    from scipy.fft import dct, idct
    return idct(factor * dct(u, type=2, norm="ortho", axis=axis),
                type=2, norm="ortho", axis=axis)


@dataclass
class Diffusion:
    D1: float = 50.0
    D2: float = 50.0
    D3: float = 50.0
    D4: float = 50.0

    def as_tuple(self):
        return (self.D1, self.D2, self.D3, self.D4)

    def label(self):
        v = self.as_tuple()
        return f"D={v[0]:g}" if len(set(v)) == 1 else \
               "D=(" + ",".join(f"{z:g}" for z in v) + ")"

    def length_scale(self, T=T_MAX):
        """sqrt(2 D T): how far mass spreads over the window, per field."""
        return tuple(np.sqrt(2.0 * D * T) for D in self.as_tuple())


# --------------------------------------------------------------------------
# Initial condition
# --------------------------------------------------------------------------
@dataclass
class InitialCondition:
    S0: np.ndarray
    I0: np.ndarray
    P0: np.ndarray
    E0: np.ndarray
    I_agg_nodal: np.ndarray
    P_agg_nodal: np.ndarray
    S_nodal: np.ndarray
    opts: dict = field(default_factory=dict)


def _lognormal_profile(v, mean, sd, w):
    """Log-normal density on the grid v, moment-matched to (mean, sd), truncated
    to the grid and renormalized to unit mass under the weights w."""
    s2 = np.log(1.0 + (sd / mean) ** 2)
    mu = np.log(mean) - 0.5 * s2
    s = np.sqrt(s2)
    g = np.zeros_like(v)
    pos = v > 0
    g[pos] = np.exp(-(np.log(v[pos]) - mu) ** 2 / (2 * s2)) / (v[pos] * s * np.sqrt(2 * np.pi))
    tot = float(np.sum(w * g))
    return g / tot if tot > 0 else g


def build_initial_condition(grid: Grid, xg, y_first, cum_first,
                            truncation_correction: bool = False,
                            age_profile: str = "flat",
                            p_from_model: bool = False) -> InitialCondition:
    """Data-anchored initial state.

    truncation_correction=False reproduces the project identity
        I_agg0 = y0 / (mu3+gamma+delta),  P_agg0 = alpha0 I_agg0 / mu4,
    which inverts an UNtruncated mean duration. True uses the occupancy the
    truncated system actually holds -- smaller by 1.5753 (I) and 1.5820 (P).

    age_profile: 'flat' (project baseline, adds no assumption the synthetic IC did
    not already carry), 'equilibrium' (exponential, consistent with the stock
    identity), or 'lognormal' (mean/SD 3.5/2.0 d for I and 10/6 d for P, the
    profile the methodology records as the available refinement).
    """
    tau_I = TAU_I_TRUNC if truncation_correction else TAU_I_NAIVE
    tau_P = TAU_P_TRUNC if truncation_correction else TAU_P_NAIVE

    y_first = np.asarray(y_first, float)
    I_agg = y_first * tau_I
    P_agg = ALPHA0 * I_agg * tau_P
    R_frac = np.maximum(0.0, np.asarray(cum_first, float) - I_agg)
    S_nodal = np.maximum(0.0, 1.0 - I_agg - R_frac)

    order = np.argsort(xg)
    xs = np.asarray(xg, float)[order]

    def lift(vals):
        """Nodal governorate values -> field on the x grid: piecewise linear
        between reporting coordinates, held flat beyond the farthest one."""
        v = np.asarray(vals, float)[order]
        if grid.nx == 1:
            return np.array([v[0]])
        return np.interp(grid.x, xs, v, left=v[0], right=v[-1])

    S0 = lift(S_nodal); I_agg_f = lift(I_agg); P_agg_f = lift(P_agg)

    if age_profile == "flat":
        pI = np.ones_like(grid.a) / A_MAX
        pP = np.ones_like(grid.b) / B_MAX
    elif age_profile == "equilibrium":
        pI = np.exp(-R_I * grid.a); pI /= np.sum(grid.wa * pI)
        pP = np.exp(-MU4 * grid.b); pP /= np.sum(grid.wb * pP)
    elif age_profile == "lognormal":
        pI = _lognormal_profile(grid.a, 3.5, 2.0, grid.wa)
        pP = _lognormal_profile(grid.b, 10.0, 6.0, grid.wb)
    else:
        raise ValueError(f"unknown age_profile {age_profile!r}")

    I0 = pI[:, None] * I_agg_f[None, :]
    P0 = pP[:, None] * P_agg_f[None, :]

    if p_from_model:
        # The project IC sets P_agg0 = alpha0 I_agg0 / mu4, treating the shedding
        # inflow as alpha0 * I_agg. The model's own renewal condition is
        # alpha0 * int f(a) I da, larger by int f da / A_MAX = 6.1527 under a flat
        # profile. This rescales P(0,.) to the stock that renewal implies, so the
        # IC and the equation it feeds agree at t=0.
        shed = grid.walpha @ I0
        cur = grid.wb @ P0
        scale = np.divide(shed * tau_P, cur, out=np.ones_like(cur), where=cur > 0)
        P0 = P0 * scale[None, :]
        P_agg = P_agg * (scale[0] if grid.nx == 1 else
                         np.interp(np.asarray(xg, float), grid.x, scale))

    E0 = np.zeros((grid.na + 1, grid.nx))
    return InitialCondition(S0, I0, P0, E0, I_agg, P_agg, S_nodal,
                            opts=dict(truncation_correction=truncation_correction,
                                      age_profile=age_profile,
                                      p_from_model=p_from_model,
                                      tau_I=tau_I, tau_P=tau_P))


# --------------------------------------------------------------------------
# Forward solve
# --------------------------------------------------------------------------
OBS_FORCE_OF_INFECTION = "foi"      # H = S [int b1 I da + int b2 P db]
OBS_INFECTIOUS_ONSET = "onset"      # H = int sigma(a) E da, the declared alternative


def solve_forward(grid: Grid, ic: InitialCondition, beta10_t, beta20_t,
                  diff: Diffusion, want_states: bool = False,
                  scheme: str = "strang", observable: str = OBS_FORCE_OF_INFECTION,
                  n_sweeps: int = 2, record_full: bool = False, mix=None):
    """March over [0, T_MAX].

    scheme
        'strang'  Strang splitting about the reaction-transport flow, with the
                  diffusion half-flows applied EXACTLY (spectral) and the renewal
                  boundary values taken at the NEW time level by a predictor-
                  corrector. Second order in dt.
        'lie'     first-order Lie splitting with the renewal inflows held at the
                  old time level: what this project's earlier results used.

    observable
        'foi'     S [int b1 I da + int b2 P db], the baseline.
        'onset'   int sigma(a) E da, the declared sensitivity alternative.

    Returns H (nt+1, nx), Sbar (nt+1,), and optionally recorded states.
    """
    if scheme not in ("strang", "lie"):
        raise ValueError(f"unknown scheme {scheme!r}")
    if observable not in (OBS_FORCE_OF_INFECTION, OBS_INFECTIOUS_ONSET):
        raise ValueError(f"unknown observable {observable!r}")
    dt = grid.dt
    Ds = diff.as_tuple()
    if grid.nx == 1 and any(Ds):
        raise ValueError("a single-location grid carries no spatial operator; "
                         "nx=1 requires D1=D2=D3=D4=0")

    S = ic.S0.copy(); E = ic.E0.copy(); I = ic.I0.copy(); P = ic.P0.copy()
    decE = np.exp(-R_E * dt); decI = np.exp(-R_I * dt); decP = np.exp(-MU4 * dt)

    # Exact diffusion propagators. Strang applies two half-steps per step; Lie one
    # full step. Precomputed once -- they do not depend on t.
    spatial = grid.nx > 1 and any(Ds)
    if spatial:
        eig = diffusion_eigenvalues(grid.nx, grid.dx)
        tau = 0.5 * dt if scheme == "strang" else dt
        fac = [np.exp(D * tau * eig) if D else None for D in Ds]

    def diffuse(S, E, I, P):
        if not spatial:
            return S, E, I, P
        if fac[0] is not None: S = diffuse_exact(S, fac[0])
        if fac[1] is not None: E = diffuse_exact(E, fac[1])
        if fac[2] is not None: I = diffuse_exact(I, fac[2])
        if fac[3] is not None: P = diffuse_exact(P, fac[3])
        return S, E, I, P

    # Infection pressure felt at each location. With `mix` (an nx-by-nx
    # row-stochastic matrix) susceptibles at location g are exposed to the
    # mix-weighted pressure of every location -- the metapopulation coupling
    # used by network_fit.py in place of diffusion along one axis.
    if mix is None:
        pI = lambda I_: grid.wf @ I_
        pP = lambda P_: grid.wh @ P_
    else:
        pI = lambda I_: mix @ (grid.wf @ I_)
        pP = lambda P_: mix @ (grid.wh @ P_)

    H = np.empty((grid.nt + 1, grid.nx))
    Sbar = np.empty(grid.nt + 1)
    want_states = want_states or record_full
    rec = {k: np.empty((grid.nt + 1, grid.nx))
           for k in ("I_agg", "P_agg", "E_agg", "foi", "S")} if want_states else None
    if record_full:
        # The full age-resolved fields. Only sane at small nx -- for a single
        # location this is (nt+1, na+1), which is what a city-level I(t,a)
        # surface figure needs.
        rec["I_full"] = np.empty((grid.nt + 1, grid.na + 1, grid.nx))
        rec["E_full"] = np.empty((grid.nt + 1, grid.na + 1, grid.nx))
        rec["P_full"] = np.empty((grid.nt + 1, grid.nb + 1, grid.nx))

    for k in range(grid.nt + 1):
        lamI = beta10_t[k] * pI(I)
        lamP = beta20_t[k] * pP(P)
        foi0 = lamI + lamP
        H[k] = S * foi0 if observable == OBS_FORCE_OF_INFECTION else (grid.wsig @ E)
        Sbar[k] = S.mean()
        if want_states:
            rec["I_agg"][k] = grid.wa @ I
            rec["P_agg"][k] = grid.wb @ P
            rec["E_agg"][k] = grid.wa @ E
            rec["foi"][k] = foi0
            rec["S"][k] = S
            if record_full:
                rec["I_full"][k] = I; rec["E_full"][k] = E; rec["P_full"][k] = P

        if k == grid.nt:
            break

        if scheme == "strang":
            S, E, I, P = diffuse(S, E, I, P)
            foi0 = (beta10_t[k] * pI(I)
                    + beta20_t[k] * pP(P))                  # after the half-flow

        inE0 = S * foi0
        inI0 = grid.wsig @ E
        inP0 = grid.walpha @ I

        # age/duration transport: exact characteristic shift plus exact decay
        En = np.empty_like(E); In = np.empty_like(I); Pn = np.empty_like(P)
        En[1:] = E[:-1] * decE; In[1:] = I[:-1] * decI; Pn[1:] = P[:-1] * decP
        En[0] = inE0; In[0] = inI0; Pn[0] = inP0            # predictor

        if scheme == "lie":
            S = S + dt * (-S * foi0 + LAMBDA_REC - MU1 * S)
            np.clip(S, 0.0, None, out=S)
        else:
            b10h, b20h = beta10_t[k + 1], beta20_t[k + 1]
            Snew = S
            for _ in range(max(1, n_sweeps)):
                foi1 = b10h * pI(In) + b20h * pP(Pn)
                # S by the trapezoid rule, solved semi-implicitly so the sink uses
                # the new-time S rather than the old one
                Snew = ((S * (1.0 - 0.5 * dt * (foi0 + MU1)) + dt * LAMBDA_REC)
                        / (1.0 + 0.5 * dt * (foi1 + MU1)))
                np.clip(Snew, 0.0, None, out=Snew)
                # The renewal conditions are constraints at the NEW time level, so
                # the a=0 / b=0 cells take their new-time values, not a time
                # average of old and new. Swept in causal order: E(.,0) needs I,P;
                # I(.,0) needs E; P(.,0) needs I.
                En[0] = Snew * foi1
                In[0] = grid.wsig @ En
                Pn[0] = grid.walpha @ In
            S = Snew
        E, I, P = En, In, Pn

        np.clip(E, 0.0, None, out=E)
        np.clip(I, 0.0, None, out=I)
        np.clip(P, 0.0, None, out=P)

        S, E, I, P = diffuse(S, E, I, P)

    if want_states:
        rec.update(S_final=S, I_final=I, P_final=P)
        return H, Sbar, rec
    return H, Sbar, None


def R0_of_t(beta10_t, beta20_t, Sbar, kappa, J1, J2):
    """R0(t) = Sbar(t) kappa_EI J1 (beta10 + alpha0 beta20 J2).

    Always call this rather than re-deriving the expression inline: reusing the
    formula or its constituent integrals across reductions without rederiving is
    a documented failure mode in this project's earlier work, and dropping Sbar
    silently compares two different definitions.
    """
    return np.asarray(Sbar) * kappa * J1 * (np.asarray(beta10_t)
                                            + ALPHA0 * np.asarray(beta20_t) * J2)


def route_share(beta10_t, beta20_t, J2, grid=None):
    """Environmental share of the next-generation contribution, in [0, 1]:

        alpha0 beta20 J2 / (beta10 + alpha0 beta20 J2)

    Reported in preference to the ratio alpha0 beta20 J2 / beta10, whose POINTWISE
    mean is dominated by the stretches where beta10 sits on the positivity floor
    and is therefore an artifact of that floor rather than a statement about the
    routes. The time-integrated ratio is returned alongside for continuity with
    the earlier column.
    """
    b10 = np.asarray(beta10_t, float); b20 = np.asarray(beta20_t, float)
    env = ALPHA0 * b20 * J2
    share_t = env / np.maximum(b10 + env, 1e-300)
    if grid is None:
        share = float(np.mean(share_t))
        ratio = float(np.sum(env) / max(np.sum(b10), 1e-300))
    else:
        w = np.full(grid.t.size, grid.dt); w[0] *= 0.5; w[-1] *= 0.5
        share = float(np.sum(w * share_t) / np.sum(w))
        ratio = float(np.sum(w * env) / max(np.sum(w * b10), 1e-300))
    return share, ratio, share_t