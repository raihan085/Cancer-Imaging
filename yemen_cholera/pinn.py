"""
pinn.py -- Dual-network physics-informed neural networks for the age-structured
SEIP system, in the two variants this project needs to compare.

ARCHITECTURES (both are DUAL networks, per this project's own Sec. 10 design:
one network for the states, a second for the parameters, coupled only through a
SOFT residual penalty -- the coupling whose failure mode the per-region memo
diagnosed).

    DPINN    FieldNet: plain tanh MLP on the normalized coordinates -> softplus
                       outputs (S, E, I, P).
             BetaNet:  plain tanh MLP on normalized t -> softplus (b10, b20).

    FFPINN   FieldNet: the SAME MLP, but each normalized coordinate is first
                       lifted by a deterministic Fourier feature map
                           Phi_K(u) = [u, sin(pi u), cos(pi u), ..., sin(K pi u), cos(K pi u)]
             BetaNet:  a truncated Fourier series with a linear trend, passed
                       through softplus.

The two changes point in OPPOSITE directions and the paper must say so. The
embedding on FieldNet RAISES the frequency content the state representation can
reach (a coordinate MLP is biased toward low frequencies). The series on BetaNet
LOWERS it -- it is a hard low-pass restriction to M harmonics, with M bounded by
the sampling: a harmonic of index m has period T/m and cannot be resolved from
samples more than half its period apart, so M <= T/(2*max gap) = 6.5 here.

OBJECTIVE. The data term is the Poisson deviance on counts, not weighted least
squares. On a panel matching the record's 22% exact-zero fraction the specified
weight 1/sqrt(count) puts ~82% of the objective on zero rows, and a fit then
collapses toward predicting zero; see weighting_test.py. Weighted MSE is kept as
`objective="wls"` for the declared comparison.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

import seip

torch.set_default_dtype(torch.float64)
DEV = torch.device("cpu")


# --------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------
def gauss_legendre(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def age_nodes(n_a=12):
    """Gauss-Legendre on [0, A_MAX] for the infection-age integrals."""
    return gauss_legendre(0.0, seip.A_MAX, n_a)


def path_nodes(n1=4, n2=4, n3=10):
    """Gauss-Legendre on [0,1], [1,2], [2,30] so h(b)'s two breakpoints are not
    straddled by a single rule -- the split the methodology specifies. h is
    constant on the last panel and P decays smoothly there, so a short rule is
    accurate; the node counts are checked against the exact J2 below."""
    xs, ws = [], []
    for (lo, hi, n) in ((0.0, 1.0, n1), (1.0, 2.0, n2), (2.0, seip.B_MAX, n3)):
        x, w = gauss_legendre(lo, hi, n)
        xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------
class FourierEmbed(nn.Module):
    """Deterministic Fourier feature map applied PER COORDINATE.

    Deterministic rather than random: with one input dimension per physical
    coordinate and a modest K, the deterministic ladder spans the same band
    without a seed-dependent draw, so a DPINN/FF-PINN comparison is not
    confounded by which random matrix was sampled.
    """

    def __init__(self, orders):
        super().__init__()
        self.orders = list(orders)

    @property
    def out_dim(self):
        return sum(1 + 2 * k for k in self.orders)

    def forward(self, z):
        out = []
        for j, K in enumerate(self.orders):
            u = z[:, j:j + 1]
            out.append(u)
            for m in range(1, K + 1):
                out.append(torch.sin(m * math.pi * u))
                out.append(torch.cos(m * math.pi * u))
        return torch.cat(out, dim=1)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=64, depth=4):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class RawNet(nn.Module):
    """Unconstrained scalar output; positivity is imposed by the HardIC wrapper."""

    def __init__(self, d_in, width=64, depth=4, fourier_orders=None):
        super().__init__()
        self.embed = FourierEmbed(fourier_orders) if fourier_orders else None
        d = self.embed.out_dim if self.embed else d_in
        self.mlp = MLP(d, 1, width, depth)

    def forward(self, z):
        h = self.embed(z) if self.embed else z
        return self.mlp(h).squeeze(-1)


class FieldNet(nn.Module):
    """Normalized coordinates -> one non-negative field.

    Softplus output: smooth everywhere, so the derivatives the physics loss
    needs are not corrupted by a kink, and non-negative by architecture rather
    than by clamping, which would cut the gradient path autograd depends on.
    """

    def __init__(self, d_in, width=64, depth=4, fourier_orders=None, scale=1.0):
        super().__init__()
        self.embed = FourierEmbed(fourier_orders) if fourier_orders else None
        d = self.embed.out_dim if self.embed else d_in
        self.mlp = MLP(d, 1, width, depth)
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self, z):
        h = self.embed(z) if self.embed else z
        return torch.exp(self.log_scale) * nn.functional.softplus(self.mlp(h)).squeeze(-1)


# --------------------------------------------------------------------------
# Architecture registry: the 2x2 design
#
#                       beta = free MLP      beta = order-M Fourier series
#   states plain MLP    "dpinn"              "dpinn_fs"
#   states Fourier      "fdpinn"             "ff"
#
# "dpinn" and "ff" are the two arms this project reported first; because they
# differ in BOTH factors at once, neither effect was attributable from them.
# The two added cells complete the factorial.
# --------------------------------------------------------------------------
ARCH = {
    "dpinn":    (False, "mlp"),        # DPINN
    "dpinn_fs": (False, "fourier"),    # DPINN-FS
    "fdpinn":   (True,  "mlp"),        # F-DPINN
    "ff":       (True,  "fourier"),    # FF-PINN
}
ARCH_NAME = {
    "dpinn":    "DPINN (plain states, MLP beta)",
    "dpinn_fs": "DPINN-FS (plain states, Fourier beta)",
    "fdpinn":   "F-DPINN (Fourier states, MLP beta)",
    "ff":       "FF-PINN (Fourier states, Fourier beta)",
}


class BetaNetMLP(nn.Module):
    """DPINN parameter network: normalized t -> (b10, b20), softplus, with NO
    hand-picked multiplicative scale on the output. A learned log-scale is used
    instead so the magnitude comes from the loss rather than from a constant
    chosen before the first gradient step."""

    def __init__(self, width=48, depth=3, scale10=1e-2, scale20=1e-5):
        super().__init__()
        self.mlp = MLP(1, 2, width, depth)
        self.ls = nn.Parameter(torch.tensor([math.log(scale10), math.log(scale20)]))

    def forward(self, tau):
        out = nn.functional.softplus(self.mlp(tau))
        return (torch.exp(self.ls[0]) * out[:, 0] + 1e-14,
                torch.exp(self.ls[1]) * out[:, 1] + 1e-14)


class BetaNetFourier(nn.Module):
    """FF-PINN parameter model: truncated Fourier series with a linear trend.

    The trend is present because a pure Fourier basis is periodic at the domain
    edges and would otherwise force the first and last fitted values to agree.
    M is Nyquist-bounded by the worst-served governorate's reporting gap.
    """

    def __init__(self, M=4, scale10=1e-2, scale20=1e-5):
        super().__init__()
        self.M = M
        self.c = nn.Parameter(torch.zeros(2, 2 + 2 * M))
        with torch.no_grad():
            self.c[0, 0] = math.log(math.expm1(1.0))
            self.c[1, 0] = math.log(math.expm1(1.0))
        self.ls = nn.Parameter(torch.tensor([math.log(scale10), math.log(scale20)]))

    def basis(self, tau):
        t = 0.5 * (tau + 1.0)                       # tau in [-1,1] -> t/T in [0,1]
        cols = [torch.ones_like(t), t]
        for m in range(1, self.M + 1):
            cols += [torch.sin(2 * math.pi * m * t), torch.cos(2 * math.pi * m * t)]
        return torch.cat(cols, dim=1)

    def forward(self, tau):
        X = self.basis(tau)
        g = X @ self.c.T
        out = nn.functional.softplus(g)
        return (torch.exp(self.ls[0]) * out[:, 0] + 1e-14,
                torch.exp(self.ls[1]) * out[:, 1] + 1e-14)


class HardIC(nn.Module):
    """Wraps a raw network so the initial condition holds EXACTLY, by
    construction, instead of being enforced by a penalty.

    Why this matters here and is not a numerical nicety. The problem statement's
    Sec. 5 amplitude scaling

        (E, I, P, b10, b20) -> (cE, cI, cP, b10/c, b20/c)

    leaves the observation operator and every transport equation invariant and
    is broken only by the renewal condition for E. In a soft-penalty PINN that
    relation is one more residual term, so c is nearly free and the LEVEL of the
    transmission estimate is set by the loss weights rather than by the data.
    Pinning the initial condition exactly removes c: measured on the city twin, a
    soft IC returns a reproduction number inflated by a factor of 2 to 3 with the
    states correspondingly too small, while this parameterization does not.

        ramp(t) = 1 - exp(-t/lam)          ramp(0) = 0

        zero IC      u = ramp * scale * softplus(net)
        positive IC  u = u0 * exp(ramp * clamp(net))
        decreasing   u = u0 * exp(-ramp * softplus(net))      (used for S)

    Each is non-negative by construction and exact at t = 0.
    """

    def __init__(self, net, mode, u0=None, scale=1.0, lam=2.0, clamp=12.0):
        super().__init__()
        self.net = net
        self.mode = mode
        self.scale = scale
        self.lam = lam
        self.clamp = clamp
        if u0 is not None and not torch.is_tensor(u0):
            u0 = torch.tensor(u0, dtype=torch.get_default_dtype())
        self.u0_const = u0
        if mode == "decreasing":
            # start near S = S0: softplus(-4) = 0.018, so S barely declines until
            # the data asks it to, rather than collapsing in the first days.
            with torch.no_grad():
                for m_ in reversed(list(self.net.mlp.net)):
                    if isinstance(m_, nn.Linear):
                        m_.bias.fill_(-4.0); break

    def ramp(self, t):
        return 1.0 - torch.exp(-t / self.lam)

    def forward(self, z, t, u0=None):
        g = self.net(z)
        r = self.ramp(t)
        base = self.u0_const if u0 is None else u0
        if self.mode == "zero":
            return r * self.scale * nn.functional.softplus(g)
        if self.mode == "positive":
            return base * torch.exp(r * torch.clamp(g, -self.clamp, self.clamp))
        if self.mode == "decreasing":
            return base * torch.exp(-r * nn.functional.softplus(g))
        raise ValueError(self.mode)


def d_dx(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


# --------------------------------------------------------------------------
# Single-location (city) PINN:  S(t), E(t,a), I(t,a), P(t,b),  D = 0
# --------------------------------------------------------------------------
class CityPINN(nn.Module):
    """The age-structured SEIP system at one governorate with diffusion off.

    With D = 0 the spatial operator vanishes and the system decouples into an
    initial-boundary-value problem in (t, a) and (t, b). That is the reduction
    this project's per-region work already uses, and it makes I(t, a) a genuine
    two-dimensional surface -- the object the age structure actually adds, and
    the one the two architectures can be compared on.
    """

    def __init__(self, kind="ff", width=64, depth=4, M=4, K_t=8, K_a=4, K_b=6,
                 scaleI=1e-5, scaleP=1e-4, seed=0, hard_ic=True,
                 S0=1.0, I0=1e-5, P0=1e-4):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        # The two design factors are independent and must be set independently:
        # an earlier version keyed both off a single flag, which confounded the
        # state embedding with the beta parameterisation and made the DPINN vs
        # FF-PINN difference unattributable to either.
        state_ff, beta_kind = ARCH[kind]
        self.state_ff, self.beta_kind = state_ff, beta_kind
        ff2 = [K_t, K_a] if state_ff else None
        ff2b = [K_t, K_b] if state_ff else None
        ff1 = [K_t] if state_ff else None
        self.hard_ic = hard_ic
        if hard_ic:
            # S0, I0 and P0 are supplied at build time so the IC is exact.
            self._S = HardIC(RawNet(1, width, depth, ff1), "decreasing", u0=S0)
            self._E = HardIC(RawNet(2, width, depth, ff2), "zero", scale=scaleI)
            self._I = HardIC(RawNet(2, width, depth, ff2), "positive", u0=I0)
            self._P = HardIC(RawNet(2, width, depth, ff2b), "positive", u0=P0)
        else:
            self._S = FieldNet(1, width, depth, ff1, scale=1.0)
            self._E = FieldNet(2, width, depth, ff2, scale=scaleI)
            self._I = FieldNet(2, width, depth, ff2, scale=scaleI)
            self._P = FieldNet(2, width, depth, ff2b, scale=scaleP)
        self.beta = (BetaNetFourier(M=M) if beta_kind == "fourier" else BetaNetMLP())

        aq, wq = age_nodes()
        bq, vq = path_nodes()
        self.register_buffer("aq", torch.tensor(aq))
        self.register_buffer("wf", torch.tensor(wq * seip.f_kernel(aq)))
        self.register_buffer("wsig", torch.tensor(wq * seip.SIGMA0))
        self.register_buffer("walpha", torch.tensor(wq * seip.ALPHA0 * seip.f_kernel(aq)))
        self.register_buffer("bq", torch.tensor(bq))
        self.register_buffer("wh", torch.tensor(vq * seip.h_kernel(bq)))

    # -- coordinate normalization -----------------------------------------
    @staticmethod
    def nt(t):
        return 2.0 * t / seip.T_MAX - 1.0

    @staticmethod
    def na(a):
        return 2.0 * a / seip.A_MAX - 1.0

    @staticmethod
    def nb(b):
        return 2.0 * b / seip.B_MAX - 1.0

    # -- field evaluation --------------------------------------------------
    def S_of(self, t):
        z = self.nt(t).reshape(-1, 1)
        return self._S(z, t) if self.hard_ic else self._S(z)

    def E(self, z, t):
        return self._E(z, t) if self.hard_ic else self._E(z)

    def I(self, z, t):
        return self._I(z, t) if self.hard_ic else self._I(z)

    def P(self, z, t):
        return self._P(z, t) if self.hard_ic else self._P(z)

    def _grid(self, t, q):
        """Outer product of times t (n,) and nodes q (m,) -> flat (n*m,) pairs."""
        n, m = t.numel(), q.numel()
        return (t.reshape(-1, 1).expand(n, m).reshape(-1),
                q.reshape(1, -1).expand(n, m).reshape(-1), n, m)

    def EIP_on_nodes(self, t):
        """E, I on the a-quadrature nodes and P on the b-nodes, for times t."""
        ta, aa, n, ma = self._grid(t, self.aq)
        za = torch.stack([self.nt(ta), self.na(aa)], 1)
        Emat = self.E(za, ta).reshape(n, ma)
        Imat = self.I(za, ta).reshape(n, ma)
        tb, bb, _, mb = self._grid(t, self.bq)
        zb = torch.stack([self.nt(tb), self.nb(bb)], 1)
        Pmat = self.P(zb, tb).reshape(n, mb)
        return Emat, Imat, Pmat

    def lambdas(self, t, Imat=None, Pmat=None):
        if Imat is None:
            _, Imat, Pmat = self.EIP_on_nodes(t)
        b10, b20 = self.beta(self.nt(t).reshape(-1, 1))
        lamI = b10 * (Imat @ self.wf)
        lamP = b20 * (Pmat @ self.wh)
        return lamI, lamP, b10, b20

    def IP_on_nodes(self, t):
        """I on the a-nodes and P on the b-nodes only. The observation operator
        does not involve E, and evaluating it there was a third of the cost."""
        ta, aa, n, ma = self._grid(t, self.aq)
        Imat = self.I(torch.stack([self.nt(ta), self.na(aa)], 1), ta).reshape(n, ma)
        tb, bb, _, mb = self._grid(t, self.bq)
        Pmat = self.P(torch.stack([self.nt(tb), self.nb(bb)], 1), tb).reshape(n, mb)
        return Imat, Pmat

    def observable(self, t):
        """H(t) = S(t) [ int b1 I da + int b2 P db ] -- what a bulletin measures."""
        Imat, Pmat = self.IP_on_nodes(t)
        b10, b20 = self.beta(self.nt(t).reshape(-1, 1))
        return self.S_of(t) * (b10 * (Imat @ self.wf) + b20 * (Pmat @ self.wh))

    # -- residuals ---------------------------------------------------------
    def residuals(self, t_col):
        t = t_col.clone().requires_grad_(True)

        # S: ordinary differential residual
        S = self.S_of(t)
        dS = d_dx(S, t)
        ta, aa, n, ma = self._grid(t, self.aq)
        ta = ta.clone().requires_grad_(True); aa = aa.clone().requires_grad_(True)
        za = torch.stack([self.nt(ta), self.na(aa)], 1)
        Ef = self.E(za, ta); If = self.I(za, ta)
        Emat, Imat = Ef.reshape(n, ma), If.reshape(n, ma)

        tb, bb, _, mb = self._grid(t, self.bq)
        tb = tb.clone().requires_grad_(True); bb = bb.clone().requires_grad_(True)
        zb = torch.stack([self.nt(tb), self.nb(bb)], 1)
        Pf = self.P(zb, tb); Pmat = Pf.reshape(n, mb)

        b10, b20 = self.beta(self.nt(t).reshape(-1, 1))
        lamI = b10 * (Imat @ self.wf)
        lamP = b20 * (Pmat @ self.wh)
        foi = lamI + lamP

        r_S = dS + S * foi - seip.LAMBDA_REC + seip.MU1 * S
        r_E = d_dx(Ef, ta) + d_dx(Ef, aa) + seip.R_E * Ef
        r_I = d_dx(If, ta) + d_dx(If, aa) + seip.R_I * If
        r_P = d_dx(Pf, tb) + d_dx(Pf, bb) + seip.MU4 * Pf

        # renewal conditions -- the non-local boundary conditions at a = 0, b = 0,
        # which is where the unknowns enter the system at all
        z0 = torch.stack([self.nt(t), torch.full_like(t, -1.0)], 1)
        E0 = self.E(z0, t); I0 = self.I(z0, t); P0 = self.P(z0, t)
        r_E0 = E0 - S * foi
        r_I0 = I0 - (Emat @ self.wsig)
        r_P0 = P0 - (Imat @ self.walpha)
        return r_S, r_E, r_I, r_P, r_E0, r_I0, r_P0


# --------------------------------------------------------------------------
# Data term
# --------------------------------------------------------------------------
class CityData:
    """Interval-averaged observation operator and the two data objectives."""

    def __init__(self, panel, gi, n_q=2, objective="poisson"):
        m = panel.gov == gi
        self.t0 = torch.tensor(panel.t0[m]); self.t1 = torch.tensor(panel.t1[m])
        self.y = torch.tensor(panel.y[m]); self.c = torch.tensor(panel.dC[m])
        self.k = (torch.tensor(np.asarray(panel.nb_k)[m])
                  if getattr(panel, "nb_k", None) is not None else None)
        self.w = torch.tensor(panel.w[m])
        self.N = float(panel.pop[gi])
        self.name = panel.names[gi]
        self.objective = objective
        x, w = gauss_legendre(-1.0, 1.0, n_q)
        self.qx = torch.tensor(x); self.qw = torch.tensor(w) / 2.0
        self.n_obs = int(m.sum())
        self.Ndt = self.N * (self.t1 - self.t0)
        self.dev_scale = 1.0
        self.yscale = float(np.sqrt(np.mean((panel.w[m] * panel.y[m]) ** 2)))

    def eval_times(self):
        mid = 0.5 * (self.t0 + self.t1); half = 0.5 * (self.t1 - self.t0)
        return (mid.reshape(-1, 1) + half.reshape(-1, 1) * self.qx.reshape(1, -1)).reshape(-1)

    def average(self, H):
        return (H.reshape(self.n_obs, -1) * self.qw.reshape(1, -1)).sum(1)

    def residual(self, yhat):
        if self.objective == "wls":
            return self.w * (yhat - self.y) / self.yscale
        lam = torch.clamp(yhat * self.Ndt, min=1e-10)
        c = self.c
        term = torch.where(c > 0, c * torch.log(torch.clamp(c, min=1e-300) / lam),
                           torch.zeros_like(c))
        if self.objective == "nb":
            k = self.k
            d2 = torch.clamp(2.0 * (term - (c + k) * torch.log((c + k) / (lam + k))), min=0.0)
        else:
            d2 = torch.clamp(2.0 * (term - (c - lam)), min=0.0)
        return torch.sign(c - lam) * torch.sqrt(d2) / self.dev_scale


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def city_initial_condition(panel, gi, grid, **ic_opts):
    """The same data-anchored IC the deterministic arm uses, at one location."""
    ic = seip.build_initial_condition(seip.Grid(dt=grid.dt, nx=1),
                                      np.array([0.0]),
                                      np.array([panel.y_first[gi]]),
                                      np.array([panel.cum_first[gi]]), **ic_opts)
    return (float(ic.S0[0]), ic.I0[:, 0].copy(), ic.P0[:, 0].copy(),
            float(ic.I_agg_nodal[0]), float(ic.P_agg_nodal[0]), ic.opts)


def train_city(model: CityPINN, data: CityData, S0, Iagg0, Pagg0,
               n_col=192, epochs=3000, lr=3e-3, w_pde=1.0, w_ren=20.0, w_ic=50.0,
               w_smooth=3e-2, lbfgs=250, log_every=500, verbose=True,
               pretrain=800):
    """Staged: (1) fit the fields to the initial condition and the renewal
    relations with beta frozen, (2) release everything under Adam, (3) refine
    with L-BFGS. Starting the joint fit cold lets the two networks trade against
    each other before either means anything.

    w_ren is deliberately large. The renewal condition E(t,0) = S(t)[int b1 I +
    int b2 P] is the ONLY relation in which beta appears at all, so it is the
    only thing tying BetaNet to FieldNet. Under-weight it and the two networks
    decouple: the data term constrains their product and never the factors, and
    the optimizer can cheapen the loss by inflating beta instead of growing I and
    P. That is precisely the dual-network failure this project's per-region work
    diagnosed, so the coupling weight is reported, not left at a default."""
    T = seip.T_MAX
    aq_t = torch.tensor(np.linspace(0, seip.A_MAX, 33))
    bq_t = torch.tensor(np.linspace(0, seip.B_MAX, 41))
    I0_flat = Iagg0 / seip.A_MAX
    P0_flat = Pagg0 / seip.B_MAX

    def ic_loss():
        if getattr(model, "hard_ic", False):
            # exact by construction; the penalty would only add a zero term
            return torch.zeros((), dtype=torch.get_default_dtype())
        z = torch.zeros(1)
        s = model.S_of(z)
        za = torch.stack([model.nt(torch.zeros_like(aq_t)), model.na(aq_t)], 1)
        zb = torch.stack([model.nt(torch.zeros_like(bq_t)), model.nb(bq_t)], 1)
        E0 = model.E(za); I0 = model.I(za); P0 = model.P(zb)
        sc_I = max(I0_flat, 1e-12); sc_P = max(P0_flat, 1e-12)
        return (((s - S0) / max(S0, 1e-9)) ** 2).mean() \
            + ((E0 / sc_I) ** 2).mean() \
            + (((I0 - I0_flat) / sc_I) ** 2).mean() \
            + (((P0 - P0_flat) / sc_P) ** 2).mean()

    hist = []

    def physics(n):
        tc = torch.rand(n) * T
        tc = torch.cat([tc, torch.rand(n // 4) * 14.0])      # oversample near t=0
        rS, rE, rI, rP, rE0, rI0, rP0 = model.residuals(tc)
        sI = max(Iagg0 / seip.A_MAX, 1e-12); sP = max(Pagg0 / seip.B_MAX, 1e-12)
        pde = (rS ** 2).mean() + ((rE / sI) ** 2).mean() + ((rI / sI) ** 2).mean() \
            + ((rP / sP) ** 2).mean()
        ren = ((rE0 / sI) ** 2).mean() + ((rI0 / sI) ** 2).mean() \
            + ((rP0 / sP) ** 2).mean()
        return pde, ren

    def smooth():
        tt = torch.linspace(0, T, 200).reshape(-1, 1).requires_grad_(True)
        b10, b20 = model.beta(model.nt(tt))
        d1 = d_dx(b10, tt); d2 = d_dx(b20, tt)
        return ((d1 / torch.clamp(b10.mean(), min=1e-300)) ** 2).mean() \
            + ((d2 / torch.clamp(b20.mean(), min=1e-300)) ** 2).mean()

    # ---- stage 1: fields to the IC and the renewal relations ----------------
    for p in model.beta.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for it in range(pretrain):
        opt.zero_grad()
        pde, ren = physics(n_col // 2)
        loss = w_ic * ic_loss() + 0.1 * pde + w_ren * ren
        loss.backward(); opt.step()
    for p in model.beta.parameters():
        p.requires_grad_(True)

    # ---- stage 2: joint Adam ------------------------------------------------
    te = data.eval_times()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 50)
    for it in range(epochs):
        opt.zero_grad()
        yhat = data.average(model.observable(te))
        r = data.residual(yhat)
        dloss = (r ** 2).mean()
        pde, ren = physics(n_col)
        loss = dloss + w_pde * pde + w_ren * ren + w_ic * ic_loss() + w_smooth * smooth()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
        if verbose and (it % log_every == 0 or it == epochs - 1):
            print(f"    adam {it:5d}  total {loss.item():.4e}  data {dloss.item():.4e}  "
                  f"pde {pde.item():.3e}  renewal {ren.item():.3e}")
        hist.append(float(loss.item()))

    # ---- stage 3: L-BFGS ----------------------------------------------------
    if lbfgs:
        optL = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs, history_size=50,
                                 tolerance_grad=1e-12, tolerance_change=1e-14,
                                 line_search_fn="strong_wolfe")
        tc_fixed = torch.linspace(0.0, T, n_col)

        def closure():
            optL.zero_grad()
            yhat = data.average(model.observable(te))
            r = data.residual(yhat)
            rS, rE, rI, rP, rE0, rI0, rP0 = model.residuals(tc_fixed)
            sI = max(Iagg0 / seip.A_MAX, 1e-12); sP = max(Pagg0 / seip.B_MAX, 1e-12)
            pde = (rS ** 2).mean() + ((rE / sI) ** 2).mean() + ((rI / sI) ** 2).mean() \
                + ((rP / sP) ** 2).mean()
            ren = ((rE0 / sI) ** 2).mean() + ((rI0 / sI) ** 2).mean() \
                + ((rP0 / sP) ** 2).mean()
            loss = (r ** 2).mean() + w_pde * pde + w_ren * ren + w_ic * ic_loss() \
                + w_smooth * smooth()
            loss.backward()
            return loss

        try:
            optL.step(closure)
        except RuntimeError as e:
            if verbose:
                print("    L-BFGS stopped:", e)
    return hist


@torch.no_grad()
def city_predict(model: CityPINN, t_grid, a_grid, b_grid):
    """Everything a figure needs, on a regular grid."""
    t = torch.tensor(np.asarray(t_grid, float))
    S = model.S_of(t).numpy()
    b10, b20 = model.beta(model.nt(t).reshape(-1, 1))
    Emat, Imat, Pmat = model.EIP_on_nodes(t)
    lamI = b10 * (Imat @ model.wf); lamP = b20 * (Pmat @ model.wh)
    H = (model.S_of(t) * (lamI + lamP)).numpy()

    def surf(net, tg, qg, norm):
        tt = torch.tensor(np.repeat(np.asarray(tg, float), len(qg)))
        qq = torch.tensor(np.tile(np.asarray(qg, float), len(tg)))
        z = torch.stack([model.nt(tt), norm(qq)], 1)
        return net(z, tt).reshape(len(tg), len(qg)).numpy()

    return dict(S=S, b10=b10.numpy(), b20=b20.numpy(), H=H,
                I_surface=surf(model.I, t_grid, a_grid, model.na),
                E_surface=surf(model.E, t_grid, a_grid, model.na),
                P_surface=surf(model.P, t_grid, b_grid, model.nb),
                I_agg=(Imat @ torch.tensor(age_nodes()[1])).numpy())


def R0_from(model, t_grid, kappa, J1, J2):
    with torch.no_grad():
        t = torch.tensor(np.asarray(t_grid, float))
        S = model.S_of(t).numpy()
        b10, b20 = model.beta(model.nt(t).reshape(-1, 1))
    return seip.R0_of_t(b10.numpy(), b20.numpy(), S, kappa, J1, J2)


# --------------------------------------------------------------------------
# Full-domain PINN:  S(t,x), E(t,a,x), I(t,a,x), P(t,b,x)
# --------------------------------------------------------------------------
class DomainPINN(nn.Module):
    """The full age-structured transport-diffusion system on [0, X_max].

    Same dual-network design as CityPINN; the fields now carry a spatial
    coordinate, so the PDE residuals pick up D d2/dx2 and the homogeneous
    Neumann conditions are enforced as a soft penalty at the two ends.
    """

    def __init__(self, kind="ff", width=64, depth=4, M=4,
                 K_t=8, K_x=4, K_a=4, K_b=6, scaleI=1e-5, scaleP=1e-4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        state_ff, beta_kind = ARCH[kind]
        self.state_ff, self.beta_kind = state_ff, beta_kind
        ff_tx = [K_t, K_x] if state_ff else None
        ff_tax = [K_t, K_a, K_x] if state_ff else None
        ff_tbx = [K_t, K_b, K_x] if state_ff else None
        self.S = FieldNet(2, width, depth, ff_tx, scale=1.0)
        self.E = FieldNet(3, width, depth, ff_tax, scale=scaleI)
        self.I = FieldNet(3, width, depth, ff_tax, scale=scaleI)
        self.P = FieldNet(3, width, depth, ff_tbx, scale=scaleP)
        self.beta = (BetaNetFourier(M=M) if beta_kind == "fourier" else BetaNetMLP())
        aq, wq = age_nodes(); bq, vq = path_nodes()
        self.register_buffer("aq", torch.tensor(aq))
        self.register_buffer("wf", torch.tensor(wq * seip.f_kernel(aq)))
        self.register_buffer("wsig", torch.tensor(wq * seip.SIGMA0))
        self.register_buffer("walpha", torch.tensor(wq * seip.ALPHA0 * seip.f_kernel(aq)))
        self.register_buffer("bq", torch.tensor(bq))
        self.register_buffer("wh", torch.tensor(vq * seip.h_kernel(bq)))

    def __repr_name__(self):
        return self.kind

    @staticmethod
    def nt(t):
        return 2.0 * t / seip.T_MAX - 1.0

    @staticmethod
    def na(a):
        return 2.0 * a / seip.A_MAX - 1.0

    @staticmethod
    def nb(b):
        return 2.0 * b / seip.B_MAX - 1.0

    def nx(self, x):
        return 2.0 * x / self.x_max - 1.0

    def set_domain(self, x_max):
        self.x_max = float(x_max)

    def S_of(self, t, x):
        return self.S(torch.stack([self.nt(t), self.nx(x)], 1))

    def _tile(self, t, x, q):
        n, m = t.numel(), q.numel()
        tt = t.reshape(-1, 1).expand(n, m).reshape(-1)
        xx = x.reshape(-1, 1).expand(n, m).reshape(-1)
        qq = q.reshape(1, -1).expand(n, m).reshape(-1)
        return tt, xx, qq, n, m

    def IP_on_nodes(self, t, x):
        tt, xx, aa, n, ma = self._tile(t, x, self.aq)
        Imat = self.I(torch.stack([self.nt(tt), self.na(aa), self.nx(xx)], 1)).reshape(n, ma)
        tb, xb, bb, _, mb = self._tile(t, x, self.bq)
        Pmat = self.P(torch.stack([self.nt(tb), self.nb(bb), self.nx(xb)], 1)).reshape(n, mb)
        return Imat, Pmat

    def observable(self, t, x):
        Imat, Pmat = self.IP_on_nodes(t, x)
        b10, b20 = self.beta(self.nt(t).reshape(-1, 1))
        return self.S_of(t, x) * (b10 * (Imat @ self.wf) + b20 * (Pmat @ self.wh))

    def residuals(self, t_col, x_col, diff: seip.Diffusion):
        D1, D2, D3, D4 = diff.as_tuple()
        sx = 2.0 / self.x_max                      # d/dx = sx * d/dxi
        t = t_col.clone().requires_grad_(True)
        x = x_col.clone().requires_grad_(True)

        S = self.S_of(t, x)
        dSt = d_dx(S, t)
        dSx = d_dx(S, x)
        dSxx = d_dx(dSx, x)

        tt, xx, aa, n, ma = self._tile(t, x, self.aq)
        tt = tt.clone().requires_grad_(True); xx = xx.clone().requires_grad_(True)
        aa = aa.clone().requires_grad_(True)
        za = torch.stack([self.nt(tt), self.na(aa), self.nx(xx)], 1)
        Ef = self.E(za); If = self.I(za)
        Emat, Imat = Ef.reshape(n, ma), If.reshape(n, ma)

        tb, xb, bb, _, mb = self._tile(t, x, self.bq)
        tb = tb.clone().requires_grad_(True); xb = xb.clone().requires_grad_(True)
        bb = bb.clone().requires_grad_(True)
        zb = torch.stack([self.nt(tb), self.nb(bb), self.nx(xb)], 1)
        Pf = self.P(zb); Pmat = Pf.reshape(n, mb)

        b10, b20 = self.beta(self.nt(t).reshape(-1, 1))
        foi = b10 * (Imat @ self.wf) + b20 * (Pmat @ self.wh)

        def lap(f, v):
            g1 = d_dx(f, v)
            return d_dx(g1, v)

        r_S = dSt - D1 * dSxx + S * foi - seip.LAMBDA_REC + seip.MU1 * S
        r_E = d_dx(Ef, tt) + d_dx(Ef, aa) - D2 * lap(Ef, xx) + seip.R_E * Ef
        r_I = d_dx(If, tt) + d_dx(If, aa) - D3 * lap(If, xx) + seip.R_I * If
        r_P = d_dx(Pf, tb) + d_dx(Pf, bb) - D4 * lap(Pf, xb) + seip.MU4 * Pf

        z0 = torch.stack([self.nt(t), torch.full_like(t, -1.0), self.nx(x)], 1)
        r_E0 = self.E(z0) - S * foi
        r_I0 = self.I(z0) - (Emat @ self.wsig)
        r_P0 = self.P(z0) - (Imat @ self.walpha)
        return r_S, r_E, r_I, r_P, r_E0, r_I0, r_P0

    def neumann(self, t_b):
        """Homogeneous Neumann on all four fields at x = 0 and x = X_max."""
        out = []
        for xv in (0.0, self.x_max):
            x = torch.full_like(t_b, xv).requires_grad_(True)
            S = self.S_of(t_b, x)
            out.append(d_dx(S, x))
            for net, q, norm in ((self.E, self.aq, self.na), (self.I, self.aq, self.na),
                                 (self.P, self.bq, self.nb)):
                tt, xx, qq, n, m = self._tile(t_b, x, q)
                xx = xx.clone().requires_grad_(True)
                f = net(torch.stack([self.nt(tt), norm(qq), self.nx(xx)], 1))
                out.append(d_dx(f, xx))
        return torch.cat(out)