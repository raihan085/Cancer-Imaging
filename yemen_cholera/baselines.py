"""
baselines.py -- two simple references every architecture has to beat.

spline  Poisson P-spline regression of the per-capita incidence on time. No
        epidemic mechanism at all: it is the floor any curve-fitter reaches,
        so an architecture that cannot beat it out of sample is adding
        flexibility, not information. Smoothing chosen by AIC on the training
        rows only.

Both take an optional negative-binomial size k; without it the likelihood is
Poisson.

sir     A plain SIR model with a time-varying transmission rate
            S' = -beta(t) S I,   I' = beta(t) S I - gamma I,   H = beta S I
        beta(t) = softplus(order-M Fourier series + trend), gamma = 1/7 d,
        Poisson deviance. The simplest mechanistic model with a reproduction
        number, R(t) = beta(t) S(t) / gamma, and no age structure, no
        environmental route.

Both are fitted to interval-AVERAGED rates on exactly the rows the other arms
see, and both predict every interval so they slot into the same holdout.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, minimize

import seip

GAMMA_SIR = seip.GAMMA + seip.DELTA + seip.MU3


def _dev(c, lam, k=None):
    from fit import deviance_residual
    return deviance_residual(c, lam, k)


def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w / 2.0


def _interval_nodes(t0, t1, nq=8):
    x, w = _gl(nq)
    mid, half = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
    return mid[:, None] + half[:, None] * x[None, :], w


def poisson_deviance(c, lam):
    lam = np.maximum(lam, 1e-12)
    term = np.where(c > 0, c * np.log(np.maximum(c, 1e-300) / lam), 0.0)
    return 2.0 * np.sum(term - (c - lam))


# --------------------------------------------------------------------------
# P-spline Poisson regression
# --------------------------------------------------------------------------
class SplineBaseline:
    def __init__(self, n_basis=24, degree=3, T=seip.T_MAX):
        self.k = degree
        inner = np.linspace(0.0, T, n_basis - degree + 1)
        self.knots = np.r_[[0.0] * degree, inner, [T] * degree]
        self.nb = n_basis
        D = np.diff(np.eye(n_basis), 2, axis=0)
        self.P = D.T @ D

    def basis(self, t):
        t = np.clip(np.asarray(t, float), 0.0, self.knots[-1] - 1e-9)
        return BSpline.design_matrix(t.ravel(), self.knots, self.k).toarray()

    def _setup(self, t0, t1, Ndt):
        tq, w = _interval_nodes(t0, t1)
        B = self.basis(tq.ravel()).reshape(len(t0), tq.shape[1], self.nb)
        return B, w, Ndt

    @staticmethod
    def _lam(theta, B, w, Ndt):
        r = np.exp(np.clip(B @ theta, -60, 5))            # (n, q)
        return Ndt * (r @ w), r

    def _fit_pen(self, B, w, Ndt, c, pen, theta0, k=None):
        """Penalised Poisson (k None) or negative-binomial (size k) fit."""
        def f(th):
            lam, r = self._lam(th, B, w, Ndt)
            lam = np.maximum(lam, 1e-12)
            if k is None:
                val = np.sum(lam - c * np.log(lam))
                dl = 1.0 - c / lam
            else:
                val = np.sum((c + k) * np.log(lam + k) - c * np.log(lam))
                dl = (c + k) / (lam + k) - c / lam
            val += pen * th @ self.P @ th
            G = Ndt[:, None] * np.einsum("nq,q,nqb->nb", r, w, B)
            grad = G.T @ dl + 2.0 * pen * self.P @ th
            return val, grad
        res = minimize(f, theta0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=3000))
        th = res.x
        lam, r = self._lam(th, B, w, Ndt)
        lam = np.maximum(lam, 1e-12)
        G = Ndt[:, None] * np.einsum("nq,q,nqb->nb", r, w, B)
        var = lam if k is None else lam + lam ** 2 / k
        F = G.T @ (G / var[:, None])
        A = np.linalg.solve(F + 2.0 * pen * self.P + 1e-12 * np.eye(self.nb), F)
        edf = float(np.trace(A))
        dev = float(np.sum(_dev(c, lam, k) ** 2))
        return th, edf, dev

    def fit(self, t0, t1, c, N, pens=(1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4), k=None):
        Ndt = N * (t1 - t0)
        B, w, _ = self._setup(t0, t1, Ndt)
        y0 = max(c.sum() / Ndt.sum(), 1e-9)
        th0 = np.full(self.nb, np.log(y0))
        best = None
        for pen in pens:
            th, edf, dev = self._fit_pen(B, w, Ndt, c, pen, th0, k)
            aic = dev + 2.0 * edf
            if best is None or aic < best[0]:
                best = (aic, pen, th, edf)
        self.aic, self.pen, self.theta, self.edf = best
        return self

    def predict(self, t0, t1):
        tq, w = _interval_nodes(t0, t1)
        B = self.basis(tq.ravel()).reshape(len(t0), tq.shape[1], self.nb)
        return np.exp(np.clip(B @ self.theta, -60, 5)) @ w

    def rate(self, t):
        return np.exp(np.clip(self.basis(t) @ self.theta, -60, 5))


# --------------------------------------------------------------------------
# SIR with time-varying beta
# --------------------------------------------------------------------------
class SIRBaseline:
    def __init__(self, M=4, dt=0.25, T=seip.T_MAX):
        self.M, self.dt, self.T = M, dt, T
        self.t = np.linspace(0.0, T, int(round(T / dt)) + 1)
        self.X = self.design(self.t)

    def design(self, t):
        cols = [np.ones_like(t), t / self.T]
        for m in range(1, self.M + 1):
            cols += [np.sin(2 * np.pi * m * t / self.T), np.cos(2 * np.pi * m * t / self.T)]
        return np.column_stack(cols)

    def beta(self, theta):
        return np.logaddexp(0.0, self.X @ theta) + 1e-12

    def solve(self, theta, S0, I0):
        b = self.beta(theta)
        n = len(self.t)
        S = np.empty(n); I = np.empty(n)
        S[0], I[0] = S0, I0
        h, g = self.dt, GAMMA_SIR
        for k in range(n - 1):
            bm = 0.5 * (b[k] + b[k + 1])

            def f(s, i, bb):
                inf = bb * s * i
                return -inf, inf - g * i
            s, i = S[k], I[k]
            k1 = f(s, i, b[k]); k2 = f(s + h / 2 * k1[0], i + h / 2 * k1[1], bm)
            k3 = f(s + h / 2 * k2[0], i + h / 2 * k2[1], bm)
            k4 = f(s + h * k3[0], i + h * k3[1], b[k + 1])
            S[k + 1] = max(s + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 0.0)
            I[k + 1] = max(i + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 0.0)
        return S, I, b * S * I, b

    def _avg(self, H, t0, t1):
        out = np.empty(len(t0))
        for r, (a, c) in enumerate(zip(t0, t1)):
            k0 = int(np.floor(a / self.dt + 1e-9)); k1 = max(int(np.ceil(c / self.dt - 1e-9)), k0 + 1)
            seg = H[k0:k1 + 1]
            wt = np.full(len(seg), 1.0); wt[0] = wt[-1] = 0.5
            out[r] = np.sum(wt * seg) / wt.sum()
        return out

    def fit(self, t0, t1, c, N, y_first, cum_first, max_nfev=300, k=None):
        I0 = max(y_first / GAMMA_SIR, 1e-9)
        S0 = max(1.0 - I0 - cum_first, 1e-6)
        self.S0, self.I0 = S0, I0
        Ndt = N * (t1 - t0)
        sc = np.sqrt(max(len(c), 1))

        def resid(th):
            _, _, H, _ = self.solve(th, S0, I0)
            lam = np.maximum(self._avg(H, t0, t1) * Ndt, 1e-10)
            return _dev(c, lam, k) / sc

        # seed: best constant beta on a coarse grid
        best = None
        for b0 in np.geomspace(0.03, 1.5, 25):
            th = np.zeros(self.X.shape[1]); th[0] = np.log(np.expm1(b0))
            v = float(np.sum(resid(th) ** 2))
            if best is None or v < best[0]:
                best = (v, th)
        sol = least_squares(resid, best[1], method="trf", x_scale="jac",
                            diff_step=1e-4, max_nfev=max_nfev)
        self.theta = sol.x
        self.S, self.I, self.H, self.b = self.solve(sol.x, S0, I0)
        self.R = self.b * self.S / GAMMA_SIR
        return self

    def predict(self, t0, t1):
        return self._avg(self.H, t0, t1)


def nb_k_by_governorate(panel):
    """(n_obs,) negative-binomial size: one k per governorate, by method of
    moments around that governorate's penalised-spline Poisson fit."""
    from fit import nb_size
    k = np.empty(panel.n_obs)
    for g in range(panel.ng):
        m = panel.gov == g
        N = float(panel.pop[g])
        sp = SplineBaseline().fit(panel.t0[m], panel.t1[m], panel.dC[m], N)
        mu = sp.predict(panel.t0[m], panel.t1[m]) * N * (panel.t1[m] - panel.t0[m])
        k[m] = nb_size(panel.dC[m], mu)
    return k
