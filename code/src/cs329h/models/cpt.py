"""M2: cumulative prospect theory with logit noise, person parameters by importance sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from ..data import ChoiceData
from .base import SupportModel, lapse_prob, normalize, person_loglik, rating_z

# Person parameters are theta = (log alpha, log lambda, log gamma); values are clipped to these
# ranges before use (gamma above 0.28 keeps the weighting function increasing).
ALPHA_RANGE = (0.05, 3.0)
LAMBDA_RANGE = (0.1, 10.0)
GAMMA_RANGE = (0.28, 2.5)


def weight(p: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Tversky--Kahneman (1992) weighting ``p^g / (p^g + (1 - p)^g)^(1 / g)``."""
    p = np.clip(p, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = p**gamma, (1 - p) ** gamma
        w = a / (a + b) ** (1 / gamma)
    return np.where(p <= 0, 0.0, np.where(p >= 1, 1.0, w))


def cpt_value(x: np.ndarray, p: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """``(S, Q, 2)`` prospect-theory value of each option of each question at each ``theta``.

    Args:
        x: ``(Q, 2, 2)`` outcomes (question, option, outcome), scaled to [-1, 1].
        p: ``(Q, 2, 2)`` their probabilities.
        theta: ``(S, 3)`` rows of ``(log alpha, log lambda, log gamma)``.

    Value is ``x^alpha`` for gains and ``-lambda (-x)^alpha`` for losses. Two-outcome
    prospects are valued rank-dependently: over gains, ``w(p_hi) v(hi) + (1 - w(p_hi)) v(lo)``;
    over losses, ``w(p_lo) v(lo) + (1 - w(p_lo)) v(hi)``; mixed, ``w(p_gain) v(gain) +
    w(p_loss) v(loss)``.
    """
    th = np.atleast_2d(theta)
    alpha = np.clip(np.exp(th[:, 0]), *ALPHA_RANGE)[:, None, None]
    lam = np.clip(np.exp(th[:, 1]), *LAMBDA_RANGE)[:, None, None]
    gamma = np.clip(np.exp(th[:, 2]), *GAMMA_RANGE)[:, None, None]
    first_hi = x[..., 0] >= x[..., 1]
    hi = np.where(first_hi, x[..., 0], x[..., 1])  # (Q, 2)
    lo = np.where(first_hi, x[..., 1], x[..., 0])
    p_hi = np.where(first_hi, p[..., 0], p[..., 1])

    def v(o):
        """``(S, Q, 2)`` value of outcomes ``o``: ``|o|^alpha``, times ``-lambda`` for losses."""
        mag = np.abs(o)[None] ** alpha
        return np.where(o[None] >= 0, mag, -lam * mag)

    v_hi, v_lo = v(hi), v(lo)
    w_hi, w_lo = weight(p_hi[None], gamma), weight(1 - p_hi[None], gamma)
    gains = w_hi * v_hi + (1 - w_hi) * v_lo
    losses = w_lo * v_lo + (1 - w_lo) * v_hi
    mixed = w_hi * v_hi + w_lo * v_lo
    return np.where(lo[None] >= 0, gains, np.where(hi[None] <= 0, losses, mixed))


@dataclass
class CPTParams:
    """Population parameters of :class:`ProspectTheory` (vectors over the three parameters)."""

    tau: float
    eps: float
    mu0: np.ndarray
    mu1: np.ndarray
    sigma: np.ndarray

    def as_dict(self) -> dict:
        """The parameters as plain Python values, for JSON."""
        return {"tau": float(self.tau), "eps": float(self.eps), "mu0": self.mu0.tolist(),
                "mu1": self.mu1.tolist(), "sigma": self.sigma.tolist()}


class ProspectTheory(SupportModel):
    """M2: cumulative prospect theory (Tversky and Kahneman 1992) with logit noise.

    The score is ``s_iq = [V(risky; theta_i) - V(safe; theta_i)] / tau`` with person-level
    curvature ``alpha_i``, loss aversion ``lambda_i`` and probability weighting ``gamma_i``
    (one ``gamma`` for gains and losses). The population prior on
    ``theta_i = (log alpha_i, log lambda_i, log gamma_i)`` is normal with mean
    ``mu0 + mu1 * z_i`` and a diagonal covariance, ``z_i`` the centered self-rating.

    **Posterior.** Every person shares one set of ``n_samples`` points drawn from a broad normal
    proposal ``q``; a person's belief is self-normalized importance weights
    ``prior_i(theta_s) / q(theta_s) * likelihood``. Expected information gain and predictions
    are then exact weighted sums over the shared points, as for M1's grid.

    **Fitting.** Monte Carlo EM on the training people. Each round draws a fresh sample from a
    proposal centered on the current population (standard deviation 1.5 times the population
    spread, at least 0.15), fits ``tau`` and ``eps`` by L-BFGS with the prior held fixed, then
    updates the prior in closed form: ``(mu0, mu1)`` by least squares of the people's posterior
    means on ``[1, z]`` and ``sigma^2`` as the mean posterior variance plus residual variance.
    The last round's sample is kept as the support. Like M1, M2 is fitted without a penalty.

    Args:
        use_rating: Let the prior mean depend on the self-rating.
        n_samples: Number of shared importance samples.
        n_rounds: Rounds of Monte Carlo EM.
        seed: Seed for the samples.
    """

    name = "M2"

    def __init__(self, use_rating: bool = True, n_samples: int = 3000, n_rounds: int = 5, seed: int = 0):
        self.use_rating = use_rating
        self.n_samples = n_samples
        self.n_rounds = n_rounds
        self.seed = seed
        self.params: CPTParams | None = None

    def value_gap(self, theta: np.ndarray) -> np.ndarray:
        """``(S, Q)`` value of the riskier option minus the safer one."""
        v = cpt_value(self._x, self._p, theta)
        return v[..., 0] - v[..., 1]

    def choice_prob_people(self, theta: np.ndarray) -> np.ndarray:
        """``(n, Q)`` choice probabilities for people with parameters ``theta`` (``(n, 3)``)."""
        return lapse_prob(self.value_gap(theta) / self.params.tau, self.params.eps)

    def sample_people(self, rating: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 3)`` person parameters drawn from the population prior."""
        pr = self.params
        return self._prior_mean(rating_z(rating if self.use_rating else None, n), pr) + pr.sigma * rng.standard_normal((n, 3))

    @staticmethod
    def _prior_mean(z: np.ndarray, pr: CPTParams) -> np.ndarray:
        return pr.mu0[None] + z[:, None] * pr.mu1[None]

    def _log_prior_on(self, samples, log_q, z, pr) -> np.ndarray:
        mean = self._prior_mean(z, pr)  # (n, 3)
        d = (samples[None] - mean[:, None]) / pr.sigma  # (n, S, 3)
        lp = -0.5 * (d**2).sum(-1) - np.log(pr.sigma).sum()
        return lp - log_q[None]

    def log_prior(self, rating: np.ndarray | None, n: int | None = None) -> np.ndarray:
        """``(n, S)`` log importance weights before any answer."""
        n = len(rating) if rating is not None else n
        z = rating_z(rating if self.use_rating else None, n)
        return self._log_prior_on(self.support_values, self._log_q, z, self.params)

    def bind(self, data: ChoiceData) -> "ProspectTheory":
        """Attach the questions of ``data`` and, when fitted, the choice probabilities at the support."""
        self._x, self._p = data.x, data.p
        if self.params is not None:
            self._P = self.choice_prob_people(self.support_values)
        return self

    def fit(self, data: ChoiceData, rows: np.ndarray | None = None) -> "ProspectTheory":
        """Fit the population by Monte Carlo EM on ``rows`` of ``data``."""
        self.params = None
        self.bind(data)
        train = data if rows is None else data.subset(rows)
        Y = train.Y.astype(float)
        n = len(Y)
        z = rating_z(train.rating if self.use_rating else None, n)
        X = np.column_stack([np.ones(n), z]) if self.use_rating else np.ones((n, 1))
        rng = np.random.default_rng(self.seed)
        pr = CPTParams(tau=0.05, eps=0.05, mu0=np.array([-0.2, np.log(1.5), np.log(0.7)]),
                       mu1=np.zeros(3), sigma=np.array([0.5, 0.5, 0.4]))
        prop_mean, prop_sd = pr.mu0.copy(), np.full(3, 1.0)
        for rnd in range(self.n_rounds):
            samples = prop_mean + prop_sd * rng.standard_normal((self.n_samples, 3))
            log_q = (-0.5 * ((samples - prop_mean) / prop_sd) ** 2).sum(1) - np.log(prop_sd).sum()
            gap = self.value_gap(samples)
            lp = self._log_prior_on(samples, log_q, z, pr)

            def nll(v, lp=lp, gap=gap):
                """Negative log marginal likelihood per person at ``v = (log tau, logit eps)``, up to a constant.

                Estimated by importance sampling on this round's samples, with the prior held fixed.
                """
                ll = person_loglik(Y, lapse_prob(gap / np.exp(v[0]), expit(v[1])))
                return -float(logsumexp(ll + lp, axis=1).sum()) / n

            v0 = np.array([np.log(pr.tau), np.log(pr.eps / (1 - pr.eps))])
            if rnd == 0:
                starts = [np.array([np.log(t), np.log(e / (1 - e))]) for t in (0.02, 0.05, 0.15, 0.4) for e in (0.03, 0.15)]
                v0 = min(starts, key=nll)
            res = minimize(nll, v0, method="L-BFGS-B", bounds=[(np.log(1e-3), np.log(5.0)), (-8.0, 4.0)])
            pr.tau, pr.eps = float(np.exp(res.x[0])), float(expit(res.x[1]))

            ll = person_loglik(Y, lapse_prob(gap / pr.tau, pr.eps))
            for _ in range(10):  # closed-form prior updates with the likelihood held fixed
                w = normalize(ll + self._log_prior_on(samples, log_q, z, pr))
                m = w @ samples
                var = w @ samples**2 - m**2
                coef, *_ = np.linalg.lstsq(X, m, rcond=None)
                resid = m - X @ coef
                pr.mu0 = coef[0]
                pr.mu1 = coef[1] if self.use_rating else np.zeros(3)
                pr.sigma = np.sqrt(np.maximum((var + resid**2).mean(0), 1e-4))
            self.support_values, self._log_q = samples, log_q
            spread = np.sqrt(pr.sigma**2 + pr.mu1**2 * z.var())
            prop_mean, prop_sd = pr.mu0 + pr.mu1 * z.mean(), np.maximum(1.5 * spread, 0.15)
        self.params = pr
        return self.bind(data)

    def set_params(self, params: CPTParams, data: ChoiceData, seed: int = 0) -> "ProspectTheory":
        """Use given population parameters, with a support drawn from a proposal around them."""
        self.params = params
        rng = np.random.default_rng(seed)
        prop_mean, prop_sd = params.mu0, np.maximum(1.5 * np.sqrt(params.sigma**2 + params.mu1**2 * 0.25), 0.15)
        s = prop_mean + prop_sd * rng.standard_normal((self.n_samples, 3))
        self.support_values = s
        self._log_q = (-0.5 * ((s - prop_mean) / prop_sd) ** 2).sum(1) - np.log(prop_sd).sum()
        return self.bind(data)
