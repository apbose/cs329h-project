"""M1: expected utility with logit noise and constant absolute risk aversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from ..data import ChoiceData
from .base import SupportModel, lapse_prob, person_loglik, rating_z


def cara_utility(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """CARA utility ``(1 - exp(-r x)) / r`` (``x`` when ``r = 0``), broadcast over ``r``.

    Args:
        x: Outcomes, any shape.
        r: Risk-aversion coefficients, shape ``(G,)``; the result has shape ``(G, *x.shape)``.
    """
    r = np.asarray(r, float).reshape((-1,) + (1,) * x.ndim)
    small = np.abs(r) < 1e-8
    safe_r = np.where(small, 1.0, r)
    with np.errstate(over="ignore"):
        u = -np.expm1(-safe_r * x[None]) / safe_r
    return np.where(small, x[None], u)


@dataclass
class EUParams:
    """Population parameters of :class:`ExpectedUtility`."""

    tau: float
    eps: float
    mu0: float
    mu1: float
    sigma: float

    def as_dict(self) -> dict[str, float]:
        """The parameters as plain floats, for JSON."""
        return {k: float(v) for k, v in self.__dict__.items()}


class ExpectedUtility(SupportModel):
    """M1: expected utility with logit noise and constant absolute risk aversion.

    The score is ``s_iq = [EU(risky; r_i) - EU(safe; r_i)] / tau`` with utility
    ``u(x) = (1 - exp(-r_i x)) / r_i`` on outcomes scaled to [-1, 1] within each price list,
    an affine transformation of ``-exp(-r_i x)`` that stays finite at ``r_i = 0``. Each person
    has one parameter ``r_i``, held on a grid. The population prior is a normal distribution on
    the grid with mean ``mu0 + mu1 * z_i`` and standard deviation ``sigma``, where ``z_i`` is the
    centered self-rating (``mu1 = 0`` when the rating is not used).

    The five population parameters ``(tau, eps, mu0, mu1, sigma)`` are fitted by maximizing the
    marginal likelihood of the training people's answers (the person parameter integrated out on
    the grid), without a penalty.

    Args:
        use_rating: Let the prior mean depend on the self-rating.
        grid: Support points for ``r``.
    """

    name = "M1"

    def __init__(self, use_rating: bool = True, grid: np.ndarray | None = None):
        self.use_rating = use_rating
        self.grid = np.linspace(-6.0, 10.0, 161) if grid is None else np.asarray(grid, float)
        self.support_values = self.grid[:, None]
        self.params: EUParams | None = None

    def utility_gap(self, r: np.ndarray | None = None) -> np.ndarray:
        """``(G, Q)`` expected utility of the riskier option minus the safer one."""
        r = self.grid if r is None else r
        eu = (cara_utility(self._x, r) * self._p[None]).sum(-1)  # (G, Q, 2)
        return eu[..., 0] - eu[..., 1]

    def choice_prob_people(self, theta: np.ndarray) -> np.ndarray:
        """``(n, Q)`` choice probabilities for people with parameters ``theta`` (``(n, 1)``)."""
        r = np.asarray(theta, float).reshape(-1)
        return lapse_prob(self.utility_gap(r) / self.params.tau, self.params.eps)

    def sample_people(self, rating: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 1)`` risk aversion drawn from the population prior."""
        z = rating_z(rating if self.use_rating else None, n)
        pr = self.params
        return (pr.mu0 + pr.mu1 * z + pr.sigma * rng.standard_normal(n))[:, None]

    def log_prior(self, rating: np.ndarray | None, n: int | None = None) -> np.ndarray:
        """``(n, G)`` log prior weights over the grid for people with these ratings."""
        n = len(rating) if rating is not None else n
        z = rating_z(rating if self.use_rating else None, n)
        return self._log_prior(z, self.params.mu0, self.params.mu1, self.params.sigma)

    def _log_prior(self, z: np.ndarray, mu0: float, mu1: float, sigma: float) -> np.ndarray:
        mean = mu0 + mu1 * z
        lp = -0.5 * ((self.grid[None, :] - mean[:, None]) / sigma) ** 2
        return lp - logsumexp(lp, axis=1, keepdims=True)

    def bind(self, data: ChoiceData) -> "ExpectedUtility":
        """Attach the questions of ``data``."""
        self._x, self._p = data.x, data.p
        self._gap = self.utility_gap()
        if self.params is not None:
            self._P = lapse_prob(self._gap / self.params.tau, self.params.eps)
        return self

    def fit(self, data: ChoiceData, rows: np.ndarray | None = None) -> "ExpectedUtility":
        """Fit the population parameters by maximum marginal likelihood on ``rows`` of ``data``."""
        self.params = None
        self.bind(data)
        train = data if rows is None else data.subset(rows)
        Y = train.Y.astype(float)
        z = rating_z(train.rating if self.use_rating else None, train.n_people)

        def unpack(v):
            """``(tau, eps, mu0, mu1, sigma)`` from the unconstrained vector ``v``."""
            tau, eps = np.exp(v[0]), expit(v[1])
            mu1 = v[3] if self.use_rating else 0.0
            return tau, eps, v[2], mu1, np.exp(v[4])

        def objective(v):
            """Negative log marginal likelihood per person, ``r`` summed out on the grid."""
            tau, eps, mu0, mu1, sigma = unpack(v)
            ll = person_loglik(Y, lapse_prob(self._gap / tau, eps))
            lp = self._log_prior(z, mu0, mu1, sigma)
            return -float(logsumexp(ll + lp, axis=1).sum()) / len(Y)

        # Coarse search for a starting point, then L-BFGS on the unconstrained parameters.
        starts = [
            np.array([np.log(t), np.log(e / (1 - e)), m, 0.0, np.log(s)])
            for t in (0.02, 0.05, 0.15)
            for e in (0.03, 0.15)
            for m in (0.0, 2.0)
            for s in (1.0, 3.0)
        ]
        v0 = min(starts, key=objective)
        bounds = [(np.log(1e-3), np.log(5.0)), (-8.0, 4.0), (-6.0, 10.0), (-10.0, 10.0), (np.log(0.15), np.log(10.0))]
        res = minimize(objective, v0, method="L-BFGS-B", bounds=bounds)
        self.params = EUParams(*unpack(res.x))
        return self.bind(data)

    def set_params(self, params: EUParams, data: ChoiceData) -> "ExpectedUtility":
        """Use given population parameters (e.g. the true ones in a simulation)."""
        self.params = params
        return self.bind(data)
