"""Pieces shared by every choice model, and the support-point belief.

Every model gives the probability that person ``i`` chooses the riskier option of question
``q``, mixed with a lapse rate ``eps``::

    P(y_iq = 1) = eps / 2 + (1 - eps) * sigmoid(s_iq)

and a population prior over each person's parameters whose mean depends on the 0--10
self-rating when the rating is used. Selection and evaluation talk to a model only through its
*belief* about a set of people:

* ``init_belief(rating, n)`` -- the prior for ``n`` new people (``rating=None`` ignores it);
* ``update(belief, q, y)`` -- add person ``i``'s answer ``y[i]`` to question ``q[i]``;
* ``predict(belief)`` -- ``(n, Q)`` posterior predictive probability of the riskier option;
* ``eig(belief)`` -- ``(n, Q)`` expected information gain of each question;
* ``take(belief, rows)`` / ``put(belief, rows, part)`` -- select or replace some people;
* ``point_estimate(belief)`` -- ``(n, d)`` posterior mean of the person parameters;
* ``support_probs(belief)`` -- posterior weights over points and the choice probabilities at
  them, for rules that need the joint distribution of several answers.

:class:`SupportModel` implements these for models whose belief is a set of weights over
support points shared by everyone (a grid for M1, importance samples for M2).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

RATING_CENTER = 5.0
RATING_SCALE = 5.0


def rating_z(rating: np.ndarray | None, n: int) -> np.ndarray:
    """Center and scale a 0--10 rating to about [-1, 1]; zeros when there is no rating."""
    if rating is None:
        return np.zeros(n)
    return (np.asarray(rating, float) - RATING_CENTER) / RATING_SCALE


def lapse_prob(score: np.ndarray, eps: float) -> np.ndarray:
    """``eps / 2 + (1 - eps) * sigmoid(score)``, clipped away from 0 and 1."""
    return np.clip(eps / 2 + (1 - eps) * expit(score), 1e-9, 1 - 1e-9)


def entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def normalize(logw: np.ndarray) -> np.ndarray:
    """Weights from log weights along the last axis."""
    return np.exp(logw - logsumexp(logw, axis=-1, keepdims=True))


def person_loglik(Y: np.ndarray, P: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``(n, S)`` log likelihood of each person's answers at each support point.

    Args:
        Y: ``(n, Q)`` answers.
        P: ``(S, Q)`` probability of the riskier option at each support point.
        mask: Optional ``(n, Q)`` boolean array of the answers to count.
    """
    W1 = Y if mask is None else Y * mask
    W0 = (1 - Y) if mask is None else (1 - Y) * mask
    return W1 @ np.log(P).T + W0 @ np.log1p(-P).T


class SupportModel:
    """Belief = ``(n, S)`` log weights over support points shared by all people.

    Subclasses set ``self.support_values`` (``(S, d)`` person parameters at each point) and
    ``self._P`` (``(S, Q)`` choice probabilities) when fitted, and define ``log_prior``.
    """

    support_values: np.ndarray
    _P: np.ndarray

    def choice_prob(self) -> np.ndarray:
        """``(S, Q)`` probability of the riskier option at each support point."""
        return self._P

    def log_prior(self, rating: np.ndarray | None, n: int) -> np.ndarray:
        """``(n, S)`` log prior weights over the support points for people with these ratings."""
        raise NotImplementedError

    def init_belief(self, rating: np.ndarray | None, n: int) -> np.ndarray:
        """Belief ``(n, S)`` for ``n`` new people before any answer: the log prior weights."""
        return self.log_prior(rating, n)

    def update(self, belief: np.ndarray, q: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Add answer ``y[i]`` to question ``q[i]`` for each person ``i`` (Bayes' rule on the log weights)."""
        pq = self._P[:, q].T  # (n, S)
        return belief + np.where(np.asarray(y)[:, None] == 1, np.log(pq), np.log1p(-pq))

    def predict(self, belief: np.ndarray) -> np.ndarray:
        """``(n, Q)`` posterior predictive probability of the riskier option."""
        return normalize(belief) @ self._P

    def eig(self, belief: np.ndarray) -> np.ndarray:
        """``(n, Q)`` expected information gain of each question.

        Entropy of the predictive answer minus the posterior mean of its entropy at the support points.
        """
        w = normalize(belief)
        return entropy(w @ self._P) - w @ entropy(self._P)

    @staticmethod
    def take(belief: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Belief of the people in ``rows``."""
        return belief[rows]

    @staticmethod
    def put(belief: np.ndarray, rows: np.ndarray, part: np.ndarray) -> np.ndarray:
        """Copy of ``belief`` with the people in ``rows`` replaced by ``part``."""
        belief = belief.copy()
        belief[rows] = part
        return belief

    def point_estimate(self, belief: np.ndarray) -> np.ndarray:
        """``(n, d)`` posterior mean of the person parameters."""
        return normalize(belief) @ self.support_values

    def support_probs(self, belief: np.ndarray):
        """Posterior weights ``(n, S)`` and choice probabilities ``(S, Q)`` at the support points."""
        return normalize(belief), self._P
