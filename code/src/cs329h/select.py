"""Question selection and replay, for any model class.

A model holds a *belief* about a set of people (see :mod:`cs329h.models.base`); everything here
is vectorized over people and talks to the model only through ``update``, ``predict``,
``eig``, ``take``, ``put`` and ``point_estimate``.

Selection rules, each returning the ``k_max`` questions asked of every person:

* :func:`adaptive_order` -- before each question, ask the available one with the largest
  expected information gain about the person's parameters (Houlsby et al. 2011),
  ``EIG(q) = H[E_theta p(y | theta, q)] - E_theta H[p(y | theta, q)]``.
* :func:`predictive_order` -- (exploratory) ask the question
  whose answer is expected to tell most about the person's answers to the questions not yet
  asked: expected predictive information gain, EPIG (Bickford Smith et al. 2023), summed over
  those questions. It targets prediction, not the model's parameters.
* :func:`greedy_fixed_order` -- the best fixed order, built greedily on training people: each
  step appends the question that most lowers their mean held-out log loss.
* :func:`random_orders` -- a random order of the available questions, separately per person.
* :func:`original_orders` -- the questionnaire's own order.

"Available" means the question is not in one of the person's held-out price lists; every
person in these datasets answered every question, so selection replays their real answers.
"""

from __future__ import annotations

import numpy as np

from .models.base import entropy


def adaptive_order(model, belief0, Y: np.ndarray, available: np.ndarray, k_max: int) -> np.ndarray:
    """``(n, k_max)`` questions chosen by expected information gain, replaying answers ``Y``."""
    n = len(Y)
    belief = belief0
    avail = available.copy()
    asked = np.zeros((n, k_max), int)
    rows = np.arange(n)
    for k in range(k_max):
        gain = np.where(avail, model.eig(belief), -np.inf)
        q = gain.argmax(1)
        asked[:, k] = q
        avail[rows, q] = False
        belief = model.update(belief, q, Y[rows, q])
    return asked


def predictive_gain(w: np.ndarray, P: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``(Q,)`` EPIG of each question for one person, summed over the ``target`` questions.

    Args:
        w: ``(S,)`` posterior weights over support points.
        P: ``(S, Q)`` choice probabilities at those points.
        target: ``(Q,)`` boolean, the questions whose answers we want to predict.
    """
    pq = np.clip(w @ P, 1e-9, 1 - 1e-9)
    J = (P * w[:, None]).T @ P  # (Q, Q): probability that both answers are the riskier option
    p1 = np.clip(J / pq[:, None], 1e-9, 1 - 1e-9)  # P(y_t = 1 | y_q = 1)
    p0 = np.clip((pq[None, :] - J) / (1 - pq[:, None]), 1e-9, 1 - 1e-9)  # P(y_t = 1 | y_q = 0)
    info = entropy(pq)[None, :] - pq[:, None] * entropy(p1) - (1 - pq[:, None]) * entropy(p0)
    np.fill_diagonal(info, 0.0)
    return info[:, target].sum(1)


def predictive_order(model, belief0, Y: np.ndarray, available: np.ndarray, k_max: int) -> np.ndarray:
    """``(n, k_max)`` questions chosen by EPIG over the questions not yet asked.

    The targets are every question the person has not answered yet, held-out or not, because
    an adviser does not know which choices will be scored.
    """
    n, Q = Y.shape
    belief = belief0
    avail = available.copy()
    unasked = np.ones((n, Q), bool)
    asked = np.zeros((n, k_max), int)
    rows = np.arange(n)
    for k in range(k_max):
        w, P = model.support_probs(belief)
        for i in range(n):
            Pi = P if P.ndim == 2 else P[i]
            gain = predictive_gain(w[i], Pi, unasked[i])
            asked[i, k] = np.where(avail[i], gain, -np.inf).argmax()
        q = asked[:, k]
        avail[rows, q] = False
        unasked[rows, q] = False
        belief = model.update(belief, q, Y[rows, q])
    return asked


def first_available(order: np.ndarray, available: np.ndarray, k_max: int) -> np.ndarray:
    """``(n, k_max)``: for each person the first ``k_max`` questions of ``order`` they can be asked."""
    out = np.zeros((len(available), k_max), int)
    for i in range(len(available)):
        picks = [q for q in order if available[i, q]][:k_max]
        if len(picks) < k_max:
            raise ValueError("order too short for this person's available questions")
        out[i] = picks
    return out


def heldout_loss(pred: np.ndarray, Y: np.ndarray, heldout: np.ndarray) -> np.ndarray:
    """``(n,)`` mean held-out log loss per person."""
    pred = np.clip(pred, 1e-12, 1 - 1e-12)
    ll = Y * np.log(pred) + (1 - Y) * np.log1p(-pred)
    return -(ll * heldout).sum(1) / heldout.sum(1)


def greedy_fixed_order(model, belief0, Y: np.ndarray, heldout: np.ndarray, k_max: int) -> np.ndarray:
    """Best fixed order of questions, chosen greedily on training people.

    Each training person has their own held-out lists (``heldout``, ``(n, Q)``). A fixed order
    asks every person its first questions that are not held out for them. At each step the
    question appended is the one that most lowers the training people's mean held-out log loss
    (averaged over people), counting only people for whom the question is available and who
    have fewer than ``k_max`` answers so far. The order grows until every person has
    ``k_max`` questions.

    Returns:
        The order as an array of question indices (longer than ``k_max`` when some of its
        questions are held out for some people).
    """
    n, Q = Y.shape
    available = ~heldout
    belief = belief0
    count = np.zeros(n, int)
    order: list[int] = []
    current = heldout_loss(model.predict(belief), Y, heldout)
    while (count < k_max).any():
        active = count < k_max
        best = None
        for q in range(Q):
            if q in order:
                continue
            who = np.flatnonzero(active & available[:, q])
            if not len(who):
                continue
            part = model.update(model.take(belief, who), np.full(len(who), q), Y[who, q])
            new = current.copy()
            new[who] = heldout_loss(model.predict(part), Y[who], heldout[who])
            score = new.mean()
            if best is None or score < best[0]:
                best = (score, q, who, part, new)
        _, q, who, part, new = best
        order.append(q)
        belief = model.put(belief, who, part)
        current = new
        count[who] += 1
    return np.array(order)


def random_orders(available: np.ndarray, k_max: int, rng: np.random.Generator) -> np.ndarray:
    """``(n, k_max)`` random available questions per person, in random order."""
    out = np.zeros((len(available), k_max), int)
    for i, row in enumerate(available):
        out[i] = rng.permutation(np.flatnonzero(row))[:k_max]
    return out


def original_orders(available: np.ndarray, k_max: int) -> np.ndarray:
    """``(n, k_max)`` the first available questions in the questionnaire's own order."""
    return first_available(np.arange(available.shape[1]), available, k_max)


def replay_curve(model, belief0, Y: np.ndarray, asked: np.ndarray, heldout: np.ndarray):
    """Held-out predictions after ``k = 0 .. k_max`` answers.

    Returns:
        ``loss`` and ``acc``, each ``(n, k_max + 1)`` (mean held-out log loss and accuracy per
        person); ``preds``, ``(k_max + 1, n, Q)`` held-out predictive probabilities (``nan``
        outside the held-out questions), for calibration; and ``means``,
        ``(k_max + 1, n, d)`` the posterior mean of the person parameters.
    """
    n, k_max = asked.shape
    rows = np.arange(n)
    hm = heldout.astype(float)
    belief = belief0
    loss = np.zeros((n, k_max + 1))
    acc = np.zeros((n, k_max + 1))
    preds = np.full((k_max + 1, n, Y.shape[1]), np.nan)
    means = []
    for k in range(k_max + 1):
        pred = model.predict(belief)
        loss[:, k] = heldout_loss(pred, Y, hm)
        acc[:, k] = (((pred > 0.5) == (Y == 1)) * hm).sum(1) / hm.sum(1)
        preds[k][heldout] = pred[heldout]
        means.append(model.point_estimate(belief))
        if k < k_max:
            q = asked[:, k]
            belief = model.update(belief, q, Y[rows, q])
    return loss, acc, preds, np.array(means)
