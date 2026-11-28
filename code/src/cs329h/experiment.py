"""The evaluation loop: five person folds, two held-out lists, ``k = 0 .. 10``.

For every fold the model is fitted on the training people only. Each test person then gets
``k`` questions from their non-held-out lists under each selection rule, and the posterior
after each answer predicts their held-out choices. M3 and M4 choose their hyperparameters on
validation people drawn from the training people. The prior either ignores the self-rating
(``"none"``) or centers on it (``"rating"``, when the dataset has one).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import select
from .data import ChoiceData
from .evaluate import bootstrap_draws, calibration, interval
from .splits import Splits

PLANNED_RULES = ("adaptive", "best_fixed", "random", "original")
EXPLORATORY_RULES = ("predictive",)
RULES = PLANNED_RULES + EXPLORATORY_RULES
CALIBRATION_K = (0, 5, 10)


def run_curves(data: ChoiceData, splits: Splits, make_model: Callable[[bool], object], k_max: int = 10,
               seed: int = 329, priors: tuple[str, ...] | None = None, rules: tuple[str, ...] = RULES,
               train_means: bool = False, train_rows: Callable[[np.ndarray, int], np.ndarray] | None = None) -> dict:
    """Held-out loss and accuracy after ``k`` answers, for every prior and selection rule.

    Args:
        data: The dataset.
        splits: Folds and held-out lists for its people.
        make_model: ``make_model(use_rating)`` returns an unfitted model.
        k_max: Largest number of questions asked.
        seed: Seed for the random-order rule.
        priors: Which priors to run; default ``("none", "rating")`` when the data has a rating.
        rules: Which selection rules to run.
        train_means: Also ask each fold's training people adaptively and keep their posterior
            means, for outcome models fitted on training people (``train_post_mean[prior]``,
            one ``(rows, means)`` pair per fold, ``means`` shaped ``(k_max + 1, n_train, d)``).
        train_rows: ``train_rows(rows, fold)`` picks the people a fold's model is fitted on
            (default: all training people); for learning curves over training size.

    Returns:
        A dict with per-person arrays ``loss[prior][rule]`` and ``acc[prior][rule]``
        (``(n, k_max + 1)``), ``post_mean[prior]`` (posterior mean of the person parameters
        under the first rule, adaptive by default, ``(n, k_max + 1, d)``, ``nan``-padded when ``d``
        differs by fold), pooled held-out predictions for calibration, the fitted parameters per
        fold, and the best fixed order per fold.
    """
    if priors is None:
        priors = ("none", "rating") if data.rating is not None else ("none",)
    n, Q = data.Y.shape
    K = k_max + 1
    rng = np.random.default_rng(seed)
    heldout = splits.heldout_mask(data.lists)
    out = {
        "loss": {pr: {r: np.zeros((n, K)) for r in rules} for pr in priors},
        "acc": {pr: {r: np.zeros((n, K)) for r in rules} for pr in priors},
        "post_mean": {pr: None for pr in priors},
        "calib_pred": {pr: {r: {k: [] for k in CALIBRATION_K} for r in rules} for pr in priors},
        "calib_y": {pr: {r: {k: [] for k in CALIBRATION_K} for r in rules} for pr in priors},
        "fits": {pr: [] for pr in priors},
        "best_fixed_order": {pr: [] for pr in priors},
        "train_post_mean": {pr: [] for pr in priors},
    }
    for f in range(splits.n_folds):
        tr = np.flatnonzero(splits.fold != f)
        te = np.flatnonzero(splits.fold == f)
        random_asked = select.random_orders(~heldout[te], k_max, rng)
        for pr in priors:
            fit_rows = tr if train_rows is None else train_rows(tr, f)
            model = make_model(pr == "rating").fit(data, fit_rows)
            out["fits"][pr].append(model.params.as_dict())
            rating = data.rating if pr == "rating" else None
            belief = model.init_belief(rating, n)
            Y = data.Y.astype(float)
            held, avail = heldout[te], ~heldout[te]
            asked = {}
            if "best_fixed" in rules:
                order = select.greedy_fixed_order(model, model.take(belief, fit_rows), Y[fit_rows], heldout[fit_rows], k_max)
                out["best_fixed_order"][pr].append(order.tolist())
                asked["best_fixed"] = select.first_available(order, avail, k_max)
            b_te = model.take(belief, te)
            if "adaptive" in rules:
                asked["adaptive"] = select.adaptive_order(model, b_te, Y[te], avail, k_max)
            if "predictive" in rules:
                asked["predictive"] = select.predictive_order(model, b_te, Y[te], avail, k_max)
            if "random" in rules:
                asked["random"] = random_asked
            if "original" in rules:
                asked["original"] = select.original_orders(avail, k_max)
            if train_means:
                b_tr, held_tr = model.take(belief, fit_rows), heldout[fit_rows]
                a_tr = select.adaptive_order(model, b_tr, Y[fit_rows], ~held_tr, k_max)
                means_tr = select.replay_curve(model, b_tr, Y[fit_rows], a_tr, held_tr)[3]
                out["train_post_mean"][pr].append((fit_rows, means_tr))
            for rule in rules:
                loss, acc, preds, means = select.replay_curve(model, b_te, Y[te], asked[rule], held)
                out["loss"][pr][rule][te] = loss
                out["acc"][pr][rule][te] = acc
                for k in CALIBRATION_K:
                    if k > k_max:
                        continue
                    out["calib_pred"][pr][rule][k].append(preds[k][held])
                    out["calib_y"][pr][rule][k].append(Y[te][held])
                if rule == rules[0]:
                    pm, dim = out["post_mean"][pr], means.shape[-1]
                    if pm is None or pm.shape[-1] < dim:
                        grown = np.full((n, K, dim), np.nan)
                        if pm is not None:
                            grown[..., :pm.shape[-1]] = pm
                        out["post_mean"][pr] = pm = grown
                    pm[te, :, :dim] = means.transpose(1, 0, 2)
    return out


def summarize(curves: dict, seed: int = 330, n_boot: int = 2000) -> dict:
    """Means and paired 95% bootstrap intervals over test people, in JSON-ready form.

    ``loss``/``acc``: ``[mean, lower, upper]`` per ``k`` for each prior and rule.
    ``adaptive_minus_best_fixed``: paired loss difference (negative favors adaptive).
    ``rating_gain``: loss without the rating minus loss with it (positive means it helps).
    ``calibration``: expected calibration error of held-out predictions at ``k = 0, 5, 10``.
    """
    priors = list(curves["loss"])
    n = next(iter(curves["loss"][priors[0]].values())).shape[0]
    draws = bootstrap_draws(n, np.random.default_rng(seed), n_boot)
    rules = list(curves["loss"][priors[0]])
    s = {"n_people": int(n), "loss": {}, "acc": {}, "adaptive_minus_best_fixed": {}, "calibration": {}}
    for pr in priors:
        s["loss"][pr] = {r: interval(curves["loss"][pr][r], draws) for r in rules}
        s["acc"][pr] = {r: interval(curves["acc"][pr][r], draws) for r in rules}
        if {"adaptive", "best_fixed"} <= set(rules):
            diff = curves["loss"][pr]["adaptive"] - curves["loss"][pr]["best_fixed"]
            s["adaptive_minus_best_fixed"][pr] = interval(diff, draws)
        if {"adaptive", "predictive"} <= set(rules):
            s.setdefault("predictive_minus_adaptive", {})[pr] = interval(
                curves["loss"][pr]["predictive"] - curves["loss"][pr]["adaptive"], draws)
        s["calibration"][pr] = {
            r: {str(k): calibration(np.concatenate(curves["calib_y"][pr][r][k]),
                                    np.concatenate(curves["calib_pred"][pr][r][k]))["ece"]
                for k in CALIBRATION_K}
            for r in rules
        }
    if {"none", "rating"} <= set(priors):
        s["rating_gain"] = {
            r: interval(curves["loss"]["none"][r] - curves["loss"]["rating"][r], draws) for r in rules
        }
    s["fits"] = curves["fits"]
    s["best_fixed_order"] = curves["best_fixed_order"]
    return s
