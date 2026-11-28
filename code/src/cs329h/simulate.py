"""Simulated investors with known parameters.

People are drawn from a fitted population and answer every question of a template dataset
(the same lotteries as the real data), so parameter recovery and the selection code can be
checked against the truth. Works with any model class: the model supplies
``sample_people(rating, n, rng)`` and ``choice_prob_people(theta)``.
"""

from __future__ import annotations

import numpy as np

from .data import ChoiceData


def simulate(model, template: ChoiceData, n: int, rng: np.random.Generator,
             rating_pool: np.ndarray | None = None, name: str | None = None) -> tuple[ChoiceData, np.ndarray]:
    """Draw ``n`` people from ``model``'s population and their answers to ``template``'s questions.

    Each person gets a self-rating resampled from ``rating_pool`` (when given), person
    parameters from the population prior given that rating, and an answer to every question
    drawn from the model's choice probabilities.

    Returns:
        The simulated dataset and the true person parameters, ``(n, d)``.
    """
    model.bind(template)
    rating = None if rating_pool is None else rng.choice(np.asarray(rating_pool), size=n, replace=True)
    theta = model.sample_people(rating, n, rng)
    P = model.choice_prob_people(theta)
    Y = (rng.random(P.shape) < P).astype(int)
    data = ChoiceData(
        name=name or f"sim_{model.name.lower()}",
        Y=Y,
        x=template.x,
        p=template.p,
        scale=template.scale,
        lists=template.lists,
        list_names=template.list_names,
        person_ids=[f"sim{i:05d}" for i in range(n)],
        rating=rating,
    )
    return data, theta
