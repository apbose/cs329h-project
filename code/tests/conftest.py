"""Shared fixtures: small synthetic questionnaires, so tests never download data."""

import numpy as np
import pytest

from cs329h.data import ChoiceData


def _dataset(x, p, lists, name):
    q = len(lists)
    return ChoiceData(
        name=name,
        Y=np.zeros((1, q), int),
        x=np.array(x, float),
        p=np.array(p, float),
        scale=np.ones(q),
        lists=np.array(lists),
        list_names=[f"list {l}" for l in sorted(set(lists))],
        person_ids=["p0"],
        rating=np.array([5.0]),
    )


@pytest.fixture
def template() -> ChoiceData:
    """Four price lists of six rows: a 50/50 lottery (1 or 0) against a rising sure amount."""
    sure = np.linspace(0.1, 0.9, 6)
    x, p, lists = [], [], []
    for l in range(4):
        for c in sure:
            x.append([[1.0, 0.0], [c, c]])
            p.append([[0.5, 0.5], [0.5, 0.5]])
            lists.append(l)
    return _dataset(x, p, lists, "toy")


@pytest.fixture
def mixed_template() -> ChoiceData:
    """Five lists over gains, losses and mixed gambles, so all three M2 parameters matter."""
    x, p, lists = [], [], []
    for c in np.linspace(0.1, 0.9, 8):  # gains, 50/50
        x.append([[1.0, 0.0], [c, c]]); p.append([[0.5, 0.5], [0.5, 0.5]]); lists.append(0)
    for c in np.linspace(0.02, 0.5, 8):  # gains, 10/90 long shot
        x.append([[1.0, 0.0], [c, c]]); p.append([[0.1, 0.9], [0.5, 0.5]]); lists.append(1)
    for c in np.linspace(-0.9, -0.1, 8):  # losses, 50/50
        x.append([[0.0, -1.0], [c, c]]); p.append([[0.5, 0.5], [0.5, 0.5]]); lists.append(2)
    for g in np.linspace(0.2, 1.0, 8):  # mixed: win g or lose 0.4, against nothing
        x.append([[g, -0.4], [0.0, 0.0]]); p.append([[0.5, 0.5], [0.5, 0.5]]); lists.append(3)
    for c in np.linspace(0.3, 0.98, 8):  # gains, 90/10
        x.append([[1.0, 0.0], [c, c]]); p.append([[0.9, 0.1], [0.5, 0.5]]); lists.append(4)
    return _dataset(x, p, lists, "toy_mixed")
