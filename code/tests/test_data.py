"""Loader helpers that run without downloads."""

import numpy as np

from cs329h.data import _twin_amount


def test_twin_amounts_parse_signs_and_cents():
    assert _twin_amount("Lottery:$0.50".split(":")[1]) == 0.5
    assert _twin_amount("Lottery:-$5.50".split(":")[1]) == -5.5
    assert _twin_amount("Lottery with x=$7.00") == 7.0


def test_subset_keeps_questions_and_selects_people(template):
    data = template
    data.Y = np.arange(3 * data.n_questions).reshape(3, -1) % 2
    data.person_ids = ["a", "b", "c"]
    data.rating = np.array([1.0, 2.0, 3.0])
    sub = data.subset(np.array([True, False, True]))
    assert sub.person_ids == ["a", "c"]
    assert sub.rating.tolist() == [1.0, 3.0]
    assert sub.n_questions == data.n_questions
