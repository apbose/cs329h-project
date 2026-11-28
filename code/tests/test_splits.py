"""Folds and held-out lists."""

import itertools
from collections import Counter

import numpy as np

from cs329h.splits import make_splits


def test_folds_are_balanced_and_deterministic():
    a = make_splits(1000, 7)
    b = make_splits(1000, 7)
    assert (a.fold == b.fold).all() and (a.heldout_lists == b.heldout_lists).all()
    assert set(np.bincount(a.fold)) == {200}


def test_every_pair_of_lists_is_used_about_equally():
    s = make_splits(1000, 7)
    counts = Counter(map(tuple, s.heldout_lists))
    assert set(counts) == set(itertools.combinations(range(7), 2))
    assert max(counts.values()) - min(counts.values()) <= 1


def test_heldout_mask_covers_two_whole_lists():
    lists = np.repeat(np.arange(4), 6)
    s = make_splits(10, 4)
    mask = s.heldout_mask(lists)
    assert (mask.sum(1) == 12).all()
    for row, (a, b) in zip(mask, s.heldout_lists):
        assert set(lists[row]) == {a, b}
