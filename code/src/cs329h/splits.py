"""Fixed evaluation splits: five person folds, and two held-out price lists per person.

Each test person is evaluated on two whole price lists that are never asked about, so a
held-out choice cannot be read off its neighbor in the same list. Which two lists are held out
rotates over people: each person is assigned one of the ``L * (L - 1) / 2`` pairs of lists, in
a seeded random order that cycles through every pair before repeating one.

:func:`make_splits` is deterministic in ``seed``. For the real datasets the result is written
once to ``splits/<dataset>.json`` (committed) by ``scripts/make_splits.py``, and
:func:`load_splits` reads it back, so the folds stay fixed even if this code changes.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import ChoiceData

SPLIT_DIR = Path(__file__).resolve().parents[2] / "splits"
SEED = 329
N_FOLDS = 5


@dataclass
class Splits:
    """Fold and held-out lists for every person of one dataset.

    Attributes:
        fold: ``(n,)`` fold index ``0 .. n_folds - 1``.
        heldout_lists: ``(n, 2)`` the two price lists held out for each person.
    """

    fold: np.ndarray
    heldout_lists: np.ndarray

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return int(self.fold.max()) + 1

    def heldout_mask(self, lists: np.ndarray) -> np.ndarray:
        """``(n, Q)`` boolean mask of each person's held-out questions."""
        return (lists[None, :] == self.heldout_lists[:, :1]) | (lists[None, :] == self.heldout_lists[:, 1:])


def make_splits(n_people: int, n_lists: int, n_folds: int = N_FOLDS, seed: int = SEED) -> Splits:
    """Assign people to folds of near-equal size and to rotating pairs of held-out lists."""
    rng = np.random.default_rng(seed)
    fold = rng.permutation(n_people) % n_folds
    pairs = np.array(list(itertools.combinations(range(n_lists), 2)))
    order = rng.permutation(n_people)
    pair_of = np.empty(n_people, int)
    pair_of[order] = np.arange(n_people) % len(pairs)
    pair_perm = rng.permutation(len(pairs))
    return Splits(fold=fold, heldout_lists=pairs[pair_perm[pair_of]])


def save_splits(data: ChoiceData, splits: Splits, folder: Path = SPLIT_DIR) -> Path:
    """Write the splits of ``data`` keyed by person id."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{data.name}.json"
    rows = [
        f'  {json.dumps(pid)}: {{"fold": {int(f)}, "heldout_lists": [{int(a)}, {int(b)}]}}'
        for pid, f, (a, b) in zip(data.person_ids, splits.fold, splits.heldout_lists)
    ]
    head = f'{{"dataset": {json.dumps(data.name)}, "lists": {json.dumps(data.list_names)}, "people": {{\n'
    path.write_text(head + ",\n".join(rows) + "\n}}\n")
    return path


def load_splits(data: ChoiceData, folder: Path = SPLIT_DIR) -> Splits:
    """Read the committed splits for ``data``, in the order of ``data.person_ids``."""
    saved = json.loads((folder / f"{data.name}.json").read_text())["people"]
    missing = [pid for pid in data.person_ids if pid not in saved]
    if missing:
        raise KeyError(f"{len(missing)} people have no saved split, e.g. {missing[:3]}")
    fold = np.array([saved[pid]["fold"] for pid in data.person_ids])
    held = np.array([saved[pid]["heldout_lists"] for pid in data.person_ids])
    return Splits(fold=fold, heldout_lists=held)
