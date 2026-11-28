"""Write the fixed evaluation splits of the two lottery datasets to ``splits/``.

Five person folds and two held-out price lists per person (see ``cs329h.splits``). The split
depends only on the number of people and lists and on the seed, never on the answers. Run once;
the committed files are what every analysis reads.

    python scripts/make_splits.py
"""

from cs329h.data import load_basel_berlin, load_twin
from cs329h.splits import make_splits, save_splits


def main() -> None:
    """Make and save the splits of both lottery datasets."""
    for data in (load_basel_berlin(), load_twin()):
        path = save_splits(data, make_splits(data.n_people, data.n_lists))
        print(f"{data.name}: {data.n_people} people, {data.n_lists} lists -> {path}")


if __name__ == "__main__":
    main()
