"""Which price lists the adaptive questions come from (exploratory diagnostic for H2).

For every fold, the factor model M3 is fitted (prior without the self-rating) on the training
people, and each test person is asked ten adaptive questions from the lists that are not held out.
Reports how many distinct price lists the first five questions cover, on average, and how often
each list appears among them, next to the lists of the first five questions of the best fixed
order.

    python scripts/run_list_coverage.py --data twin2k500

Writes results/<data>/list_coverage.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
K = 5  # questions whose lists are counted
K_MAX = 10


def coverage(data_name: str) -> dict:
    """Distinct lists among each test person's first ``K`` adaptive questions, over all folds."""
    from cs329h import select
    from cs329h.data import load_basel_berlin, load_twin
    from cs329h.models import make_model
    from cs329h.splits import load_splits

    data = {"basel_berlin": load_basel_berlin, "twin2k500": load_twin}[data_name]()
    splits = load_splits(data)
    held = splits.heldout_mask(data.lists)
    fixed = json.loads((RESULTS / data_name / "curves_M3.json").read_text())["best_fixed_order"]["none"]
    distinct, counts, fixed_lists = [], np.zeros(int(data.lists.max()) + 1, int), []
    for f in range(splits.n_folds):
        tr, te = np.flatnonzero(splits.fold != f), np.flatnonzero(splits.fold == f)
        model = make_model("M3", False)
        model.fit(data, tr)
        belief = model.take(model.init_belief(None, data.n_people), te)
        asked = np.asarray(select.adaptive_order(model, belief, data.Y[te], ~held[te], K_MAX))
        lists = data.lists[asked[:, :K]]  # (test people, K) list index of each question asked
        distinct += [len(set(row)) for row in lists]
        counts += np.bincount(lists.ravel(), minlength=len(counts))
        fixed_lists.append(data.lists[fixed[f][:K]].tolist())
        print(f"{data_name} fold {f}: mean distinct lists {np.mean(distinct):.2f}", flush=True)
    return {"data": data_name, "model": "M3", "prior": "none", "k": K,
            "mean_distinct_lists": float(np.mean(distinct)),
            "list_counts": counts.tolist(), "best_fixed_lists_by_fold": fixed_lists}


def main() -> None:
    """Command-line entry point: run one dataset and write its JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=("basel_berlin", "twin2k500"))
    args = ap.parse_args()
    out = coverage(args.data)
    path = RESULTS / args.data / "list_coverage.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
