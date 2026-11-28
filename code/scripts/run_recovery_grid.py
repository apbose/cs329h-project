"""The 4 x 4 grid of true by fitted model class (Figure 1, right; Table 4).

For each true class, the class is fitted (rating in the prior) to all Basel--Berlin people, and
as many simulated investors are drawn from it, each with a resampled Basel--Berlin self-rating and
answers to the same 66 questions. Every class is then fitted on the training folds of those
answers and evaluated on the test folds exactly as on real data (five folds, two held-out price
lists per person, rating in the prior), with adaptive selection, the best fixed order and a
random order.

Per cell it reports the held-out loss after k = 0, 5 and 10 answers, its excess over the loss
under the true model with each person's true parameters (the floor), and adaptive minus the best
fixed order, which shows whether adaptive selection loses its edge when the fitted class is not
the true one.

    python scripts/run_recovery_grid.py            # all rows, one process each; ~2 h on 4 cores
    python scripts/run_recovery_grid.py --true M2  # one row
    python scripts/run_recovery_grid.py --collect  # combine rows already written

Writes results/recovery_grid/<true>.json and, once all rows exist, results/recovery_grid.json.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

CLASSES = ("M1", "M2", "M3", "M4")
OUT = Path(__file__).resolve().parents[1] / "results"
SEED = 329
K_MAX = 10
RULES = ("adaptive", "best_fixed", "random")


def run_row(true_name: str) -> dict:
    """One row of the grid: simulate from true class ``true_name`` and evaluate every class on it.

    Returns the true parameters, the floor loss under the truth, and per fitted class the held-out
    loss and excess loss at ``k = 0, 5, 10`` for each rule and adaptive minus best fixed (paired).
    """
    from cs329h.data import load_basel_berlin
    from cs329h.evaluate import bootstrap_draws, interval
    from cs329h.experiment import run_curves
    from cs329h.models import make_model
    from cs329h.simulate import simulate
    from cs329h.splits import make_splits

    bb = load_basel_berlin()
    truth = make_model(true_name, True).fit(bb)
    rng = np.random.default_rng(SEED + CLASSES.index(true_name))
    sim, theta = simulate(truth, bb, bb.n_people, rng, rating_pool=bb.rating)
    splits = make_splits(sim.n_people, sim.n_lists, seed=SEED)
    held = splits.heldout_mask(sim.lists)
    P_true = truth.choice_prob_people(theta)
    Y = sim.Y.astype(float)
    ll = Y * np.log(P_true) + (1 - Y) * np.log1p(-P_true)
    oracle = -(ll * held).sum(1) / held.sum(1)
    draws = bootstrap_draws(sim.n_people, np.random.default_rng(SEED + 1))
    row = {"true": true_name, "true_params": truth.params.as_dict(), "oracle_loss": interval(oracle, draws), "fitted": {}}
    for fit_name in CLASSES:
        curves = run_curves(sim, splits, lambda ur, f=fit_name: make_model(f, ur), k_max=K_MAX, seed=SEED,
                            priors=("rating",), rules=RULES)
        cell = {"fits": curves["fits"]["rating"]}
        for rule in RULES:
            loss = curves["loss"]["rating"][rule]
            cell[rule] = {
                "loss": {str(k): interval(loss[:, k], draws) for k in (0, 5, 10)},
                "excess_loss": {str(k): interval(loss[:, k] - oracle, draws) for k in (0, 5, 10)},
            }
        diff = curves["loss"]["rating"]["adaptive"] - curves["loss"]["rating"]["best_fixed"]
        cell["adaptive_minus_best_fixed"] = {str(k): interval(diff[:, k], draws) for k in (0, 5, 10)}
        row["fitted"][fit_name] = cell
        print(f"true {true_name} fitted {fit_name}: adaptive excess loss k=0,5,10 "
              + " ".join(f"{cell['adaptive']['excess_loss'][k][0]:+.4f}" for k in ("0", "5", "10")), flush=True)
    return row


def main() -> None:
    """Run one row, or every row in parallel, then combine them and print excess loss at ``k = 5``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", choices=CLASSES, help="run one row only")
    ap.add_argument("--collect", action="store_true", help="only combine the rows already written")
    args = ap.parse_args()
    folder = OUT / "recovery_grid"
    folder.mkdir(parents=True, exist_ok=True)
    if args.true:
        row = run_row(args.true)
        (folder / f"{args.true}.json").write_text(json.dumps(row, indent=1) + "\n")
        return
    if not args.collect:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        procs = [subprocess.Popen([sys.executable, __file__, "--true", t], env=env) for t in CLASSES]
        if any(p.wait() for p in procs):
            raise SystemExit("a row failed")
    grid = {t: json.loads((folder / f"{t}.json").read_text()) for t in CLASSES}
    (OUT / "recovery_grid.json").write_text(json.dumps(grid, indent=1) + "\n")
    print("excess held-out loss after 5 adaptive answers (rows: true class, columns: fitted class)")
    print("      " + "  ".join(f"{f:>8s}" for f in CLASSES))
    for t in CLASSES:
        print(f"{t:5s} " + "  ".join(f"{grid[t]['fitted'][f]['adaptive']['excess_loss']['5'][0]:+8.4f}" for f in CLASSES))


if __name__ == "__main__":
    main()
