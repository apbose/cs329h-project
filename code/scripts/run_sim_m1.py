"""M1 end to end on simulated investors (Figure 1, left; Section 3, "Sanity check").

Fits M1, with the self-rating in the prior, to all Basel--Berlin people; draws as many
simulated investors from that population, each with a resampled Basel--Berlin self-rating and
answers to the same 66 questions; and runs the main evaluation on them: five folds, two
held-out price lists per person, ``k = 0 .. 10`` questions under adaptive selection, the best
fixed order, a random order and the original order, with and without the rating in the prior.

Also reports how well the truth is recovered: fitted population parameters per fold against
the true ones, the correlation of each test person's posterior mean with their true risk
aversion, and the held-out loss under the true parameters (the floor no method can beat).

    python scripts/run_sim_m1.py          # writes results/sim_m1.json
"""

import json
from pathlib import Path

import numpy as np

from cs329h.data import load_basel_berlin
from cs329h.evaluate import bootstrap_draws, interval
from cs329h.experiment import PLANNED_RULES, run_curves, summarize
from cs329h.models import ExpectedUtility
from cs329h.simulate import simulate
from cs329h.splits import make_splits

OUT = Path(__file__).resolve().parents[1] / "results" / "sim_m1.json"
SEED = 329
K_MAX = 10


def main() -> None:
    """Simulate investors from M1, run the evaluation on them, and write and print the summary."""
    bb = load_basel_berlin()
    truth_model = ExpectedUtility(use_rating=True).fit(bb)
    truth = truth_model.params
    rng = np.random.default_rng(SEED)
    sim, theta = simulate(truth_model, bb, bb.n_people, rng, rating_pool=bb.rating, name="sim_m1")
    r_true = theta[:, 0]
    splits = make_splits(sim.n_people, sim.n_lists, seed=SEED)

    curves = run_curves(sim, splits, lambda use_rating: ExpectedUtility(use_rating=use_rating), k_max=K_MAX, seed=SEED,
                        rules=PLANNED_RULES)
    summary = summarize(curves)

    held = splits.heldout_mask(sim.lists)
    P_true = truth_model.choice_prob_people(theta)
    ll = sim.Y * np.log(P_true) + (1 - sim.Y) * np.log1p(-P_true)
    oracle = -(ll * held).sum(1) / held.sum(1)
    draws = bootstrap_draws(sim.n_people, np.random.default_rng(SEED + 1))
    summary["oracle_loss"] = interval(oracle, draws)
    summary["true_params"] = truth.as_dict()
    summary["recovery_corr"] = {
        pr: [float(np.corrcoef(curves["post_mean"][pr][:, k, 0], r_true)[0, 1]) if k else None for k in range(K_MAX + 1)]
        for pr in curves["post_mean"]
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1) + "\n")

    print("true", {k: round(v, 3) for k, v in truth.as_dict().items()})
    for pr, fits in summary["fits"].items():
        for f in fits:
            print(f"fit[{pr}]", {k: round(v, 3) for k, v in f.items()})
    print(f"oracle held-out loss {summary['oracle_loss'][0]:.4f}")
    for pr in summary["loss"]:
        for rule, c in summary["loss"][pr].items():
            print(f"{pr:6s} {rule:10s} loss k=0,1,3,5,10:", " ".join(f"{c[k][0]:.4f}" for k in (0, 1, 3, 5, 10)))
        d = summary["adaptive_minus_best_fixed"][pr]
        print(f"{pr:6s} adaptive - best fixed at k=5: {d[5][0]:+.4f} [{d[5][1]:+.4f}, {d[5][2]:+.4f}]")
        print(f"{pr:6s} corr(posterior mean, true r) k=1,5,10:",
              " ".join(f"{summary['recovery_corr'][pr][k]:.3f}" for k in (1, 5, 10)))
    if "rating_gain" in summary:
        g = summary["rating_gain"]["adaptive"]
        print("rating gain (adaptive) k=0,5,10:", " ".join(f"{g[k][0]:+.4f}[{g[k][1]:+.4f},{g[k][2]:+.4f}]" for k in (0, 5, 10)))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
