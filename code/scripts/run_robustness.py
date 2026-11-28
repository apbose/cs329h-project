"""Robustness of the learning curves (Table 2): lapse rate, prior strength and rank K.

Each check changes one modeling choice and reruns the k = 0..10 curve under adaptive
selection, the best fixed order and a random order, with the main analysis's folds and held-out lists
(prior as in deployment: with the self-rating on Basel--Berlin, without it on Twin-2K-500).

* **Lapse rate** (M1, M3): after fitting, the lapse rate ``eps`` is set to 0 (M1 only), 0.05 or
  0.15 and everything else is kept, so the posterior and the question choice assume that much
  random error. M3's fitted lapse rate is already close to 0.
* **Prior strength** (M1, M3): after fitting, the prior's standard deviation is halved or
  doubled (for M3, ``Sigma`` scaled by 0.25 or 4), making the prior more or less confident.
* **Rank** (M3): K fixed at 1, 2, 3, 4 or 5, with the penalty still chosen on validation
  people from {1, 10}; the main analysis chooses K from {1, 2, 3}.

On Twin-2K-500 these runs are exploratory: its one confirmatory analysis is the main run.

    python scripts/run_robustness.py --data basel_berlin     # every check, 4 at a time

Writes results/<data>/robustness/<check>.npz and .json; ``scripts/analyze.py`` reads them.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SEED = 329
K_MAX = 10
RULES = ("adaptive", "best_fixed", "random")
RESULTS = Path(__file__).resolve().parents[1] / "results"
CHECKS = {
    **{f"{m}_eps{e:g}": (m, {"eps": e}) for m in ("M1", "M3") for e in (0.0, 0.05, 0.15) if (m, e) != ("M3", 0.0)},
    **{f"{m}_prior{s:g}": (m, {"prior_scale": s}) for m in ("M1", "M3") for s in (0.5, 2.0)},
    **{f"M3_K{k}": ("M3", {"K": k}) for k in (1, 2, 3, 4, 5)},
}


def make_factory(model_name: str, change: dict):
    """A ``make_model(use_rating)`` for run_curves that applies ``change`` after each fit."""
    from cs329h.models import make_model

    def make(use_rating):
        """A new model of class ``model_name`` whose ``fit`` applies ``change`` afterwards."""
        model = make_model(model_name, use_rating)
        if "K" in change:
            model.hyper_grid = [{"K": change["K"], "lam": lam} for lam in (1.0, 10.0)]
        fit = model.fit

        def fit_then_change(d, rows=None):
            """Fit as usual, then set the lapse rate or rescale the prior as ``change`` says, and rebind."""
            fit(d, rows)
            p = model.params
            if "eps" in change:
                p.eps = change["eps"]
            if "prior_scale" in change:
                s = change["prior_scale"]
                if model_name == "M1":
                    p.sigma = p.sigma * s
                else:
                    p.Sigma = p.Sigma * s**2
            return model.bind(d)

        model.fit = fit_then_change
        return model

    return make


def run_check(data_name: str, check: str) -> None:
    """Rerun the curves under robustness check ``check`` and write its losses and metadata."""
    from cs329h.experiment import run_curves, summarize
    from cs329h.splits import load_splits

    from run_curves import load

    data = load(data_name)
    splits = load_splits(data)
    model_name, change = CHECKS[check]
    prior = "rating" if data.rating is not None else "none"
    t0 = time.time()
    curves = run_curves(data, splits, make_factory(model_name, change), k_max=K_MAX, seed=SEED,
                        priors=(prior,), rules=RULES)
    summary = summarize(curves)
    out_dir = RESULTS / data_name / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{check}.npz", **{f"loss__{r}": curves["loss"][prior][r] for r in RULES})
    meta = {"data": data_name, "check": check, "model": model_name, "change": change, "prior": prior,
            "calibration": summary["calibration"][prior], "fits": summary["fits"][prior],
            "seconds": round(time.time() - t0)}
    (out_dir / f"{check}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"done {data_name} {check} in {meta['seconds']} s", flush=True)


def main() -> None:
    """Run one check, or every check whose results are missing, ``--workers`` subprocesses at a time."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=("basel_berlin", "twin2k500"))
    ap.add_argument("--check", choices=tuple(CHECKS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.check:
        run_check(args.data, args.check)
        return
    jobs = [c for c in CHECKS if not (RESULTS / args.data / "robustness" / f"{c}.npz").exists()]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running, failed = [], []
    while jobs or running:
        while jobs and len(running) < args.workers:
            c = jobs.pop(0)
            running.append((c, subprocess.Popen([sys.executable, __file__, "--data", args.data, "--check", c], env=env)))
        time.sleep(5)
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                if item[1].returncode:
                    failed.append(item[0])
    if failed:
        raise SystemExit(f"failed checks: {failed}")


if __name__ == "__main__":
    main()
