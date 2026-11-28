"""Three further exploratory checks of the main results (Appendix B).

* **Payoff scale** (``scale_<class>``, M1 and M2, both datasets): the main analysis scales the
  outcomes of each price list to ``[-1, 1]`` separately, so lists with very different stakes
  (3.85 to 200 in Basel--Berlin, in the study's currency) share one scale. Here outcomes are
  instead divided by one constant per dataset (its largest stake), so a person's risk aversion
  applies to money in the same units in every list. Prior as in deployment; adaptive, best fixed
  and random order.
* **Financial self-rating in the prior** (``finrating_<class>``, Basel--Berlin, M1 to M3): the
  main analysis puts the general 0--10 rating in the prior; here the financial rating replaces it,
  with adaptive selection.
* **More draws of training people** (``ndraw_<class>_s<seed>``, both datasets, every class): the
  learning curve over training size uses one random draw of training people per size; these
  jobs repeat it at N = 50, 100 and 200 with four further draws, adaptive selection, k = 5.

Folds, held-out lists and seeds are those of the main analysis.

    python scripts/run_further_checks.py --data basel_berlin     # every check, 4 at a time
    python scripts/run_further_checks.py --data twin2k500 --check scale_M1

Writes results/<data>/further/<check>.npz and .json; ``scripts/analyze.py`` reads them.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SEED = 329
K_MAX = 10
N_SMALL = (50, 100, 200)
DRAW_SEEDS = (1, 2, 3, 4)
CLASSES = ("M1", "M2", "M3", "M4", "M4-pre")
RESULTS = Path(__file__).resolve().parents[1] / "results"


def checks_for(data_name: str) -> list[str]:
    """Names of the checks that apply to ``data_name``, cheapest first."""
    names = [f"scale_{m}" for m in ("M1", "M2")]
    if data_name == "basel_berlin":
        names += [f"finrating_{m}" for m in ("M1", "M2", "M3")]
    names += [f"ndraw_{m}_s{s}" for s in DRAW_SEEDS for m in ("M4-pre", "M4", "M2", "M3", "M1")]
    return names


def global_scale(data):
    """The dataset with outcomes divided by its largest stake instead of each list's own."""
    money = data.x * data.scale[:, None, None]
    return dataclasses.replace(data, x=money / data.scale.max())


def run_check(data_name: str, check: str) -> None:
    """Run one check and write its per-person losses and metadata."""
    from cs329h.experiment import run_curves, summarize
    from cs329h.models import make_model
    from cs329h.splits import load_splits

    from run_curves import load

    data = load(data_name)
    splits = load_splits(data)
    kind, model_name = check.split("_")[0], check.split("_")[1]
    deploy = "rating" if data.rating is not None else "none"
    make = lambda use_rating: make_model(model_name, use_rating)
    t0 = time.time()
    arrays, meta = {}, {"data": data_name, "check": check, "model": model_name}
    if kind == "scale":
        rules = ("adaptive", "best_fixed", "random")
        curves = run_curves(global_scale(data), splits, make, k_max=K_MAX, seed=SEED, priors=(deploy,), rules=rules)
        arrays = {f"loss__{r}": curves["loss"][deploy][r] for r in rules}
        meta.update({"prior": deploy, "fits": summarize(curves)["fits"][deploy]})
    elif kind == "finrating":
        fin = dataclasses.replace(data, rating=data.rating_fin)
        curves = run_curves(fin, splits, make, k_max=K_MAX, seed=SEED, priors=("rating",), rules=("adaptive",))
        arrays = {"loss__finrating__adaptive": curves["loss"]["rating"]["adaptive"]}
        meta["fits"] = summarize(curves)["fits"]["rating"]
    elif kind == "ndraw":
        seed = int(check.split("_s")[-1])
        for N in N_SMALL:
            rng = np.random.default_rng(SEED + N + 1000 * seed)
            pick = lambda tr, f, N=N, rng=rng: np.sort(rng.permutation(tr)[:N])
            curves = run_curves(data, splits, make, k_max=5, seed=SEED, priors=(deploy,), rules=("adaptive",),
                                train_rows=pick)
            arrays[f"loss_k5__N{N}"] = curves["loss"][deploy]["adaptive"][:, 5]
        meta.update({"prior": deploy, "draw_seed": seed})
    else:
        raise ValueError(f"unknown check {check!r}")
    meta["seconds"] = round(time.time() - t0)
    out_dir = RESULTS / data_name / "further"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{check}.npz", **arrays)
    (out_dir / f"{check}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"done {data_name} {check} in {meta['seconds']} s", flush=True)


def main() -> None:
    """Run one check, or every check whose results are missing, ``--workers`` subprocesses at a time."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=("basel_berlin", "twin2k500"))
    ap.add_argument("--check", help="one check, e.g. scale_M1, finrating_M3 or ndraw_M4_s2")
    ap.add_argument("--only", help="run only checks whose name starts with this, e.g. ndraw")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    names = checks_for(args.data)
    if args.check:
        if args.check not in names:
            raise SystemExit(f"unknown check {args.check!r} for {args.data}; choose from {', '.join(names)}")
        run_check(args.data, args.check)
        return
    jobs = [c for c in names if not (RESULTS / args.data / "further" / f"{c}.npz").exists()
            and (args.only is None or c.startswith(args.only))]
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
