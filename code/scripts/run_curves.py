"""Learning curves on real data (Figures 2 and 3, Table 1): every class, both priors, every rule.

Two curves per model class:

* over questions asked, ``k = 0 .. 10``, at full training size, under every selection rule
  and with and without the self-rating in the prior (``curves``);
* over training size ``N`` in {50, 100, 200, 400, 800, all} at ``k = 5`` with adaptive
  selection, prior as in deployment (with the rating when the data has one) (``ncurve``).

Folds and held-out lists are the committed ones in ``splits/``. For Basel--Berlin the curves job
also keeps each fold's training people's posterior means, for the outcome models.

    python scripts/run_curves.py --data basel_berlin                    # every job, 4 at a time
    python scripts/run_curves.py --data basel_berlin --job curves --model M2   # one job

Writes results/<data>/<job>_<model>.npz and .json; ``scripts/analyze.py`` reads them.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CLASSES = ("M1", "M2", "M3", "M4", "M4-pre")
N_GRID = (50, 100, 200, 400, 800)
SEED = 329
K_MAX = 10
RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(name):
    """Load dataset ``name`` (``basel_berlin`` or ``twin2k500``)."""
    from cs329h.data import load_basel_berlin, load_twin

    return {"basel_berlin": load_basel_berlin, "twin2k500": load_twin}[name]()


def run_job(data_name: str, job: str, model_name: str) -> None:
    """Run one job (``curves`` or ``ncurve``) for one model class and write its ``.npz`` and ``.json``."""
    from cs329h.experiment import run_curves, summarize
    from cs329h.models import make_model
    from cs329h.splits import load_splits

    data = load(data_name)
    splits = load_splits(data)
    out_dir = RESULTS / data_name
    out_dir.mkdir(parents=True, exist_ok=True)
    make = lambda use_rating: make_model(model_name, use_rating)
    t0 = time.time()
    arrays, meta = {}, {"data": data_name, "model": model_name, "job": job}
    if job == "curves":
        curves = run_curves(data, splits, make, k_max=K_MAX, seed=SEED, train_means=data.rating is not None)
        for pr in curves["loss"]:
            for rule in curves["loss"][pr]:
                arrays[f"loss__{pr}__{rule}"] = curves["loss"][pr][rule]
                arrays[f"acc__{pr}__{rule}"] = curves["acc"][pr][rule]
            arrays[f"post_mean__{pr}"] = curves["post_mean"][pr]
            for f, (rows, means) in enumerate(curves["train_post_mean"][pr]):
                arrays[f"train_rows__{pr}__{f}"] = rows
                arrays[f"train_means__{pr}__{f}"] = means
        summary = summarize(curves)
        meta.update({k: summary[k] for k in ("calibration", "fits", "best_fixed_order")})
    else:
        pr = "rating" if data.rating is not None else "none"
        for N in N_GRID:
            rng = np.random.default_rng(SEED + N)
            pick = lambda tr, f, N=N, rng=rng: np.sort(rng.permutation(tr)[:N])
            curves = run_curves(data, splits, make, k_max=5, seed=SEED, priors=(pr,), rules=("adaptive",),
                                train_rows=pick)
            arrays[f"loss_k5__N{N}"] = curves["loss"][pr]["adaptive"][:, 5]
            meta.setdefault("fits", {})[str(N)] = curves["fits"][pr]
            print(f"{data_name} {model_name} N={N}: loss k=5 {arrays[f'loss_k5__N{N}'].mean():.4f}", flush=True)
    meta["seconds"] = round(time.time() - t0)
    np.savez_compressed(out_dir / f"{job}_{model_name}.npz", **arrays)
    (out_dir / f"{job}_{model_name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"done {data_name} {job} {model_name} in {meta['seconds']} s", flush=True)


def main() -> None:
    """Run one job, or every job whose results are missing, ``--workers`` subprocesses at a time."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=("basel_berlin", "twin2k500"))
    ap.add_argument("--job", choices=("curves", "ncurve"))
    ap.add_argument("--model", choices=CLASSES)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.job and args.model:
        run_job(args.data, args.job, args.model)
        return
    jobs = [(j, m) for m in ("M4-pre", "M4", "M2", "M3", "M1") for j in ("curves", "ncurve")
            if not (RESULTS / args.data / f"{j}_{m}.npz").exists()]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running, failed = [], []
    while jobs or running:
        while jobs and len(running) < args.workers:
            j, m = jobs.pop(0)
            cmd = [sys.executable, __file__, "--data", args.data, "--job", j, "--model", m]
            running.append(((j, m), subprocess.Popen(cmd, env=env)))
        time.sleep(5)
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                if item[1].returncode:
                    failed.append(item[0])
    if failed:
        raise SystemExit(f"failed jobs: {failed}")


if __name__ == "__main__":
    main()
