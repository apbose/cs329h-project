"""Hypotheses H1--H4 of the pre-analysis plan from the saved learning curves.

Reads ``results/<data>/{curves,ncurve}_<model>.npz`` (written by ``run_curves.py``) and writes
``results/<data>/analysis.json``. Every interval is a 95% paired bootstrap over test people.

* **H1** (model class and data): held-out loss at ``k = 5`` (adaptive, prior as in deployment)
  for every class and training size ``N``; each flexible class (M3, M4, M4-pre) minus the
  better of M1 and M2, and the first ``N`` at which that difference is at most zero.
* **H2** (adaptive versus fixed): adaptive minus best fixed order at ``k = 5`` under the class
  with the lowest loss at full ``N``; supported if the interval lies below zero.
* **H3** (Basel--Berlin only): loss without the rating minus loss with it, adaptive selection,
  for ``k = 0 .. 10``; the smallest ``k`` whose upper end is below 0.01 nats per choice.
* **H4** (Basel--Berlin only): a logistic regression of each outcome (stock investment, the
  equity fund choice) on the posterior mean after ``k`` adaptive questions, prior without the
  rating, fitted on each fold's training people and applied to its test people; AUC (over
  pairs within a fold) compared with a regression on the general self-rating alone, the one
  the prior uses, and, as a further benchmark, on the financial self-rating alone.

Also reported, as exploratory: the prediction-targeted rule minus the parameter-targeted one,
calibration (expected calibration error) and accuracy at ``k = 0, 5, 10``, the split-half
reliability of the lottery answers (Basel--Berlin), and the checks of ``run_robustness.py`` and
``run_further_checks.py`` when their results exist.

    python scripts/analyze.py --data basel_berlin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cs329h.evaluate import bootstrap_draws, interval
from cs329h.scf import weighted_auc, weighted_logit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLASSES = ("M1", "M2", "M3", "M4", "M4-pre")
SIMPLE, FLEXIBLE = ("M1", "M2"), ("M3", "M4", "M4-pre")
N_GRID = (50, 100, 200, 400, 800)
K_REF = 5
H3_MARGIN = 0.01
SEED, N_BOOT = 330, 2000
# Reliability of a 0--10 self-rating for the correction for attenuation: the mean six-month
# retest correlation of the self-report measures in Frey et al. (2017).
RATING_RELIABILITY = 0.68


def load(data_name: str) -> dict:
    """Curves, training-size curves and metadata of every class that has both result files."""
    out = {}
    for m in CLASSES:
        c, n = RESULTS / data_name / f"curves_{m}.npz", RESULTS / data_name / f"ncurve_{m}.npz"
        if c.exists() and n.exists():
            out[m] = {"curves": dict(np.load(c)), "ncurve": dict(np.load(n)),
                      "meta": json.loads(c.with_suffix(".json").read_text())}
    return out


def h1(res: dict, prior: str, draws: np.ndarray, n_all: int) -> dict:
    """H1: loss at ``k = 5`` by class and training size, and each flexible class against the simple ones.

    Args:
        res: Results from :func:`load`.
        prior: Prior of the full-size curves to use (as in deployment).
        draws: Bootstrap draws over test people.
        n_all: Training people per fold at full size, reported alongside.

    Returns:
        Intervals of the loss per class and size, the best class and best simple class per size,
        each flexible class minus the best simple class, and the first size at which that
        difference's point estimate is at most zero.
    """
    sizes = [str(N) for N in N_GRID] + ["all"]
    loss = {m: {**{str(N): r["ncurve"][f"loss_k5__N{N}"] for N in N_GRID},
                "all": r["curves"][f"loss__{prior}__adaptive"][:, K_REF]} for m, r in res.items()}
    table = {m: {N: interval(loss[m][N], draws) for N in sizes} for m in res}
    best_simple, gaps, first = {}, {m: {} for m in FLEXIBLE if m in res}, {}
    for N in sizes:
        simple = min((m for m in SIMPLE if m in res), key=lambda m: loss[m][N].mean())
        best_simple[N] = simple
        for m in gaps:
            gaps[m][N] = interval(loss[m][N] - loss[simple][N], draws)
    for m in gaps:
        first[m] = next((N for N in sizes if gaps[m][N][0] <= 0), None)
    best = {N: min(res, key=lambda m: loss[m][N].mean()) for N in sizes}
    return {"n_train_all": n_all, "loss_k5": table, "best_class": best, "best_simple": best_simple,
            "flexible_minus_best_simple": gaps, "first_N_matching": first}


def h2(res: dict, best: str, priors: list[str], draws: np.ndarray) -> dict:
    """H2: adaptive minus best fixed order at ``k = 5`` for every class and prior.

    Supported if the interval for class ``best`` under the last prior (as in deployment) lies below
    zero.
    """
    out = {"class": best, "k": K_REF}
    for m, r in res.items():
        for pr in priors:
            d = r["curves"][f"loss__{pr}__adaptive"][:, K_REF] - r["curves"][f"loss__{pr}__best_fixed"][:, K_REF]
            out.setdefault("adaptive_minus_best_fixed", {}).setdefault(m, {})[pr] = interval(d, draws)
    ci = out["adaptive_minus_best_fixed"][best][priors[-1]]
    out["supported"] = bool(ci[2] < 0)
    return out


def h3(res: dict, best: str, draws: np.ndarray) -> dict:
    """H3: loss without minus loss with the rating, adaptive selection, ``k = 0 .. 10``, per class.

    Also the smallest ``k`` whose upper end is below ``H3_MARGIN``, per class and for ``best``.
    """
    gains, smallest = {}, {}
    for m, r in res.items():
        g = r["curves"]["loss__none__adaptive"] - r["curves"]["loss__rating__adaptive"]
        gains[m] = interval(g, draws)
        smallest[m] = next((k for k, (_, _, hi) in enumerate(gains[m]) if hi < H3_MARGIN), None)
    return {"class": best, "margin": H3_MARGIN, "rating_gain": gains, "smallest_k": smallest,
            "smallest_k_best_class": smallest[best]}


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


def _cross_fit_outcome(r: dict, fold: np.ndarray, rating: np.ndarray, y: np.ndarray, k: int, features: str):
    """Out-of-fold predicted probability of the outcome for every person (``nan`` if missing)."""
    pred = np.full(len(y), np.nan)
    post = r["curves"]["post_mean__none"]
    for f in range(int(fold.max()) + 1):
        rows_tr = r["curves"][f"train_rows__none__{f}"]
        te = np.flatnonzero(fold == f)
        dim = r["curves"][f"train_means__none__{f}"].shape[-1]
        parts_tr, parts_te = [], []
        if features in ("posterior", "both"):
            parts_tr.append(r["curves"][f"train_means__none__{f}"][k])
            parts_te.append(post[te, k, :dim])
        if features in ("rating", "both"):
            parts_tr.append(rating[rows_tr, None])
            parts_te.append(rating[te, None])
        X_tr, X_te = _standardize(np.column_stack(parts_tr), np.column_stack(parts_te))
        ok_tr, ok_te = np.isfinite(y[rows_tr]), np.isfinite(y[te])
        b = weighted_logit(X_tr[ok_tr], y[rows_tr][ok_tr], np.ones(ok_tr.sum()), ridge=1e-3)
        pred[te[ok_te]] = 1 / (1 + np.exp(-(b[0] + X_te[ok_te] @ b[1:])))
    return pred


def within_fold_auc(y: np.ndarray, p: np.ndarray, fold: np.ndarray) -> float:
    """AUC over (positive, negative) pairs from the same fold.

    Each fold's predictions come from its own fitted model, so pairs across folds would compare
    scores on different scales (for M3 and M4, posterior means on different latent axes).
    """
    num = den = 0.0
    for f in np.unique(fold):
        m = fold == f
        pairs = y[m].sum() * (1 - y[m]).sum()
        if pairs > 0:
            num += pairs * weighted_auc(y[m], p[m], np.ones(m.sum()))
            den += pairs
    return num / den


def _auc_with_interval(y, p, fold, draws):
    boot = np.array([within_fold_auc(y[d], p[d], fold[d]) for d in draws])
    return [within_fold_auc(y, p, fold), *np.quantile(boot, [0.025, 0.975]).tolist()], boot


def h4(res: dict, best: str, data, fold: np.ndarray, k_values: range, n_boot: int = 1000) -> dict:
    """Outcome prediction from posterior means; ``k = 0`` is left out (every mean is the same)."""
    out = {"class": best, "k": list(k_values), "outcomes": {}}
    ratings = {"rating_alone": data.rating.astype(float), "rating_fin_alone": data.rating_fin.astype(float)}
    r0 = next(iter(res.values()))
    for name, y_all in data.outcomes.items():
        ok = np.isfinite(y_all)
        y, fo = y_all[ok], fold[ok]
        draws = bootstrap_draws(ok.sum(), np.random.default_rng(SEED), n_boot)
        entry = {"n": int(ok.sum()), "n_positive": int(y.sum()), "classes": {}}
        boot_ref = {}
        for key, rating in ratings.items():
            p = _cross_fit_outcome(r0, fold, rating, y_all, 0, "rating")[ok]
            auc, boot_ref[key] = _auc_with_interval(y, p, fo, draws)
            entry[key] = {"auc": auc}
        for m, r in res.items():
            ks = k_values if m == best else (K_REF, k_values[-1])
            rows = {}
            for k in ks:
                row = {}
                for features in ("posterior", "both"):
                    p = _cross_fit_outcome(r, fold, ratings["rating_alone"], y_all, k, features)[ok]
                    auc, boot = _auc_with_interval(y, p, fo, draws)
                    row[features] = {"auc": auc}
                    if features == "posterior":
                        for key in ratings:
                            d = boot - boot_ref[key]
                            row[f"posterior_minus_{key}_auc"] = [auc[0] - entry[key]["auc"][0],
                                                                 *np.quantile(d, [0.025, 0.975]).tolist()]
                rows[str(k)] = row
            entry["classes"][m] = rows
        out["outcomes"][name] = entry
    return out


def ratings_from_lotteries(res: dict, data, fold: np.ndarray, draws: np.ndarray, ks=(10,)) -> dict:
    """How well lottery answers predict the two self-ratings (exploratory).

    For each class and ``k``, the rating is regressed (least squares) on the posterior means after
    ``k`` adaptive questions, prior without the rating, fitted on each fold's training people and
    applied to its test people. Reports the correlation of these out-of-fold predictions with the
    rating, with a bootstrap interval over people.
    """
    ratings = {"general": data.rating.astype(float), "financial": data.rating_fin.astype(float)}
    out = {}
    for m, r in res.items():
        c = r["curves"]
        out[m] = {}
        for k in ks:
            row = {}
            for name, y in ratings.items():
                pred = np.full(len(y), np.nan)
                for f in range(int(fold.max()) + 1):
                    rows_tr = c[f"train_rows__none__{f}"]
                    x_tr = c[f"train_means__none__{f}"][k]
                    dim = x_tr.shape[-1]
                    te = np.flatnonzero(fold == f)
                    b = np.linalg.lstsq(np.column_stack([np.ones(len(x_tr)), x_tr]), y[rows_tr], rcond=None)[0]
                    pred[te] = np.column_stack([np.ones(len(te)), c["post_mean__none"][te, k, :dim]]) @ b
                boot = [np.corrcoef(pred[d], y[d])[0, 1] for d in draws]
                row[name] = [float(np.corrcoef(pred, y)[0, 1]), *np.quantile(boot, [0.025, 0.975]).tolist()]
            out[m][str(k)] = row
    return out


def exploratory(res: dict, priors: list[str], draws: np.ndarray) -> dict:
    """Exploratory results per class: predictive minus adaptive, calibration, loss under every rule."""
    out = {"predictive_minus_adaptive": {}, "calibration": {}, "loss": {}, "accuracy": {}}
    for m, r in res.items():
        c = r["curves"]
        out["predictive_minus_adaptive"][m] = {
            pr: interval(c[f"loss__{pr}__predictive"] - c[f"loss__{pr}__adaptive"], draws) for pr in priors}
        out["calibration"][m] = r["meta"]["calibration"]
        out["loss"][m] = {pr: {rule: interval(c[f"loss__{pr}__{rule}"], draws)
                               for rule in ("adaptive", "best_fixed", "random", "original", "predictive")}
                          for pr in priors}
        out["accuracy"][m] = {pr: {rule: {str(k): interval(c[f"acc__{pr}__{rule}"][:, k], draws) for k in (0, K_REF, 10)}
                                   for rule in ("adaptive", "best_fixed", "random")}
                              for pr in priors}
    return out


def lottery_reliability(data, rating_corr: dict) -> dict:
    """Split-half reliability of the lottery answers and the rating correlations corrected for it.

    The share of risky choices on the odd-numbered price lists is correlated with the share on
    the even-numbered ones, and the Spearman--Brown formula gives the reliability of all lists
    together. The correlation of each self-rating with its prediction from M3 after ten questions
    (from :func:`ratings_from_lotteries`) is then divided by the square root of the product of
    the two reliabilities, with ``RATING_RELIABILITY`` for the rating.
    """
    Y = data.Y.astype(float)
    odd = np.isin(data.lists, np.arange(0, data.n_lists, 2))
    r = float(np.corrcoef(Y[:, odd].mean(1), Y[:, ~odd].mean(1))[0, 1])
    rel = 2 * r / (1 + r)
    corrected = {name: rating_corr[name][0] / np.sqrt(rel * RATING_RELIABILITY) for name in ("general", "financial")}
    return {"split_half_r": r, "spearman_brown": rel, "rating_reliability": RATING_RELIABILITY,
            "observed_M3_k10": {name: rating_corr[name] for name in ("general", "financial")},
            "corrected_M3_k10": corrected}


def further(res: dict, data_name: str, prior: str, draws: np.ndarray) -> dict:
    """The exploratory checks of ``run_further_checks.py`` next to the main run, when they exist.

    * ``scale``: loss with one payoff scale per dataset minus the main loss (per-list scaling),
      per rule at ``k = 0, 5, 10``.
    * ``finrating``: adaptive loss with the financial rating in the prior minus that with the
      general rating, and minus that with no rating, at ``k = 0, 5, 10``.
    * ``ndraw``: for each training size, each flexible class minus the better simple class at
      ``k = 5`` in each of five draws of training people (the main draw and four more): the point
      estimate per draw, and the interval for the mean over draws.
    """
    folder = RESULTS / data_name / "further"
    if not folder.exists():
        return {}
    out = {"scale": {}, "finrating": {}, "ndraw": {}}
    rules = ("adaptive", "best_fixed", "random")
    for m in SIMPLE:
        path = folder / f"scale_{m}.npz"
        if m in res and path.exists():
            arr, c = np.load(path), res[m]["curves"]
            out["scale"][m] = {rule: {str(k): interval(arr[f"loss__{rule}"][:, k] - c[f"loss__{prior}__{rule}"][:, k], draws)
                                      for k in (0, K_REF, 10)} for rule in rules}
    for m in ("M1", "M2", "M3"):
        path = folder / f"finrating_{m}.npz"
        if m in res and path.exists():
            fin, c = np.load(path)["loss__finrating__adaptive"], res[m]["curves"]
            out["finrating"][m] = {
                "minus_general": {str(k): interval(fin[:, k] - c["loss__rating__adaptive"][:, k], draws) for k in (0, K_REF, 10)},
                "minus_none": {str(k): interval(fin[:, k] - c["loss__none__adaptive"][:, k], draws) for k in (0, K_REF, 10)},
            }
    seeds = sorted({int(p.stem.split("_s")[-1]) for p in folder.glob("ndraw_*_s*.npz")})
    have = [m for m in CLASSES if m in res and all((folder / f"ndraw_{m}_s{s}.npz").exists() for s in seeds)]
    if seeds and all(m in have for m in SIMPLE):
        sizes = sorted({int(k.split("N")[-1]) for k in np.load(folder / f"ndraw_M1_s{seeds[0]}.npz").files})
        loss = {m: {N: [res[m]["ncurve"][f"loss_k5__N{N}"]] + [np.load(folder / f"ndraw_{m}_s{s}.npz")[f"loss_k5__N{N}"]
                                                              for s in seeds] for N in sizes} for m in have}
        for m in (f for f in FLEXIBLE if f in have):
            out["ndraw"][m] = {}
            for N in sizes:
                gaps = [loss[m][N][d] - min((loss[s_][N][d] for s_ in SIMPLE), key=np.mean)
                        for d in range(len(seeds) + 1)]
                out["ndraw"][m][str(N)] = {"per_draw": [float(g.mean()) for g in gaps],
                                           "mean_over_draws": interval(np.mean(gaps, axis=0), draws)}
        out["ndraw_draws"] = len(seeds) + 1
    return out


def robustness(res: dict, data_name: str, prior: str, draws: np.ndarray) -> dict:
    """Each check of ``run_robustness.py`` next to the main run of the same class.

    Per check: adaptive loss at k = 5 and 10, adaptive minus best fixed and adaptive minus
    random at the same k (paired intervals), and calibration of the adaptive rule.
    """
    folder = RESULTS / data_name / "robustness"
    rows = {}
    runs = {f"{m}_main": (m, {r: res[m]["curves"][f"loss__{prior}__{r}"] for r in ("adaptive", "best_fixed", "random")},
                          res[m]["meta"]["calibration"][prior]) for m in ("M1", "M3") if m in res}
    for path in sorted(folder.glob("*.npz")) if folder.exists() else []:
        meta = json.loads(path.with_suffix(".json").read_text())
        arr = np.load(path)
        runs[path.stem] = (meta["model"], {r: arr[f"loss__{r}"] for r in ("adaptive", "best_fixed", "random")},
                           meta["calibration"])
    for name, (model, loss, calib) in runs.items():
        row = {"model": model}
        for k in (K_REF, 10):
            row[f"adaptive_k{k}"] = interval(loss["adaptive"][:, k], draws)
            row[f"adaptive_minus_best_fixed_k{k}"] = interval(loss["adaptive"][:, k] - loss["best_fixed"][:, k], draws)
            row[f"adaptive_minus_random_k{k}"] = interval(loss["adaptive"][:, k] - loss["random"][:, k], draws)
            row[f"ece_adaptive_k{k}"] = calib["adaptive"][str(k)]
        rows[name] = row
    return rows


def main() -> None:
    """Run H1--H4 and the exploratory analyses for one dataset and write ``analysis.json``."""
    from cs329h.data import load_basel_berlin, load_twin
    from cs329h.splits import load_splits

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=("basel_berlin", "twin2k500"))
    args = ap.parse_args()
    data = {"basel_berlin": load_basel_berlin, "twin2k500": load_twin}[args.data]()
    res = load(args.data)
    missing = [m for m in CLASSES if m not in res]
    if not res:
        raise SystemExit("no results yet; run scripts/run_curves.py first")
    priors = ["none", "rating"] if data.rating is not None else ["none"]
    deploy = priors[-1]
    fold = load_splits(data).fold
    n_all = int((fold != 0).sum())  # training people per fold at full size (fold 0)
    draws = bootstrap_draws(data.n_people, np.random.default_rng(SEED), N_BOOT)
    out = {"data": args.data, "missing_classes": missing, "H1": h1(res, deploy, draws, n_all)}
    best = out["H1"]["best_class"]["all"]
    out["H2"] = h2(res, best, priors, draws)
    if data.rating is not None:
        out["H3"] = h3(res, best, draws)
    if data.outcomes:
        out["H4"] = h4(res, best, data, fold, range(1, 11))
        out["ratings_from_lotteries"] = ratings_from_lotteries(res, data, fold, draws)
        if "M3" in res:
            out["lottery_reliability"] = lottery_reliability(data, out["ratings_from_lotteries"]["M3"]["10"])
    out["exploratory"] = exploratory(res, priors, draws)
    out["robustness"] = robustness(res, args.data, deploy, draws)
    out["further_checks"] = further(res, args.data, deploy, draws)
    path = RESULTS / args.data / "analysis.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}" + (f" (missing {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
