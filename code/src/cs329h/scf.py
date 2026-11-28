"""What the 0--10 self-rating says about stock ownership in the 2022 Survey of Consumer Finances.

A survey-weighted logistic regression of stock ownership (directly held stocks, ``owns_stock``)
with and without the rating. Covariates: age, age squared, log income, log net worth (signed
``log1p``, since both can be zero or negative) and education (four levels, as indicators). The
rating enters linearly.

The weighted AUC with and without the rating, and its change, come from five-fold cross-fitting
by household, with the folds the same in every implicate. Each estimate's within-implicate
variance is a bootstrap over households (the fitted predictions held fixed), and the five
implicates are combined with Rubin's rules (Rubin 1987). The rating's coefficient comes from a
fit on all households, with a sandwich standard error that ignores the survey's strata (the
public file has none).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def signed_log(x: np.ndarray) -> np.ndarray:
    """``sign(x) * log1p(|x|)``: a log scale that allows zero and negative values."""
    return np.sign(x) * np.log1p(np.abs(x))


def design(df, with_rating: bool) -> np.ndarray:
    """Covariates (standardized) plus education indicators, and the rating when asked."""
    cols = [df.age.values, df.age.values ** 2, signed_log(df.income.values), signed_log(df.networth.values)]
    X = np.column_stack([(c - c.mean()) / c.std() for c in cols])
    edu = np.column_stack([(df.educ.values == lv).astype(float) for lv in (2, 3, 4)])
    parts = [X, edu]
    if with_rating:
        parts.append(df.risk10.values.astype(float)[:, None])
    return np.column_stack(parts)


def weighted_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Newton's method for the weighted logistic likelihood (intercept added)."""
    X = np.column_stack([np.ones(len(X)), X])
    w = w / w.mean()
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (w * (p - y)) + ridge * b
        H = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-10:
            break
    return b


def logit_sandwich_se(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sandwich standard errors of weighted logistic coefficients ``b`` (intercept first)."""
    X = np.column_stack([np.ones(len(X)), X])
    w = w / w.sum()
    p = 1 / (1 + np.exp(-X @ b))
    H = (X * (w * p * (1 - p))[:, None]).T @ X
    S = (X * (w * (y - p))[:, None]).T @ (X * (w * (y - p))[:, None])
    Hi = np.linalg.inv(H)
    return np.sqrt(np.diag(Hi @ S @ Hi))


def weighted_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Weighted probability that a random owner scores above a random non-owner (ties count half)."""
    order = np.argsort(p, kind="mergesort")
    y, p, w = y[order], p[order], w[order]
    pos, neg = w * y, w * (1 - y)
    _, start, counts = np.unique(p, return_index=True, return_counts=True)
    neg_tie = np.add.reduceat(neg, start)
    pos_tie = np.add.reduceat(pos, start)
    neg_below = np.cumsum(neg_tie) - neg_tie
    return float((pos_tie * (neg_below + 0.5 * neg_tie)).sum() / (pos.sum() * neg.sum()))


def cross_fit(fit, predict, X, y, w, fold):
    """``(n,)`` out-of-fold predictions.

    For each fold, ``fit(X, y, w)`` on the other folds and ``predict(coefficients, X)`` on this one.
    """
    out = np.zeros(len(y))
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        out[te] = predict(fit(X[tr], y[tr], w[tr]), X[te])
    return out


def rubin(estimates: np.ndarray, variances: np.ndarray) -> dict:
    """Combine one estimate per implicate with Rubin's rules: mean and a t-based 95% interval."""
    M = len(estimates)
    qbar, ubar = float(np.mean(estimates)), float(np.mean(variances))
    between = float(np.var(estimates, ddof=1))
    total = ubar + (1 + 1 / M) * between
    if between > 0:
        dof = (M - 1) * (1 + ubar / ((1 + 1 / M) * between)) ** 2
    else:
        dof = np.inf
    half = stats.t.ppf(0.975, dof) * np.sqrt(total)
    return {"estimate": qbar, "lower": qbar - half, "upper": qbar + half, "se": float(np.sqrt(total)),
            "within_var": ubar, "between_var": between, "dof": float(dof)}


def _logit_predict(b, X):
    return 1 / (1 + np.exp(-(b[0] + X @ b[1:])))


def analyze(scf, n_folds: int = 5, n_boot: int = 500, seed: int = 329) -> dict:
    """The stock-ownership regression on every implicate, combined with Rubin's rules."""
    households = np.sort(scf.household.unique())
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(households, rng.permutation(len(households)) % n_folds))
    implicates = sorted(scf.implicate.unique())
    keys = ("auc_without", "auc_with", "d_auc")
    per = {key: [] for key in keys + ("coef_own",)}
    var = {key: [] for key in per}
    sizes = {}
    for m in implicates:
        df = scf[scf.implicate == m].sort_values("household").reset_index(drop=True)
        fold = df.household.map(fold_of).values
        w, y = df.weight.values.astype(float), df.owns_stock.values.astype(float)
        X0, X1 = design(df, False), design(df, True)
        p0 = cross_fit(weighted_logit, _logit_predict, X0, y, w, fold)
        p1 = cross_fit(weighted_logit, _logit_predict, X1, y, w, fold)
        sizes[int(m)] = int(len(df))

        def metrics(idx):
            """AUC without and with the rating, and its change, on households ``idx``."""
            a0, a1 = weighted_auc(y[idx], p0[idx], w[idx]), weighted_auc(y[idx], p1[idx], w[idx])
            return a0, a1, a1 - a0

        full = metrics(np.arange(len(y)))
        boot = np.array([metrics(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
        for key, est, v in zip(keys, full, boot.var(0, ddof=1)):
            per[key].append(est)
            var[key].append(v)
        b = weighted_logit(X1, y, w)
        per["coef_own"].append(b[-1])
        var["coef_own"].append(logit_sandwich_se(X1, y, w, b)[-1] ** 2)
    combined = {key: rubin(np.array(per[key]), np.array(var[key])) for key in per}
    combined["odds_ratio_per_point"] = {k: float(np.exp(combined["coef_own"][k])) for k in ("estimate", "lower", "upper")}
    return {"households": sizes, "per_implicate": {k: [float(x) for x in v] for k, v in per.items()}, "combined": combined}
