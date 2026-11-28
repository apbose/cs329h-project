"""Calibration and bootstrap intervals.

Intervals are percentile intervals from a bootstrap over test people. A *paired* bootstrap
resamples people once and applies the same draw to both arms of a comparison, so the interval
for a difference uses each person's own difference.
"""

from __future__ import annotations

import numpy as np


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Reliability table and expected calibration error over equal-width probability bins."""
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    keep = ~np.isnan(p)
    y, p = y[keep], p[keep]
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            table.append({"bin": [float(edges[b]), float(edges[b + 1])], "n": int(m.sum()),
                          "mean_pred": float(p[m].mean()), "share_risky": float(y[m].mean())})
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"ece": float(ece), "table": table}


def bootstrap_draws(n: int, rng: np.random.Generator, n_boot: int = 2000) -> np.ndarray:
    """``(n_boot, n)`` resampled person indices, to reuse across paired comparisons."""
    return rng.integers(0, n, size=(n_boot, n))


def interval(values: np.ndarray, draws: np.ndarray, level: float = 0.95) -> list[float]:
    """``[mean, lower, upper]`` of per-person ``values`` (``(n,)`` or ``(n, K)``).

    For a 2-D input the result is one triple per column.
    """
    values = np.asarray(values, float)
    means = values[draws].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2], axis=0)
    est = values.mean(0)
    if values.ndim == 1:
        return [float(est), float(lo), float(hi)]
    return [[float(a), float(b), float(c)] for a, b, c in zip(est, lo, hi)]
