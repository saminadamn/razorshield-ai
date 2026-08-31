"""Probability calibration.

Phase 3 found the selected model is systematically under-confident at the top
of its range: when it says 0.90 the observed fraud rate is 0.98. Ranking
metrics (PR-AUC, ROC-AUC) cannot see this, because they only care about order.
A risk *score* does care -- if the number shown to an analyst is going to be
read as "how likely is this to be fraud", it has to mean that.

Isotonic regression fixes the miscalibration without touching the ranking: it
is monotone, so every ordering-based metric is unchanged by construction.
Fitted on validation, never on test.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression mapping raw probability to observed frequency."""
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(p, y)
    return calibrator


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, bins: int = 20
) -> tuple[float, float]:
    """Quantile-binned ECE and the worst single-bin deviation.

    Quantile bins rather than equal-width: at a 3% base rate almost every
    prediction sits near zero, so equal-width bins would put 99% of the data
    in one bucket and report a meaningless number.
    """
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0, 0.0
    total, worst = 0.0, 0.0
    n = p.size
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        last = i == edges.size - 2
        mask = (p >= lo) & (p <= hi if last else p < hi)
        if not mask.any():
            continue
        deviation = abs(p[mask].mean() - y[mask].mean())
        total += mask.sum() / n * deviation
        worst = max(worst, deviation)
    return float(total), float(worst)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    ece, worst = expected_calibration_error(y, p)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece,
        "max_bin_deviation": worst,
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }
