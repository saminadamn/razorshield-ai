"""The 0-100 risk score and its action bands.

Two things are easy to get wrong here.

**The score.** Showing the calibrated probability times 100 would put almost
every transaction between 0 and 5, because the base rate is 3%. The score is
therefore piecewise-linear in *log-odds*, anchored so that the band boundaries
land exactly on the familiar 0 / 30 / 60 / 80 / 100 cut points. Log-odds is the
right space: it is the scale the model actually works in, and it spreads the
interesting top end out instead of crushing it against zero.

**The boundaries.** 0-29 LOW / 30-59 MEDIUM / 60-79 HIGH / 80-100 CRITICAL is a
presentation choice. Where those bands fall in probability is not -- it is
fitted on validation data from what each band is supposed to *mean*:

    CRITICAL  the tail above it is >=90% fraud -- safe to block outright
    HIGH      down to the Phase 2 cost-optimal threshold -- worth reviewing
    MEDIUM    at least 3x the base rate -- step-up challenge, not a block
    LOW       everything else -- let it through

The ladder is ordered by *action*, and the cost-optimal threshold is the
boundary of acting at all, so it sits at the bottom of HIGH rather than the
bottom of CRITICAL. Anchoring CRITICAL there instead makes CRITICAL swallow the
entire action set and leaves HIGH with nothing in it.

CRITICAL is set at 90% rather than a bare majority because it is the only band
that acts without a human: blocking a band that is 67% fraud means one in three
blocked customers did nothing wrong.

So the score is a rescaling of a calibrated probability, and the cut points are
an operating decision measured on held-out data -- not four round numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

BAND_EDGES = ((0.0, "LOW"), (30.0, "MEDIUM"), (60.0, "HIGH"), (80.0, "CRITICAL"))
BAND_NAMES = [name for _, name in BAND_EDGES]

CRITICAL_BAND_FRAUD_RATE = 0.90
MEDIUM_BAND_LIFT = 3.0
MIN_BAND_ROWS = 50


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _extend_down(
    p: np.ndarray,
    y: np.ndarray,
    upper: float,
    target_rate: float,
    grid: int = 300,
) -> float:
    """Lowest threshold at which the band [t, upper) still hits `target_rate`.

    Scans every candidate rather than stopping at the first miss: isotonic
    calibration produces a step function with heavy ties, so the band rate is
    only roughly monotone in the threshold and an early break lands on noise.
    """
    below = p[p < upper]
    if below.size < MIN_BAND_ROWS:
        return upper
    candidates = np.unique(np.quantile(below, np.linspace(0.0, 1.0, grid)))
    best = upper
    for t in candidates:  # ascending, so the first pass is the lowest
        mask = (p >= t) & (p < upper)
        if mask.sum() < MIN_BAND_ROWS:
            continue
        if y[mask].mean() >= target_rate:
            return float(t)
    return best


def _threshold_for_cumulative_rate(
    p: np.ndarray, y: np.ndarray, target_rate: float
) -> float:
    """Lowest threshold whose entire tail above it is at least `target_rate` fraud.

    Used for the CRITICAL boundary, where the question is about the whole top
    band rather than a slice, and which is robust to the ties isotonic leaves.
    """
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    counts = np.arange(1, ys.size + 1)
    cumulative = np.cumsum(ys) / counts
    ok = np.flatnonzero((cumulative >= target_rate) & (counts >= MIN_BAND_ROWS))
    if ok.size == 0:
        return float(ps[0])
    return float(ps[ok[-1]])


@dataclass
class RiskScale:
    """Maps calibrated probability to a 0-100 score and an action band."""

    p_medium: float
    p_high: float
    p_critical: float
    p_floor: float = 1e-5
    p_ceiling: float = 0.999

    def anchors(self) -> tuple[np.ndarray, np.ndarray]:
        points = [
            self.p_floor, self.p_medium, self.p_high, self.p_critical, self.p_ceiling
        ]
        # Nudge any collapsed boundary so the interpolation stays strictly
        # increasing (happens when a band comes out empty).
        for i in range(1, len(points)):
            if points[i] <= points[i - 1]:
                points[i] = points[i - 1] * 1.0001 + 1e-12
        return np.array(points), np.array([0.0, 30.0, 60.0, 80.0, 100.0])

    def score(self, p: np.ndarray) -> np.ndarray:
        xs, ys = self.anchors()
        clipped = np.clip(np.asarray(p, dtype=float), self.p_floor, self.p_ceiling)
        return np.round(np.interp(_logit(clipped), _logit(xs), ys), 1)

    def band(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=float)
        out = np.full(score.shape, BAND_NAMES[0], dtype=object)
        for edge, name in BAND_EDGES[1:]:
            out[score >= edge] = name
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def fit_scale(
    p: np.ndarray,
    y: np.ndarray,
    cost_threshold: float,
    critical_rate: float = CRITICAL_BAND_FRAUD_RATE,
    medium_lift: float = MEDIUM_BAND_LIFT,
) -> RiskScale:
    """Choose band boundaries on validation data."""
    base_rate = float(y.mean())
    # Acting at all starts here, so this is the floor of HIGH, not of CRITICAL.
    p_high = float(cost_threshold)
    p_critical = max(_threshold_for_cumulative_rate(p, y, critical_rate), p_high)
    p_medium = _extend_down(p, y, p_high, medium_lift * base_rate)
    return RiskScale(p_medium=p_medium, p_high=p_high, p_critical=p_critical)


def band_stats(
    p: np.ndarray, y: np.ndarray, scale: RiskScale, amount: np.ndarray | None = None
) -> list[dict]:
    """What each band actually contains -- volume, purity, and fraud captured."""
    scores = scale.score(p)
    bands = scale.band(scores)
    base_rate = float(y.mean())
    total_fraud = float(y.sum())

    rows = []
    for name in BAND_NAMES:
        mask = bands == name
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "band": name, "n": 0, "share": 0.0, "fraud_rate": 0.0,
                "lift": 0.0, "fraud_captured": 0.0, "recall": 0.0,
                "score_range": "-", "amount_at_risk": 0.0,
            })
            continue
        fraud = float(y[mask].sum())
        rows.append({
            "band": name,
            "n": n,
            "share": n / p.size,
            "fraud_rate": fraud / n,
            "lift": (fraud / n) / base_rate if base_rate else 0.0,
            "fraud_captured": fraud,
            "recall": fraud / total_fraud if total_fraud else 0.0,
            "score_range": f"{scores[mask].min():.0f}-{scores[mask].max():.0f}",
            "amount_at_risk": float(amount[mask][y[mask] == 1].sum())
            if amount is not None
            else 0.0,
        })
    return rows
