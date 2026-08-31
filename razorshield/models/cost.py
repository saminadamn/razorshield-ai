"""The business objective: what a wrong decision actually costs.

Picking the model with the best F1 is a modelling decision dressed up as a
business one. A missed fraud costs the transaction amount plus a dispute fee.
A false positive costs a review, and annoys a real customer who may abandon
the purchase. Those are different currencies and different magnitudes, and the
ratio between them decides both which model to ship and where to threshold it.

Every number here is an assumption, stated in one place so it can be argued
with and varied. `sensitivity` exists because the honest answer to "which
model is best" is sometimes "it depends on the cost ratio".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Rupee costs per decision outcome."""

    # Missed fraud: the amount is lost, plus the issuer's dispute fee.
    chargeback_fee: float = 500.0
    # Every alert costs analyst time, whether or not it turns out to be fraud.
    review_cost: float = 40.0
    # A blocked genuine customer: abandoned basket, support contact, goodwill.
    false_decline_cost: float = 250.0
    # Share of a flagged fraud that review actually stops.
    review_catch_rate: float = 1.0

    def evaluate(
        self, y_true: np.ndarray, flagged: np.ndarray, amount: np.ndarray
    ) -> dict[str, float]:
        """Total cost of one set of decisions, plus the confusion counts."""
        y_true = y_true.astype(bool)
        flagged = flagged.astype(bool)

        tp = flagged & y_true
        fp = flagged & ~y_true
        fn = ~flagged & y_true

        # Caught fraud still costs a review, and whatever review fails to stop.
        caught_loss = (1.0 - self.review_catch_rate) * (
            amount[tp].sum() + self.chargeback_fee * tp.sum()
        )
        cost_tp = self.review_cost * tp.sum() + caught_loss
        cost_fp = (self.review_cost + self.false_decline_cost) * fp.sum()
        cost_fn = amount[fn].sum() + self.chargeback_fee * fn.sum()

        # Doing nothing at all: every fraud goes through.
        baseline = amount[y_true].sum() + self.chargeback_fee * y_true.sum()
        total = cost_tp + cost_fp + cost_fn

        return {
            "total_cost": float(total),
            "cost_missed_fraud": float(cost_fn),
            "cost_false_positives": float(cost_fp),
            "cost_reviews": float(cost_tp),
            "baseline_cost": float(baseline),
            "net_saving": float(baseline - total),
            "saving_pct": float((baseline - total) / baseline) if baseline else 0.0,
            "true_positives": int(tp.sum()),
            "false_positives": int(fp.sum()),
            "false_negatives": int(fn.sum()),
            "alerts": int(flagged.sum()),
            "alert_rate": float(flagged.mean()),
        }


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    cost: CostModel,
    n_candidates: int = 400,
) -> tuple[float, dict[str, float]]:
    """Cost-minimising threshold, chosen on validation data only."""
    candidates = np.unique(
        np.quantile(y_prob, np.linspace(0.50, 0.9999, n_candidates))
    )
    best_t, best = candidates[0], None
    for t in candidates:
        result = cost.evaluate(y_true, y_prob >= t, amount)
        if best is None or result["total_cost"] < best["total_cost"]:
            best_t, best = float(t), result
    return best_t, best


def threshold_at_budget(y_prob: np.ndarray, budget: float) -> float:
    """Threshold that flags a fixed share of traffic -- the capacity view.

    Review teams have a headcount, not a cost function. This is the operating
    point you get when the constraint is how many alerts a day can be worked.
    """
    return float(np.quantile(y_prob, 1.0 - budget))


def sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    base: CostModel,
    multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> list[dict[str, float]]:
    """How the optimal operating point moves as false positives get pricier."""
    rows = []
    for m in multipliers:
        variant = CostModel(
            chargeback_fee=base.chargeback_fee,
            review_cost=base.review_cost,
            false_decline_cost=base.false_decline_cost * m,
            review_catch_rate=base.review_catch_rate,
        )
        t, result = choose_threshold(y_true, y_prob, amount, variant)
        rows.append({
            "fp_cost_multiplier": m,
            "false_decline_cost": variant.false_decline_cost,
            "threshold": t,
            "alert_rate": result["alert_rate"],
            "recall": result["true_positives"]
            / max(result["true_positives"] + result["false_negatives"], 1),
            "saving_pct": result["saving_pct"],
        })
    return rows
