"""Online feature computation -- the same definitions, one event at a time.

`features.py` computes features by replaying a whole log. This computes them
for a single incoming payment against whatever history the store holds. The two
have to agree exactly, or the model is scored on inputs it was never trained
on. `parity.py` asserts that they do.

Timestamps are Unix seconds in UTC everywhere. The seeder converts simulation
time to the same basis so that a replayed row lands on the same wall clock as
the batch pipeline gave it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.features import (
    DAY_NAMES,
    FEATURE_COLUMNS,
    MAX_DEVIATION_RATIO,
    velocity_score,
)

HOUR = 3_600.0
DAY = 86_400.0
WEEK = 7 * DAY


@dataclass(frozen=True)
class ServingConstants:
    """Values the online path must share with training. From `manifest.json`."""

    cold_start_amount: float
    chargeback_lag_days: float
    merchant_prior_alpha: float
    merchant_prior_beta: float

    @classmethod
    def from_manifest(cls, path: Path) -> ServingConstants:
        data = json.loads(Path(path).read_text())["serving_constants"]
        return cls(
            cold_start_amount=float(data["cold_start_amount"]),
            chargeback_lag_days=float(data["chargeback_lag_days"]),
            merchant_prior_alpha=float(data["merchant_prior_alpha"]),
            merchant_prior_beta=float(data["merchant_prior_beta"]),
        )

    @property
    def lag_seconds(self) -> float:
        return self.chargeback_lag_days * DAY


@dataclass
class RawEvent:
    """A payment attempt, after the webhook payload has been unpacked."""

    transaction_id: str
    customer_id: str
    merchant_id: str
    ts: float
    amount: float
    payment_method: str
    device_id: str
    device_type: str
    location: str
    declined: bool = False
    # Set when a dispute is already known. Visibility is still gated by the
    # confirmation lag at query time, so recording it early is safe.
    chargeback_confirmed_at: float | None = None


class OnlineFeatureBuilder:
    def __init__(self, store, constants: ServingConstants):
        self.store = store
        self.constants = constants

    def build(self, event: RawEvent) -> dict:
        """The 18 model features for one event, from stored history."""
        now = event.ts
        store = self.store
        history = store.customer_history(event.customer_id, now, WEEK)

        window_ts = np.asarray(history.window_ts, dtype=float)
        declined = np.asarray(history.window_declined, dtype=int)

        # Rolling counts over prior attempts only; the store already excluded
        # the current event, which has not been written yet.
        in_1h = window_ts >= now - HOUR
        in_24h = window_ts >= now - DAY
        n_1h = int(in_1h.sum())
        n_24h = int(in_24h.sum())
        failed_24h = int(declined[in_24h].sum()) if declined.size else 0

        prior_count = history.prior_count
        if prior_count > 0:
            avg_amount = history.prior_amount_sum / prior_count
        else:
            # No history: the population prior measured on the training window.
            avg_amount = self.constants.cold_start_amount
        # Deviation divides by the *unrounded* mean, and the mean is rounded
        # only for display -- the batch pipeline does it in that order, and
        # doing it the other way round shifts the ratio in the 4th decimal.
        deviation = float(np.round(
            np.clip(event.amount / max(avg_amount, 1.0), 0.0, MAX_DEVIATION_RATIO), 4
        ))
        # np.round, not round(): they disagree on exact .xx5 values and the
        # batch pipeline uses np.round. Matching it is what keeps parity exact.
        avg_amount = float(np.round(avg_amount, 2))

        # Distinct devices and locations include the current event: at scoring
        # time we already know what this transaction is coming from.
        unique_devices = len(set(history.window_device) | {event.device_id})
        unique_locations = len(set(history.window_location) | {event.location})

        # Observed rate uses history only, so it is zero on a first transaction.
        if history.first_ts is None:
            observed_rate = 0.0
        else:
            days_seen = max((now - history.first_ts) / DAY, 1.0)
            observed_rate = prior_count / days_seen

        tenure = store.customer_tenure(event.customer_id)
        if tenure is None:
            account_age = 0
        else:
            first_seen, age_at_first_seen = tenure
            account_age = int(age_at_first_seen + (now - first_seen) / DAY)

        # A device we have never seen is new to us. device_age_days = 0 is the
        # single strongest signal the model has, so this default is deliberate.
        device_seen = store.device_first_seen(event.customer_id, event.device_id)
        device_age = (
            0 if device_seen is None else int(max(0.0, (now - device_seen) / DAY))
        )

        matured, confirmed = store.merchant_exposure(
            event.merchant_id, now, self.constants.lag_seconds
        )
        alpha, beta = (
            self.constants.merchant_prior_alpha,
            self.constants.merchant_prior_beta,
        )
        merchant_risk = float(np.round((alpha + confirmed) / (alpha + beta + matured), 6))

        when = datetime.fromtimestamp(float(np.round(now)), tz=UTC)

        return {
            "amount": event.amount,
            "transaction_hour": when.hour,
            "day_of_week": DAY_NAMES[when.weekday()],
            "payment_method": event.payment_method,
            "device_type": event.device_type,
            "device_age_days": device_age,
            "account_age_days": account_age,
            "transactions_last_1h": n_1h,
            "transactions_last_24h": n_24h,
            "avg_transaction_amount": avg_amount,
            "amount_deviation_ratio": deviation,
            "failed_attempts_24h": failed_24h,
            "unique_devices_7d": unique_devices,
            "unique_locations_7d": unique_locations,
            "previous_transaction_count": prior_count,
            "previous_chargeback_count": history.confirmed_chargebacks,
            "merchant_risk_score": merchant_risk,
            "velocity_score": float(np.round(velocity_score(n_1h, n_24h, observed_rate), 4)),
        }

    def build_frame(self, event: RawEvent) -> pd.DataFrame:
        """Feature row as the single-row frame the scorer expects."""
        return pd.DataFrame([self.build(event)])[FEATURE_COLUMNS]

    def observe(self, event: RawEvent, commit: bool = True) -> None:
        """Write the event to the store so later transactions can see it.

        Called *after* scoring: a transaction must never be part of its own
        history.
        """
        self.store.register_customer(event.customer_id, event.ts)
        self.store.register_device(
            event.customer_id, event.device_id, event.ts, event.device_type
        )
        self.store.record_event(
            transaction_id=event.transaction_id,
            customer_id=event.customer_id,
            merchant_id=event.merchant_id,
            ts=event.ts,
            amount=event.amount,
            payment_method=event.payment_method,
            device_id=event.device_id,
            device_type=event.device_type,
            location=event.location,
            declined=event.declined,
            chargeback_confirmed_at=event.chargeback_confirmed_at,
        )
        if commit:
            self.store.commit()
