"""Feature computation -- the single definition used by training and serving.

Every feature here is computable from information available *strictly before*
the transaction being scored. Two rules keep it honest:

1.  Rolling windows count prior attempts only. The current row never counts
    itself, except for the `unique_*_7d` features where the current device and
    location are legitimately known at scoring time.
2.  Chargebacks are only visible after `chargeback_lag_days`. A dispute raised
    on day 40 does not exist for a model scoring on day 20, so both
    `previous_chargeback_count` and `merchant_risk_score` apply the lag.

The window helpers at the top take (prior events, now) rather than a whole
frame, so the online path can call them with a customer's recent history
pulled from the event store and get identical numbers. `build_features` is the
batch replay over the simulated log.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GeneratorConfig
from .entities import DEVICE_TYPES, PAYMENT_METHODS, SECONDS_PER_DAY

HOUR = 3_600
DAY = SECONDS_PER_DAY
WEEK = 7 * SECONDS_PER_DAY

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# The 20 model features, in the order they appear in the released file.
FEATURE_COLUMNS = [
    "amount",
    "transaction_hour",
    "day_of_week",
    "payment_method",
    "device_type",
    "device_age_days",
    "account_age_days",
    "transactions_last_1h",
    "transactions_last_24h",
    "avg_transaction_amount",
    "amount_deviation_ratio",
    "failed_attempts_24h",
    "unique_devices_7d",
    "unique_locations_7d",
    "previous_transaction_count",
    "previous_chargeback_count",
    "merchant_risk_score",
    "velocity_score",
]
CATEGORICAL_COLUMNS = ["day_of_week", "payment_method", "device_type"]
# Present in the file for joins and temporal splitting; never model inputs.
IDENTIFIER_COLUMNS = ["transaction_id", "timestamp", "customer_id", "merchant_id"]
TARGET = "is_fraud"

MAX_DEVIATION_RATIO = 100.0


# --------------------------------------------------------------------------
# window primitives -- shared by batch replay and online scoring
# --------------------------------------------------------------------------

def count_in_window(prior_ts: np.ndarray, now: float, window: float) -> int:
    """Number of prior events in [now - window, now)."""
    lo = np.searchsorted(prior_ts, now - window, side="left")
    return int(prior_ts.size - lo)


def sum_in_window(
    prior_ts: np.ndarray, prior_values: np.ndarray, now: float, window: float
) -> float:
    lo = np.searchsorted(prior_ts, now - window, side="left")
    return float(prior_values[lo:].sum())


def unique_in_window(
    prior_ts: np.ndarray,
    prior_codes: np.ndarray,
    now: float,
    window: float,
    current_code: int,
) -> int:
    """Distinct values over the trailing window, including the current event."""
    lo = np.searchsorted(prior_ts, now - window, side="left")
    return int(np.unique(np.append(prior_codes[lo:], current_code)).size)


def velocity_score(n_1h: float, n_24h: float, observed_rate_per_day: float) -> float:
    """Bounded 0-1 burst score: recent counts against the customer's own rate.

    Deliberately a heuristic composite rather than a tuned threshold -- the
    numbers below are scale factors, not calibrated cut-offs.
    """
    r1 = n_1h / (observed_rate_per_day / 24.0 + 0.05)
    r24 = n_24h / (observed_rate_per_day + 0.5)
    return float(1.0 - np.exp(-(0.20 * np.log1p(r1) + 0.25 * np.log1p(r24))))


def _rolling_unique(codes: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Distinct count over a sliding window [lo[i], i], for non-decreasing lo."""
    n = codes.size
    out = np.empty(n, dtype=np.int32)
    counts: dict[int, int] = {}
    left = 0
    distinct = 0
    for i in range(n):
        c = int(codes[i])
        if counts.get(c, 0) == 0:
            distinct += 1
        counts[c] = counts.get(c, 0) + 1
        while left < lo[i]:
            c2 = int(codes[left])
            counts[c2] -= 1
            if counts[c2] == 0:
                distinct -= 1
            left += 1
        out[i] = distinct
    return out


# --------------------------------------------------------------------------
# batch replay
# --------------------------------------------------------------------------

def _customer_features(
    events: pd.DataFrame, out: dict[str, np.ndarray], cfg: GeneratorConfig
) -> None:
    """Per-customer rolling history, written back in original row order."""
    cust = events["cust_idx"].to_numpy()
    ts = events["ts"].to_numpy()
    amount = events["amount"].to_numpy()
    declined = events["is_declined"].to_numpy().astype(np.int64)
    chargeback = events["chargeback"].to_numpy()
    device_code = events["device_code"].to_numpy()
    location = events["location"].to_numpy()

    order = np.lexsort((ts, cust))
    bounds = np.flatnonzero(np.diff(cust[order])) + 1
    lag = cfg.chargeback_lag_days * DAY

    for grp in np.split(order, bounds):
        t = ts[grp]
        a = amount[grp]
        n = t.size
        rank = np.arange(n)

        # Prior counts: rows strictly before i, within the window.
        lo_1h = np.searchsorted(t, t - HOUR, side="left")
        lo_24h = np.searchsorted(t, t - DAY, side="left")
        lo_7d = np.searchsorted(t, t - WEEK, side="left")
        out["transactions_last_1h"][grp] = rank - lo_1h
        out["transactions_last_24h"][grp] = rank - lo_24h

        # Prior declines in the last 24h, via prefix sums.
        fail_cum = np.concatenate([[0], np.cumsum(declined[grp])])
        out["failed_attempts_24h"][grp] = fail_cum[rank] - fail_cum[lo_24h]

        # Expanding mean of prior amounts; row 0 has no history.
        amt_cum = np.concatenate([[0.0], np.cumsum(a)])
        with np.errstate(invalid="ignore", divide="ignore"):
            prior_mean = np.where(rank > 0, amt_cum[rank] / np.maximum(rank, 1), np.nan)
        out["avg_transaction_amount"][grp] = prior_mean

        out["previous_transaction_count"][grp] = rank

        # Distinct devices / locations over 7 days, current row included.
        out["unique_devices_7d"][grp] = _rolling_unique(device_code[grp], lo_7d)
        out["unique_locations_7d"][grp] = _rolling_unique(location[grp], lo_7d)

        # Chargebacks the customer has raised that are confirmed by now.
        cb = np.sort(t[chargeback[grp]] + lag)
        out["previous_chargeback_count"][grp] = np.searchsorted(cb, t, side="left")

        # Observed rate uses only history, so it is zero on the first row.
        days_seen = np.maximum((t - t[0]) / DAY, 1.0)
        out["_observed_rate"][grp] = rank / days_seen


def _merchant_risk(
    events: pd.DataFrame, out: dict[str, np.ndarray], cfg: GeneratorConfig
) -> None:
    """Beta-posterior chargeback rate per merchant, respecting the dispute lag.

    At time t the score uses chargebacks confirmed before t over transactions
    old enough to have been disputed (before t - lag). Cold-start merchants sit
    at the prior mean until they accumulate history.
    """
    merch = events["merchant_idx"].to_numpy()
    ts = events["ts"].to_numpy()
    chargeback = events["chargeback"].to_numpy()
    lag = cfg.chargeback_lag_days * DAY
    a, b = cfg.merchant_prior_alpha, cfg.merchant_prior_beta

    order = np.lexsort((ts, merch))
    bounds = np.flatnonzero(np.diff(merch[order])) + 1

    for grp in np.split(order, bounds):
        t = ts[grp]
        matured = np.searchsorted(t, t - lag, side="left")
        cb_conf = np.sort(t[chargeback[grp]] + lag)
        cb_before = np.searchsorted(cb_conf, t, side="left")
        out["merchant_risk_score"][grp] = (a + cb_before) / (a + b + matured)


def build_features(
    events: pd.DataFrame,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    cfg: GeneratorConfig,
) -> pd.DataFrame:
    """Turn the raw event log into the released feature table.

    Expects `events` sorted by `ts` with a `chargeback` column already set.
    """
    events = events.reset_index(drop=True)
    n = len(events)

    # Stable integer codes for the sliding-window distinct counts.
    events["device_code"] = pd.factorize(events["device_id"])[0]

    out: dict[str, np.ndarray] = {
        "transactions_last_1h": np.zeros(n, dtype=np.int32),
        "transactions_last_24h": np.zeros(n, dtype=np.int32),
        "failed_attempts_24h": np.zeros(n, dtype=np.int32),
        "avg_transaction_amount": np.zeros(n, dtype=np.float64),
        "previous_transaction_count": np.zeros(n, dtype=np.int32),
        "unique_devices_7d": np.zeros(n, dtype=np.int32),
        "unique_locations_7d": np.zeros(n, dtype=np.int32),
        "previous_chargeback_count": np.zeros(n, dtype=np.int32),
        "merchant_risk_score": np.zeros(n, dtype=np.float64),
        "_observed_rate": np.zeros(n, dtype=np.float64),
    }
    _customer_features(events, out, cfg)
    _merchant_risk(events, out, cfg)

    ts = events["ts"].to_numpy()
    # Whole seconds: sub-second precision here is simulation artefact, not signal.
    timestamp = pd.Timestamp(cfg.start_date) + pd.to_timedelta(np.round(ts), unit="s")

    # A customer with no history gets the population prior, measured on the
    # warmup window only so the released data contains no look-ahead. The same
    # constant must be used at serving time -- it is saved to the manifest.
    warm = ts < cfg.warmup_days * DAY
    cold_start_amount = float(np.median(events["amount"].to_numpy()[warm]))
    avg_amount = np.where(
        np.isnan(out["avg_transaction_amount"]),
        cold_start_amount,
        out["avg_transaction_amount"],
    )

    deviation = np.clip(
        events["amount"].to_numpy() / np.maximum(avg_amount, 1.0), 0.0, MAX_DEVIATION_RATIO
    )

    device_age = np.maximum(
        0.0, (ts - events["device_install_ts"].to_numpy()) / DAY
    ).astype(np.int32)
    account_age = (
        customers["account_age_days_start"].to_numpy()[events["cust_idx"].to_numpy()]
        + (ts / DAY)
    ).astype(np.int32)

    velocity = np.array([
        velocity_score(n1, n24, rate)
        for n1, n24, rate in zip(
            out["transactions_last_1h"],
            out["transactions_last_24h"],
            out["_observed_rate"], strict=False,
        )
    ])

    frame = pd.DataFrame({
        "timestamp": timestamp,
        "customer_id": customers["customer_id"].to_numpy()[events["cust_idx"].to_numpy()],
        "merchant_id": merchants["merchant_id"].to_numpy()[events["merchant_idx"].to_numpy()],
        "amount": events["amount"].to_numpy(),
        "transaction_hour": timestamp.hour.to_numpy().astype(np.int16),
        "day_of_week": [DAY_NAMES[d] for d in timestamp.weekday],
        "payment_method": [PAYMENT_METHODS[i] for i in events["method_idx"]],
        "device_type": [DEVICE_TYPES[i] for i in events["device_type_idx"]],
        "device_age_days": device_age,
        "account_age_days": account_age,
        "transactions_last_1h": out["transactions_last_1h"],
        "transactions_last_24h": out["transactions_last_24h"],
        "avg_transaction_amount": np.round(avg_amount, 2),
        "amount_deviation_ratio": np.round(deviation, 4),
        "failed_attempts_24h": out["failed_attempts_24h"],
        "unique_devices_7d": out["unique_devices_7d"],
        "unique_locations_7d": out["unique_locations_7d"],
        "previous_transaction_count": out["previous_transaction_count"],
        "previous_chargeback_count": out["previous_chargeback_count"],
        "merchant_risk_score": np.round(out["merchant_risk_score"], 6),
        "velocity_score": np.round(velocity, 4),
        # The observed label: what the business actually recorded, noise and all.
        "is_fraud": events["is_fraud_observed"].to_numpy(),
    })

    meta = pd.DataFrame({
        "true_fraud": events["true_fraud"].to_numpy(),
        "label_flipped": (
            events["is_fraud_observed"].to_numpy() != events["true_fraud"].to_numpy()
        ),
        "fraud_archetype": events["archetype"].to_numpy(),
        "is_hard_negative": events["is_hard_negative"].to_numpy(),
        "is_declined": events["is_declined"].to_numpy(),
        "chargeback": events["chargeback"].to_numpy(),
        "device_id": events["device_id"].to_numpy(),
        "location": events["location"].to_numpy(),
        "merchant_category": merchants["category"].to_numpy()[
            events["merchant_idx"].to_numpy()
        ],
    })
    frame.attrs["cold_start_amount"] = cold_start_amount
    return frame, meta
