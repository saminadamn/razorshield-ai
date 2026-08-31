"""Turning SHAP contributions into a sentence, deterministically.

No language model is involved anywhere in this file. SHAP decides *which*
factors mattered and in which direction; these templates only put the factor
and its actual value into words. That distinction matters -- an explanation
that a model invented is not an explanation, and in a payments context it is a
liability.
"""

from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    "amount": "Transaction amount",
    "transaction_hour": "Time of day",
    "day_of_week": "Day of week",
    "payment_method": "Payment method",
    "device_type": "Device type",
    "device_age_days": "Device age",
    "account_age_days": "Account age",
    "transactions_last_1h": "Transaction velocity (1h)",
    "transactions_last_24h": "Transaction velocity (24h)",
    "avg_transaction_amount": "Customer average amount",
    "amount_deviation_ratio": "Amount deviation",
    "failed_attempts_24h": "Failed attempts",
    "unique_devices_7d": "Devices used (7d)",
    "unique_locations_7d": "Locations used (7d)",
    "previous_transaction_count": "Account history",
    "previous_chargeback_count": "Prior chargebacks",
    "merchant_risk_score": "Merchant risk",
    "velocity_score": "Velocity score",
}


def _plural(n: float, one: str, many: str) -> str:
    return one if abs(n - 1) < 1e-9 else many


def _ordinal_hour(h: float) -> str:
    hour = int(h)
    if hour == 0:
        return "midnight"
    if hour == 12:
        return "midday"
    suffix = "am" if hour < 12 else "pm"
    display = hour if hour <= 12 else hour - 12
    return f"{display}{suffix}"


# Each entry renders the *evidence*, not the verdict: what the value was, in
# plain words. Direction is supplied separately by the sign of the SHAP value.
PHRASES: dict[str, callable] = {
    "amount": lambda v: (
        f"a Rs{v:,.2f} transaction" if v < 100 else f"a Rs{v:,.0f} transaction"
    ),
    "transaction_hour": lambda v: f"a transaction at {_ordinal_hour(v)}",
    "day_of_week": lambda v: f"a {v} transaction",
    "payment_method": lambda v: f"payment by {v}",
    "device_type": lambda v: f"a {str(v).lower()} device",
    "device_age_days": lambda v: (
        "a device first seen today"
        if v < 1
        else f"a device first seen {int(v)} {_plural(int(v), 'day', 'days')} ago"
    ),
    "account_age_days": lambda v: f"an account {int(v)} {_plural(int(v), 'day', 'days')} old",
    "transactions_last_1h": lambda v: (
        "no prior activity in the last hour"
        if v == 0
        else f"{int(v)} {_plural(int(v), 'transaction', 'transactions')} in the previous hour"
    ),
    "transactions_last_24h": lambda v: (
        "no prior activity in the last 24 hours"
        if v == 0
        else f"{int(v)} {_plural(int(v), 'transaction', 'transactions')} in the last 24 hours"
    ),
    "avg_transaction_amount": lambda v: f"a customer average of Rs{v:,.0f}",
    "amount_deviation_ratio": lambda v: f"an amount {v:.1f}x the customer's average",
    "failed_attempts_24h": lambda v: (
        "no failed attempts"
        if v == 0
        else f"{int(v)} failed {_plural(int(v), 'attempt', 'attempts')} in 24 hours"
    ),
    "unique_devices_7d": lambda v: (
        f"{int(v)} {_plural(int(v), 'device', 'devices')} used this week"
    ),
    "unique_locations_7d": lambda v: (
        f"{int(v)} {_plural(int(v), 'location', 'locations')} this week"
    ),
    "previous_transaction_count": lambda v: (
        "no prior transaction history"
        if v == 0
        else f"{int(v)} prior {_plural(int(v), 'transaction', 'transactions')}"
    ),
    "previous_chargeback_count": lambda v: (
        "no prior chargebacks"
        if v == 0
        else f"{int(v)} prior {_plural(int(v), 'chargeback', 'chargebacks')}"
    ),
    "merchant_risk_score": lambda v: f"a merchant risk score of {v:.3f}",
    "velocity_score": lambda v: f"a velocity score of {v:.2f}",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def phrase(feature: str, value) -> str:
    renderer = PHRASES.get(feature)
    if renderer is None:
        return f"{label(feature).lower()} of {value}"
    try:
        return renderer(value)
    except (TypeError, ValueError):
        return f"{label(feature).lower()} of {value}"


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def summarise(factors: list, band: str, top_k: int = 3) -> str:
    """Build the one-line explanation from the strongest signed factors."""
    pushing_up = [f for f in factors if f.contribution > 0][:top_k]
    pushing_down = [f for f in factors if f.contribution < 0][:2]

    if not pushing_up:
        drivers = "nothing unusual in this transaction"
    else:
        drivers = _join([phrase(f.feature, f.value) for f in pushing_up])

    # "Low risk driven by ..." reads as a contradiction. Below the alert
    # threshold the factors are evidence that was weighed, not a verdict.
    elevated = band.lower() not in ("low", "minimal")
    if elevated:
        sentence = f"{band} risk driven by {drivers}."
    else:
        sentence = f"{band} risk. Strongest signals were {drivers}."

    if pushing_down:
        offsets = _join([phrase(f.feature, f.value) for f in pushing_down])
        sentence += f" Offset by {offsets}."
    return sentence
