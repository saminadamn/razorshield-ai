"""Customer, merchant and device populations.

Transactions are generated *from* these entities rather than sampled as
independent rows. That is what makes the correlations in the final dataset
(spend level vs merchant category vs device churn) look like behaviour instead
of noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GeneratorConfig

MAX_DEVICES = 4
MAX_TRIPS = 2
SECONDS_PER_DAY = 86_400

# weight = share of traffic, ticket = multiplier on the customer's typical
# amount, risk = how attractive the category is to a fraudster,
# decline = base authorisation failure rate.
MERCHANT_CATEGORIES: dict[str, dict[str, float]] = {
    "grocery":       {"weight": 0.16, "ticket": 0.60, "risk": 0.15, "decline": 0.020},
    "food_delivery": {"weight": 0.18, "ticket": 0.35, "risk": 0.20, "decline": 0.020},
    "utilities":     {"weight": 0.10, "ticket": 0.80, "risk": 0.10, "decline": 0.015},
    "fashion":       {"weight": 0.12, "ticket": 1.20, "risk": 0.45, "decline": 0.030},
    "electronics":   {"weight": 0.08, "ticket": 4.00, "risk": 0.85, "decline": 0.040},
    "travel":        {"weight": 0.07, "ticket": 5.00, "risk": 0.70, "decline": 0.050},
    "gaming":        {"weight": 0.09, "ticket": 0.50, "risk": 0.90, "decline": 0.060},
    "gift_cards":    {"weight": 0.04, "ticket": 2.00, "risk": 1.00, "decline": 0.070},
    "crypto_onramp": {"weight": 0.03, "ticket": 3.00, "risk": 1.00, "decline": 0.080},
    "education":     {"weight": 0.06, "ticket": 2.50, "risk": 0.20, "decline": 0.030},
    "subscriptions": {"weight": 0.07, "ticket": 0.30, "risk": 0.35, "decline": 0.030},
}
CATEGORIES = list(MERCHANT_CATEGORIES)

PAYMENT_METHODS = ["UPI", "Card", "NetBanking", "Wallet", "EMI"]
METHOD_BASE = np.array([0.52, 0.28, 0.09, 0.08, 0.03])

# Multiplicative tilt applied to a customer's own method preference, per
# merchant category. Columns follow PAYMENT_METHODS.
METHOD_BY_CATEGORY = np.array([
    [1.3, 0.9, 0.4, 1.2, 0.1],   # grocery
    [1.4, 0.9, 0.2, 1.3, 0.1],   # food_delivery
    [1.2, 0.8, 1.4, 0.7, 0.1],   # utilities
    [1.0, 1.2, 0.6, 0.9, 0.8],   # fashion
    [0.7, 1.4, 0.9, 0.4, 2.5],   # electronics
    [0.6, 1.6, 1.1, 0.4, 1.8],   # travel
    [1.1, 1.3, 0.3, 1.4, 0.2],   # gaming
    [0.9, 1.5, 0.4, 1.2, 0.1],   # gift_cards
    [0.8, 1.3, 1.2, 0.6, 0.1],   # crypto_onramp
    [0.9, 1.1, 1.5, 0.4, 2.0],   # education
    [1.2, 1.4, 0.3, 0.8, 0.1],   # subscriptions
])

DEVICE_TYPES = ["Mobile", "Desktop", "Tablet", "POS"]
DEVICE_TYPE_P = np.array([0.78, 0.14, 0.05, 0.03])

# Diurnal shape indexed by hour of day.
DIURNAL = np.array([
    0.60, 0.35, 0.20, 0.15, 0.15, 0.25, 0.50, 0.90,
    1.30, 1.60, 1.80, 1.90, 1.90, 1.70, 1.60, 1.60,
    1.70, 1.90, 2.10, 2.30, 2.40, 2.00, 1.40, 0.90,
])
# Monday .. Sunday
WEEKDAY_FACTOR = np.array([1.00, 1.00, 1.02, 1.05, 1.15, 1.20, 1.05])


def build_merchants(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Sample the merchant population, with traffic share following a long tail."""
    cat_weights = np.array([MERCHANT_CATEGORIES[c]["weight"] for c in CATEGORIES])
    cat_weights = cat_weights / cat_weights.sum()
    cat_idx = rng.choice(len(CATEGORIES), size=cfg.n_merchants, p=cat_weights)

    # A handful of merchants carry most of the volume.
    traffic = rng.pareto(1.6, size=cfg.n_merchants) + 0.15

    ticket = np.array([MERCHANT_CATEGORIES[CATEGORIES[i]]["ticket"] for i in cat_idx])
    ticket = ticket * np.exp(rng.normal(0.0, 0.35, size=cfg.n_merchants))

    risk = np.array([MERCHANT_CATEGORIES[CATEGORIES[i]]["risk"] for i in cat_idx])
    risk = np.clip(risk * np.exp(rng.normal(0.0, 0.45, size=cfg.n_merchants)), 0.02, 3.0)

    decline = np.array([MERCHANT_CATEGORIES[CATEGORIES[i]]["decline"] for i in cat_idx])
    decline = np.clip(
        decline * np.exp(rng.normal(0.0, 0.30, size=cfg.n_merchants)), 0.005, 0.25
    )

    return pd.DataFrame({
        "merchant_id": [f"MERCH_{i:04d}" for i in range(cfg.n_merchants)],
        "category": [CATEGORIES[i] for i in cat_idx],
        "category_idx": cat_idx,
        "traffic_weight": traffic / traffic.sum(),
        "ticket_factor": ticket,
        "fraud_attractiveness": risk,
        "decline_base": decline,
    })


def build_customers(
    cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Sample the customer population plus their device and travel histories.

    Returns the scalar attributes as a frame, and the per-customer matrices
    (hour profile, device install times, trips) separately since they are only
    used by the simulator.
    """
    n = cfg.n_customers

    # Account age at t=0: about a third of the base is new, the rest is aged.
    new_cohort = rng.random(n) < 0.32
    age = np.where(
        new_cohort,
        rng.exponential(90.0, size=n),
        180.0 + rng.gamma(shape=2.2, scale=380.0, size=n),
    )
    account_age_days_start = np.clip(age, 1.0, 3600.0).astype(np.int32)

    # Typical spend, lognormal. exp(6.40) is roughly a 600 rupee median basket.
    spend_mu = rng.normal(6.40, 0.62, size=n)
    spend_sigma = rng.uniform(0.45, 1.10, size=n)

    # Transactions per day, calibrated so the run lands near cfg.n_transactions
    # once the warmup window is dropped.
    target_rate = cfg.n_transactions / (n * cfg.released_days)
    sigma = 0.80
    activity_rate = rng.lognormal(np.log(target_rate) - sigma**2 / 2, sigma, size=n)
    activity_rate = np.clip(activity_rate, 0.004, 3.0)

    # Per-customer hour profile: the shared diurnal shape, jittered, with a
    # night-owl minority.
    hour_w = DIURNAL[None, :] * np.exp(rng.normal(0.0, 0.35, size=(n, 24)))
    night_owl = rng.random(n) < 0.15
    late_hours = np.array([h for h in range(24) if h >= 22 or h <= 3])
    hour_w[np.ix_(np.where(night_owl)[0], late_hours)] *= 3.0
    hour_cdf = np.cumsum(hour_w / hour_w.sum(axis=1, keepdims=True), axis=1)

    # Per-customer category taste and payment method preference.
    base_cat = np.array([MERCHANT_CATEGORIES[c]["weight"] for c in CATEGORIES])
    cat_w = base_cat[None, :] * rng.gamma(2.5, 1.0, size=(n, len(CATEGORIES)))
    cat_cdf = np.cumsum(cat_w / cat_w.sum(axis=1, keepdims=True), axis=1)

    method_w = METHOD_BASE[None, :] * rng.gamma(3.0, 1.0, size=(n, len(PAYMENT_METHODS)))
    method_w = method_w / method_w.sum(axis=1, keepdims=True)

    device_type_idx = rng.choice(len(DEVICE_TYPES), size=n, p=DEVICE_TYPE_P)

    # Devices as install timestamps in seconds relative to t=0; negative means
    # the device predates the simulated window. Unused slots are +inf so they
    # are never available. A device installed mid-run produces a genuine
    # "new device" signal that is not fraud.
    install = np.full((n, MAX_DEVICES), np.inf)
    install[:, 0] = -account_age_days_start * rng.uniform(0.55, 1.0, size=n) * SECONDS_PER_DAY

    n_extra = rng.choice([0, 1, 2], size=n, p=[0.42, 0.42, 0.16])
    for slot in (1, 2):
        has_slot = n_extra >= slot
        offset = rng.uniform(0.15, 1.0, size=n) * account_age_days_start * SECONDS_PER_DAY
        install[:, slot] = np.where(has_slot, install[:, 0] + offset, np.inf)

    upgrades = rng.random(n) < 0.18  # bought a new phone mid-window
    install[:, 3] = np.where(
        upgrades, rng.uniform(0.0, cfg.days, size=n) * SECONDS_PER_DAY, np.inf
    )

    # Travel: short windows where the customer transacts from another location.
    home = rng.integers(0, cfg.n_locations, size=n)
    trip_start = np.full((n, MAX_TRIPS), np.inf)
    trip_end = np.full((n, MAX_TRIPS), -np.inf)
    trip_loc = np.zeros((n, MAX_TRIPS), dtype=np.int64)
    n_trips = rng.choice([0, 1, 2], size=n, p=[0.55, 0.33, 0.12])
    for t in range(MAX_TRIPS):
        takes = n_trips > t
        start = rng.uniform(0.0, cfg.days, size=n) * SECONDS_PER_DAY
        length = rng.uniform(2.0, 8.0, size=n) * SECONDS_PER_DAY
        trip_start[:, t] = np.where(takes, start, np.inf)
        trip_end[:, t] = np.where(takes, start + length, -np.inf)
        trip_loc[:, t] = rng.integers(0, cfg.n_locations, size=n)

    customers = pd.DataFrame({
        "customer_id": [f"CUST_{i:06d}" for i in range(n)],
        "account_age_days_start": account_age_days_start,
        "spend_mu": spend_mu,
        "spend_sigma": spend_sigma,
        "activity_rate": activity_rate,
        "device_type_idx": device_type_idx,
        "home_location": home,
    })

    arrays = {
        "hour_cdf": hour_cdf,
        "cat_cdf": cat_cdf,
        "method_w": method_w,
        "device_install": install,
        "trip_start": trip_start,
        "trip_end": trip_end,
        "trip_loc": trip_loc,
    }
    return customers, arrays
