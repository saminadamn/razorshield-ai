"""Event-stream simulation: legitimate traffic, fraud episodes, hard negatives.

The generator never assigns `is_fraud` from a formula over the released
features. Instead it simulates behaviour -- a background stream of genuine
transactions, plus injected episodes -- and the features are *measured* off
that stream afterwards in `features.py`. A model therefore has to recover the
behaviour, not reverse-engineer a labelling function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GeneratorConfig
from .entities import (
    CATEGORIES,
    DEVICE_TYPES,
    MAX_DEVICES,
    MAX_TRIPS,
    METHOD_BY_CATEGORY,
    PAYMENT_METHODS,
    SECONDS_PER_DAY,
    WEEKDAY_FACTOR,
    build_customers,
    build_merchants,
)

# Share of fraud *episodes* by archetype. These are weighted so the resulting
# share of fraud *rows* is roughly balanced: card testing emits ~25 rows per
# episode while merchant collusion emits ~2, so equal episode weights would
# leave the label dominated by the easiest pattern to detect.
# Target row mix is roughly ato 32 / testing 25 / swap 13 / bleed 20 / ring 10.
# `slow_bleed` and `merchant_collusion` are deliberately hard: they overlap
# with ordinary behaviour and exist to stop recall from being free.
FRAUD_ARCHETYPES = {
    "ato_burst": 0.21,
    "card_testing": 0.07,
    "device_swap_drain": 0.21,
    "slow_bleed": 0.18,
    "merchant_collusion": 0.33,
}

# Fraudsters lean on instruments they can use without a second factor.
FRAUD_METHOD_P = np.array([0.35, 0.45, 0.05, 0.12, 0.03])

EVENT_COLUMNS = [
    "cust_idx", "merchant_idx", "ts", "amount", "method_idx", "device_id",
    "device_install_ts", "device_type_idx", "location", "is_declined",
    "true_fraud", "archetype", "is_hard_negative",
]


def _sample_from_cdf(cdf_rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one categorical index per row of a matrix of cumulative weights."""
    u = rng.random(cdf_rows.shape[0])
    return (u[:, None] > cdf_rows).sum(axis=1)


def _start_weekday(cfg: GeneratorConfig) -> int:
    return pd.Timestamp(cfg.start_date).weekday()


def simulate_legit(
    cfg: GeneratorConfig,
    customers: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    merchants: pd.DataFrame,
    rng: np.random.Generator,
    device_type_slot: np.ndarray,
) -> pd.DataFrame:
    """Generate the background stream of genuine transaction attempts."""
    n_cust = len(customers)
    counts = rng.poisson(customers["activity_rate"].to_numpy() * cfg.days)
    cust_idx = np.repeat(np.arange(n_cust), counts)
    n = cust_idx.size

    # --- when -------------------------------------------------------------
    weekday = (np.arange(cfg.days) + _start_weekday(cfg)) % 7
    day_p = WEEKDAY_FACTOR[weekday]
    day_p = day_p / day_p.sum()
    day = rng.choice(cfg.days, size=n, p=day_p)
    hour = _sample_from_cdf(arrays["hour_cdf"][cust_idx], rng)
    ts = day * SECONDS_PER_DAY + hour * 3600 + rng.random(n) * 3600

    # --- where ------------------------------------------------------------
    cat = _sample_from_cdf(arrays["cat_cdf"][cust_idx], rng)
    merchant_idx = np.empty(n, dtype=np.int64)
    m_cat = merchants["category_idx"].to_numpy()
    m_traffic = merchants["traffic_weight"].to_numpy()
    for c in range(len(CATEGORIES)):
        members = np.where(m_cat == c)[0]
        mask = cat == c
        k = int(mask.sum())
        if k == 0:
            continue
        if members.size == 0:
            # No merchant sampled in this category; fall back to global traffic.
            merchant_idx[mask] = rng.choice(
                len(merchants), size=k, p=m_traffic / m_traffic.sum()
            )
            continue
        w = m_traffic[members]
        merchant_idx[mask] = rng.choice(members, size=k, p=w / w.sum())

    # --- how much ---------------------------------------------------------
    ticket = merchants["ticket_factor"].to_numpy()[merchant_idx]
    mu = customers["spend_mu"].to_numpy()[cust_idx] + np.log(ticket)
    sigma = customers["spend_sigma"].to_numpy()[cust_idx]
    amount = np.round(rng.lognormal(mu, sigma), 2)

    # --- how --------------------------------------------------------------
    method_w = arrays["method_w"][cust_idx] * METHOD_BY_CATEGORY[cat]
    method_cdf = np.cumsum(method_w / method_w.sum(axis=1, keepdims=True), axis=1)
    method_idx = _sample_from_cdf(method_cdf, rng)

    # --- which device -----------------------------------------------------
    # Only devices already installed are eligible; newer devices dominate once
    # they exist, which is what produces natural device churn.
    install = arrays["device_install"][cust_idx]
    available = install <= ts[:, None]
    slot_pref = np.array([0.35, 0.50, 0.75, 1.00])
    dev_w = available * slot_pref[None, :]
    total = dev_w.sum(axis=1, keepdims=True)
    # A customer whose first device postdates the event falls back to slot 0.
    dev_w = np.where(total > 0, dev_w, np.eye(MAX_DEVICES)[0][None, :])
    dev_cdf = np.cumsum(dev_w / dev_w.sum(axis=1, keepdims=True), axis=1)
    device_slot = _sample_from_cdf(dev_cdf, rng)
    device_install_ts = install[np.arange(n), device_slot]
    device_install_ts = np.where(np.isinf(device_install_ts), ts, device_install_ts)

    # --- where from -------------------------------------------------------
    location = customers["home_location"].to_numpy()[cust_idx].copy()
    for t in range(MAX_TRIPS):
        in_trip = (ts >= arrays["trip_start"][cust_idx, t]) & (
            ts <= arrays["trip_end"][cust_idx, t]
        )
        location = np.where(in_trip, arrays["trip_loc"][cust_idx, t], location)
    wandering = rng.random(n) < 0.02
    location = np.where(wandering, rng.integers(0, cfg.n_locations, size=n), location)

    # --- authorised or declined -------------------------------------------
    device_age_days = np.maximum(0.0, (ts - device_install_ts) / SECONDS_PER_DAY)
    p_decline = merchants["decline_base"].to_numpy()[merchant_idx].copy()
    p_decline = p_decline * np.where(method_idx == 1, 1.6, 1.0)      # cards fail more
    p_decline = p_decline * np.where(device_age_days < 3, 1.4, 1.0)  # fresh device
    p_decline = p_decline * np.where((hour >= 0) & (hour <= 5), 1.2, 1.0)
    is_declined = rng.random(n) < np.clip(p_decline, 0.0, 0.6)

    return pd.DataFrame({
        "cust_idx": cust_idx,
        "merchant_idx": merchant_idx,
        "ts": ts,
        "amount": amount,
        "method_idx": method_idx,
        "device_id": [f"DEV_{c:06d}_{s}" for c, s in zip(cust_idx, device_slot, strict=False)],
        "device_install_ts": device_install_ts,
        "device_type_idx": device_type_slot[cust_idx, device_slot],
        "location": location,
        "is_declined": is_declined,
        "true_fraud": np.zeros(n, dtype=np.int8),
        "archetype": "none",
        "is_hard_negative": np.zeros(n, dtype=bool),
    })


def _latest_device(install_row: np.ndarray, t: float) -> int:
    """Slot of the newest device the customer already owns at time `t`."""
    eligible = np.where(install_row <= t)[0]
    if eligible.size == 0:
        return 0
    return int(eligible[np.argmax(install_row[eligible])])


def _night_hour(rng: np.random.Generator) -> float:
    """Pick an hour-of-day offset in seconds, skewed towards the small hours."""
    if rng.random() < 0.60:
        return float(rng.uniform(0, 6) * 3600)
    return float(rng.uniform(6, 24) * 3600)


def _episode_rows(
    rows: list[dict],
    cust: int,
    times: np.ndarray,
    merchant_pick: np.ndarray,
    amounts: np.ndarray,
    declined: np.ndarray,
    device_id: str,
    device_install_ts: float,
    device_type_idx: int,
    location: int,
    method_idx: np.ndarray,
    true_fraud: int,
    archetype: str,
    hard_negative: bool,
) -> None:
    for i in range(len(times)):
        rows.append({
            "cust_idx": cust,
            "merchant_idx": int(merchant_pick[i]),
            "ts": float(times[i]),
            "amount": float(round(amounts[i], 2)),
            "method_idx": int(method_idx[i]),
            "device_id": device_id,
            "device_install_ts": device_install_ts,
            "device_type_idx": device_type_idx,
            "location": int(location),
            "is_declined": bool(declined[i]),
            "true_fraud": true_fraud,
            "archetype": archetype,
            "is_hard_negative": hard_negative,
        })


def simulate_episodes(
    cfg: GeneratorConfig,
    customers: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    merchants: pd.DataFrame,
    rng: np.random.Generator,
    device_type_slot: np.ndarray,
    n_legit: int,
) -> pd.DataFrame:
    """Inject fraud episodes and genuine-but-anomalous (hard negative) episodes."""
    horizon = cfg.days * SECONDS_PER_DAY
    baseline = np.exp(customers["spend_mu"].to_numpy())
    home = customers["home_location"].to_numpy()
    install = arrays["device_install"]

    attract = merchants["fraud_attractiveness"].to_numpy()
    traffic = merchants["traffic_weight"].to_numpy()
    fraud_merchant_p = attract * np.power(traffic, 0.3)
    fraud_merchant_p = fraud_merchant_p / fraud_merchant_p.sum()

    # Low-ticket merchants are where cards get tested.
    m_cat = merchants["category_idx"].to_numpy()
    testing_cats = [CATEGORIES.index(c) for c in ("gaming", "subscriptions", "gift_cards")]
    testing_pool = np.where(np.isin(m_cat, testing_cats))[0]
    if testing_pool.size == 0:
        testing_pool = np.arange(len(merchants))

    # A small ring of colluding merchants, drawn from the attractive tail.
    ring_size = max(4, len(merchants) // 50)
    ring = rng.choice(len(merchants), size=ring_size, replace=False, p=fraud_merchant_p)

    # Customers who are more active and spend more are likelier to be targeted.
    target_w = (
        customers["activity_rate"].to_numpy() ** 0.5
        * baseline ** 0.3
        * rng.gamma(2.0, 1.0, size=len(customers))
    )
    order = np.argsort(-target_w)

    archetypes = list(FRAUD_ARCHETYPES)
    arch_p = np.array([FRAUD_ARCHETYPES[a] for a in archetypes])
    arch_p = arch_p / arch_p.sum()

    # Solve for the fraud row count that hits the target rate once the fraud
    # rows are themselves part of the denominator.
    p = cfg.target_fraud_rate
    target_fraud_rows = int(round(p * n_legit / (1.0 - p)))

    rows: list[dict] = []
    fraud_rows = 0
    cursor = 0
    n_hard_negatives = int(round(cfg.hard_negative_rate * cfg.n_customers))

    # (customer, end of their last episode). Being defrauded once makes you a
    # known-good target, so a share of episodes re-hit an earlier victim once
    # their first dispute has had time to surface.
    victims: list[tuple[int, float]] = []
    episode_no = 0
    lag = cfg.chargeback_lag_days * SECONDS_PER_DAY
    # Widest margin any archetype needs at the end of the window (slow_bleed).
    repeat_margin = 11 * SECONDS_PER_DAY

    while fraud_rows < target_fraud_rows and cursor < len(order):
        repeat = None
        if victims and rng.random() < cfg.repeat_victim_rate:
            eligible = [v for v in victims if v[1] + lag < horizon - repeat_margin]
            if eligible:
                repeat = eligible[int(rng.integers(len(eligible)))]

        if repeat is not None:
            cust, t_lo = repeat[0], repeat[1] + lag
        else:
            cust = int(order[cursor])
            cursor += 1
            t_lo = 0.0

        arch = archetypes[int(rng.choice(len(archetypes), p=arch_p))]
        base = float(baseline[cust])
        # Episode-scoped so a repeat victim's second compromise is a genuinely
        # different device rather than the same id with two install times.
        episode_no += 1
        new_dev_id = f"DEV_{cust:06d}_f{episode_no}"
        new_dev_type = int(rng.choice([0, 1], p=[0.72, 0.28]))
        elsewhere = int(rng.integers(0, cfg.n_locations))

        if arch == "ato_burst":
            t0 = float(rng.uniform(t_lo, horizon - SECONDS_PER_DAY))
            t0 = t0 - (t0 % SECONDS_PER_DAY) + _night_hour(rng)
            n_fail = int(rng.integers(1, 5))
            n_ok = int(rng.integers(3, 13))
            fail_times = t0 - np.sort(rng.uniform(60, 1800, size=n_fail))[::-1]
            ok_times = t0 + np.cumsum(rng.uniform(60, 900, size=n_ok))
            times = np.concatenate([fail_times, ok_times])
            amounts = base * rng.uniform(3.0, 15.0, size=times.size) * np.exp(
                rng.normal(0, 0.3, size=times.size)
            )
            declined = np.concatenate([
                np.ones(n_fail, dtype=bool),
                rng.random(n_ok) < 0.20,
            ])
            picks = rng.choice(len(merchants), size=times.size, p=fraud_merchant_p)
            device_id, dev_install = new_dev_id, t0 - rng.uniform(0, 2 * SECONDS_PER_DAY)
            dev_type, loc = new_dev_type, elsewhere

        elif arch == "card_testing":
            t0 = float(rng.uniform(t_lo, horizon - SECONDS_PER_DAY))
            n = int(rng.integers(6, 31))
            times = t0 + np.cumsum(rng.uniform(5, 90, size=n))
            amounts = rng.uniform(1.0, 50.0, size=n)
            declined = rng.random(n) < 0.65
            picks = rng.choice(testing_pool, size=n)
            if rng.random() < 0.55:  # a successful test is followed by a real hit
                n_hit = int(rng.integers(1, 3))
                hit_times = times[-1] + np.cumsum(rng.uniform(120, 900, size=n_hit))
                times = np.concatenate([times, hit_times])
                amounts = np.concatenate(
                    [amounts, base * rng.uniform(4.0, 20.0, size=n_hit)]
                )
                declined = np.concatenate([declined, rng.random(n_hit) < 0.15])
                picks = np.concatenate([
                    picks, rng.choice(len(merchants), size=n_hit, p=fraud_merchant_p)
                ])
            device_id, dev_install = new_dev_id, t0 - rng.uniform(0, SECONDS_PER_DAY)
            dev_type, loc = new_dev_type, elsewhere

        elif arch == "device_swap_drain":
            t0 = float(rng.uniform(t_lo, horizon - SECONDS_PER_DAY))
            n = int(rng.integers(2, 7))
            times = t0 + np.cumsum(rng.uniform(300, 3600, size=n))
            amounts = base * rng.uniform(2.0, 6.0, size=n) * np.exp(
                rng.normal(0, 0.25, size=n)
            )
            declined = rng.random(n) < 0.15
            picks = rng.choice(len(merchants), size=n, p=fraud_merchant_p)
            # New device but the usual location: harder to separate.
            device_id, dev_install = new_dev_id, t0 - rng.uniform(0, 3 * SECONDS_PER_DAY)
            dev_type, loc = new_dev_type, int(home[cust])

        elif arch == "slow_bleed":
            t0 = float(rng.uniform(t_lo, horizon - 10 * SECONDS_PER_DAY))
            n = int(rng.integers(4, 11))
            times = t0 + np.cumsum(
                rng.uniform(0.3, 1.5, size=n) * SECONDS_PER_DAY
            )
            amounts = base * rng.uniform(1.2, 2.5, size=n) * np.exp(
                rng.normal(0, 0.2, size=n)
            )
            declined = rng.random(n) < 0.05
            picks = rng.choice(len(merchants), size=n, p=fraud_merchant_p)
            slot = _latest_device(install[cust], t0)
            device_id = f"DEV_{cust:06d}_{slot}"
            dev_install = float(install[cust, slot])
            dev_install = t0 - SECONDS_PER_DAY if np.isinf(dev_install) else dev_install
            dev_type, loc = int(device_type_slot[cust, slot]), int(home[cust])

        else:  # merchant_collusion
            t0 = float(rng.uniform(t_lo, horizon - SECONDS_PER_DAY))
            n = int(rng.integers(1, 4))
            times = t0 + np.cumsum(rng.uniform(600, 7200, size=n))
            # Round-number amounts are a tell for scripted merchant-side fraud.
            amounts = np.round(base * rng.uniform(1.5, 5.0, size=n) / 500.0) * 500.0
            amounts = np.maximum(amounts, 500.0)
            declined = rng.random(n) < 0.08
            picks = rng.choice(ring, size=n)
            slot = _latest_device(install[cust], t0)
            device_id = f"DEV_{cust:06d}_{slot}"
            dev_install = float(install[cust, slot])
            dev_install = t0 - SECONDS_PER_DAY if np.isinf(dev_install) else dev_install
            dev_type, loc = int(device_type_slot[cust, slot]), int(home[cust])

        keep = (times >= 0) & (times < horizon)
        if not keep.any():
            continue
        times, amounts, declined, picks = (
            times[keep], amounts[keep], declined[keep], picks[keep]
        )
        methods = rng.choice(len(PAYMENT_METHODS), size=times.size, p=FRAUD_METHOD_P)
        _episode_rows(
            rows, cust, times, picks, amounts, declined, device_id, dev_install,
            dev_type, loc, methods, 1, arch, False,
        )
        fraud_rows += int(times.size)
        victims.append((cust, float(times.max())))

    # --- hard negatives ---------------------------------------------------
    # Genuine customers having an unusual day: new phone, travelling, a big
    # purchase, a couple of declines. Structurally an ATO burst, labelled 0.
    hn_pool = order[cursor : cursor + n_hard_negatives * 3]
    hn_chosen = rng.choice(
        hn_pool, size=min(n_hard_negatives, hn_pool.size), replace=False
    )
    for cust in hn_chosen:
        cust = int(cust)
        base = float(baseline[cust])
        t0 = float(rng.uniform(0, horizon - SECONDS_PER_DAY))
        t0 = t0 - (t0 % SECONDS_PER_DAY) + _night_hour(rng)
        n_fail = int(rng.integers(0, 4))
        n_ok = int(rng.integers(2, 7))
        fail_times = t0 - np.sort(rng.uniform(60, 2400, size=n_fail))[::-1]
        ok_times = t0 + np.cumsum(rng.uniform(120, 2400, size=n_ok))
        times = np.concatenate([fail_times, ok_times])
        amounts = base * rng.uniform(2.5, 8.0, size=times.size) * np.exp(
            rng.normal(0, 0.3, size=times.size)
        )
        declined = np.concatenate([
            np.ones(n_fail, dtype=bool), rng.random(n_ok) < 0.12
        ])
        picks = rng.choice(len(merchants), size=times.size, p=fraud_merchant_p)
        keep = (times >= 0) & (times < horizon)
        if not keep.any():
            continue
        times, amounts, declined, picks = (
            times[keep], amounts[keep], declined[keep], picks[keep]
        )
        methods = rng.choice(len(PAYMENT_METHODS), size=times.size, p=FRAUD_METHOD_P)
        _episode_rows(
            rows, cust, times, picks, amounts, declined,
            f"DEV_{cust:06d}_h", t0 - rng.uniform(0, 2 * SECONDS_PER_DAY),
            int(rng.choice([0, 1], p=[0.72, 0.28])),
            int(rng.integers(0, cfg.n_locations)), methods, 0, "hard_negative", True,
        )

    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(rows)


def simulate(cfg: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full simulation. Returns (events, customers, merchants)."""
    rng = np.random.default_rng(cfg.seed)
    merchants = build_merchants(cfg, rng)
    customers, arrays = build_customers(cfg, rng)

    # Device type per customer per slot: usually the same kind of device.
    primary = customers["device_type_idx"].to_numpy()
    device_type_slot = np.tile(primary[:, None], (1, MAX_DEVICES))
    switch = rng.random((cfg.n_customers, MAX_DEVICES)) < 0.30
    device_type_slot = np.where(
        switch,
        rng.choice(len(DEVICE_TYPES), size=(cfg.n_customers, MAX_DEVICES)),
        device_type_slot,
    )
    device_type_slot[:, 0] = primary

    legit = simulate_legit(cfg, customers, arrays, merchants, rng, device_type_slot)
    episodes = simulate_episodes(
        cfg, customers, arrays, merchants, rng, device_type_slot, len(legit)
    )

    events = pd.concat([legit, episodes], ignore_index=True)
    events = events.sort_values("ts", kind="mergesort").reset_index(drop=True)
    return events, customers, merchants
