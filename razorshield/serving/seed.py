"""Populate the event store with history so the demo means something.

    python -m razorshield.serving.seed --db data/razorshield.db --score 300

A brand-new customer has no velocity, no device age and no history, so every
first transaction scores about the same. Seeding the store with the simulated
population gives the demo customers a past, which is what makes a live
transaction score differently depending on who is making it.

The same helpers are used by `parity.py`, so the seeded store and the parity
proof are built by identical code.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.config import GeneratorConfig
from ..data.entities import DEVICE_TYPES, PAYMENT_METHODS
from ..data.generate import apply_label_noise
from ..data.simulate import simulate
from .online import DAY, OnlineFeatureBuilder, RawEvent, ServingConstants
from .store import EventStore

EPOCH = pd.Timestamp("1970-01-01")


def epoch_offset(cfg: GeneratorConfig) -> float:
    """Seconds between the Unix epoch and the simulation's start date."""
    return (pd.Timestamp(cfg.start_date) - EPOCH).total_seconds()


def build_population(cfg: GeneratorConfig):
    """Simulate the event stream and attach observed labels."""
    events, customers, merchants = simulate(cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    return apply_label_noise(events, cfg, rng), customers, merchants


def register_entities(
    store: EventStore, customers: pd.DataFrame, events: pd.DataFrame, offset: float
) -> None:
    """Register customers at signup and devices at their install time.

    Customers are registered with the tenure they already had at t=0, which is
    what a real deployment knows from its own signup records. Devices carry
    their install time; in production only first-seen is knowable, so a live
    `device_age_days` is a lower bound on the true one.
    """
    customer_ids = customers["customer_id"].to_numpy()
    for cid, age in zip(customer_ids, customers["account_age_days_start"].to_numpy(), strict=False):
        store.register_customer(str(cid), offset, float(age))

    devices = events[["cust_idx", "device_id", "device_install_ts", "device_type_idx"]]
    devices = devices.drop_duplicates(subset=["cust_idx", "device_id"])
    for cust_idx, device_id, install, type_idx in devices.itertuples(index=False):
        store.register_device(
            str(customer_ids[cust_idx]),
            str(device_id),
            float(install) + offset,
            DEVICE_TYPES[type_idx],
        )
    store.commit()


def event_stream(
    events: pd.DataFrame,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    cfg: GeneratorConfig,
    offset: float,
    id_prefix: str = "TXN",
) -> Iterator[RawEvent]:
    """Yield the simulated stream as RawEvents in chronological order."""
    customer_ids = customers["customer_id"].to_numpy()
    merchant_ids = merchants["merchant_id"].to_numpy()
    lag = cfg.chargeback_lag_days * DAY

    for i, row in enumerate(events.reset_index(drop=True).itertuples(index=False)):
        yield RawEvent(
            transaction_id=f"{id_prefix}_{i:07d}",
            customer_id=str(customer_ids[row.cust_idx]),
            merchant_id=str(merchant_ids[row.merchant_idx]),
            ts=float(row.ts) + offset,
            amount=float(row.amount),
            payment_method=PAYMENT_METHODS[row.method_idx],
            device_type=DEVICE_TYPES[row.device_type_idx],
            device_id=str(row.device_id),
            location=str(row.location),
            declined=bool(row.is_declined),
            chargeback_confirmed_at=(
                float(row.ts) + offset + lag if row.chargeback else None
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the online event store.")
    parser.add_argument("--db", default="data/razorshield.db", type=Path)
    parser.add_argument("--manifest", default="data/raw/manifest.json", type=Path)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument(
        "--score",
        type=int,
        default=0,
        help=(
            "Also score N events sampled across the window, so the console "
            "opens with a representative distribution."
        ),
    )
    parser.add_argument(
        "--bundle", default="models/razorshield_model.joblib", type=Path
    )
    parser.add_argument("--scorer", default="models/risk_scorer.joblib", type=Path)
    args = parser.parse_args()

    cfg = GeneratorConfig(seed=args.seed)
    offset = epoch_offset(cfg)

    if args.db.exists():
        args.db.unlink()  # seeding is a rebuild, not an append
    store = EventStore(args.db)
    constants = ServingConstants.from_manifest(args.manifest)
    builder = OnlineFeatureBuilder(store, constants)

    print("simulating population ...")
    events, customers, merchants = build_population(cfg)
    print(f"  {len(events):,} events, {len(customers):,} customers")

    register_entities(store, customers, events, offset)
    print(f"  registered {len(customers):,} customers and their devices")

    stream = list(event_stream(events, customers, merchants, cfg, offset))

    # Score a random sample spread across the whole window rather than the last
    # N events. The tail of the stream is not a representative sample -- taking
    # it leaves the console showing almost nothing but LOW. Each sampled event
    # is still scored at its own position in the replay, so its features see
    # exactly the history it would have had.
    rng = np.random.default_rng(cfg.seed)
    to_score = set()
    if args.score:
        k = min(args.score, len(stream))
        to_score = set(rng.choice(len(stream), size=k, replace=False).tolist())

    scorer = None
    if args.score:
        from ..scoring.score import RiskScorer

        scorer = RiskScorer.load(args.bundle, args.scorer)

    print("replaying events ...")
    for i, event in enumerate(stream):
        if scorer is not None and i in to_score:
            features = builder.build_frame(event)
            payload = scorer.assess_one(features, event.transaction_id)
            payload["customer_id"] = event.customer_id
            payload["amount"] = event.amount
            store.record_assessment(payload, event.customer_id, event.ts, event.amount)
        builder.observe(event, commit=False)
        if i % 20_000 == 0 and i:
            store.commit()
            print(f"  {i:,} events")
    store.commit()

    summary = store.summary()
    print()
    print(f"  events       {summary['events']:,}")
    print(f"  customers    {summary['customers']:,}")
    print(f"  assessments  {summary['assessments']:,}  {summary['bands']}")
    print(f"  written to   {args.db.resolve()}")


if __name__ == "__main__":
    main()
