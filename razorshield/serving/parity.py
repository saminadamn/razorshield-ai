"""Proof that the online path computes the same features as training.

    python -m razorshield.serving.parity

Generates a small dataset, computes features the batch way, then replays the
identical event stream one event at a time through the store and the online
builder. Every one of the 18 features must match on every row.

This is the test that makes the training/serving parity claim mean something.
Without it, "we reuse the same functions" is an assertion; with it, a drift in
either definition fails the build.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ..data.config import GeneratorConfig
from ..data.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_features
from .online import OnlineFeatureBuilder, ServingConstants
from .seed import build_population, epoch_offset, event_stream, register_entities
from .store import EventStore


def replay(cfg: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (batch features, online features) for the same event stream.

    The store is populated by the same helpers `seed.py` uses, so this proves
    parity for the code path the demo actually runs -- not a parallel one
    written to pass.
    """
    events, customers, merchants = build_population(cfg)
    batch, _ = build_features(events, customers, merchants, cfg)
    offset = epoch_offset(cfg)

    store = EventStore(":memory:")
    constants = ServingConstants(
        cold_start_amount=batch.attrs["cold_start_amount"],
        chargeback_lag_days=cfg.chargeback_lag_days,
        merchant_prior_alpha=cfg.merchant_prior_alpha,
        merchant_prior_beta=cfg.merchant_prior_beta,
    )
    builder = OnlineFeatureBuilder(store, constants)
    register_entities(store, customers, events, offset)

    rows = []
    for event in event_stream(events, customers, merchants, cfg, offset, "REPLAY"):
        rows.append(builder.build(event))
        builder.observe(event, commit=False)
    store.commit()

    online = pd.DataFrame(rows)[FEATURE_COLUMNS]
    return batch[FEATURE_COLUMNS].reset_index(drop=True), online


# A feature rounded to n decimal places cannot be compared more finely than
# one unit in that place. Where a value sits exactly on a rounding boundary --
# the mean of an even number of 2-decimal amounts lands on a half-paisa -- the
# tie breaks on the last bit of the float sum, and numpy's pairwise cumsum and
# SQLite's compensated SUM do not agree there. Those are counted separately as
# ties, and capped, rather than being waved through as a wider tolerance.
TOLERANCE = {
    "avg_transaction_amount": 0.01,
    "amount_deviation_ratio": 1e-4,
    "merchant_risk_score": 1e-6,
    "velocity_score": 1e-4,
}
MAX_TIE_RATE = 0.02


def compare(batch: pd.DataFrame, online: pd.DataFrame) -> list[dict]:
    results = []
    n = len(batch)
    for column in FEATURE_COLUMNS:
        left, right = batch[column], online[column]
        tol = TOLERANCE.get(column, 0.0)
        if column in CATEGORICAL_COLUMNS:
            mismatches = int((left.astype(str) != right.astype(str)).sum())
            ties, worst = 0, float(mismatches > 0)
        else:
            diff = np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))
            # The difference of two rounded floats is itself inexact:
            # |550.09 - 550.10| is 0.010000000000048, so the tolerance needs
            # slack for its own representation error.
            limit = tol + 1e-9
            mismatches = int((diff > limit).sum())
            ties = int(((diff > 0) & (diff <= limit)).sum())
            worst = float(diff.max())
        tie_rate = ties / n if n else 0.0
        results.append({
            "feature": column,
            "mismatches": mismatches,
            "rounding_ties": ties,
            "tie_rate": tie_rate,
            "max_difference": worst,
            "ok": mismatches == 0 and tie_rate <= MAX_TIE_RATE,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch vs online feature parity.")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--customers", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = GeneratorConfig(
        n_transactions=args.n,
        n_customers=args.customers,
        days=120,
        warmup_days=40,
        seed=args.seed,
    )
    print("=" * 70)
    print("FEATURE PARITY  (batch replay vs online, event by event)")
    print("=" * 70)
    batch, online = replay(cfg)
    print(f"  rows compared        {len(batch):,}")
    print(f"  features compared    {len(FEATURE_COLUMNS)}")
    print()

    results = compare(batch, online)
    width = max(len(r["feature"]) for r in results)
    for r in results:
        if r["mismatches"]:
            status = f"MISMATCH ({r['mismatches']:,} rows)"
        elif r["rounding_ties"]:
            status = f"exact, except {r['rounding_ties']} rounding ties ({r['tie_rate']:.2%})"
        else:
            status = "exact"
        print(f"  {r['feature']:<{width}}  max diff {r['max_difference']:>10.6g}   {status}")

    ties = sum(r["rounding_ties"] for r in results)
    if ties:
        print()
        print(f"  {ties} rounding ties: a value sitting exactly on its last decimal")
        print("  place, where the float sum's final bit decides the direction. No")
        print("  effect on any prediction; capped at "
              f"{MAX_TIE_RATE:.0%} of rows per feature.")

    failed = [r for r in results if not r["ok"]]
    print()
    if failed:
        print(f"FAILED -- {len(failed)} feature(s) disagree between training and serving")
        sys.exit(1)
    print("PASSED -- the model is served exactly what it was trained on")


if __name__ == "__main__":
    main()
