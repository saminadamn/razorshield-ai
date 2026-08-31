"""CLI entry point: simulate, label, featurise, write.

    python -m razorshield.data.generate --out data/raw --n 100000 --seed 42
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GeneratorConfig
from .features import TARGET, build_features
from .simulate import SECONDS_PER_DAY, simulate


def apply_label_noise(
    events: pd.DataFrame, cfg: GeneratorConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Derive the observed label and the chargeback flag from ground truth.

    Real labels are imperfect in both directions: some fraud is never reported,
    and some genuine transactions get disputed by the cardholder. The label the
    dataset ships is the *observed* one, and chargebacks are then generated from
    that label so features derived from disputes stay consistent with it.
    """
    truth = events["true_fraud"].to_numpy().astype(np.int8)
    observed = truth.copy()

    is_fraud = truth == 1
    missed = is_fraud & (rng.random(truth.size) < cfg.label_noise_missed_fraud)
    observed[missed] = 0

    is_legit = truth == 0
    disputed = is_legit & (rng.random(truth.size) < cfg.label_noise_false_alarm)
    observed[disputed] = 1

    chargeback = np.where(
        observed == 1,
        rng.random(truth.size) < cfg.fraud_chargeback_rate,
        rng.random(truth.size) < cfg.legit_chargeback_rate,
    )

    events = events.copy()
    events["is_fraud_observed"] = observed
    events["chargeback"] = chargeback
    return events


def generate(cfg: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Produce the released dataset, its metadata sidecar, and a run manifest."""
    events, customers, merchants = simulate(cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    events = apply_label_noise(events, cfg, rng)

    frame, meta = build_features(events, customers, merchants, cfg)
    cold_start_amount = frame.attrs["cold_start_amount"]

    # Drop the warmup window: those rows exist only to warm up merchant risk
    # scores and customer histories.
    released = events["ts"].to_numpy() >= cfg.warmup_days * SECONDS_PER_DAY
    frame, meta = frame[released], meta[released]

    # Trim from the tail so the released window stays contiguous in time.
    if len(frame) > cfg.n_transactions:
        frame = frame.iloc[: cfg.n_transactions]
        meta = meta.iloc[: cfg.n_transactions]

    frame = frame.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    txn_ids = [f"TXN_{i:06d}" for i in range(1, len(frame) + 1)]
    frame.insert(0, "transaction_id", txn_ids)
    meta.insert(0, "transaction_id", txn_ids)

    manifest = {
        "config": dataclasses.asdict(cfg),
        "n_rows": int(len(frame)),
        "fraud_rate_observed": float(frame[TARGET].mean()),
        "fraud_rate_true": float(meta["true_fraud"].mean()),
        "labels_flipped": int(meta["label_flipped"].sum()),
        "window_start": str(frame["timestamp"].min()),
        "window_end": str(frame["timestamp"].max()),
        "n_customers_seen": int(frame["customer_id"].nunique()),
        "n_merchants_seen": int(frame["merchant_id"].nunique()),
        "archetype_rows": meta["fraud_archetype"].value_counts().to_dict(),
        # Serving-time constants. The online path must reuse these or the
        # features it computes will not match the ones the model trained on.
        "serving_constants": {
            "cold_start_amount": cold_start_amount,
            "chargeback_lag_days": cfg.chargeback_lag_days,
            "merchant_prior_alpha": cfg.merchant_prior_alpha,
            "merchant_prior_beta": cfg.merchant_prior_beta,
        },
    }
    return frame, meta, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RazorShield dataset.")
    parser.add_argument("--out", default="data/raw", type=Path)
    parser.add_argument("--n", type=int, default=GeneratorConfig.n_transactions)
    parser.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    parser.add_argument(
        "--fraud-rate", type=float, default=GeneratorConfig.target_fraud_rate
    )
    args = parser.parse_args()

    cfg = GeneratorConfig(
        n_transactions=args.n, seed=args.seed, target_fraud_rate=args.fraud_rate
    )
    frame, meta, manifest = generate(cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "transactions.csv", index=False)
    meta.to_csv(args.out / "transactions_meta.csv", index=False)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"rows           {manifest['n_rows']:,}")
    print(f"fraud rate     {manifest['fraud_rate_observed']:.4f}")
    print(f"window         {manifest['window_start']} -> {manifest['window_end']}")
    print(f"customers      {manifest['n_customers_seen']:,}")
    print(f"written to     {args.out.resolve()}")


if __name__ == "__main__":
    main()
