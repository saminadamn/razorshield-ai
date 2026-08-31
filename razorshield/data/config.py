"""Configuration for the synthetic transaction generator.

Every knob that shapes the dataset lives here so a run is fully described by
one frozen object plus a seed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorConfig:
    # --- volume -------------------------------------------------------------
    n_transactions: int = 100_000
    n_customers: int = 10_000
    n_merchants: int = 300
    n_locations: int = 60

    # --- timeline -----------------------------------------------------------
    # `warmup_days` are simulated but dropped from the released dataset. They
    # exist so that merchant risk scores and customer histories are already
    # warm on day 1 of the data we actually train on.
    days: int = 180
    warmup_days: int = 60
    start_date: str = "2025-01-01"

    # --- labels -------------------------------------------------------------
    target_fraud_rate: float = 0.03
    # Chargebacks are only *known* after the issuer raises them. Any feature
    # derived from a chargeback must respect this lag or it is time-travel.
    chargeback_lag_days: int = 35
    fraud_chargeback_rate: float = 0.72
    legit_chargeback_rate: float = 0.003

    # Real labels are imperfect: some fraud is never reported, and some
    # genuine transactions get disputed by the cardholder ("friendly fraud").
    label_noise_missed_fraud: float = 0.02
    label_noise_false_alarm: float = 0.0006

    # --- realism ------------------------------------------------------------
    # Genuine customers who behave like fraudsters for one episode (new phone
    # + travel + big purchase + a couple of declines). These are the rows that
    # make precision hard, and they are the whole point.
    hard_negative_rate: float = 0.015

    # Share of fraud episodes that re-target someone already defrauded. Without
    # this, one episode per customer makes a prior chargeback a *protective*
    # signal, which is backwards and teaches the model the wrong lesson.
    repeat_victim_rate: float = 0.15

    # Beta prior for merchant risk. Cold-start merchants sit near
    # prior_alpha / (prior_alpha + prior_beta) until they accumulate history.
    merchant_prior_alpha: float = 0.5
    merchant_prior_beta: float = 60.0

    seed: int = 42

    @property
    def released_days(self) -> int:
        return self.days - self.warmup_days
