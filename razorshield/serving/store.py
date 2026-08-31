"""SQLite event store -- the state the online features are computed from.

Razorpay's webhook tells us about one payment. It cannot tell us how many
transactions this customer made in the last hour, how old their device is, or
whether they have a dispute on file. That history is ours to keep, and this is
where it lives.

SQLite because it needs no setup, survives a restart, and a hackathon demo will
never outgrow it. The access patterns are all "recent events for one customer"
and "lifetime counts for one merchant", both of which are index lookups.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id              TEXT PRIMARY KEY,
    first_seen               REAL NOT NULL,
    account_age_at_first_seen REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS devices (
    customer_id TEXT NOT NULL,
    device_id   TEXT NOT NULL,
    first_seen  REAL NOT NULL,
    device_type TEXT NOT NULL,
    PRIMARY KEY (customer_id, device_id)
);

CREATE TABLE IF NOT EXISTS events (
    transaction_id          TEXT PRIMARY KEY,
    customer_id             TEXT NOT NULL,
    merchant_id             TEXT NOT NULL,
    ts                      REAL NOT NULL,
    amount                  REAL NOT NULL,
    payment_method          TEXT NOT NULL,
    device_id               TEXT NOT NULL,
    device_type             TEXT NOT NULL,
    location                TEXT NOT NULL,
    declined                INTEGER NOT NULL DEFAULT 0,
    chargeback_confirmed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_events_customer ON events (customer_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_merchant ON events (merchant_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_chargeback
    ON events (merchant_id, chargeback_confirmed_at);

CREATE TABLE IF NOT EXISTS assessments (
    transaction_id TEXT PRIMARY KEY,
    ts             REAL NOT NULL,
    customer_id    TEXT NOT NULL,
    amount         REAL NOT NULL,
    probability    REAL NOT NULL,
    risk_score     REAL NOT NULL,
    band           TEXT NOT NULL,
    payload        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assessments_ts ON assessments (ts);

-- Razorpay retries a delivery until it gets a 2xx, and the same event can
-- arrive more than once. Deduping on the event id is what Razorpay recommends:
-- keying on the payment id instead would wrongly swallow a *different* event
-- about the same payment (an authorize followed by a capture).
-- Orders we created. The payment-signature check must be computed against
-- the order id from our own records; verifying a browser-supplied order id
-- against a browser-supplied signature would prove nothing.
CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT PRIMARY KEY,
    amount      REAL NOT NULL,
    customer_id TEXT NOT NULL,
    device_id   TEXT NOT NULL,
    device_type TEXT NOT NULL,
    location    TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS webhook_events (
    event_id       TEXT PRIMARY KEY,
    event_name     TEXT NOT NULL,
    received_at    REAL NOT NULL,
    transaction_id TEXT
);
"""


@dataclass
class CustomerHistory:
    """Everything the feature builder needs about one customer, as of `now`."""

    prior_count: int
    prior_amount_sum: float
    first_ts: float | None
    window_ts: list[float]
    window_amount: list[float]
    window_declined: list[int]
    window_device: list[str]
    window_location: list[str]
    confirmed_chargebacks: int


class EventStore:
    def __init__(self, path: Path | str = "data/razorshield.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- registration -------------------------------------------------------

    def register_customer(
        self, customer_id: str, ts: float, account_age_days: float = 0.0
    ) -> None:
        """First sighting wins; later events never rewrite the tenure clock."""
        self.conn.execute(
            "INSERT OR IGNORE INTO customers (customer_id, first_seen, "
            "account_age_at_first_seen) VALUES (?, ?, ?)",
            (customer_id, ts, account_age_days),
        )

    def register_device(
        self, customer_id: str, device_id: str, ts: float, device_type: str
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO devices (customer_id, device_id, first_seen, "
            "device_type) VALUES (?, ?, ?, ?)",
            (customer_id, device_id, ts, device_type),
        )

    def customer_tenure(self, customer_id: str) -> tuple[float, float] | None:
        row = self.conn.execute(
            "SELECT first_seen, account_age_at_first_seen FROM customers "
            "WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        return (row["first_seen"], row["account_age_at_first_seen"]) if row else None

    def device_first_seen(self, customer_id: str, device_id: str) -> float | None:
        row = self.conn.execute(
            "SELECT first_seen FROM devices WHERE customer_id = ? AND device_id = ?",
            (customer_id, device_id),
        ).fetchone()
        return row["first_seen"] if row else None

    # -- reads used by the feature builder ----------------------------------

    def customer_history(
        self, customer_id: str, now: float, window: float
    ) -> CustomerHistory:
        """Lifetime aggregates plus the raw rows inside the widest window.

        Two queries rather than one: the expanding mean needs every prior
        event, the rolling windows only need the last seven days.
        """
        agg = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS total, MIN(ts) AS first_ts "
            "FROM events WHERE customer_id = ? AND ts < ?",
            (customer_id, now),
        ).fetchone()

        rows = self.conn.execute(
            "SELECT ts, amount, declined, device_id, location FROM events "
            "WHERE customer_id = ? AND ts < ? AND ts >= ? ORDER BY ts",
            (customer_id, now, now - window),
        ).fetchall()

        chargebacks = self.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE customer_id = ? "
            "AND chargeback_confirmed_at IS NOT NULL AND chargeback_confirmed_at < ?",
            (customer_id, now),
        ).fetchone()["n"]

        return CustomerHistory(
            prior_count=agg["n"],
            prior_amount_sum=agg["total"],
            first_ts=agg["first_ts"],
            window_ts=[r["ts"] for r in rows],
            window_amount=[r["amount"] for r in rows],
            window_declined=[r["declined"] for r in rows],
            window_device=[r["device_id"] for r in rows],
            window_location=[r["location"] for r in rows],
            confirmed_chargebacks=chargebacks,
        )

    def merchant_exposure(
        self, merchant_id: str, now: float, lag_seconds: float
    ) -> tuple[int, int]:
        """(transactions old enough to have been disputed, disputes confirmed).

        Mirrors the batch definition exactly: the denominator only counts
        transactions that have had time to be charged back, so a merchant's
        score cannot be inflated by traffic too recent to have failed yet.
        """
        matured = self.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE merchant_id = ? AND ts < ?",
            (merchant_id, now - lag_seconds),
        ).fetchone()["n"]
        confirmed = self.conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE merchant_id = ? "
            "AND chargeback_confirmed_at IS NOT NULL AND chargeback_confirmed_at < ?",
            (merchant_id, now),
        ).fetchone()["n"]
        return matured, confirmed

    # -- writes -------------------------------------------------------------

    def record_event(
        self,
        transaction_id: str,
        customer_id: str,
        merchant_id: str,
        ts: float,
        amount: float,
        payment_method: str,
        device_id: str,
        device_type: str,
        location: str,
        declined: bool = False,
        chargeback_confirmed_at: float | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO events (transaction_id, customer_id, merchant_id, "
            "ts, amount, payment_method, device_id, device_type, location, declined, "
            "chargeback_confirmed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (transaction_id, customer_id, merchant_id, ts, amount, payment_method,
             device_id, device_type, location, int(declined), chargeback_confirmed_at),
        )

    def record_assessment(self, payload: dict, customer_id: str, ts: float,
                          amount: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO assessments (transaction_id, ts, customer_id, "
            "amount, probability, risk_score, band, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (payload["transaction_id"], ts, customer_id, amount,
             payload["probability"], payload["risk_score"], payload["band"],
             json.dumps(payload, default=str)),
        )

    def record_order(
        self, order_id: str, amount: float, customer_id: str, device_id: str,
        device_type: str, location: str, created_at: float,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO orders (order_id, amount, customer_id, "
            "device_id, device_type, location, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (order_id, amount, customer_id, device_id, device_type, location,
             created_at),
        )

    def get_order(self, order_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        return dict(row) if row else None

    def seen_event(self, event_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT event_id, event_name, received_at, transaction_id FROM "
            "webhook_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return dict(row) if row else None

    def record_webhook_event(
        self, event_id: str, event_name: str, received_at: float,
        transaction_id: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO webhook_events (event_id, event_name, "
            "received_at, transaction_id) VALUES (?, ?, ?, ?)",
            (event_id, event_name, received_at, transaction_id),
        )

    def mark_chargeback(self, transaction_id: str, confirmed_at: float) -> bool:
        cursor = self.conn.execute(
            "UPDATE events SET chargeback_confirmed_at = ? WHERE transaction_id = ?",
            (confirmed_at, transaction_id),
        )
        return cursor.rowcount > 0

    def commit(self) -> None:
        self.conn.commit()

    # -- reads used by the dashboard ----------------------------------------

    def assessment(self, transaction_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT payload FROM assessments WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def recent_assessments(self, limit: int = 50, band: str | None = None) -> list[dict]:
        sql = ("SELECT transaction_id, ts, customer_id, amount, probability, "
               "risk_score, band FROM assessments")
        params: list = []
        if band:
            sql += " WHERE band = ?"
            params.append(band)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def summary(self) -> dict:
        total = self.conn.execute(
            "SELECT COUNT(*) AS n FROM assessments"
        ).fetchone()["n"]
        bands = {
            r["band"]: r["n"]
            for r in self.conn.execute(
                "SELECT band, COUNT(*) AS n FROM assessments GROUP BY band"
            ).fetchall()
        }
        events = self.conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        customers = self.conn.execute(
            "SELECT COUNT(*) AS n FROM customers"
        ).fetchone()["n"]
        return {
            "assessments": total,
            "events": events,
            "customers": customers,
            "bands": bands,
        }
