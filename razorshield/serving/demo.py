"""End-to-end demonstration of the serving path.

    python -m razorshield.serving.demo

Boots the API in-process, scores a normal transaction for a customer with real
history, then walks that same customer through an account-takeover pattern and
shows the risk score climb. Finishes by posting a correctly signed Razorpay
webhook and a tampered one, to show verification actually rejects.

No network and no Razorpay account required.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

WEBHOOK_SECRET = "demo_webhook_secret_not_a_real_one"


def pick_customer(db: Path) -> tuple[str, float, str]:
    """A customer with plenty of history, plus the end of the seeded window."""
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT customer_id, COUNT(*) AS n, MAX(ts) AS last_ts FROM events "
        "GROUP BY customer_id HAVING n > 25 ORDER BY n DESC LIMIT 1"
    ).fetchone()
    horizon = conn.execute("SELECT MAX(ts) AS t FROM events").fetchone()["t"]
    device = conn.execute(
        "SELECT device_id FROM events WHERE customer_id = ? ORDER BY ts DESC LIMIT 1",
        (row["customer_id"],),
    ).fetchone()["device_id"]
    conn.close()
    return row["customer_id"], float(horizon), str(device)


def show(label: str, result: dict) -> None:
    factors = ", ".join(
        f"{f['label']} {f['contribution']:+.2f}"
        for f in result["factors"][:3]
    )
    print(f"  {label:<34} score {result['risk_score']:>5.1f}  "
          f"{result['band']:<9} p={result['probability']:.4f}")
    print(f"      {factors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the serving path.")
    parser.add_argument("--db", default="data/razorshield.db", type=Path)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(
            f"{args.db} not found. Run: python -m razorshield.serving.seed"
        )

    # Run against a throwaway copy of the seeded store. Writing into the real
    # one makes the demo unrepeatable: the previous run's "attack" transactions
    # become part of the customer's normal spend, the deviation signal
    # collapses, and the same script quietly stops escalating.
    scratch = Path(tempfile.mkdtemp(prefix="razorshield-demo-")) / "demo.db"
    shutil.copy2(args.db, scratch)

    os.environ["RAZORPAY_WEBHOOK_SECRET"] = WEBHOOK_SECRET
    os.environ["RAZORSHIELD_DB"] = str(scratch)

    from fastapi.testclient import TestClient

    from .app import app

    customer, horizon, device = pick_customer(scratch)
    # Scoring is idempotent per transaction id, which is what makes webhook
    # retries safe -- but it also means a rerun with fixed ids would replay the
    # previous run's answers against a store that has since moved on. Each run
    # gets its own ids so the escalation below is always freshly computed.
    run = int(time.time())

    with TestClient(app) as client:
        print("=" * 72)
        print("HEALTH")
        print("=" * 72)
        health = client.get("/health").json()
        print(f"  model                {health['model']}")
        print(f"  webhook verification {health['webhook_verification']}")
        print(f"  events in store      {health['store']['events']:,}")
        print(f"  customers            {health['store']['customers']:,}")

        print()
        print("=" * 72)
        print(f"NORMAL TRANSACTION  ({customer}, known device, typical amount)")
        print("=" * 72)
        normal = client.post("/score", json={
            "transaction_id": f"DEMO_{run}_NORMAL",
            "customer_id": customer,
            "amount": 640.0,
            "payment_method": "UPI",
            "device_id": device,
            "device_type": "Mobile",
            "location": "LOC_01",
            "ts": horizon + 3600,
        }).json()
        show("everyday purchase", normal)

        print()
        print("=" * 72)
        print("ACCOUNT TAKEOVER  (same customer, new device, rapid high-value run)")
        print("=" * 72)
        print("  Each attempt is scored before it is stored, so the velocity")
        print("  features build up exactly as they would in production.")
        print()
        for i in range(6):
            result = client.post("/score", json={
                "transaction_id": f"DEMO_{run}_ATO_{i}",
                "customer_id": customer,
                "amount": 4500.0 + i * 1500,
                "payment_method": "Card",
                "device_id": "DEV_STOLEN_9001",
                "device_type": "Mobile",
                "location": "LOC_57",
                "declined": i in (0, 1),
                "ts": horizon + 7200 + i * 180,
            }).json()
            label = f"attempt {i + 1}" + ("  (declined)" if i in (0, 1) else "")
            show(label, result)

        print()
        print("=" * 72)
        print("RAZORPAY WEBHOOK  (Test Mode payload)")
        print("=" * 72)
        body = json.dumps({
            "event": "payment.authorized",
            "payload": {"payment": {"entity": {
                "id": f"pay_Demo{run}",
                "amount": 845000,  # paise
                "currency": "INR",
                "status": "authorized",
                "method": "upi",
                "created_at": int(horizon + 9000),
                "notes": {
                    "customer_id": customer,
                    "device_id": "DEV_STOLEN_9001",
                    "device_type": "Mobile",
                    "location": "LOC_57",
                    "merchant_id": "MERCH_0007",
                },
            }}},
        }).encode()
        signature = hmac.new(
            WEBHOOK_SECRET.encode(), body, hashlib.sha256
        ).hexdigest()

        def deliver(content: bytes, event_id: str):
            return client.post(
                "/webhooks/razorpay",
                content=content,
                headers={
                    "X-Razorpay-Signature": signature,
                    "X-Razorpay-Event-Id": event_id,
                    "Content-Type": "application/json",
                },
            )

        ok = deliver(body, f"evt_{run}_authorized")
        result = ok.json()
        print(f"  valid signature      HTTP {ok.status_code}")
        if result.get("scored"):
            assessment = result["assessment"]
            print(f"  Rs845,000 paise -> Rs{assessment['amount']:,.2f} "
                  "(Razorpay reports paise)")
            show(f"pay_Demo{run}", assessment)

        tampered = deliver(body.replace(b"845000", b"100"), f"evt_{run}_tampered")
        print(f"  tampered body        HTTP {tampered.status_code} "
              f"({tampered.json()['detail']})")

        # Razorpay retries until it gets a 2xx. The same event id must not be
        # processed twice.
        retry = deliver(body, f"evt_{run}_authorized")
        print(f"  retried delivery     HTTP {retry.status_code}, "
              f"deduped on event id: {retry.json().get('duplicate')}")

        # A *different* event about the same payment must still be accepted --
        # which is why the dedupe key is the event id, not the payment id.
        captured = body.replace(b"payment.authorized", b"payment.captured")
        signature = hmac.new(WEBHOOK_SECRET.encode(), captured, hashlib.sha256).hexdigest()
        follow = deliver(captured, f"evt_{run}_captured")
        print(f"  capture of same pay  HTTP {follow.status_code}, "
              f"processed: {follow.json().get('scored')}")

        print()
        print("=" * 72)
        print("RECENT ALERTS")
        print("=" * 72)
        for row in client.get("/transactions", params={"limit": 6}).json()[
            "transactions"
        ]:
            print(f"  {row['transaction_id']:<22} {row['band']:<9} "
                  f"score {row['risk_score']:>5.1f}  Rs{row['amount']:>10,.2f}")

    shutil.rmtree(scratch.parent, ignore_errors=True)
    print()
    print(f"  (scored against a disposable copy of {args.db}; "
          "the seeded store is untouched)")


if __name__ == "__main__":
    main()
