"""Tests for the two Razorpay signature checks.

These are the only code paths that decide whether an inbound payment is
treated as genuine, so they get tests rather than a manual once-over. The two
schemes are different constructions and are not interchangeable:

    webhook   HMAC-SHA256 over the raw request body
    payment   HMAC-SHA256 over "order_id|payment_id"

Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from razorshield.serving.razorpay import (
    extract_event,
    verify_payment_signature,
    verify_signature,
)
from razorshield.serving.store import EventStore

SECRET = "a_test_secret"
ORDER = "order_TestOrder0001"
PAYMENT = "pay_TestPayment001"


def payment_signature(order: str = ORDER, payment: str = PAYMENT,
                      secret: str = SECRET) -> str:
    return hmac.new(
        secret.encode(), f"{order}|{payment}".encode(), hashlib.sha256
    ).hexdigest()


def webhook_body(event: str = "payment.authorized", amount: int = 845000) -> bytes:
    return json.dumps({
        "event": event,
        "payload": {"payment": {"entity": {
            "id": PAYMENT, "amount": amount, "currency": "INR",
            "status": "authorized", "method": "upi",
            "created_at": int(time.time()),
            "notes": {"customer_id": "CUST_000001", "device_id": "DEV_1",
                      "device_type": "Mobile", "location": "LOC_01"},
        }}},
    }).encode()


# --- payment signature (Checkout handler) ---------------------------------

def test_payment_signature_accepts_a_genuine_signature():
    assert verify_payment_signature(ORDER, PAYMENT, payment_signature(), SECRET)


@pytest.mark.parametrize(
    "order, payment, signature, secret",
    [
        (ORDER, PAYMENT, payment_signature()[:-1] + "0", SECRET),  # tampered
        ("order_Different", PAYMENT, payment_signature(), SECRET),  # other order
        (ORDER, "pay_Different", payment_signature(), SECRET),      # other payment
        (ORDER, PAYMENT, payment_signature(secret="wrong"), SECRET),
        (ORDER, PAYMENT, "", SECRET),
        (ORDER, PAYMENT, payment_signature(), ""),
        ("", PAYMENT, payment_signature(), SECRET),
    ],
)
def test_payment_signature_rejects_everything_else(order, payment, signature, secret):
    assert not verify_payment_signature(order, payment, signature, secret)


def test_payment_and_webhook_signatures_are_not_interchangeable():
    """A webhook-style digest must not pass the payment check, or vice versa."""
    body = webhook_body()
    webhook_sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert not verify_payment_signature(ORDER, PAYMENT, webhook_sig, SECRET)
    assert not verify_signature(body, payment_signature(), SECRET)


# --- webhook signature ------------------------------------------------------

def test_webhook_signature_accepts_a_genuine_body():
    body = webhook_body()
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert verify_signature(body, signature, SECRET)


def test_webhook_signature_rejects_a_tampered_body():
    body = webhook_body()
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert not verify_signature(body.replace(b"845000", b"100"), signature, SECRET)


def test_webhook_signature_rejects_missing_inputs():
    body = webhook_body()
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert not verify_signature(body, "", SECRET)
    assert not verify_signature(body, signature, "")


def test_webhook_signature_is_computed_over_raw_bytes():
    """Re-serialising the parsed JSON must not still verify.

    The digest covers the exact bytes Razorpay sent. Round-tripping through
    json.loads/json.dumps changes separators and key order, so a handler that
    verifies the re-serialised form would reject genuine deliveries -- and one
    that normalises before hashing would accept forged ones.
    """
    body = webhook_body()
    signature = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    reserialised = json.dumps(json.loads(body), indent=2, sort_keys=True).encode()
    assert reserialised != body
    assert not verify_signature(reserialised, signature, SECRET)


# --- payload extraction -----------------------------------------------------

def test_amounts_are_converted_from_paise():
    event, name = extract_event(json.loads(webhook_body(amount=845000)))
    assert name == "payment.authorized"
    assert event.amount == 8450.0


def test_failed_payments_are_marked_declined():
    event, _ = extract_event(json.loads(webhook_body(event="payment.failed")))
    assert event.declined is True


def test_unscorable_events_are_ignored():
    assert extract_event(json.loads(webhook_body(event="payment.pending"))) is None


# --- event-id idempotency ---------------------------------------------------

def test_event_ids_dedupe_but_payment_ids_do_not_collide():
    """Two events about the same payment are distinct and must both survive."""
    store = EventStore(":memory:")
    store.record_webhook_event("evt_A", "payment.authorized", time.time(), PAYMENT)
    store.record_webhook_event("evt_B", "payment.captured", time.time(), PAYMENT)
    store.commit()

    assert store.seen_event("evt_A") is not None
    assert store.seen_event("evt_B") is not None
    assert store.seen_event("evt_C") is None
    store.close()
