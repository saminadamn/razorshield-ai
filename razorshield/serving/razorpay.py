"""Razorpay webhook verification and payload unpacking.

Deliberately not using the Razorpay SDK. Signature verification is six lines of
HMAC and it is the one piece of this file that must be obviously correct, so it
is better read than imported.

Test Mode only. The client refuses to start with a key id that is not
`rzp_test_`, because the difference between test and live keys is one character
and the consequence is real money.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path

from .online import RawEvent

# The repository root, where a developer's .env lives.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_dotenv_loaded = False


def load_env_file() -> Path | None:
    """Load `.env` into the environment, once, if it exists.

    Without this, `.env.example` tells you to copy the file to `.env` and the
    keys are then silently ignored -- the app reads `os.environ` and nothing
    ever put them there. Real environment variables always win, so an explicit
    `RAZORPAY_KEY_ID=... uvicorn ...` still overrides the file.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return None
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional dependency; env vars still work
        return None
    for candidate in (Path.cwd() / ".env", _REPO_ROOT / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return candidate
    return None


# Events worth scoring. An authorisation is the moment a decision can still be
# made; a failure is evidence for the next transaction's velocity features.
SCORABLE_EVENTS = {"payment.authorized", "payment.captured", "payment.failed"}

# Razorpay's method strings mapped to the values the model was trained on.
METHOD_MAP = {
    "upi": "UPI",
    "card": "Card",
    "netbanking": "NetBanking",
    "wallet": "Wallet",
    "emi": "EMI",
}
DEFAULT_METHOD = "Card"

TEST_KEY_PREFIX = "rzp_test_"


def verify_signature(body: bytes, signature: str, secret: str) -> bool:
    """Constant-time check of Razorpay's `X-Razorpay-Signature` header.

    The digest must be computed over the *raw* request bytes. Re-serialising
    the parsed JSON changes key order and whitespace and will never match.
    """
    if not signature or not secret:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_payment_signature(
    order_id: str, payment_id: str, signature: str, secret: str
) -> bool:
    """Verify the signature Checkout hands back after a successful payment.

    HMAC-SHA256 over `order_id|payment_id`, keyed with the API secret. This is
    a different construction from the webhook signature, which is taken over
    the whole raw body -- they are not interchangeable.

    `order_id` must be the id read back from our own records, never the
    `razorpay_order_id` the browser supplied: verifying a client-supplied order
    against a client-supplied signature proves nothing.
    """
    if not (order_id and payment_id and signature and secret):
        return False
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@dataclass(frozen=True)
class Settings:
    key_id: str = ""
    key_secret: str = ""
    webhook_secret: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        load_env_file()
        return cls(
            key_id=os.environ.get("RAZORPAY_KEY_ID", ""),
            key_secret=os.environ.get("RAZORPAY_KEY_SECRET", ""),
            webhook_secret=os.environ.get("RAZORPAY_WEBHOOK_SECRET", ""),
        )

    @property
    def configured(self) -> bool:
        return bool(self.key_id and self.key_secret)

    @property
    def is_test_mode(self) -> bool:
        return self.key_id.startswith(TEST_KEY_PREFIX)

    def require_test_mode(self) -> None:
        if self.configured and not self.is_test_mode:
            raise RuntimeError(
                "RAZORPAY_KEY_ID is not a test key. RazorShield is a demo and "
                "refuses to talk to a live Razorpay account."
            )


def extract_event(
    payload: dict, default_merchant: str = "MERCH_LIVE"
) -> tuple[RawEvent, str] | None:
    """Unpack a webhook body into a scorable event.

    Returns None for events we do not score. The customer and device fields
    come from the order's `notes`, which our own checkout page sets -- Razorpay
    has no idea who the customer is in our terms, or what device they are on.
    """
    event_name = payload.get("event", "")
    if event_name not in SCORABLE_EVENTS:
        return None

    entity = (
        payload.get("payload", {}).get("payment", {}).get("entity")
    )
    if not entity:
        return None

    notes = entity.get("notes") or {}
    # Razorpay reports amounts in paise. Dividing by 100 is not optional.
    amount = float(entity.get("amount", 0)) / 100.0

    event = RawEvent(
        transaction_id=str(entity.get("id", "")),
        customer_id=str(notes.get("customer_id") or "CUST_UNKNOWN"),
        merchant_id=str(notes.get("merchant_id") or default_merchant),
        ts=float(entity.get("created_at", 0)),
        amount=amount,
        payment_method=METHOD_MAP.get(
            str(entity.get("method", "")).lower(), DEFAULT_METHOD
        ),
        device_id=str(notes.get("device_id") or "DEV_UNKNOWN"),
        device_type=str(notes.get("device_type") or "Mobile"),
        location=str(notes.get("location") or "UNKNOWN"),
        declined=event_name == "payment.failed",
    )
    return event, event_name


class RazorpayClient:
    """Minimal Test Mode client: create an order, read a payment back."""

    BASE_URL = "https://api.razorpay.com/v1"

    def __init__(self, settings: Settings):
        settings.require_test_mode()
        self.settings = settings

    def _auth(self) -> tuple[str, str]:
        return (self.settings.key_id, self.settings.key_secret)

    def create_order(
        self, amount_rupees: float, notes: dict, receipt: str | None = None
    ) -> dict:
        import httpx

        response = httpx.post(
            f"{self.BASE_URL}/orders",
            auth=self._auth(),
            json={
                "amount": int(round(amount_rupees * 100)),  # paise
                "currency": "INR",
                "receipt": receipt or "razorshield-demo",
                "notes": notes,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()

    def fetch_payment(self, payment_id: str) -> dict:
        import httpx

        response = httpx.get(
            f"{self.BASE_URL}/payments/{payment_id}",
            auth=self._auth(),
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()
