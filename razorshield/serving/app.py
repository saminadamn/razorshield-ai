"""RazorShield API: receive a payment event, score it, explain it, store it.

    uvicorn razorshield.serving.app:app --reload

The webhook is the integration point with Razorpay Test Mode. `/score` does the
same work from a plain JSON body, so the system can be demonstrated without a
Razorpay account at all.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..scoring.score import RiskScorer
from .online import OnlineFeatureBuilder, RawEvent, ServingConstants
from .razorpay import (
    RazorpayClient,
    Settings,
    extract_event,
    load_env_file,
    verify_payment_signature,
    verify_signature,
)
from .store import EventStore

MODEL_BUNDLE = Path(os.environ.get("RAZORSHIELD_MODEL", "models/razorshield_model.joblib"))
SCORER_BUNDLE = Path(os.environ.get("RAZORSHIELD_SCORER", "models/risk_scorer.joblib"))
MANIFEST = Path(os.environ.get("RAZORSHIELD_MANIFEST", "data/raw/manifest.json"))
DB_PATH = Path(os.environ.get("RAZORSHIELD_DB", "data/razorshield.db"))
REPORTS = Path(os.environ.get("RAZORSHIELD_REPORTS", "reports"))

# Fail closed: an unverified webhook is refused unless this is explicitly set
# for local testing. It should never be set anywhere reachable from outside.
ALLOW_UNSIGNED = os.environ.get("RAZORSHIELD_ALLOW_UNSIGNED", "").lower() in {"1", "true"}

DISPUTE_EVENTS = {"payment.dispute.created", "payment.dispute.lost"}


class ScoreRequest(BaseModel):
    """A payment attempt described directly, bypassing Razorpay."""

    transaction_id: str
    customer_id: str
    amount: float = Field(gt=0)
    merchant_id: str = "MERCH_DEMO"
    payment_method: str = "UPI"
    device_id: str = "DEV_DEMO"
    device_type: str = "Mobile"
    location: str = "LOC_00"
    declined: bool = False
    ts: float | None = None


class VerifyRequest(BaseModel):
    """Exactly what Razorpay Checkout hands back on a successful payment."""

    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class OrderRequest(BaseModel):
    amount: float = Field(gt=0)
    customer_id: str
    device_id: str = "DEV_DEMO"
    device_type: str = "Mobile"
    location: str = "LOC_00"


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [p for p in (MODEL_BUNDLE, SCORER_BUNDLE, MANIFEST) if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing artifacts: "
            + ", ".join(str(p) for p in missing)
            + ". Run the data, compare and scoring stages first."
        )
    app.state.scorer = RiskScorer.load(MODEL_BUNDLE, SCORER_BUNDLE)
    app.state.store = EventStore(DB_PATH)
    app.state.builder = OnlineFeatureBuilder(
        app.state.store, ServingConstants.from_manifest(MANIFEST)
    )
    app.state.env_file = load_env_file()
    app.state.settings = Settings.from_env()
    app.state.settings.require_test_mode()
    yield
    app.state.store.close()


app = FastAPI(
    title="RazorShield AI",
    description="Transaction risk scoring with SHAP explanations.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_state(request: Request):
    return request.app.state


def assess(state, event: RawEvent) -> dict:
    """Score one event, then record it. Order matters.

    The event is written to the store only after it has been scored, so a
    transaction can never appear in its own velocity or history features.
    """
    existing = state.store.assessment(event.transaction_id)
    if existing is not None:
        # Razorpay retries webhooks; scoring twice would also let the first
        # copy leak into the second one's history.
        return existing

    features = state.builder.build_frame(event)
    payload = state.scorer.assess_one(features, event.transaction_id)
    payload["customer_id"] = event.customer_id
    payload["amount"] = event.amount

    state.builder.observe(event, commit=False)
    state.store.record_assessment(payload, event.customer_id, event.ts, event.amount)
    state.store.commit()
    return payload


STATIC_DIR = Path(__file__).parent / "static"
# Serves the shared stylesheet both pages link, so the theme has one definition
# rather than two that drift.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/checkout", response_class=HTMLResponse)
def checkout_page() -> str:
    """Demo checkout: create a Test Mode order, pay, see the risk assessment."""
    return (STATIC_DIR / "checkout.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def landing_page() -> str:
    """The project's front door: what it does, what it scores, what it misses."""
    return (STATIC_DIR / "landing.html").read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page() -> str:
    """Operations view: live risk distribution, model quality, alert detail."""
    return (STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")


@app.get("/health")
def health(state=Depends(get_state)) -> dict:
    return {
        "status": "ok",
        "model": state.scorer.explainer.model_name,
        "razorpay_configured": state.settings.configured,
        "razorpay_test_mode": state.settings.is_test_mode,
        # Where the keys came from, so a missing .env is one request to diagnose.
        "env_file": str(state.env_file) if state.env_file else None,
        "razorpay_key_id": (
            state.settings.key_id[:12] + "..." if state.settings.key_id else None
        ),
        "webhook_verification": bool(state.settings.webhook_secret) or (
            "DISABLED" if ALLOW_UNSIGNED else False
        ),
        "store": state.store.summary(),
    }


@app.post("/score")
def score(body: ScoreRequest, state=Depends(get_state)) -> dict:
    """Score a payment described directly. No Razorpay account required."""
    event = RawEvent(
        transaction_id=body.transaction_id,
        customer_id=body.customer_id,
        merchant_id=body.merchant_id,
        ts=body.ts if body.ts is not None else time.time(),
        amount=body.amount,
        payment_method=body.payment_method,
        device_id=body.device_id,
        device_type=body.device_type,
        location=body.location,
        declined=body.declined,
    )
    return assess(state, event)


@app.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request, state=Depends(get_state)) -> dict:
    """Receive a Razorpay Test Mode event, verify it, and score it."""
    raw = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")
    secret = state.settings.webhook_secret

    if secret:
        if not verify_signature(raw, signature, secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )
    elif not ALLOW_UNSIGNED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "RAZORPAY_WEBHOOK_SECRET is not configured. Refusing to accept "
                "unverified webhooks. Set RAZORSHIELD_ALLOW_UNSIGNED=1 only for "
                "local testing."
            ),
        )

    payload = await request.json()
    event_name = payload.get("event", "")

    # Razorpay retries until it gets a 2xx, so the same delivery can arrive
    # more than once. Dedupe on the event id rather than the payment id: two
    # different events (authorized, then captured) share a payment id and both
    # need processing.
    event_id = request.headers.get("x-razorpay-event-id", "")
    if event_id:
        seen = state.store.seen_event(event_id)
        if seen is not None:
            return {
                "event": event_name,
                "duplicate": True,
                "first_received_at": seen["received_at"],
                "assessment": state.store.assessment(seen["transaction_id"] or ""),
            }

    if event_name in DISPUTE_EVENTS:
        entity = (
            payload.get("payload", {}).get("dispute", {}).get("entity", {})
        )
        payment_id = str(entity.get("payment_id", ""))
        # A dispute is recorded now but stays invisible to the features until
        # the confirmation lag has passed -- same rule as training.
        found = state.store.mark_chargeback(payment_id, time.time())
        if event_id:
            state.store.record_webhook_event(
                event_id, event_name, time.time(), payment_id
            )
        state.store.commit()
        return {"event": event_name, "payment_id": payment_id, "recorded": found}

    extracted = extract_event(payload)
    if extracted is None:
        if event_id:
            state.store.record_webhook_event(event_id, event_name, time.time())
            state.store.commit()
        return {"event": event_name, "scored": False, "reason": "event not scorable"}

    event, _ = extracted
    result = assess(state, event)
    if event_id:
        state.store.record_webhook_event(
            event_id, event_name, time.time(), event.transaction_id
        )
        state.store.commit()
    return {"event": event_name, "scored": True, "assessment": result}


@app.post("/payments/verify")
def verify_and_assess(body: VerifyRequest, state=Depends(get_state)) -> dict:
    """Verify Checkout's payment signature, then fetch and score the payment.

    The webhook is the production path, but it needs a publicly reachable URL
    and a laptop has none, so the demo checkout page calls this instead. It is
    *not* a shortcut past authentication: Razorpay requires the signature to be
    verified server-side before a payment is treated as genuine, and a failed
    check means the payment is rejected outright rather than retried.

    Trusting a bare payment id here -- an earlier version of this endpoint did --
    would let anyone post a fabricated id and have it fetched and scored as a
    real payment.
    """
    if not state.settings.configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not configured.",
        )

    # The order must be one we created; the browser does not get to name it.
    order = state.store.get_order(body.razorpay_order_id)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown order. It was not created by this service.",
        )

    if not verify_payment_signature(
        order["order_id"],
        body.razorpay_payment_id,
        body.razorpay_signature,
        state.settings.key_secret,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment signature verification failed. Payment rejected.",
        )

    payment = RazorpayClient(state.settings).fetch_payment(body.razorpay_payment_id)
    # Shape it like a webhook body so exactly one extraction path exists.
    extracted = extract_event({
        "event": "payment.captured"
        if payment.get("status") == "captured"
        else "payment.authorized",
        "payload": {"payment": {"entity": payment}},
    })
    if extracted is None:
        raise HTTPException(status_code=400, detail="Payment is not scorable")
    event, _ = extracted
    return assess(state, event)


@app.post("/orders")
def create_order(body: OrderRequest, state=Depends(get_state)) -> dict:
    """Create a Razorpay Test Mode order for the demo checkout page.

    The customer and device identifiers travel in `notes` so they come back on
    the webhook -- Razorpay does not know the customer in our terms.
    """
    if not state.settings.configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not configured.",
        )
    client = RazorpayClient(state.settings)
    order = client.create_order(
        amount_rupees=body.amount,
        notes={
            "customer_id": body.customer_id,
            "device_id": body.device_id,
            "device_type": body.device_type,
            "location": body.location,
        },
    )
    state.store.record_order(
        order_id=str(order["id"]),
        amount=body.amount,
        customer_id=body.customer_id,
        device_id=body.device_id,
        device_type=body.device_type,
        location=body.location,
        created_at=time.time(),
    )
    state.store.commit()
    return {"order": order, "key_id": state.settings.key_id}


@app.get("/transactions/{transaction_id}")
def transaction(transaction_id: str, state=Depends(get_state)) -> dict:
    payload = state.store.assessment(transaction_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No assessment for that id")
    return payload


@app.get("/transactions")
def transactions(
    limit: int = 50, band: str | None = None, state=Depends(get_state)
) -> dict:
    return {"transactions": state.store.recent_assessments(min(limit, 500), band)}


@app.get("/model")
def model_card() -> dict:
    """Held-out evaluation of the shipped model.

    These metrics come from the offline test split, not from the live
    transactions in the store -- live payments have no labels, so precision and
    recall cannot be computed on them. The dashboard must say so.
    """
    comparison_path = REPORTS / "model_comparison.json"
    bands_path = REPORTS / "risk_bands.json"
    if not comparison_path.exists() or not bands_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reports not found. Run the compare and scoring stages.",
        )
    comparison = json.loads(comparison_path.read_text())
    bands = json.loads(bands_path.read_text())

    selected = comparison["selected_model"]
    detail = comparison["models"][selected]
    operating = detail["test"]["at_cost_threshold"]

    return {
        "model": selected,
        "selected_on": "lowest expected cost on the validation split",
        "evaluation": {
            "source": "held-out test split, never used for fitting or tuning",
            "rows": comparison["splits"]["test"]["rows"],
            "fraud_rate": comparison["splits"]["test"]["fraud_rate"],
            "pr_auc": detail["test"]["pr_auc"],
            "roc_auc": detail["test"]["roc_auc"],
            "brier": detail["test"]["brier"],
            "precision": operating["precision"],
            "recall": operating["recall"],
            "f1": operating["f1"],
            "true_positives": operating["true_positives"],
            "false_positives": operating["false_positives"],
            "false_negatives": operating["false_negatives"],
            "net_saving": operating["net_saving"],
            "saving_pct": operating["saving_pct"],
        },
        "cost_model": comparison["cost_model"],
        "all_models": {
            name: {
                "pr_auc": m["test"]["pr_auc"],
                "precision": m["test"]["at_cost_threshold"]["precision"],
                "recall": m["test"]["at_cost_threshold"]["recall"],
                "f1": m["test"]["at_cost_threshold"]["f1"],
            }
            for name, m in comparison["models"].items()
        },
        "archetype_recall": comparison["selected_detail"]["archetype_recall"],
        "band_policy": bands["band_policy"],
        "bands_test": bands["bands_test"],
    }


@app.get("/stats")
def stats(state=Depends(get_state)) -> dict:
    summary = state.store.summary()
    scale = state.scorer.scale
    return {
        "model": state.scorer.explainer.model_name,
        **summary,
        "band_boundaries": {
            "MEDIUM": scale.p_medium,
            "HIGH": scale.p_high,
            "CRITICAL": scale.p_critical,
        },
    }
