# Deploying RazorShield AI

## What actually has to reach the server

The model, calibrator, reports and event store are **generated, not committed**.
A clone alone cannot serve traffic — the app refuses to start and tells you which
artifacts are missing.

| | Size | Needed at runtime? |
| --- | --- | --- |
| `models/razorshield_model.joblib` | 320 KB | yes |
| `models/risk_scorer.joblib` | 1.5 KB | yes |
| `data/raw/manifest.json` | 1.2 KB | yes — serving constants |
| `reports/*.json` | 310 KB | yes — the console's model card |
| `data/razorshield.db` | 34 MB | yes — customer history |
| `data/raw/transactions.csv` | 14 MB | **no** — training only |
| `data/raw/transactions_meta.csv` | 6.7 MB | **no** — analysis only |

So serving needs about **630 KB of artifacts plus the event store**. The 20 MB of
training CSVs stay in the build.

## Resource envelope, measured

| Stage | Time | Peak memory |
| --- | --- | --- |
| `generate` | 14 s | 161 MB |
| `compare` (4 models) | 29 s | 394 MB |
| `explain` | 35 s | — |
| `score` | 25 s | — |
| `seed` | ~2 min | — |
| **Serving** | — | **267 MB resident** |

The build peaks near 400 MB; serving settles around 270 MB. That matters: a
512 MB free tier can *run* this comfortably but may struggle to *build* it. Hence
the multi-stage Dockerfile — the pipeline runs at image-build time, and the host
only ever runs the server.

## Recommended: Docker

The `Dockerfile` runs the full pipeline in the build stage, including the quality
gate and the training/serving parity check. If either fails, the build fails
rather than shipping a model nobody verified.

```bash
docker build -t razorshield .
```

```bash
docker run -p 8077:8077 -e RAZORPAY_KEY_ID=rzp_test_xxx -e RAZORPAY_KEY_SECRET=xxx razorshield
```

The container reads `$PORT` and binds `0.0.0.0`, so it works unmodified on any
host that injects a port. CI builds this image and smoke-tests every page and the
scoring endpoint on each push.

### Hosts this runs on as-is

- **Hugging Face Spaces (Docker SDK)** — free, generous memory, public HTTPS URL.
  The most natural home for an ML demo. Set `app_port: 8077` in the Space's
  README front-matter, and add the Razorpay keys as Space secrets.
- **Fly.io** — `fly launch` detects the Dockerfile. Free allowance covers a demo.
- **Render** — Docker runtime. The free instance sleeps after inactivity, so the
  first request after a gap takes ~30 s.
- **Railway, Cloud Run, any container host** — no changes needed.

## Without Docker

Run the pipeline once on the host, then serve:

```bash
pip install -e . && python -m razorshield.data.generate --out data/raw && python -m razorshield.models.compare --data data/raw --out reports && python -m razorshield.explain.run --data data/raw --out reports && python -m razorshield.scoring.score --data data/raw --out reports && python -m razorshield.serving.seed --db data/razorshield.db --score 800
```

```bash
uvicorn razorshield.serving.app:app --host 0.0.0.0 --port $PORT
```

Budget ~4 minutes and 400 MB for that build step.

## Environment variables

| Variable | Required | Notes |
| --- | --- | --- |
| `PORT` | no | Injected by most hosts. Defaults to 8077. |
| `RAZORPAY_KEY_ID` | for the payment demo | Must start with `rzp_test_`. The app refuses to start with a live key. |
| `RAZORPAY_KEY_SECRET` | for the payment demo | Set as a host secret, never in the image. |
| `RAZORPAY_WEBHOOK_SECRET` | for webhooks | Without it the webhook endpoint **fails closed** and rejects everything. |
| `RAZORSHIELD_ALLOW_UNSIGNED` | never in production | Local testing only. |

`.env` is loaded if present, but real environment variables win — so host-injected
secrets override the file. `.env` is gitignored and excluded from the image.

Check what the server actually loaded:

```bash
curl -s https://your-host/health
```

It reports `razorpay_configured`, `razorpay_test_mode` and which `env_file` it
found, so a missing secret is one request to diagnose.

## The webhook only works once deployed

This is the real payoff of deploying. Razorpay needs a publicly reachable HTTPS
URL to deliver webhooks, which localhost is not — that is why the demo checkout
uses `POST /payments/verify` instead.

Once you have a public URL:

1. Razorpay Dashboard → Account & Settings → **Webhooks** → Add New Webhook
2. URL: `https://your-host/webhooks/razorpay`
3. Choose a secret, and set the same string as `RAZORPAY_WEBHOOK_SECRET`
4. Subscribe to `payment.authorized`, `payment.captured`, `payment.failed`,
   and `payment.dispute.created`

Deliveries are verified by HMAC over the raw body, deduped on the
`x-razorpay-event-id` header, and a failed signature is rejected with 401.

## Two things to know before you demo it live

**The event store is ephemeral on most free tiers.** Container filesystems reset
on redeploy, so scored transactions vanish. That is fine for a demo — the image
ships with 800 seeded assessments, so the console is never empty — but attach a
volume if you want live scores to persist.

**Free instances sleep.** Render and similar spin down after inactivity and take
around 30 seconds to wake. Load the page once before you start recording.
