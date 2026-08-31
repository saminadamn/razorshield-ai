# RazorShield AI

**Real-time transaction risk scoring with explanations an analyst can act on.**

Razorpay AI Buildathon — **Track 02, AI Risk Manager**. A detector for one class of
loss (payment fraud), with measured precision and recall on a held-out test set, an
explicit false-positive cost, and a per-transaction reason for every decision.

The model is trained and evaluated on a controlled synthetic dataset. Razorpay Test Mode
is the integration layer that demonstrates the system reacting to real payment events.
The synthetic transactions are never presented as real Razorpay traffic.

### Defense-only

This system detects and explains fraud. It contains no capability to commit, simulate
against a live target, or evade fraud controls. Concretely:

- The synthetic data generator produces **labelled training data**, not working attacks.
  Its fraud "archetypes" are statistical shapes in a CSV — velocity bursts, amount
  deviations — with no payment instrument, no target, and no execution path.
- The Razorpay integration is **Test Mode only**, and the app raises on startup if given
  a `rzp_live_` key. It reads payments and creates test orders; it moves no money.
- Every credential path is inbound verification: HMAC signature checks that decide
  whether to *trust* a request. Nothing generates credentials or probes an endpoint.

### Headline results

Held-out test split — 20,000 transactions the model never saw during fitting or tuning.

| | |
| --- | --- |
| PR-AUC (selected model) | **0.801** |
| Precision / Recall | **64.5% / 81.4%** at the cost-optimal threshold |
| Models compared | 4 — logistic regression, random forest, XGBoost, LightGBM |
| Selected on | lowest expected cost, not best F1 |
| Feature parity, batch vs live | **18/18 exact** |
| Where it fails | `slow_bleed` recall **0.331**; 98.9% of hard negatives flagged |

Those last two rows are the point. A fraud model that reports only its wins is not
telling you what it costs to run.

## Status

Seven phases, all complete and verified end to end.

| Phase | What | State |
| --- | --- | --- |
| 1 | Synthetic dataset generator + quality gate | **done** |
| 2 | Exploratory analysis | **done** |
| 3 | Model comparison (LogReg / RF / XGBoost / LightGBM) | **done** |
| 4 | SHAP explanations on the selected model | **done** |
| 5 | Risk score + calibrated bands | **done** |
| 6 | Razorpay Test Mode webhook integration | **done** |
| 7 | Risk console | **done** |

Lint (`ruff`), 17 unit tests, an end-to-end pipeline run and a byte-identical
reproducibility check all run in CI on every push.

## What broke, and how I got out

Four failures worth reporting. Each was found by a check, not by luck, and the check
is still in the repo.

### 1. The model learned that fraud victims are safe

`explain/run.py` correlates every feature's value against its own SHAP contribution.
`previous_chargeback_count` came back at **−0.65**: customers with prior disputes were
being scored as *lower* risk. Empirically fraud fell from 3.1% at zero chargebacks to
**0.07%** at three.

The cause was my own generator. It gave each customer at most one fraud episode, so
being defrauded once made you statistically immune. Real victims are re-targeted more,
not less — the model had faithfully learned a fiction.

Fixed by re-targeting 15% of episodes at earlier victims, gated so the first dispute has
had time to surface. The gradient now runs the right way (2.6% → 4.5% → 9.1%), and the
dataset got **harder**: probe PR-AUC fell 0.853 → 0.791 once the shortcut was gone.

The direction check is now a permanent part of the explain stage. It is the single most
valuable thing I built, and it found a data bug rather than a model bug.

### 2. Training and serving disagreed on four features

The claim "we reuse the same feature code" is worth nothing unless something checks it.
`serving/parity.py` replays an identical event stream through the batch pipeline and the
live one, event by event, and compares all 18 features. First run: **4 mismatched**.

Three distinct causes, none of which I would have guessed:

- **`np.round` and Python's `round` disagree.** `np.round(523.865, 2)` is `523.86`;
  `round(523.865, 2)` is `523.87`. The cold-start constant landed exactly on that
  boundary, so 878 rows differed by one paisa.
- **Order of operations.** Batch divides by the *unrounded* mean and rounds the mean
  only for display. I had rounded first, which shifted the ratio in the 4th decimal.
- **A device id collided with itself.** Repeat-victim customers reused
  `DEV_<cust>_f` across two episodes with different install times, so one id had two
  ages — an internal inconsistency in the generated data, not just the serving path.

Now 17 of 18 features match **exactly**, and the last matches except for 50 documented
rounding ties where the mean of an even number of 2-decimal amounts lands precisely on
a half-paisa and the tie breaks on the last bit of the float sum. Those are counted and
capped rather than hidden under a wider tolerance.

### 3. I shipped an endpoint that would score forged payments

The demo checkout originally called `POST /payments/{id}/assess` with a bare payment id
and no signature check. Anyone could have posted a fabricated id and had it fetched from
the Razorpay API and scored as genuine.

Reading Razorpay's integration docs made it obvious: the payment signature **must** be
verified server-side, and a failed check means reject, not retry. It is now
`POST /payments/verify`, taking all three fields Checkout returns, with the order id read
back from our own records — verifying a browser-supplied order against a browser-supplied
signature proves nothing. `tests/test_signatures.py` covers both signature schemes,
including the case where a webhook digest is offered to the payment check.

In a fraud-detection product, that was the wrong bug to ship. It is the one I am most
glad a checklist caught.

### 4. The console showed zero critical alerts

The seeder scored the *last* N events chronologically — a biased tail, not a sample — so
the review queue opened empty of anything actionable. Now it samples across the whole
window, with each event still scored at its own position so its features see the correct
history. Distribution went from `{LOW: 293, HIGH: 5, CRITICAL: 0}` to
`{LOW: 761, MEDIUM: 2, HIGH: 20, CRITICAL: 17}`.

## Where I chose not to use AI

Three places, deliberately:

**The explanations are not generated.** SHAP decides which factors mattered and in which
direction; `explain/narrative.py` puts the factor and its actual value into words from
fixed templates. An explanation a language model invented is not an explanation — in a
payments context it is a liability, because it can sound right while being unrelated to
what the model did. The additivity identity is asserted on every run, so the numbers
shown account for the prediction exactly.

**The risk score is arithmetic, not a second model.** Probability maps to 0–100 through a
piecewise-linear transform in log-odds. A learned scorer would add a component that can
drift, needs its own evaluation, and cannot be checked by hand.

**The webhook is synchronous.** The obvious "production-grade" move is a background
queue. Measured latency is **53 ms median, 59 ms p95** against Razorpay's ~5 s budget —
100× headroom. Going async would need a single-worker queue to preserve event ordering
(velocity features depend on processing order) and would add a restart-loses-the-queue
failure mode, to buy latency nobody needs. Being production-aware means measuring first.

Machine learning is used where the problem is genuinely statistical: ranking a rare,
overlapping, imbalanced class. Everywhere else, simpler tools are easier to verify.

## Quickstart

```bash
pip install -e ".[dev,notebooks]"
```

```bash
python -m razorshield.data.generate --out data/raw
```

```bash
python -m razorshield.data.validate --data data/raw
```

```bash
python -m razorshield.models.compare --data data/raw --out reports
```

```bash
python -m razorshield.explain.run --data data/raw --out reports
```

```bash
python -m razorshield.scoring.score --data data/raw --out reports
```

Prove the online path computes the same features as training, then seed the store
and run the end-to-end demo:

```bash
python -m razorshield.serving.parity
```

```bash
python -m razorshield.serving.seed --db data/razorshield.db --score 300
```

```bash
python -m razorshield.serving.demo
```

Run the tests:

```bash
python -m pytest tests -q
```

Rebuild the exploratory analysis, and open the console:

```bash
python notebooks/build_eda.py
```

```bash
python -m uvicorn razorshield.serving.app:app --port 8077
```

Then visit `http://127.0.0.1:8077/dashboard` for the risk console and
`/checkout` for the Test Mode payment demo.

Generation takes ~10s and writes ~21MB; the full comparison takes ~25s. Outputs:

- `data/raw/transactions.csv` — 100,000 rows, the released dataset
- `data/raw/transactions_meta.csv` — ground truth sidecar (archetype, chargeback, device
  id, location). **Not for training** — it is for error analysis and per-archetype recall.
- `data/raw/manifest.json` — config, achieved rates, serving constants
- `reports/model_comparison.json`, `reports/explanations.json`, `reports/risk_bands.json`
- `models/razorshield_model.joblib` — model, preprocessor, and frozen threshold
- `models/risk_scorer.joblib` — calibrator and band boundaries

## The dataset

100,000 transactions over a ~115-day window, 10,000 customers, 300 merchants,
**3.08% fraud**. No names, phone numbers, emails, addresses, card numbers, UPI IDs or
account numbers exist anywhere in the generator — every identifier is a synthetic counter
(`CUST_000421`, `MERCH_0091`).

### Data dictionary

| Column | Type | Notes |
| --- | --- | --- |
| `transaction_id` | string | `TXN_000123`, assigned in time order |
| `timestamp` | datetime | **Split key, not a feature.** Use it for temporal holdout |
| `customer_id` | string | synthetic |
| `merchant_id` | string | synthetic |
| `amount` | float | rupees |
| `transaction_hour` | int | 0–23 |
| `day_of_week` | categorical | `Monday` … `Sunday` |
| `payment_method` | categorical | UPI / Card / NetBanking / Wallet / EMI |
| `device_type` | categorical | Mobile / Desktop / Tablet / POS |
| `device_age_days` | int | days since this device was first seen |
| `account_age_days` | int | at transaction time |
| `transactions_last_1h` | int | prior attempts only |
| `transactions_last_24h` | int | prior attempts only |
| `avg_transaction_amount` | float | expanding mean of prior amounts |
| `amount_deviation_ratio` | float | `amount / avg`, clipped at 100 |
| `failed_attempts_24h` | int | prior declines |
| `unique_devices_7d` | int | includes the current device |
| `unique_locations_7d` | int | includes the current location |
| `previous_transaction_count` | int | |
| `previous_chargeback_count` | int | **confirmed disputes only** (see lag below) |
| `merchant_risk_score` | float | lagged Beta-posterior chargeback rate |
| `velocity_score` | float | 0–1 burst score vs the customer's own rate |
| `is_fraud` | binary | target — the *observed* label, noise included |

## How the labels are generated

Fraud is **not** assigned by a formula over the released features. If it were, a model
would just recover the labelling function and every metric would be meaningless.

Instead the generator simulates behaviour. Customers have spend distributions, activity
rates, hour-of-day habits, device histories and travel windows. A background stream of
genuine transactions is drawn from those. Fraud is then injected as *episodes* on
compromised accounts, and the released features are **measured off the resulting event
stream** afterwards.

| Archetype | Shape | Share of fraud rows | Difficulty |
| --- | --- | --- | --- |
| `ato_burst` | new device, new location, rapid high-value run, preceded by declines | 33.6% | easy |
| `slow_bleed` | 1.2–2.5× normal amounts over days, own device, normal hours | 21.0% | very hard |
| `card_testing` | dozens of tiny amounts in minutes, mostly declined, then a real hit | 20.1% | easy |
| `device_swap_drain` | new device but the usual location, moderate amounts | 15.4% | medium |
| `merchant_collusion` | round amounts at a small ring of merchants, normal device | 10.0% | medium |

Episode weights target a balanced share of fraud *rows*, not episodes — card testing emits
~25 rows per episode and merchant collusion ~2, so equal episode weights would leave the
label dominated by the easiest pattern.

15% of episodes re-target someone already defrauded, once their first dispute has had time
to surface. Without that, one episode per customer would make a prior chargeback a
*protective* signal. See "What the direction check caught" below — this was a real bug.

### Hard negatives

~150 genuine customers get an episode *structurally identical to an ATO burst*: new phone,
travelling, a big purchase, a couple of declines. Labelled `0`. 489 rows. These are the
reason precision is not free — the selected model flags 98.9% of them.

### Label noise

The shipped `is_fraud` is the **observed** label: 2% of fraud is never reported (labelled
0) and a small number of genuine transactions are disputed by the cardholder (labelled 1).
109 rows are flipped. Ground truth is in the sidecar, so you can separate model error from
label error.

## Two rules that keep it honest

**1. No time travel.** Every feature is computable strictly before the transaction it
describes. Rolling windows count prior attempts only.

**2. Chargebacks are lagged.** A dispute raised on day 40 does not exist for a model
scoring on day 20. Both `previous_chargeback_count` and `merchant_risk_score` apply a
35-day confirmation lag. `merchant_risk_score` is a Beta posterior over disputes confirmed
before *t*, across transactions old enough to have been disputed — so cold-start merchants
sit at the prior instead of silently encoding the answer.

The first 60 simulated days are dropped from the released file. They exist only so
merchant risk scores and customer histories are already warm on day 1.

## Training/serving parity

The webhook payload from Razorpay gives `amount`, `method`, `status`, `created_at`,
`order_id`. It does **not** give `account_age_days`, `unique_devices_7d` or
`transactions_last_1h` — those are RazorShield's own state, computed from an event store
we maintain.

So the window primitives in `features.py` (`count_in_window`, `unique_in_window`,
`velocity_score`) take *(prior events, now)* rather than a whole dataframe. Batch replay
and online scoring call the same functions. `manifest.json` carries the constants the
online path must reuse — most importantly `cold_start_amount`, the population prior used
when a customer has no history, measured on the warmup window only.

Getting this wrong is the classic way a demo silently scores garbage: the model trains on
one definition of "velocity" and the webhook computes another.

This is now enforced rather than intended -- see **Proving training/serving
parity** below, which compares all 18 features across both paths on every row
and fails the build if they drift.

## Exploratory analysis

`notebooks/01_eda.ipynb` is generated by `notebooks/build_eda.py` rather than
hand-edited, so re-running it after regenerating the data re-runs the analysis
instead of leaving stale numbers in a committed notebook.

What it establishes, before any model is fitted:

- **1 : 31 imbalance.** "Always legitimate" scores 96.9% accuracy and catches
  nothing, which is why every metric downstream is PR-AUC.
- **No missing values, no duplicate ids** across all 23 columns.
- **Night is the sharpest single slice.** 00:00–05:59 carries a 14.6% fraud rate,
  4.7x the base rate.
- **Nothing is a relabelled target.** The strongest single feature reaches ROC-AUC
  0.862 — high, but nowhere near the 0.95 leak threshold the gate enforces.
- **Some features are near-duplicates.** `transactions_last_1h` and
  `failed_attempts_24h` correlate at +0.945, and the two velocity counts at
  +0.906. That inflates a linear model's variance without adding information, and
  it is why the tree models open up such a gap over logistic regression.
- **The two hard cases are visible in the data itself.** Median feature values
  for `hard_negative` (labelled legitimate) sit on top of `ato_burst`, and
  `slow_bleed` sits on top of ordinary traffic.

Charts follow the same rules as the console: fraud *rate* rather than fraud
*count* (counts just track traffic volume), group sizes printed beside every rate
so a high rate on a thin slice is visible as one, a log axis for amounts, and a
diverging scale for the correlation matrix because correlation is signed.

## Is the data hard enough?

`validate.py` is a gate, not a report. It exits non-zero if any single feature separates
the classes (ROC-AUC > 0.95), if a quick model exceeds 0.95 PR-AUC (too easy), or if it
falls below 0.15 (no signal).

Probe models, temporal 70/30, thresholded at a 3% alert budget:

| Probe model | PR-AUC | ROC-AUC | Precision | Recall |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.655 | 0.940 | 0.650 | 0.650 |
| Gradient boosting | 0.791 | 0.968 | 0.740 | 0.740 |

Note the ROC-AUC vs PR-AUC gap. ROC-AUC looks excellent at 0.97 and is close to
meaningless at a 3% base rate.

## Model comparison

**Protocol.** Chronological 60/20/20. Fit on train. Decide *everything* on validation —
early stopping, the operating threshold, and which model ships. Touch test once.

**Imbalance is handled at the threshold, not by reweighting.** Class weights and
`scale_pos_weight` distort predicted probabilities, and Phase 4 turns those into a 0–100
risk score. All four models are trained unweighted, ranked on PR-AUC, thresholded on cost.

Held-out test set, 20,000 transactions, 3.55% fraud:

| Model | PR-AUC | ROC-AUC | Brier | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.681 | 0.940 | 0.0176 | 0.425 | 0.780 | 0.551 |
| Random forest | 0.796 | 0.964 | 0.0133 | 0.620 | 0.825 | 0.708 |
| XGBoost | 0.804 | 0.970 | 0.0131 | 0.686 | 0.798 | 0.738 |
| **LightGBM** (selected) | 0.801 | 0.969 | 0.0134 | 0.645 | 0.814 | 0.720 |

Precision and recall are at each model's own cost-optimal threshold, so they are not
comparable across rows — the PR-AUC column is.

**On the selection.** XGBoost has the best validation PR-AUC. LightGBM has the best
validation *expected cost*, and cost is the objective, so LightGBM ships. On test the two
are separated by 0.003 PR-AUC, which is noise. Re-picking after seeing test would make the
held-out number meaningless.

### The business objective

- missed fraud → transaction amount + ₹500 dispute fee
- false positive → ₹40 review + ₹250 for blocking a real customer
- caught fraud → ₹40 review

| Model | Alerts | Caught | Missed | Net saving | vs doing nothing |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 1,300 | 553 | 156 | ₹2,375,201 | 76.8% |
| Random forest | 943 | 585 | 124 | ₹2,686,114 | 86.9% |
| XGBoost | 825 | 566 | 143 | ₹2,670,721 | 86.4% |
| **LightGBM** | 894 | 577 | 132 | ₹2,687,447 | 86.9% |

Doing nothing costs ₹3,092,545 over the test window.

**The 86.9% assumes review stops 100% of flagged fraud** (`review_catch_rate = 1.0`). That
is the most optimistic assumption in the project; the script prints the caveat itself.

Under a fixed **1% alert budget** — when the constraint is analyst headcount, not money —
LightGBM gets 0.963 precision at 0.329 recall. Same model, very different operating point,
which is why the threshold is a business input and not a hyperparameter.

As false positives get more expensive the optimum moves: at ₹62 per false decline the
model flags 5.7% of traffic and catches 87% of fraud; at ₹1,000 it flags 2.9% and catches
77%.

### Calibration

Predicted vs observed fraud rate on test, binned finely at the top where the risk bands
will sit:

| Score bucket | n | Predicted | Observed |
| --- | --- | --- | --- |
| 0.0037–0.0050 | 10,000 | 0.0040 | 0.0010 |
| 0.0099–0.0227 | 2,000 | 0.0146 | 0.0145 |
| 0.0826–0.5162 | 500 | 0.2484 | 0.3120 |
| 0.5162–0.8553 | 300 | 0.7038 | 0.7967 |
| 0.8997–0.9153 | 63 | 0.9002 | 0.9841 |

Systematically **under-confident at the top** — when it says 0.90 the observed rate is
0.98. Phase 4 fixes this with isotonic recalibration.

## Explanations

`explain/` runs TreeSHAP on the selected model. Contributions are in **log-odds**, the
space where SHAP is additive:

```
base_value + sum(contributions) = log-odds of the predicted probability
```

That identity is asserted on every run (`additivity check`, max error 1.07e-14). If it
ever fails the run aborts rather than printing explanations nobody can trust.

The model sees 31 columns because the categoricals were one-hot encoded. Contributions are
summed back onto the original 18 features before display — summing is valid precisely
because SHAP is additive. Nobody should have to read `cat__payment_method_UPI +0.03` in an
alert queue.

**No language model is involved.** SHAP decides which factors mattered and in which
direction; `narrative.py` only puts the factor and its actual value into words, from fixed
templates. An explanation a model invented is not an explanation.

Global importance (mean |contribution|): `amount` 0.30, `device_age_days` 0.26,
`merchant_risk_score` 0.22, `amount_deviation_ratio` 0.18, `velocity_score` 0.14.

### What the direction check caught

`signed_effect` correlates each feature's value against its own SHAP contribution. Higher
values of a risk feature should push risk *up*. Three features came back looking wrong, and
they turned out to be three different things:

**A real bug.** `previous_chargeback_count` correlated **−0.65**: prior victims were being
scored as *safer*. The generator gave each customer at most one fraud episode, so being
defrauded once made you immune. Empirically fraud fell from 3.1% at zero chargebacks to
0.07% at three. Real victims get re-targeted more, not less. Fixed by re-targeting 15% of
episodes at earlier victims; the gradient now runs the right way (2.6% → 4.5% → 9.1%), and
the dataset got *harder* — probe PR-AUC dropped from 0.853 to 0.791 once the shortcut was
gone.

**An interaction, not a bug.** `account_age_days` correlated **+0.73**, but the marginal
fraud rate is flat across all ten deciles. An old account on a brand-new device is
suspicious in a way a new account on a new device is not; the model is reading the
interaction, and a marginal correlation cannot see it.

**Working as designed.** `previous_transaction_count` correlated **+0.72**, and the
marginal fraud rate genuinely climbs 0.6% → 5.7%. Fraudsters target active, high-spend
customers, which is exactly what the generator encodes.

The lesson worth keeping: explainability earned its place here by finding a data bug, not
by producing a nice chart.

### Worked examples

A caught card-testing attempt — the whole signal is one feature:

```
TXN_094466   fraud probability 0.9153       actual: FRAUD (card_testing), Rs2.54
  Device age                    0.00   +5.268  +######################
  Velocity score                0.53   +0.955  +####
  Transaction amount            2.54   +0.585  +##
  "Critical risk driven by a device first seen today, a velocity score of
   0.53 and a Rs2.54 transaction."
```

A false positive — a real customer, and the explanation shows exactly why the model could
not know better:

```
TXN_092322   fraud probability 0.8435       actual: legitimate (hard_negative), Rs3,877
  Device age                    1.00   +3.391  +######################
  Velocity score                0.76   +0.953  +######
  Transaction velocity (1h)     4.00   +0.827  +#####
```

A missed fraud — and it is obvious why nothing fired:

```
TXN_083395   fraud probability 0.0178       actual: FRAUD (slow_bleed), Rs9,480
  Transaction amount        9,480.55   +0.352  +######################
  Amount deviation              0.73   +0.275  +#################
  Device age                1,687.00   -0.245  -###############
```

₹9,480 against a customer average of ₹13,010 — *below* their normal spend, on a device
1,687 days old. There is no signal to find. This is why `slow_bleed` recall is 0.331 and
why catching it needs sequence modelling rather than a better classifier.

### Recall by archetype (selected model, test, chosen threshold)

| Archetype | Recall |
| --- | --- |
| `ato_burst` | 1.000 |
| `card_testing` | 1.000 |
| `device_swap_drain` | 0.987 |
| `merchant_collusion` | 0.905 |
| `slow_bleed` | **0.331** |
| `hard_negative` *(label 0 — these are false positives)* | 0.989 |

## Risk score and bands

### Calibration

Phase 3 found the model under-confident at the top of its range. Isotonic regression,
fitted on validation and measured on test:

| Metric | Raw | Calibrated |
| --- | --- | --- |
| Brier | 0.01335 | 0.01353 |
| ECE | 0.00713 | **0.00480** |
| Max bin deviation | 0.07690 | **0.05496** |

Isotonic is monotone, so PR-AUC and ROC-AUC are unchanged *by construction* — the ranking
is identical and only the numbers' meaning improves. Brier is marginally worse while ECE
improves by a third: the step function adds a little noise but removes the systematic
under-confidence, and ECE is what matters for a number that has to mean what it says.

The displayed probability is clamped to 0.999. Isotonic maps a pure top bin to exactly
1.0, and showing an analyst "fraud probability 1.0000" claims a certainty the model does
not have.

### The score

Showing calibrated probability × 100 would put almost every transaction between 0 and 5,
because the base rate is 3%. The score is instead **piecewise-linear in log-odds**,
anchored so the band boundaries land exactly on 0 / 30 / 60 / 80 / 100.

0–29 LOW / 30–59 MEDIUM / 60–79 HIGH / 80–100 CRITICAL is a presentation choice. Where
those bands fall in probability is not — each boundary is fitted on validation from what
the band has to *mean*:

| Band | Boundary | Chosen because | Action |
| --- | --- | --- | --- |
| MEDIUM | p ≥ 0.073 | band is ≥3× the base rate | step-up auth |
| HIGH | p ≥ 0.135 | the Phase 2 cost-optimal threshold | manual review |
| CRITICAL | p ≥ 0.667 | the tail above is ≥90% fraud | block |

The ladder is ordered by *action*, so the cost-optimal threshold — the boundary of acting
at all — sits at the bottom of HIGH, not of CRITICAL. Anchoring CRITICAL there instead
made CRITICAL swallow the entire action set and left HIGH empty; that was the first
version and it was wrong.

CRITICAL is set at 90% purity rather than a bare majority because it is the only band that
acts without a human. At the cost-optimal threshold precision is already 67% — blocking
there would mean one in three blocked customers did nothing wrong.

### What the bands actually contain

Held-out test set. Boundaries were fitted on validation and not touched again:

| Band | Score | Count | Share | Fraud rate | Lift | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| LOW | 0–30 | 18,925 | 94.6% | 0.006 | 0.2× | 0.164 |
| MEDIUM | 30–38 | 193 | 1.0% | 0.098 | 2.8× | 0.027 |
| HIGH | 60–79 | 448 | 2.2% | 0.420 | 11.8× | 0.265 |
| CRITICAL | 80–100 | 434 | 2.2% | **0.889** | 25.1× | 0.544 |

CRITICAL was fitted to 90% purity on validation and lands at 88.9% on test, so the
boundary generalises. **HIGH + CRITICAL is 4.4% of traffic and captures 81% of fraud**,
covering ₹2,502,152 of the amount at risk.

The honest caveat: **LOW still contains 16.4% of all fraud.** That is mostly `slow_bleed`,
and no threshold fixes it — the signal is not in these features.

### Reference rendering

```
  TXN_080103

    Amount              Rs4,193.58
    Payment method      UPI
    Transaction hour    03:00
    Device              Mobile, 1d old

    Fraud probability   0.9990
    Risk score          100 / 100
    Risk level          CRITICAL
    [####################]

    Top risk factors
      Device age                 ############
      Merchant risk              #####
      Velocity score             ##
      Transaction velocity (1h)  ##

    "Critical risk driven by a device first seen 1 day ago, a merchant risk
     score of 0.171 and a velocity score of 0.74."
```

`RiskScorer.assess_one()` returns exactly this as a dict — score, band, factors and
summary. It is the payload Phase 5's webhook and Phase 6's dashboard consume.

## Serving

### The problem this has to solve

A Razorpay webhook says: amount, method, status, `created_at`, `order_id`. It does not say
how many transactions this customer made in the last hour, how old their device is, or
whether they have a dispute on file. **Fourteen of the eighteen features are ours to
keep**, not Razorpay's to send. `store.py` is a SQLite event store holding exactly that
history; `online.py` computes the features from it one event at a time.

Two things Razorpay genuinely cannot tell us, and how the demo supplies them:

- **Who the customer is, in our terms.** The checkout page mints a `customer_id` and a
  device fingerprint and puts them in the order's `notes`, which come back on the webhook.
- **Merchant risk.** In Test Mode there is one merchant with no dispute history, so
  `merchant_risk_score` falls back to the Beta prior until history accumulates — exactly
  the cold-start path the batch pipeline already models.

### Proving training/serving parity

Claiming "we reuse the same functions" is cheap. `parity.py` generates a dataset, computes
features the batch way, then replays the identical stream event-by-event through the store
and the online builder, comparing all 18 features on all 9,133 rows:

```
  17 features                exact
  avg_transaction_amount     exact, except 50 rounding ties (0.55%)

PASSED -- the model is served exactly what it was trained on
```

Both paths are populated by the same helpers `seed.py` uses, so this tests the code the
demo actually runs, not a parallel implementation written to pass.

Getting there took four fixes, each a bug the test caught:

| Symptom | Cause |
| --- | --- |
| `avg_transaction_amount` off by ₹0.01 on 878 rows | `np.round(523.865, 2)` is 523.86, Python's `round` gives 523.87. The cold-start constant sat exactly on that boundary. |
| `amount_deviation_ratio` off on 558 rows | Batch divides by the *unrounded* mean and rounds the mean only for display. I had rounded first. |
| `device_age_days` off by up to 108 days | Generator bug: a repeat fraud victim reused device id `DEV_<cust>_f` with two different install times. |
| `transaction_hour` off by 1 on one row | Batch rounds timestamps to whole seconds before deriving the hour; the online path did not. |

The 50 remaining ties are the mean of an *even* number of 2-decimal amounts landing exactly
on a half-paisa, where the tie breaks on the last bit of the float sum — numpy's pairwise
`cumsum` and SQLite's compensated `SUM` disagree there. That is a representation floor, not
a definition mismatch, so it is counted and capped at 2% rather than hidden behind a wider
tolerance.

### Webhook handling

`POST /webhooks/razorpay` verifies Razorpay's `X-Razorpay-Signature` as an HMAC-SHA256 of
the **raw request bytes** (re-serialising the parsed JSON changes key order and never
matches), compared with `hmac.compare_digest`. No SDK — signature verification is six lines
and is better read than imported.

- **Fails closed.** With no `RAZORPAY_WEBHOOK_SECRET` configured the endpoint returns 503
  rather than accepting unverified events. `RAZORSHIELD_ALLOW_UNSIGNED=1` exists for local
  testing only.
- **Amounts are paise.** Razorpay reports ₹8,450 as `845000`. Dividing by 100 is not optional.
- **Idempotent, in two layers.** Razorpay retries until it gets a 2xx, so dedupe is keyed
  on the `x-razorpay-event-id` header — *not* the payment id, because `payment.authorized`
  and `payment.captured` share a payment id and both need processing. A repeated payment id
  additionally returns the stored assessment rather than rescoring, which stops the first
  delivery leaking into the second one's velocity features.
- **Score, then store.** The event is written only after it has been scored, so a
  transaction can never appear in its own history.
- **Disputes close the loop.** `payment.dispute.created` marks a chargeback, which stays
  invisible to the features until the 35-day confirmation lag has passed — the same rule as
  training.
- **Test Mode enforced.** The client refuses to start on a key id that is not `rzp_test_`.

### Verifying the payment signature

Razorpay uses **two different signature schemes**, and they are not
interchangeable:

| | Covers | Construction |
| --- | --- | --- |
| Webhook | the whole raw request body | `HMAC-SHA256(body, webhook_secret)` |
| Payment | one completed payment | `HMAC-SHA256("order_id\|payment_id", key_secret)` |

Checkout's handler returns `razorpay_payment_id`, `razorpay_order_id` and
`razorpay_signature`, and Razorpay requires the signature to be verified
server-side before the payment is treated as genuine. A failed check means the
payment is rejected outright — not retried, not fulfilled.

Two details that are easy to get wrong and that the implementation gets right:

- **The order id must come from our own records.** Verifying a browser-supplied
  order id against a browser-supplied signature proves nothing, so `POST /orders`
  persists every order it creates and the check reads the id back from there. An
  order we never created is rejected before any signature work happens.
- **The webhook digest is over the raw bytes.** Re-serialising the parsed JSON
  changes separators and key order, so a handler that normalises before hashing
  would reject genuine deliveries and accept forged ones. There is a test for
  exactly that.

`tests/test_signatures.py` covers both schemes — genuine, tampered, wrong order,
wrong payment, missing secret, and the cross-scheme case where a webhook digest
is offered to the payment check. 17 tests:

```bash
python -m pytest tests -q
```

An earlier version of this endpoint took a bare payment id and fetched it
straight from the Razorpay API. That let anyone post a fabricated id and have it
scored as a real payment — in a fraud-detection service, the wrong bug to ship.
It is now `POST /payments/verify` and takes all three fields.

### Why the webhook is synchronous

The usual advice is to acknowledge fast and do the work in the background. Measured on the
seeded store, scoring end to end — feature build, model, SHAP, calibration, write — is:

```
n=40   median 52.9 ms   p95 59.1 ms   max 175.7 ms
```

against a webhook budget of roughly five seconds. That is about 100x of headroom, so the
handler stays synchronous on purpose. Making it async would need a **single-worker queue**
to preserve arrival order, because velocity features depend on the order events are
processed in — and it would add a failure mode where a restart loses queued events. Being
production-aware here means measuring before adding machinery, not adding it reflexively.

If throughput ever outgrows this, the queue is the right change and the ordering constraint
is the thing to design around.

### Endpoints

| Route | Purpose |
| --- | --- |
| `POST /webhooks/razorpay` | verify, score and store a Test Mode payment event |
| `POST /score` | same scoring from a plain JSON body — no Razorpay account needed |
| `POST /orders` | create a Test Mode order, with ids in `notes` for the checkout page |
| `POST /payments/verify` | verify Checkout's payment signature, then fetch and score |
| `GET /model` | held-out evaluation of the shipped model |
| `GET /checkout` | demo checkout page |
| `GET /` | landing page |
| `GET /dashboard` | risk console |
| `GET /transactions/{id}` | full assessment payload |
| `GET /transactions` | recent alerts, filterable by band |
| `GET /stats` | totals and band distribution |
| `GET /health` | model, store counts, whether webhook verification is on |

### End to end

`python -m razorshield.serving.demo` boots the API in-process against a seeded store
(156,887 events, 10,000 customers) and walks one customer with real history through an
account takeover:

```
NORMAL TRANSACTION  (CUST_008736, known device, typical amount)
  everyday purchase        score  27.6  LOW       p=0.0368

ACCOUNT TAKEOVER  (same customer, new device, rapid high-value run)
  attempt 1  (declined)    score  82.7  CRITICAL  p=0.8219
  attempt 2  (declined)    score  82.7  CRITICAL  p=0.8219
  attempt 3                score  82.7  CRITICAL  p=0.8219
  attempt 4                score  72.1  HIGH      p=0.4211
  attempt 5                score  85.9  CRITICAL  p=0.9259
  attempt 6                score 100.0  CRITICAL  p=0.9990

RAZORPAY WEBHOOK  (Test Mode payload)
  valid signature      HTTP 200
  Rs845,000 paise -> Rs8,450.00
  tampered body        HTTP 401 (Invalid webhook signature)
  retried delivery     HTTP 200, deduped on event id: True
  capture of same pay  HTTP 200, processed: True
```

Each attempt is scored before it is stored, so the velocity features build exactly as they
would in production. Repeated scores are isotonic step ties; the dip at attempt 4 is real
model non-monotonicity, covered under Known limitations.

### The demo checkout page

`GET /checkout` serves a small Razorpay-inspired page: set an amount, a customer id and a
device id, then pay with Razorpay Checkout in Test Mode. The risk assessment renders beside
the form — score, band, and the SHAP factors with their signed contributions.

Test Mode instruments, from Razorpay's documented set: Visa `4100 2800 0000 1007`,
Mastercard `5500 6700 0000 1002`, RuPay `6527 6589 0000 1005` (any future expiry, any CVV),
or UPI `success@razorpay` / `failure@razorpay`.

**The webhook needs a publicly reachable URL, and a laptop does not have one.** Rather than
require an ngrok tunnel for the demo to work at all, Checkout's success handler posts all
three fields to `POST /payments/verify`, which checks the payment signature, then fetches
the payment from the Razorpay API and pushes it through *the same* extraction, features and
model. Only the transport differs — the webhook remains the production path and is
exercised by `demo.py`. This is not a shortcut past authentication: the signature check is
the same one Razorpay mandates.

A declined attempt is real signal — it feeds `failed_attempts_24h` — so `payment.failed` is
handled too. The browser only *displays* the failure; the decline is recorded from the
signature-verified webhook, because accepting declines from an unauthenticated client would
let anyone poison a customer's velocity features.

A second button scores the same inputs through `/score` with no Razorpay account at all, so
the page is useful before any keys exist.

## The risk console

`GET /dashboard` is the operations view. Static HTML served by FastAPI — two
pages did not justify a React build step, and it deploys as-is.

**The honesty problem it has to solve.** A fraud dashboard wants to show
precision and recall next to a live transaction count, which quietly implies the
two are related. They are not: live payments carry no labels, so no accuracy
figure can be computed from them. The console keeps them in separate, explicitly
titled sections — *Live scoring* (unlabelled, count and distribution only) and
*Model quality — held-out test split*, which states outright that those are not
the live transactions above.

It shows:

- **Live tiles** — transactions scored, how many were actionable, events held,
  customers tracked.
- **Risk distribution** across the four bands, with a table underneath giving each
  band's action, fraud rate, lift and share of fraud caught on the test split.
- **Model quality** — PR-AUC, precision, recall, F1, plus the four-model
  comparison with the shipped model marked and a note that precision and recall
  are each at that model's own threshold, so only PR-AUC compares across rows.
- **Alert queue** filtered by severity, opening on CRITICAL, with a detail panel
  showing the score, band, and the SHAP factors behind it.

### Look and feel

Three pages share one stylesheet, `static/theme.css`, served at `/static/theme.css`
so the theme has a single definition rather than three that drift:

| Route | What it is |
| --- | --- |
| `/` | Landing page — what the system does, the evidence, a code sample |
| `/dashboard` | Risk console — live scoring, model quality, the review queue |
| `/checkout` | Test Mode payment demo |

It follows razorpay.com's product language: a white plane, navy headlines,
`#3395ff` as the single accent, generous whitespace, hairline-bordered cards, a
sticky white nav and a deep navy footer. A marketing-style hero belongs on a
landing page and nowhere else, so the console and checkout carry the same nav and
footer chrome without one.

Two deliberate choices:

- **The name, tagline and copy are ours.** No Razorpay logo, no "Outgrow
  Ordinary". This is a submission *to* Razorpay, and a page implying official
  affiliation is worse than one that merely looks related. Every page footer
  states the project is independent and unaffiliated.
- **The plane is white and every mark still sits on a card.** The data-viz
  palette was validated against a white surface; tinted section bands are chrome
  and never carry data.

On colour for the data itself: risk bands are an ordered severity ladder, so they
use the reserved status roles rather than series colours. Two of those (MEDIUM and
HIGH) sit below the contrast floor on a light surface, and the adjacent pair
measures ΔE 13.6 for normal vision — under the 15 floor. The mitigation is that
colour never carries meaning alone: every band ships its name as text plus a count
and share, and the distribution has a table view. Signed SHAP contributions use a
diverging pair with the value printed, so direction never depends on hue.

### The contrast audit

Dark mode is a selected palette, not an inverted one — the plane becomes the brand
navy. A scripted audit walks every text node on all three pages in both modes,
compositing translucent backgrounds down the ancestor chain before measuring, and
applies the WCAG AA thresholds (4.5:1, or 3:1 for large text). It found five real
failures that eyeballing would have missed:

| What | Was | Now |
| --- | --- | --- |
| Wordmark in dark mode | navy on navy, invisible | 14.6:1 |
| Nav links in dark mode | 2.91:1 | 8.1:1 |
| Primary buttons, both modes | 3.05:1 | 4.5:1 |
| Muted body ink on white | 3.69:1 | 5.1:1 |
| `LOW` pill, both modes | 3.35:1 | 5.4:1 |

Two of those are worth explaining, because the fix was not "pick a darker colour":

- **White on Razorpay's `#3395ff` is 3.05:1.** That is fine for the large headline
  accent, where the threshold is 3:1, and it fails for 15px button text. So the
  button *fill* steps down to `#1a73e8` while `#3395ff` stays the accent
  everywhere it is not carrying text.
- **The `LOW` pill.** `MEDIUM` and `HIGH` already solve this with dark ink on a
  light fill; `LOW` and `CRITICAL` use white. `CRITICAL` passes at 4.79:1 and
  `LOW` did not, so the pill fill — and only the pill fill — steps darker. The
  documented status hex is still what draws the swatches and bars.

The audit now reports **zero failures on all three pages in both modes**. The one
remaining flagged element is Razorpay's own "Test Mode" badge, injected into
`div.razorpay-backdrop` by `checkout.js` — third-party markup, excluded rather
than restyled.

A methodological note, since it nearly cost a wrong fix: the first version of the
audit read `rgba()` backgrounds without compositing them, and scored the eyebrow
pill at 1.48:1 when the true figure was 7.82:1. The underlying issue was real
(3.65:1 before the fix, under the floor) but the number was nonsense. Measure the
composited colour, not the declared one.

### Configuration

Test Mode only — test keys start with `rzp_test_`. Copy `.env.example` to `.env`, which is
gitignored. The key secret must never reach the frontend or the repository.

```
RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxx
RAZORPAY_KEY_SECRET=
RAZORPAY_WEBHOOK_SECRET=
```

### Known limitations

- **`device_age_days` is a lower bound in production.** The batch pipeline used a device's
  true install time; a live deployment only knows when it first *saw* the device, so a
  long-standing device looks new on its first appearance and scores higher than it should
  until it has been seen once.
- **Seeding is required for a meaningful demo.** Without history every customer is new, and
  new customers all score alike.
- **SQLite is a demo choice.** The access patterns are index lookups and it survives a
  restart, but nothing here is built for concurrent writers.
- **The risk score is not monotonic across an attack.** In the run above, every feature
  moves the right way (1h velocity 1 to 5, deviation 5.3x to 12.4x) yet the raw probability
  dips at attempt 4, from 0.749 to 0.545. A gradient-boosted ensemble is a sum of step
  functions, so it is not monotonic in any single feature, and isotonic calibration
  amplifies the visible dip by mapping 0.545 onto a lower step. Both scores are still in
  actionable bands, so the decision does not change — but if a monotonic guarantee is ever
  required, LightGBM's `monotone_constraints` would enforce it on velocity and deviation.
- **The demo scores against a disposable copy of the store.** Writing into the seeded one
  makes it unrepeatable: the previous run's "attack" transactions become part of that
  customer's normal spend, the deviation signal collapses, and the same script quietly
  stops escalating. That is worth knowing generally — a fraud model's own history is state,
  and replaying against dirty state is how a demo silently stops working.

## Layout

```
razorshield/data/
  config.py      every generator knob, one frozen dataclass
  entities.py    customer / merchant / device populations
  simulate.py    event stream: legit traffic, fraud episodes, hard negatives
  features.py    the single feature definition used by training and serving
  generate.py    CLI: simulate -> label -> featurise -> write
  validate.py    quality gate

razorshield/models/
  dataset.py     temporal splits and shared preprocessing
  cost.py        the business objective: cost model and threshold selection
  compare.py     CLI: train four models, select one, persist it

razorshield/explain/
  explainer.py   TreeSHAP, aggregated back to human-readable features
  narrative.py   deterministic templates -- no LLM
  run.py         CLI: global importance, direction check, worked examples

razorshield/scoring/
  calibrate.py   isotonic calibration and calibration metrics
  bands.py       0-100 log-odds score and data-fitted action bands
  score.py       RiskScorer: the payload Phase 5 and 6 consume

razorshield/serving/
  store.py       SQLite event store -- the history Razorpay cannot send us
  online.py      per-event feature computation, same definitions as training
  parity.py      proves batch and online agree on all 18 features
  razorpay.py    webhook signature verification and Test Mode client
  app.py         FastAPI: webhook, scoring, alerts, stats
  seed.py        populate the store so the demo has real history
  demo.py        end-to-end walkthrough, no network required
  static/        theme.css plus the landing, console and checkout pages

notebooks/
  build_eda.py   generates and executes the analysis notebook
  01_eda.ipynb   exploratory analysis, outputs committed

tests/
  test_signatures.py   both Razorpay signature schemes, and payload extraction
```

Runs are reproducible: `--seed` fixes the generator, and two runs at the same seed produce
byte-identical files.
