"""CLI: global importance, sanity checks, and worked examples.

    python -m razorshield.explain.run --data data/raw --out reports

Explanations are generated for the held-out test split only -- the same rows
the reported metrics came from.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from ..models.dataset import load, temporal_split
from .explainer import Explainer

# shap warns that LightGBM binary output is now a list of arrays. Explainer
# handles both shapes, and the additivity check below proves it.
warnings.filterwarnings(
    "ignore", message=".*LightGBM binary classifier with TreeExplainer.*"
)

BAR_WIDTH = 22
N_EXPORTED = 200


def bar(share: float, positive: bool) -> str:
    """ASCII bar. Plain '#' rather than block characters so Windows consoles
    do not have to guess at an encoding."""
    width = max(1, int(round(share * BAR_WIDTH)))
    return ("+" if positive else "-") + "#" * width


def show(explanation, extra: str = "") -> None:
    print(f"  {explanation.transaction_id}    "
          f"fraud probability {explanation.probability:.4f}"
          f"    (population base rate {explanation.base_probability:.4f})")
    if extra:
        print(f"  {extra}")
    print()
    for f in explanation.factors:
        value = f.value
        numeric = isinstance(value, int | float | np.number)
        shown = f"{value:,.2f}" if numeric else str(value)
        print(f"    {f.label:<26} {shown:>12}   "
              f"{f.contribution:+7.3f}  {bar(f.share, f.contribution > 0)}")
    print()
    print(f"    {explanation.summary}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain predictions with SHAP.")
    parser.add_argument("--data", default="data/raw", type=Path)
    parser.add_argument("--out", default="reports", type=Path)
    parser.add_argument(
        "--bundle", default="models/razorshield_model.joblib", type=Path
    )
    args = parser.parse_args()

    frame, meta = load(args.data)
    _, _, test = temporal_split(frame, meta)
    explainer = Explainer(args.bundle)

    test_ids = frame.iloc[-len(test):]["transaction_id"].reset_index(drop=True)
    X = test.X
    y = test.y.to_numpy()

    proba = explainer.model.predict_proba(explainer.encode(X))[:, 1]
    flagged = proba >= explainer.threshold

    print("=" * 78)
    print(f"SHAP EXPLANATIONS  ({explainer.model_name}, "
          f"{len(test):,} held-out transactions)")
    print("=" * 78)

    # --- correctness -------------------------------------------------------
    sample = X.iloc[:2000]
    additivity = explainer.check_additivity(sample)
    status = "PASS" if additivity["passed"] else "FAIL"
    print(f"\nadditivity check   {status}   "
          f"max error {additivity['max_error']:.2e}")
    print("  base value + sum(contributions) reconstructs the model's log-odds,")
    print("  so the numbers below account for the prediction exactly.")
    if not additivity["passed"]:
        raise SystemExit("SHAP additivity check failed -- explanations are unreliable")

    # --- global ------------------------------------------------------------
    print()
    print("=" * 78)
    print("GLOBAL IMPORTANCE  (mean |contribution|, log-odds)")
    print("=" * 78)
    importance = explainer.global_importance(X)
    top = importance.max() or 1.0
    for feature, score in importance.items():
        width = max(1, int(round(score / top * BAR_WIDTH)))
        print(f"  {feature:<28} {score:>7.4f}  {'#' * width}")

    print()
    print("=" * 78)
    print("DIRECTION CHECK  (correlation between a feature's value and its own")
    print("contribution -- positive means higher values push risk up)")
    print("=" * 78)
    effects = explainer.signed_effect(X).head(10)
    for _, row in effects.iterrows():
        print(f"  {row['feature']:<28} {row['value_vs_contribution']:+.3f}")

    # --- worked examples ---------------------------------------------------
    archetype = test.meta["fraud_archetype"].to_numpy()

    def pick(mask: np.ndarray, by: np.ndarray, want_max: bool = True) -> int | None:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        return int(idx[np.argmax(by[idx])] if want_max else idx[np.argmin(by[idx])])

    # Exclude label-flipped rows from the miss example: a disputed genuine
    # transaction the model scored low is a labelling artefact, not a miss.
    genuine = ~test.meta["label_flipped"].to_numpy().astype(bool)

    cases = [
        ("CAUGHT FRAUD  (true positive)", pick(flagged & (y == 1), proba), "Critical"),
        ("FALSE POSITIVE  (a genuine customer, flagged)",
         pick(flagged & (y == 0) & (archetype == "hard_negative"), proba), "Critical"),
        ("MISSED FRAUD  (false negative)",
         pick((~flagged) & (y == 1) & genuine, test.amount.to_numpy()), "Low"),
        ("ORDINARY TRANSACTION  (true negative)",
         pick((~flagged) & (y == 0), proba, want_max=False), "Low"),
    ]

    for title, idx, band in cases:
        if idx is None:
            continue
        print()
        print("=" * 78)
        print(title)
        print("=" * 78)
        note = f"actual label: {'FRAUD' if y[idx] else 'legitimate'}"
        if archetype[idx] != "none":
            note += f"   pattern: {archetype[idx]}"
        note += f"   amount: Rs{test.amount.iloc[idx]:,.2f}"
        # A label-flipped row is not a model error: the label itself is noise.
        if bool(test.meta["label_flipped"].iloc[idx]):
            note += "\n  LABEL NOISE: ground truth disagrees with the shipped label"
        explanation = explainer.explain(
            X.iloc[[idx]], [test_ids.iloc[idx]], bands=[band]
        )[0]
        show(explanation, note)

    # --- export ------------------------------------------------------------
    order = np.argsort(-proba)[:N_EXPORTED]
    exported = explainer.explain(
        X.iloc[order],
        [test_ids.iloc[i] for i in order],
        bands=["Critical"] * len(order),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": explainer.model_name,
        "threshold": explainer.threshold,
        "additivity_check": additivity,
        "global_importance": {k: float(v) for k, v in importance.items()},
        "direction_check": explainer.signed_effect(X).to_dict("records"),
        "explanations": [e.to_dict() for e in exported],
    }
    path = args.out / "explanations.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  top {N_EXPORTED} explanations -> {path.resolve()}")


if __name__ == "__main__":
    main()
