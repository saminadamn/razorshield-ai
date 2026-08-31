"""RiskScorer: the object Phase 5 and Phase 6 actually consume.

    python -m razorshield.scoring.score --data data/raw --out reports

Fits the calibrator and the band boundaries on validation, reports what the
bands contain on test, and persists everything needed to score a live payment
event: model, preprocessor, calibrator, boundaries, and the SHAP explainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..explain.explainer import Explainer
from ..models.cost import CostModel, choose_threshold
from ..models.dataset import load, temporal_split
from .bands import RiskScale, band_stats, fit_scale
from .calibrate import calibration_metrics, fit_calibrator

BAR_WIDTH = 20


class RiskScorer:
    """Feature row in, risk score and explanation out."""

    def __init__(self, explainer: Explainer, calibrator, scale: RiskScale):
        self.explainer = explainer
        self.calibrator = calibrator
        self.scale = scale

    @classmethod
    def load(cls, bundle_path: Path, scorer_path: Path) -> RiskScorer:
        state = joblib.load(scorer_path)
        return cls(
            explainer=Explainer(bundle_path),
            calibrator=state["calibrator"],
            scale=RiskScale(**state["scale"]),
        )

    def save(self, path: Path) -> None:
        joblib.dump(
            {"calibrator": self.calibrator, "scale": self.scale.to_dict()}, path
        )

    def probabilities(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = self.explainer.model.predict_proba(self.explainer.encode(rows))[:, 1]
        # Isotonic maps a pure top bin to exactly 1.0. Showing an analyst
        # "fraud probability 1.0000" claims certainty the model does not have,
        # so the displayed probability is clamped short of both endpoints.
        calibrated = np.clip(self.calibrator.predict(raw), 1e-6, 0.999)
        return raw, calibrated

    def assess(self, rows: pd.DataFrame) -> pd.DataFrame:
        raw, calibrated = self.probabilities(rows)
        score = self.scale.score(calibrated)
        return pd.DataFrame({
            "raw_probability": raw,
            "probability": calibrated,
            "risk_score": score,
            "band": self.scale.band(score),
        })

    def assess_one(self, rows: pd.DataFrame, transaction_id: str) -> dict:
        """Full payload for one transaction: score, band, factors, summary."""
        assessment = self.assess(rows).iloc[0]
        band = str(assessment["band"])
        explanation = self.explainer.explain(
            rows, [transaction_id], bands=[band.capitalize()]
        )[0]
        payload = explanation.to_dict()
        payload.update({
            "probability": float(assessment["probability"]),
            "raw_probability": float(assessment["raw_probability"]),
            "risk_score": float(assessment["risk_score"]),
            "band": band,
        })
        return payload


def render(payload: dict, row: pd.Series) -> str:
    """Reference rendering -- the shape the dashboard should show."""
    score = payload["risk_score"]
    filled = int(round(score / 100 * BAR_WIDTH))
    lines = [
        f"  {payload['transaction_id']}",
        "",
        f"    Amount              Rs{row['amount']:,.2f}",
        f"    Payment method      {row['payment_method']}",
        f"    Transaction hour    {int(row['transaction_hour']):02d}:00",
        f"    Device              {row['device_type']}, {int(row['device_age_days'])}d old",
        "",
        f"    Fraud probability   {payload['probability']:.4f}",
        f"    Risk score          {score:.0f} / 100",
        f"    Risk level          {payload['band']}",
        f"    [{'#' * filled}{'.' * (BAR_WIDTH - filled)}]",
        "",
        "    Top risk factors",
    ]
    for factor in payload["factors"][:4]:
        if factor["contribution"] <= 0:
            continue
        width = max(1, int(round(factor["share"] * 12)))
        lines.append(f"      {factor['label']:<26} {'#' * width}")
    lines += ["", f"    {payload['summary']}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and report the risk scale.")
    parser.add_argument("--data", default="data/raw", type=Path)
    parser.add_argument("--out", default="reports", type=Path)
    parser.add_argument("--models", default="models", type=Path)
    parser.add_argument(
        "--bundle", default="models/razorshield_model.joblib", type=Path
    )
    args = parser.parse_args()

    frame, meta = load(args.data)
    _, val, test = temporal_split(frame, meta)
    explainer = Explainer(args.bundle)
    cost = CostModel()

    raw_val = explainer.model.predict_proba(explainer.encode(val.X))[:, 1]
    raw_test = explainer.model.predict_proba(explainer.encode(test.X))[:, 1]
    y_val, y_test = val.y.to_numpy(), test.y.to_numpy()

    # --- calibration -------------------------------------------------------
    calibrator = fit_calibrator(raw_val, y_val)
    cal_val = calibrator.predict(raw_val)
    cal_test = calibrator.predict(raw_test)

    print("=" * 74)
    print(f"CALIBRATION  ({explainer.model_name}, fitted on validation)")
    print("=" * 74)
    before = calibration_metrics(y_test, raw_test)
    after = calibration_metrics(y_test, cal_test)
    print(f"  {'metric':<22}{'raw':>12}{'calibrated':>14}")
    for key in ("brier", "ece", "max_bin_deviation", "mean_predicted"):
        print(f"  {key:<22}{before[key]:>12.5f}{after[key]:>14.5f}")
    print(f"  {'observed rate':<22}{before['observed_rate']:>12.5f}")
    print("\n  Isotonic regression is monotone, so PR-AUC and ROC-AUC are")
    print("  unchanged by construction -- only the numbers' meaning improves.")
    if after["brier"] > before["brier"]:
        print("  Note: Brier is marginally worse while ECE improves. The step")
        print("  function adds a little noise but removes the systematic")
        print("  under-confidence; ECE is the metric that matters for a score")
        print("  that has to mean what it says.")

    # --- bands -------------------------------------------------------------
    # The action boundary is re-derived on calibrated probabilities. Isotonic
    # preserves order, so the decision set is the same; only the cut value moves.
    threshold, _ = choose_threshold(y_val, cal_val, val.amount.to_numpy(), cost)
    scale = fit_scale(cal_val, y_val, threshold)

    print()
    print("=" * 74)
    print("BAND BOUNDARIES  (fitted on validation)")
    print("=" * 74)
    print(f"  MEDIUM   starts at p = {scale.p_medium:.4f}   "
          "band is >= 3x the base rate     -> step-up auth")
    print(f"  HIGH     starts at p = {scale.p_high:.4f}   "
          "cost-optimal action threshold   -> review")
    print(f"  CRITICAL starts at p = {scale.p_critical:.4f}   "
          "tail above is >=90% fraud       -> block")

    for name, p_probe, y_probe, amount in (
        ("VALIDATION", cal_val, y_val, val.amount.to_numpy()),
        ("TEST (held out)", cal_test, y_test, test.amount.to_numpy()),
    ):
        stats = band_stats(p_probe, y_probe, scale, amount)
        print()
        print("=" * 74)
        print(f"BANDS ON {name}")
        print("=" * 74)
        print(f"  {'band':<10}{'score':>10}{'count':>9}{'share':>9}"
              f"{'fraud rate':>12}{'lift':>8}{'recall':>9}")
        for row in stats:
            print(f"  {row['band']:<10}{row['score_range']:>10}{row['n']:>9,}"
                  f"{row['share']:>9.2%}{row['fraud_rate']:>12.3f}"
                  f"{row['lift']:>8.1f}{row['recall']:>9.3f}")
        if name.startswith("TEST"):
            test_stats = stats

    critical = next(r for r in test_stats if r["band"] == "CRITICAL")
    high = next(r for r in test_stats if r["band"] == "HIGH")
    print()
    print(f"  CRITICAL + HIGH is {critical['share'] + high['share']:.1%} of traffic "
          f"and captures {critical['recall'] + high['recall']:.1%} of fraud,")
    print(f"  covering Rs{critical['amount_at_risk'] + high['amount_at_risk']:,.0f} "
          "of the amount at risk.")

    # --- worked example ----------------------------------------------------
    scorer = RiskScorer(explainer, calibrator, scale)
    test_ids = frame.iloc[-len(test):]["transaction_id"].reset_index(drop=True)
    assessments = scorer.assess(test.X)
    idx = int(assessments["risk_score"].idxmax())

    print()
    print("=" * 74)
    print("WORKED EXAMPLE  (highest-scoring transaction on test)")
    print("=" * 74)
    payload = scorer.assess_one(test.X.iloc[[idx]], test_ids.iloc[idx])
    print(render(payload, test.X.iloc[idx]))
    print(f"\n    actual label: {'FRAUD' if test.y.iloc[idx] else 'legitimate'}")

    # --- persist -----------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    scorer.save(args.models / "risk_scorer.joblib")

    report = {
        "model": explainer.model_name,
        "calibration": {"raw": before, "calibrated": after},
        "scale": scale.to_dict(),
        "action_threshold": threshold,
        "bands_validation": band_stats(cal_val, y_val, scale, val.amount.to_numpy()),
        "bands_test": test_stats,
        "band_policy": {
            "LOW": "allow",
            "MEDIUM": "step-up authentication",
            "HIGH": "queue for manual review",
            "CRITICAL": "block before capture",
        },
    }
    (args.out / "risk_bands.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  report -> {(args.out / 'risk_bands.json').resolve()}")
    print(f"  scorer -> {(args.models / 'risk_scorer.joblib').resolve()}")


if __name__ == "__main__":
    main()
