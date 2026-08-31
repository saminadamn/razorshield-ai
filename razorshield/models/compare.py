"""Train and compare four models, then select one on the business objective.

    python -m razorshield.models.compare --data data/raw --out reports

Protocol: fit on train, decide everything (early stopping, threshold, model
selection) on validation, and touch test exactly once for the reported
numbers.

Imbalance is handled at the threshold, not by reweighting the training data.
Class weights and `scale_pos_weight` distort predicted probabilities, and
Phase 4 turns those probabilities into a 0-100 risk score -- a model that
ranks well but is badly calibrated would produce a meaningless score. All four
models are therefore trained unweighted and compared on PR-AUC, which is the
metric that respects a 3% base rate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .cost import CostModel, choose_threshold, sensitivity, threshold_at_budget
from .dataset import load, make_preprocessor, temporal_split

RANDOM_STATE = 0
ALERT_BUDGET = 0.01  # capacity view: 1% of transactions can be reviewed


def build_models() -> dict[str, dict]:
    """Model zoo. `scale` picks which preprocessing variant it is fed."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            "scale": True,
            "early_stopping": False,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "scale": False,
            "early_stopping": False,
        },
        "xgboost": {
            "estimator": XGBClassifier(
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,
                eval_metric="aucpr",
                early_stopping_rounds=100,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "scale": False,
            "early_stopping": "xgboost",
        },
        "lightgbm": {
            "estimator": lgb.LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                min_child_samples=20,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            "scale": False,
            "early_stopping": "lightgbm",
        },
    }


def fit_model(spec: dict, matrices: dict, y_train, y_val):
    """Fit one estimator, using the validation fold for early stopping."""
    key = "scaled" if spec["scale"] else "raw"
    X_train, X_val = matrices[key]["train"], matrices[key]["validation"]
    model = spec["estimator"]

    if spec["early_stopping"] == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif spec["early_stopping"] == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def at_threshold(y_true, y_prob, amount, threshold, cost: CostModel) -> dict:
    flagged = y_prob >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, flagged, average="binary", zero_division=0
    )
    result = cost.evaluate(y_true, flagged, amount)
    result.update({"threshold": float(threshold), "precision": precision,
                   "recall": recall, "f1": f1})
    return result


# Deliberately uneven: risk bands live in the top few percent of scores, so
# that is where calibration has to be checked at resolution.
CALIBRATION_QUANTILES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999, 1.0)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    """Predicted vs observed fraud rate. Phase 4's risk score depends on this."""
    edges = np.unique(np.quantile(y_prob, CALIBRATION_QUANTILES))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        last = i == len(edges) - 2
        mask = (y_prob >= lo) & (y_prob <= hi if last else y_prob < hi)
        if mask.sum() < 30:
            continue
        rows.append({
            "bucket": f"{lo:.4f}-{hi:.4f}",
            "n": int(mask.sum()),
            "predicted": float(y_prob[mask].mean()),
            "observed": float(y_true[mask].mean()),
        })
    return rows


def archetype_recall(meta: pd.DataFrame, flagged: np.ndarray) -> dict[str, float]:
    archetypes = meta["fraud_archetype"].to_numpy()
    out = {}
    for arch in np.unique(archetypes):
        if arch == "none":
            continue
        mask = archetypes == arch
        out[str(arch)] = float(flagged[mask].mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fraud models.")
    parser.add_argument("--data", default="data/raw", type=Path)
    parser.add_argument("--out", default="reports", type=Path)
    parser.add_argument("--models", default="models", type=Path)
    args = parser.parse_args()

    frame, meta = load(args.data)
    train, val, test = temporal_split(frame, meta)
    cost = CostModel()

    print("=" * 74)
    print("SPLITS  (chronological)")
    print("=" * 74)
    for split in (train, val, test):
        print(f"  {split.name:<12} {len(split):>7,} rows   "
              f"fraud {split.fraud_rate:.4f}   "
              f"{frame.timestamp.iloc[0]:%Y-%m-%d} window")
    print(f"\ncost model: missed fraud = amount + Rs{cost.chargeback_fee:.0f}, "
          f"review = Rs{cost.review_cost:.0f}, "
          f"false decline = Rs{cost.false_decline_cost:.0f}")

    # Preprocessors are fit on train only.
    matrices: dict[str, dict[str, np.ndarray]] = {}
    feature_names: dict[str, list[str]] = {}
    preprocessors = {}
    for key, scale in (("scaled", True), ("raw", False)):
        pre = make_preprocessor(scale=scale).fit(train.X)
        preprocessors[key] = pre
        names = [str(c) for c in pre.get_feature_names_out()]
        feature_names[key] = names
        # Named frames rather than bare arrays: LightGBM otherwise invents
        # "Column_0" names at fit time and sklearn warns on every predict.
        matrices[key] = {
            s.name: pd.DataFrame(pre.transform(s.X), columns=names)
            for s in (train, val, test)
        }

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    print()
    print("=" * 74)
    print("TRAINING")
    print("=" * 74)
    for name, spec in build_models().items():
        t0 = time.time()
        model = fit_model(spec, matrices, train.y, val.y)
        key = "scaled" if spec["scale"] else "raw"
        p_val = model.predict_proba(matrices[key]["validation"])[:, 1]
        p_test = model.predict_proba(matrices[key]["test"])[:, 1]

        # Threshold is chosen on validation, then frozen.
        t_cost, val_cost = choose_threshold(
            val.y.to_numpy(), p_val, val.amount.to_numpy(), cost
        )
        t_budget = threshold_at_budget(p_val, ALERT_BUDGET)

        results[name] = {
            "fit_seconds": round(time.time() - t0, 1),
            "n_estimators_used": int(
                getattr(model, "best_iteration", None)
                or getattr(model, "best_iteration_", None)
                or getattr(model, "n_estimators", 0)
            ),
            "validation": {
                "pr_auc": float(average_precision_score(val.y, p_val)),
                "chosen_threshold": t_cost,
                "saving_pct": val_cost["saving_pct"],
            },
            "test": {
                "pr_auc": float(average_precision_score(test.y, p_test)),
                "roc_auc": float(roc_auc_score(test.y, p_test)),
                "brier": float(brier_score_loss(test.y, p_test)),
                "at_cost_threshold": at_threshold(
                    test.y.to_numpy(), p_test, test.amount.to_numpy(), t_cost, cost
                ),
                "at_alert_budget": at_threshold(
                    test.y.to_numpy(), p_test, test.amount.to_numpy(), t_budget, cost
                ),
            },
        }
        fitted[name] = (model, key, p_val, p_test)
        trees = results[name]["n_estimators_used"]
        print(f"  {name:<22} {results[name]['fit_seconds']:>6.1f}s   "
              f"val PR-AUC {results[name]['validation']['pr_auc']:.4f}   "
              f"{f'trees {trees}' if trees else ''}")

    # --- selection: validation only ---------------------------------------
    selected = max(
        results, key=lambda n: results[n]["validation"]["saving_pct"]
    )
    best_by_prauc = max(results, key=lambda n: results[n]["validation"]["pr_auc"])

    print()
    print("=" * 74)
    print("TEST SET  (held out by time, touched once)")
    print("=" * 74)
    header = (
        f"  {'model':<22}{'PR-AUC':>8}{'ROC-AUC':>9}"
        f"{'Brier':>9}{'Prec':>8}{'Recall':>8}{'F1':>8}"
    )
    print(header)
    for name, r in results.items():
        op = r["test"]["at_cost_threshold"]
        mark = " *" if name == selected else "  "
        print(f"{mark}{name:<22}{r['test']['pr_auc']:>8.4f}{r['test']['roc_auc']:>9.4f}"
              f"{r['test']['brier']:>9.5f}{op['precision']:>8.3f}{op['recall']:>8.3f}"
              f"{op['f1']:>8.3f}")
    print("\n  (* selected. Precision/recall are at the cost-optimal threshold,")
    print("   which is chosen on validation and differs per model.)")

    print()
    print("=" * 74)
    print("BUSINESS OUTCOME ON TEST  (what the decision actually costs)")
    print("=" * 74)
    print(f"  {'model':<22}{'alerts':>8}{'caught':>8}{'missed':>8}"
          f"{'net saving':>13}{'vs no model':>13}")
    for name, r in results.items():
        op = r["test"]["at_cost_threshold"]
        mark = " *" if name == selected else "  "
        print(f"{mark}{name:<22}{op['alerts']:>8,}{op['true_positives']:>8,}"
              f"{op['false_negatives']:>8,}{op['net_saving']:>13,.0f}"
              f"{op['saving_pct']:>12.1%}")
    baseline = results[selected]["test"]["at_cost_threshold"]["baseline_cost"]
    print(f"\n  doing nothing costs Rs{baseline:,.0f} on the test window")
    if cost.review_catch_rate >= 1.0:
        print("  note: assumes review stops 100% of flagged fraud. Lower "
              "review_catch_rate\n        in CostModel for a more conservative "
              "number -- this is the single\n        biggest assumption behind "
              "the saving figures above.")

    print()
    print("=" * 74)
    print(f"CAPACITY VIEW  (fixed {ALERT_BUDGET:.0%} alert budget)")
    print("=" * 74)
    print(f"  {'model':<22}{'Prec':>8}{'Recall':>8}{'F1':>8}{'saving':>12}")
    for name, r in results.items():
        op = r["test"]["at_alert_budget"]
        print(f"  {name:<22}{op['precision']:>8.3f}{op['recall']:>8.3f}"
              f"{op['f1']:>8.3f}{op['saving_pct']:>11.1%}")

    # --- detail on the selected model -------------------------------------
    model, key, p_val, p_test = fitted[selected]
    flagged = p_test >= results[selected]["validation"]["chosen_threshold"]
    per_arch = archetype_recall(test.meta, flagged)
    calib = calibration_table(test.y.to_numpy(), p_test)
    sens = sensitivity(
        val.y.to_numpy(), p_val, val.amount.to_numpy(), cost
    )

    print()
    print("=" * 74)
    print(f"SELECTED: {selected}")
    print("=" * 74)
    if selected != best_by_prauc:
        print(f"  note: {best_by_prauc} has the better validation PR-AUC, but "
              f"{selected}\n        wins on expected cost. Selection follows the "
              "business objective.")
    print("\n  recall by fraud archetype (test, at the chosen threshold)")
    for arch, rec in sorted(per_arch.items(), key=lambda kv: -kv[1]):
        label = f"{arch} (label 0)" if arch == "hard_negative" else arch
        print(f"    {label:<26} {rec:.3f}")

    print("\n  calibration (predicted vs observed fraud rate)")
    for row in calib:
        print(f"    {row['bucket']:<20} n={row['n']:>6,}  "
              f"predicted {row['predicted']:.4f}  observed {row['observed']:.4f}")

    print("\n  sensitivity: as a false decline gets more expensive")
    print(f"    {'FP cost':>10}{'threshold':>12}{'alert rate':>13}{'recall':>9}{'saving':>10}")
    for row in sens:
        print(f"    {row['false_decline_cost']:>10,.0f}{row['threshold']:>12.4f}"
              f"{row['alert_rate']:>13.4f}{row['recall']:>9.3f}{row['saving_pct']:>9.1%}")

    # --- persist -----------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    report = {
        "selected_model": selected,
        "best_by_validation_pr_auc": best_by_prauc,
        "alert_budget": ALERT_BUDGET,
        "cost_model": {
            "chargeback_fee": cost.chargeback_fee,
            "review_cost": cost.review_cost,
            "false_decline_cost": cost.false_decline_cost,
            "review_catch_rate": cost.review_catch_rate,
        },
        "splits": {
            s.name: {"rows": len(s), "fraud_rate": s.fraud_rate}
            for s in (train, val, test)
        },
        "models": results,
        "selected_detail": {
            "archetype_recall": per_arch,
            "calibration": calib,
            "cost_sensitivity": sens,
        },
    }
    (args.out / "model_comparison.json").write_text(json.dumps(report, indent=2))

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessors[key],
            "feature_names": feature_names[key],
            "threshold": results[selected]["validation"]["chosen_threshold"],
            "model_name": selected,
        },
        args.models / "razorshield_model.joblib",
    )
    print(f"\n  report -> {(args.out / 'model_comparison.json').resolve()}")
    print(f"  model  -> {(args.models / 'razorshield_model.joblib').resolve()}")


if __name__ == "__main__":
    main()
