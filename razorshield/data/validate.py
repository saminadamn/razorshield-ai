"""Quality gate for a generated dataset.

A synthetic fraud dataset is only useful if it is *hard*. If a single feature
separates the classes, or a quick model scores near-perfect PR-AUC, the model
comparison in the next phase measures nothing. This script fails loudly in
those cases.

    python -m razorshield.data.validate --data data/raw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, TARGET

# A single feature scoring above this is a giveaway, not a signal.
MAX_SINGLE_FEATURE_AUC = 0.95
# A quick model above this means the generating process is too clean.
MAX_MODEL_PR_AUC = 0.95
# Below this and there is no learnable signal at all.
MIN_MODEL_PR_AUC = 0.15

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def temporal_split(frame: pd.DataFrame, train_frac: float = 0.7):
    """Split by time, never at random -- fraud has strong temporal structure."""
    cut = int(len(frame) * train_frac)
    return frame.iloc[:cut], frame.iloc[cut:]


def univariate_auc(frame: pd.DataFrame) -> pd.Series:
    y = frame[TARGET].to_numpy()
    scores = {}
    for col in NUMERIC_COLUMNS:
        x = frame[col].to_numpy(dtype=float)
        auc = roc_auc_score(y, x)
        # A feature that is strongly *negatively* predictive is equally telling.
        scores[col] = max(auc, 1 - auc)
    return pd.Series(scores).sort_values(ascending=False)


def _model_pipeline(kind: str):
    if kind == "logistic":
        pre = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ])
        return make_pipeline(
            pre, LogisticRegression(max_iter=2000, class_weight="balanced")
        )
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])
    return make_pipeline(
        pre,
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0),
    )


def quick_benchmark(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Rough difficulty probe. Not the model comparison -- that comes later."""
    rows = []
    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()
    for kind in ("logistic", "gradient_boosting"):
        model = _model_pipeline(kind)
        model.fit(train[FEATURE_COLUMNS], y_train)
        proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        # Threshold at the top 3% of scores, matching the base rate.
        cut = np.quantile(proba, 1 - y_test.mean())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, proba >= cut, average="binary", zero_division=0
        )
        rows.append({
            "model": kind,
            "pr_auc": average_precision_score(y_test, proba),
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return pd.DataFrame(rows)


def archetype_recall(
    test: pd.DataFrame, meta: pd.DataFrame, proba: np.ndarray, rate: float
) -> pd.Series:
    """Recall per fraud archetype at a fixed alert budget."""
    cut = np.quantile(proba, 1 - rate)
    flagged = proba >= cut
    joined = meta.set_index("transaction_id").loc[test["transaction_id"]]
    archetypes = joined["fraud_archetype"].to_numpy()
    out = {}
    for arch in np.unique(archetypes):
        if arch == "none":
            continue
        mask = archetypes == arch
        out[arch] = float(flagged[mask].mean())
    return pd.Series(out).sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated dataset.")
    parser.add_argument("--data", default="data/raw", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.data / "transactions.csv", parse_dates=["timestamp"])
    meta = pd.read_csv(args.data / "transactions_meta.csv")
    manifest = json.loads((args.data / "manifest.json").read_text())

    problems: list[str] = []

    print("=" * 68)
    print("SHAPE AND BALANCE")
    print("=" * 68)
    print(f"rows                {len(frame):,}")
    print(f"columns             {len(frame.columns)}")
    print(f"fraud rate          {frame[TARGET].mean():.4f}  ({int(frame[TARGET].sum()):,} rows)")
    print(f"window              {frame.timestamp.min()} -> {frame.timestamp.max()}")
    print(f"customers           {frame.customer_id.nunique():,}")
    print(f"merchants           {frame.merchant_id.nunique():,}")
    print(f"label flips         {manifest['labels_flipped']:,}")

    nulls = frame.isna().sum()
    if nulls.any():
        problems.append(f"null values present: {nulls[nulls > 0].to_dict()}")
    dupes = int(frame["transaction_id"].duplicated().sum())
    if dupes:
        problems.append(f"{dupes} duplicate transaction_ids")

    print()
    print("fraud rows by archetype")
    archetypes = meta["fraud_archetype"]
    counts = archetypes[~archetypes.isin(["none", "hard_negative"])].value_counts()
    total_fraud = max(int(counts.sum()), 1)
    for arch, k in counts.items():
        print(f"  {arch:<20} {k:>6,}  ({k / total_fraud:>5.1%} of fraud rows)")
    # Hard negatives are labelled 0 -- they are false-positive pressure, not fraud.
    print(f"  {'hard_negative (label 0)':<20} {int((archetypes == 'hard_negative').sum()):>6,}")

    print()
    print("=" * 68)
    print("UNIVARIATE SEPARATION  (single-feature ROC-AUC)")
    print("=" * 68)
    aucs = univariate_auc(frame)
    for col, auc in aucs.items():
        flag = "  <-- too strong" if auc > MAX_SINGLE_FEATURE_AUC else ""
        print(f"  {col:<28} {auc:.4f}{flag}")
    if (aucs > MAX_SINGLE_FEATURE_AUC).any():
        leaks = aucs[aucs > MAX_SINGLE_FEATURE_AUC].index.tolist()
        problems.append(f"single features separate the classes: {leaks}")

    print()
    print("=" * 68)
    print("DIFFICULTY PROBE  (temporal split, 70/30)")
    print("=" * 68)
    train, test = temporal_split(frame)
    print(f"train {len(train):,} rows  fraud {train[TARGET].mean():.4f}")
    print(f"test  {len(test):,} rows  fraud {test[TARGET].mean():.4f}")
    print()
    bench = quick_benchmark(train, test)
    print(bench.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = bench["pr_auc"].max()
    if best > MAX_MODEL_PR_AUC:
        problems.append(f"PR-AUC {best:.3f} is too high; the data is too easy")
    if best < MIN_MODEL_PR_AUC:
        problems.append(f"PR-AUC {best:.3f} is too low; there is no learnable signal")

    print()
    print("=" * 68)
    print("RECALL BY ARCHETYPE  (gradient boosting, 3% alert budget)")
    print("=" * 68)
    model = _model_pipeline("gradient_boosting")
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    per_arch = archetype_recall(test, meta, proba, 0.03)
    for arch, rec in per_arch.items():
        print(f"  {arch:<20} {rec:.3f}")

    # Hard negatives should be genuinely confusing: if none of them get flagged,
    # the false-positive pressure they were added for does not exist.
    hn = per_arch.get("hard_negative", 0.0)
    print()
    print(f"hard negatives flagged: {hn:.3f}  (should be clearly above the 3% budget)")
    if hn < 0.05:
        problems.append("hard negatives are not creating false-positive pressure")

    print()
    print("=" * 68)
    if problems:
        print("FAILED")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("PASSED -- dataset is imbalanced, non-trivial, and free of obvious leakage")


if __name__ == "__main__":
    main()
