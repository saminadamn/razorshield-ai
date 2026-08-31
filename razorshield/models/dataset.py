"""Loading, temporal splitting and preprocessing.

Splits are by time, never at random. A random split would let the model see a
customer's later transactions while predicting their earlier ones, and would
let merchant risk scores from the future leak backwards. Both inflate every
metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, TARGET

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


@dataclass
class Split:
    """One temporal fold: features, label, and the amount at risk per row."""

    X: pd.DataFrame
    y: pd.Series
    amount: pd.Series
    meta: pd.DataFrame
    name: str

    def __len__(self) -> int:
        return len(self.X)

    @property
    def fraud_rate(self) -> float:
        return float(self.y.mean())


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(data_dir / "transactions.csv", parse_dates=["timestamp"])
    meta = pd.read_csv(data_dir / "transactions_meta.csv")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("transactions.csv must be in chronological order")
    return frame, meta


def temporal_split(
    frame: pd.DataFrame,
    meta: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[Split, Split, Split]:
    """Chronological train / validation / test.

    Validation is where thresholds and early stopping are decided. Test is
    touched once, at the end, for the numbers that get reported.
    """
    n = len(frame)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    out = []
    for name, lo, hi in (
        ("train", 0, i_train),
        ("validation", i_train, i_val),
        ("test", i_val, n),
    ):
        part = frame.iloc[lo:hi]
        out.append(
            Split(
                X=part[FEATURE_COLUMNS].reset_index(drop=True),
                y=part[TARGET].reset_index(drop=True),
                amount=part["amount"].reset_index(drop=True),
                meta=meta.iloc[lo:hi].reset_index(drop=True),
                name=name,
            )
        )
    return tuple(out)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """One-hot the categoricals; scale numerics only for the linear model.

    All four models share this so the comparison is about the learner, not
    about who got the better encoding.
    """
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLUMNS),
    ])
