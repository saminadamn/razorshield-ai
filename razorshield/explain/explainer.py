"""TreeSHAP explanations, aggregated back to the features a human recognises.

The model sees 31 columns because `payment_method` was one-hot encoded into
five. Nobody wants to read "cat__payment_method_UPI +0.03" in an alert queue,
so contributions are summed back onto the original 18 features before anything
is shown. Summing is the right operation: SHAP values are additive, so the
contributions of the one-hot columns of a single categorical add up to that
categorical's total contribution.

Contributions are in **log-odds**, the space where SHAP is additive:

    base_value + sum(contributions) = log-odds of the predicted probability
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from ..data.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from .narrative import label, summarise


@dataclass
class Factor:
    """One feature's signed contribution to a single prediction."""

    feature: str
    label: str
    value: object
    contribution: float
    share: float  # |contribution| as a fraction of the largest, for bar widths

    @property
    def direction(self) -> str:
        return "increases" if self.contribution > 0 else "decreases"


@dataclass
class Explanation:
    transaction_id: str
    probability: float
    base_probability: float
    factors: list[Factor] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "probability": self.probability,
            "base_probability": self.base_probability,
            "summary": self.summary,
            "factors": [asdict(f) for f in self.factors],
        }


def _native(value):
    """numpy scalars are not JSON-serialisable; the API returns these directly."""
    return value.item() if hasattr(value, "item") else value


def _canonical_feature(encoded_name: str) -> str:
    """Map an encoded column back to the feature it came from.

    `num__amount` -> `amount`; `cat__day_of_week_Friday` -> `day_of_week`.
    Categorical names contain underscores, so match known columns explicitly
    rather than splitting on the separator.
    """
    if encoded_name.startswith("num__"):
        return encoded_name[len("num__"):]
    if encoded_name.startswith("cat__"):
        rest = encoded_name[len("cat__"):]
        for col in CATEGORICAL_COLUMNS:
            if rest.startswith(f"{col}_"):
                return col
    return encoded_name


class Explainer:
    """Wraps the persisted model bundle with TreeSHAP."""

    def __init__(self, bundle_path: Path):
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.encoded_names: list[str] = bundle["feature_names"]
        self.threshold: float = bundle["threshold"]
        self.model_name: str = bundle["model_name"]
        self.explainer = shap.TreeExplainer(self.model)
        self.groups = [_canonical_feature(n) for n in self.encoded_names]
        # Column order for aggregated output, following the dataset schema.
        self.features = [f for f in FEATURE_COLUMNS if f in set(self.groups)]

    # -- core ---------------------------------------------------------------

    def encode(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Preprocess into the named frame the model was fitted on."""
        return pd.DataFrame(
            self.preprocessor.transform(rows[FEATURE_COLUMNS]),
            columns=self.encoded_names,
        )

    def shap_matrix(self, rows: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Aggregated (n, n_features) log-odds contributions, plus the base value."""
        X = self.encode(rows)
        values = self.explainer.shap_values(X)
        if isinstance(values, list):  # older shap returns one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]

        base = self.explainer.expected_value
        base = float(np.asarray(base).ravel()[-1])

        # Sum encoded columns onto their source feature.
        agg = np.zeros((values.shape[0], len(self.features)))
        index = {f: i for i, f in enumerate(self.features)}
        for j, group in enumerate(self.groups):
            agg[:, index[group]] += values[:, j]
        return agg, base

    def check_additivity(self, rows: pd.DataFrame, tol: float = 1e-3) -> dict:
        """base + sum(contributions) must equal the model's log-odds output."""
        agg, base = self.shap_matrix(rows)
        proba = self.model.predict_proba(self.encode(rows))[:, 1]
        margin = np.log(np.clip(proba, 1e-12, 1 - 1e-12) / np.clip(1 - proba, 1e-12, 1))
        reconstructed = base + agg.sum(axis=1)
        error = np.abs(reconstructed - margin)
        return {
            "max_error": float(error.max()),
            "mean_error": float(error.mean()),
            "passed": bool(error.max() < tol),
        }

    def global_importance(self, rows: pd.DataFrame) -> pd.Series:
        """Mean |contribution| per feature -- what the model relies on overall."""
        agg, _ = self.shap_matrix(rows)
        return pd.Series(
            np.abs(agg).mean(axis=0), index=self.features
        ).sort_values(ascending=False)

    def signed_effect(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Does each feature push risk up when its value is high?

        Spearman-style check between feature value and its own contribution.
        A sign that contradicts domain intuition is worth investigating before
        anyone trusts the explanations.
        """
        agg, _ = self.shap_matrix(rows)
        out = []
        for i, feature in enumerate(self.features):
            if feature in CATEGORICAL_COLUMNS:
                continue
            values = rows[feature].to_numpy(dtype=float)
            contribution = agg[:, i]
            if values.std() == 0 or contribution.std() == 0:
                corr = float("nan")
            else:
                corr = float(
                    pd.Series(values).corr(pd.Series(contribution), method="spearman")
                )
            out.append({
                "feature": feature,
                "mean_abs_contribution": float(np.abs(contribution).mean()),
                "value_vs_contribution": corr,
            })
        return pd.DataFrame(out).sort_values(
            "mean_abs_contribution", ascending=False
        )

    # -- per transaction ----------------------------------------------------

    def explain(
        self,
        rows: pd.DataFrame,
        transaction_ids: list[str],
        bands: list[str] | None = None,
        top_k: int = 6,
    ) -> list[Explanation]:
        agg, base = self.shap_matrix(rows)
        proba = self.model.predict_proba(self.encode(rows))[:, 1]
        base_proba = float(1.0 / (1.0 + np.exp(-base)))

        out = []
        for i, txn_id in enumerate(transaction_ids):
            contributions = agg[i]
            order = np.argsort(-np.abs(contributions))[:top_k]
            largest = float(np.abs(contributions[order]).max()) or 1.0

            factors = [
                Factor(
                    feature=self.features[j],
                    label=label(self.features[j]),
                    value=_native(rows.iloc[i][self.features[j]]),
                    contribution=round(float(contributions[j]), 4),
                    share=round(abs(float(contributions[j])) / largest, 4),
                )
                for j in order
            ]
            band = bands[i] if bands else ("Elevated" if proba[i] >= self.threshold else "Low")
            out.append(
                Explanation(
                    transaction_id=txn_id,
                    probability=round(float(proba[i]), 6),
                    base_probability=round(base_proba, 6),
                    factors=factors,
                    summary=summarise(factors, band),
                )
            )
        return out
