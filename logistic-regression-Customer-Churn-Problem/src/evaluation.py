from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=4)


def precision_recall_by_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has length n-1; precision/recall have length n
    df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
    })
    return df


def choose_threshold_for_recall_and_precision(
    pr_df: pd.DataFrame,
    min_recall: float,
    min_precision: float,
    strategy: str = "first",
) -> float:
    """
    Choose a threshold satisfying BOTH:
      recall >= min_recall AND precision >= min_precision

    strategy:
      - "first": take the first threshold meeting constraints
      - "max_f1": among valid thresholds, pick the one with max F1
    """
    valid = pr_df[(pr_df["recall"] >= min_recall) & (pr_df["precision"] >= min_precision)]

    if len(valid) == 0:
        # No threshold satisfies both constraints.
        # Return a reasonable fallback: threshold that maximizes F1 overall.
        df = pr_df.copy()
        denom = (df["precision"] + df["recall"]).replace(0, 1e-12)
        df["f1"] = 2 * (df["precision"] * df["recall"]) / denom
        return float(df.loc[df["f1"].idxmax(), "threshold"])

    if strategy == "max_f1":
        df = valid.copy()
        denom = (df["precision"] + df["recall"]).replace(0, 1e-12)
        df["f1"] = 2 * (df["precision"] * df["recall"]) / denom
        return float(df.loc[df["f1"].idxmax(), "threshold"])

    # Default: first valid threshold
    return float(valid.iloc[0]["threshold"])



def apply_threshold(y_proba, threshold: float):
    return (np.asarray(y_proba) >= threshold).astype(int)


def top_coefficients(model, feature_names, top_k: int = 15) -> pd.DataFrame:
    """
    For logistic regression: returns strongest coefficients by absolute value.
    """
    coefs = pd.Series(model.coef_[0], index=feature_names)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    out = coefs.head(top_k).to_frame(name="coef")
    out["abs_coef"] = out["coef"].abs()
    return out
