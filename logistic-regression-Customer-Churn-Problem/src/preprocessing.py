from __future__ import annotations
import pandas as pd


def make_xy(df: pd.DataFrame, target: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    X = df.drop(columns=[target, "customerID"]).copy()
    y = df[target].copy()
    return X, y


def clean_telco(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    - Coerce TotalCharges to numeric (it contains blanks)
    - Drop rows where TotalCharges is missing after coercion
    - Encode y: Yes->1, No->0
    """
    X = X.copy()
    y = y.copy()

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        mask = X["TotalCharges"].notna()
        X = X.loc[mask] # here we're filtering to only keep rows were TotalCharges is TRUE
        y = y.loc[mask] # here also, we're filtering to only keep rows were TotalCharges is TRUE

    # Encode target
    y = y.map({"Yes": 1, "No": 0}) # Class 1 = customer churns (positif class) and Class 0 = customer stays (negatif class)
    if y.isna().any():
        bad = y[y.isna()].unique()
        raise ValueError(f"Unexpected target values after mapping. Got: {bad}")

    return X, y


def one_hot_encode(X: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encodes all object columns.
    """
    X = X.copy()
    categorical_cols = X.select_dtypes(include="object").columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=drop_first)
    return X_encoded
