from __future__ import annotations

import argparse
from pathlib import Path
import joblib

from src.data_loader import load_telco_csv
from src.preprocessing import make_xy, clean_telco, one_hot_encode
from src.split import stratified_split
from src.models import make_baseline, make_logistic_regression
from src.evaluation import (
    report, precision_recall_by_threshold,
    choose_threshold_for_recall_and_precision,
    apply_threshold, top_coefficients
)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to raw Telco CSV")
    parser.add_argument("--target", type=str, default="Churn")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-precision", type=float, default=0.40)
    parser.add_argument("--threshold-strategy", type=str, default="first", choices=["first", "max_f1"])
    parser.add_argument("--outdir", type=str, default="models")
    args = parser.parse_args()

    df = load_telco_csv(args.csv)
    X, y = make_xy(df, target=args.target)
    X, y = clean_telco(X, y)
    X = one_hot_encode(X, drop_first=True)

    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Baseline
    baseline = make_baseline()
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)
    print("\n=== Baseline (most_frequent) ===")
    print(report(y_test, y_base))

    # Logistic Regression
    model = make_logistic_regression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Logistic Regression @ threshold=0.5 ===")
    print(report(y_test, y_pred))

    # Threshold tuning
    y_proba = model.predict_proba(X_test)[:, 1]
    pr_df = precision_recall_by_threshold(y_test, y_proba)

    thr = choose_threshold_for_recall_and_precision(
        pr_df,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        strategy=args.threshold_strategy,
    )


    y_custom = apply_threshold(y_proba, thr) # Here i'm using the threshold obtain from thr
    print(
    f"\n=== Logistic Regression @ chosen_threshold={thr:.4f} "
    f"(min_recall={args.min_recall}, min_precision={args.min_precision}, strategy={args.threshold_strategy}) ==="
)

    print(report(y_test, y_custom))

    # Coefficients
    print("\n=== Top coefficients (by |coef|) ===")
    print(top_coefficients(model, X_train.columns, top_k=15))

    # Save artifacts
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, outdir / "logreg_model.joblib")
    joblib.dump(
    {
    "threshold": thr,
    "min_recall": args.min_recall,
    "min_precision": args.min_precision,
    "threshold_strategy": args.threshold_strategy,
    "feature_columns": list(X.columns),
    },
        outdir / "logreg_metadata.joblib"
    )
    print(f"\nSaved model + metadata to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
