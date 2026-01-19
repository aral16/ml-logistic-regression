# Customer Churn Prediction — Logistic Regression

## Problem
Predict whether a customer will churn in the next billing cycle to support retention
decisions (e.g. targeted incentives).

This is a **binary classification problem with asymmetric costs**:
- Missing a churner (false negative) → lost revenue
- Targeting a non-churner (false positive) → incentive cost

## Data
**Source:** Telco Customer Churn dataset  
**Target:** `Churn` (Yes / No)  
**Rows:** ~7,000 customers  
**Features:** Demographics, subscription, services, billing  
**Known issues:**
- Class imbalance (~26% churn)
- `TotalCharges` stored as text with missing values
- Mixed numeric and categorical features

## ML Task
Binary classification with **probabilistic output** and **decision thresholding**.

## Baseline
A `DummyClassifier` predicting the majority class (no churn) is used as a reference.

## Model
Logistic Regression with:
- One-hot encoded categorical variables
- Class weighting to address imbalance
- Interpretable coefficients
- Threshold selection based on **both recall and precision**

## Evaluation Strategy
Models are evaluated using:
- Precision / Recall per class
- Not accuracy alone
- Business-driven threshold selection

Threshold rule:
> Recall ≥ 70% **and** Precision ≥ 40% for churners

## Results Summary

### Baseline (Most Frequent)
- Accuracy: ~73%
- Recall (churn): 0%

The baseline never identifies churners and is therefore unusable.

### Logistic Regression (Threshold = 0.5)
- Recall (churn): ~80%
- Precision (churn): ~50%
- Accuracy: ~73%

This configuration significantly improves churn detection while keeping false positives reasonable.

### Logistic Regression (Business-Constrained Threshold)
- Chosen threshold: **0.255**
- Recall (churn): ~94%
- Precision (churn): ~40%
- Accuracy: ~61%

This threshold satisfies business constraints and prioritizes identifying churners
while controlling excessive false positives.

## Key Insights (Model Interpretation)
- Long-term contracts strongly reduce churn risk
- Support and security services are protective
- Fiber optic internet and electronic check payments increase churn risk
- Paperless billing and senior status correlate with higher churn probability

## Limitations
- No feature scaling (optimization warning present)
- No cross-validation
- Linear decision boundary
- No explicit cost-based optimization

## Deployment Notes
This model is suitable for:
- Ranking customers by churn risk
- Assisted decision-making for retention campaigns

Not suitable for:
- Fully automated decisions without cost calibration

## How to Run

```bash
python -m src.train \
  --csv data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --min-recall 0.70 \
  --min-precision 0.40 \
  --threshold-strategy first

