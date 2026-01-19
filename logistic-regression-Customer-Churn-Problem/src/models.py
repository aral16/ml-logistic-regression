from __future__ import annotations
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


def make_baseline():
    return DummyClassifier(strategy="most_frequent")


def make_logistic_regression(max_iter: int = 1000, class_weight: str | None = "balanced"):
    return LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
    )
