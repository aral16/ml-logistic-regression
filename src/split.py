from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y # this line forces the train and test sets to have the same class proportions as the original dataset.
    )
