from __future__ import annotations
import pandas as pd
from pathlib import Path


def load_telco_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads the Telco churn CSV exactly as-is.
    No cleaning here (raw stays raw).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df
