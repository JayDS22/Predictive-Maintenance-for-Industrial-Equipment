"""Load the turbofan dataset; generate it on demand if missing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RAW_DATA_CSV


def load_dataset(path: Path = RAW_DATA_CSV, units_if_missing: int = 100) -> pd.DataFrame:
    if not path.exists():
        from data.generate_synthetic_data import generate

        path.parent.mkdir(parents=True, exist_ok=True)
        df = generate(num_units=units_if_missing)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)
