"""Per-unit feature engineering for run-to-failure sensor streams."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import (
    EMA_SPANS,
    FAILURE_THRESHOLD,
    LAG_STEPS,
    ROLLING_WINDOWS,
    SENSOR_COLUMNS,
)


def _rolling_stats(group: pd.DataFrame, cols: Iterable[str], windows: Iterable[int]) -> pd.DataFrame:
    out = {}
    for col in cols:
        series = group[col]
        for w in windows:
            roll = series.rolling(window=w, min_periods=1)
            out[f"{col}_rmean_{w}"] = roll.mean()
            out[f"{col}_rstd_{w}"] = roll.std().fillna(0.0)
    return pd.DataFrame(out, index=group.index)


def _lag_features(group: pd.DataFrame, cols: Iterable[str], steps: Iterable[int]) -> pd.DataFrame:
    out = {}
    for col in cols:
        for s in steps:
            out[f"{col}_lag_{s}"] = group[col].shift(s).bfill()
    return pd.DataFrame(out, index=group.index)


def _ema_features(group: pd.DataFrame, cols: Iterable[str], spans: Iterable[int]) -> pd.DataFrame:
    out = {}
    for col in cols:
        for span in spans:
            out[f"{col}_ema_{span}"] = group[col].ewm(span=span, adjust=False).mean()
    return pd.DataFrame(out, index=group.index)


def build_features(df: pd.DataFrame, sensor_cols: Iterable[str] = SENSOR_COLUMNS) -> pd.DataFrame:
    """Return df augmented with rolling, lag, EMA and cycle features.

    Requires `unit_id` and `cycle` columns. All transforms are grouped by
    `unit_id`; signals never cross machines.
    """
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    frames = [df.copy()]

    grouped = df.groupby("unit_id", group_keys=False)
    frames.append(grouped[list(sensor_cols)].apply(
        lambda g: _rolling_stats(g, sensor_cols, ROLLING_WINDOWS)
    ))
    frames.append(grouped[list(sensor_cols)].apply(
        lambda g: _lag_features(g, sensor_cols, LAG_STEPS)
    ))
    frames.append(grouped[list(sensor_cols)].apply(
        lambda g: _ema_features(g, sensor_cols, EMA_SPANS)
    ))

    feat = pd.concat(frames, axis=1)
    feat = feat.loc[:, ~feat.columns.duplicated()]

    # Absolute cycle counters only. Normalising by max-cycle leaks the answer:
    # at inference time the synthesised pseudo-history would set max-cycle to
    # "now", flipping every input to end-of-life.
    feat["cycles_since_start"] = feat["cycle"].astype(float)
    feat["cycles_since_start_sq"] = feat["cycles_since_start"] ** 2

    if "RUL" in feat.columns:
        feat["failure_within_threshold"] = (feat["RUL"] <= FAILURE_THRESHOLD).astype(int)

    return feat


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-ready column names (everything except identifiers and targets)."""
    exclude = {"unit_id", "cycle", "RUL", "failure_within_threshold"}
    return [c for c in df.columns if c not in exclude]
