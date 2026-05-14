from __future__ import annotations

import numpy as np

from data.generate_synthetic_data import generate
from src.config import SENSOR_COLUMNS
from src.feature_engineering import build_features, feature_columns


def test_synthetic_dataset_schema():
    df = generate(num_units=5, seed=0)
    expected = {"unit_id", "cycle", "RUL", *SENSOR_COLUMNS}
    assert expected.issubset(set(df.columns))
    assert df["RUL"].min() == 0
    assert (df.groupby("unit_id")["cycle"].max() > 100).all()


def test_build_features_no_leak_across_units():
    df = generate(num_units=4, seed=1)
    feats = build_features(df)
    cols = feature_columns(feats)
    assert len(cols) > len(SENSOR_COLUMNS)
    # At cycle 1, rolling mean over window=5 with min_periods=1 must equal the
    # raw reading; anything else implies look-ahead.
    sample = feats[feats["cycle"] == 1].copy()
    for col in SENSOR_COLUMNS:
        assert np.allclose(sample[col], sample[f"{col}_rmean_5"], atol=1e-6)


def test_failure_flag_alignment():
    df = generate(num_units=3, seed=2)
    feats = build_features(df)
    assert "failure_within_threshold" in feats.columns
    last = feats.groupby("unit_id").tail(1)
    first = feats.groupby("unit_id").head(1)
    assert (last["failure_within_threshold"] == 1).all()
    assert (first["failure_within_threshold"] == 0).all()
