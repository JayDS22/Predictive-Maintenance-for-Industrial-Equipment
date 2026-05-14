from __future__ import annotations

import numpy as np

from data.generate_synthetic_data import generate
from src.ensemble_model import (
    ClassificationEnsemble,
    RegressionEnsemble,
    classification_metrics,
    regression_metrics,
)
from src.feature_engineering import build_features, feature_columns
from src.time_series_forecasting import forecast_rul_for_unit


def _train_test_frames():
    df = generate(num_units=10, seed=3)
    feats = build_features(df)
    cols = feature_columns(feats)
    train_units = feats["unit_id"].unique()[:7]
    train = feats[feats["unit_id"].isin(train_units)]
    test = feats[~feats["unit_id"].isin(train_units)]
    return train, test, cols


def test_regression_ensemble_beats_naive():
    train, test, cols = _train_test_frames()
    reg = RegressionEnsemble(n_estimators=80, max_depth=6).fit(train[cols], train["RUL"])
    preds = reg.predict(test[cols])
    metrics = regression_metrics(test["RUL"].values, preds)

    naive = np.full_like(test["RUL"].values, fill_value=train["RUL"].mean(), dtype=float)
    naive_mae = np.mean(np.abs(naive - test["RUL"].values))

    assert metrics["MAE"] < naive_mae


def test_classifier_returns_valid_probabilities():
    train, test, cols = _train_test_frames()
    clf = ClassificationEnsemble(n_estimators=80, max_depth=6).fit(
        train[cols], train["failure_within_threshold"]
    )
    proba = clf.predict_proba(test[cols])
    assert proba.shape == (len(test),)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_arima_forecast_is_finite_and_clipped():
    train, _, _ = _train_test_frames()
    history = train[train["unit_id"] == train["unit_id"].iloc[0]]["RUL"]
    forecast = forecast_rul_for_unit(history.head(40), horizon=10)
    assert forecast.shape == (10,)
    assert np.isfinite(forecast).all()
    assert (forecast >= 0).all()
