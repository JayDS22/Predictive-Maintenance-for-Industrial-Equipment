"""ARIMA forecasting and seasonal decomposition for RUL series.

An LSTM forecaster is exposed only when TensorFlow is importable; the Flask
demo depends only on the ARIMA path.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose


@dataclass
class ARIMAForecaster:
    order: tuple[int, int, int] = (2, 1, 2)
    model_: Optional[object] = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = ARIMA(series.astype(float).values, order=self.order).fit()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("ARIMAForecaster.fit must be called first")
        return np.asarray(self.model_.forecast(steps=steps))


def decompose_sensor(series: pd.Series, period: int = 20) -> dict:
    """Additive trend/seasonal/residual decomposition of a sensor trace."""
    series = series.astype(float).reset_index(drop=True)
    period = max(2, min(period, max(2, len(series) // 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = seasonal_decompose(series, period=period, model="additive", extrapolate_trend="freq")
    return {
        "trend": result.trend.fillna(method="bfill").fillna(method="ffill").tolist(),
        "seasonal": result.seasonal.tolist(),
        "resid": result.resid.fillna(0).tolist(),
        "observed": series.tolist(),
    }


def forecast_rul_for_unit(rul_history: pd.Series, horizon: int = 20) -> np.ndarray:
    """Forecast next `horizon` RUL values with ARIMA; fall back to linear
    extrapolation when the series is too short or ARIMA fails to converge."""
    if len(rul_history) < 10:
        slope = (rul_history.iloc[-1] - rul_history.iloc[0]) / max(len(rul_history) - 1, 1)
        return np.maximum(rul_history.iloc[-1] + slope * np.arange(1, horizon + 1), 0)
    try:
        model = ARIMAForecaster().fit(rul_history)
        return np.maximum(model.forecast(horizon), 0)
    except Exception:
        last = float(rul_history.iloc[-1])
        return np.maximum(last - np.arange(1, horizon + 1), 0)


try:
    import tensorflow as tf  # noqa: F401

    def train_lstm_forecaster(X: np.ndarray, y: np.ndarray, epochs: int = 5):
        """Small LSTM regressor. X shape: (samples, timesteps, features)."""
        from tensorflow.keras import layers, models

        model = models.Sequential(
            [
                layers.Input(shape=X.shape[1:]),
                layers.LSTM(64),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
        model.fit(X, y, epochs=epochs, batch_size=64, verbose=0)
        return model
except Exception:
    train_lstm_forecaster = None  # type: ignore[assignment]
