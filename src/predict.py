"""Inference wrappers consumed by the Flask app."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .config import (
    CLASSIFIER_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
    FAILURE_THRESHOLD,
    FEATURE_SCALER_PATH,
    MODELS_DIR,
    SENSOR_COLUMNS,
)
from .feature_engineering import build_features


@dataclass
class Predictor:
    regressor: object
    classifier: object
    scaler: object
    feature_columns: list[str]

    @classmethod
    def load(cls) -> "Predictor":
        if not ENSEMBLE_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Models not found. Run `python -m src.train` to produce them."
            )
        payload = joblib.load(FEATURE_SCALER_PATH)
        return cls(
            regressor=joblib.load(ENSEMBLE_MODEL_PATH),
            classifier=joblib.load(CLASSIFIER_MODEL_PATH),
            scaler=payload["scaler"],
            feature_columns=payload["feature_columns"],
        )

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict RUL and failure risk for a feature-engineered frame."""
        X = pd.DataFrame(
            self.scaler.transform(df[self.feature_columns]),
            columns=self.feature_columns,
            index=df.index,
        )
        rul = np.clip(self.regressor.predict(X), 0, 125)
        risk = self.classifier.predict_proba(X)
        return pd.DataFrame(
            {
                "unit_id": df.get("unit_id", pd.Series(["adhoc"] * len(df), index=df.index)),
                "cycle": df.get("cycle", pd.Series(range(1, len(df) + 1), index=df.index)),
                "predicted_rul": rul,
                "failure_risk": risk,
                "risk_band": pd.cut(
                    risk,
                    bins=[-0.001, 0.25, 0.6, 1.0],
                    labels=["healthy", "watch", "critical"],
                ).astype(str),
            }
        )

    def predict_from_history(self, history: pd.DataFrame) -> dict:
        """Score one unit given its observation history.

        `history` must contain sensor columns, `unit_id`, and `cycle`. Returns
        the latest-cycle prediction together with a 20-cycle ARIMA forecast.
        """
        from .time_series_forecasting import forecast_rul_for_unit

        feats = build_features(history, sensor_cols=SENSOR_COLUMNS)
        preds = self.predict_dataframe(feats)
        latest = preds.iloc[-1].to_dict()

        forecast = forecast_rul_for_unit(preds["predicted_rul"], horizon=20).tolist()
        latest["forecast_rul_next_20"] = forecast
        latest["actions"] = _recommend_actions(latest["failure_risk"], latest["predicted_rul"])
        return latest


def _recommend_actions(risk: float, rul: float) -> list[str]:
    if risk >= 0.7 or rul <= 10:
        return [
            "Schedule emergency inspection within 24h",
            "Prepare replacement parts and crew",
            "Reduce operating load to 70%",
        ]
    if risk >= 0.4 or rul <= FAILURE_THRESHOLD:
        return [
            "Plan maintenance within next 2 weeks",
            "Increase sensor sampling frequency",
            "Notify reliability engineering lead",
        ]
    return [
        "Continue normal operation",
        "Next routine check at scheduled interval",
    ]


def load_demo_holdout() -> Optional[pd.DataFrame]:
    path = MODELS_DIR / "demo_holdout.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)
