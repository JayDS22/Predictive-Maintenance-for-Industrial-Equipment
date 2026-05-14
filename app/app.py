"""Flask demo and REST API.

Routes
------
GET  /                  landing page
GET  /dashboard         interactive fleet dashboard
GET  /api/health        liveness check
GET  /api/metrics       latest training metrics
GET  /api/fleet         per-unit predictions for the holdout sample
GET  /api/unit/<id>     sensor traces and RUL forecast for one unit
POST /api/predict       ad-hoc prediction from a JSON sensor payload

Run:
    python -m app.app                                # development
    gunicorn -b 0.0.0.0:8000 app.app:app             # production
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ENSEMBLE_MODEL_PATH,
    METRICS_PATH,
    SENSOR_COLUMNS,
)
from src.predict import Predictor, load_demo_holdout
from src.time_series_forecasting import decompose_sensor

app = Flask(__name__, static_folder="static", template_folder="templates")

_state: dict = {"predictor": None, "holdout": None, "metrics": None, "loaded_at": None}


def _ensure_models_loaded() -> bool:
    if _state["predictor"] is not None:
        return True
    if not ENSEMBLE_MODEL_PATH.exists():
        return False
    _state["predictor"] = Predictor.load()
    _state["holdout"] = load_demo_holdout()
    if METRICS_PATH.exists():
        _state["metrics"] = json.loads(METRICS_PATH.read_text())
    _state["loaded_at"] = time.time()
    return True


@app.route("/")
def index():
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    return render_template("index.html", metrics=metrics)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "models_trained": ENSEMBLE_MODEL_PATH.exists(),
            "metrics_available": METRICS_PATH.exists(),
        }
    )


@app.route("/api/metrics")
def api_metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "metrics not available - run `python -m src.train`"}), 404
    return jsonify(json.loads(METRICS_PATH.read_text()))


@app.route("/api/fleet")
def api_fleet():
    if not _ensure_models_loaded():
        return _models_missing()
    holdout = _state["holdout"]
    if holdout is None or holdout.empty:
        return jsonify({"error": "demo holdout not available"}), 404

    predictor: Predictor = _state["predictor"]
    preds = predictor.predict_dataframe(holdout)
    fleet = (
        preds.groupby("unit_id")
        .agg(
            latest_cycle=("cycle", "max"),
            current_rul=("predicted_rul", "last"),
            failure_risk=("failure_risk", "last"),
            risk_band=("risk_band", "last"),
        )
        .reset_index()
        .sort_values("failure_risk", ascending=False)
    )
    return jsonify(
        {
            "units": fleet.to_dict(orient="records"),
            "summary": {
                "n_units": int(fleet["unit_id"].nunique()),
                "n_critical": int((fleet["risk_band"] == "critical").sum()),
                "n_watch": int((fleet["risk_band"] == "watch").sum()),
                "n_healthy": int((fleet["risk_band"] == "healthy").sum()),
            },
        }
    )


@app.route("/api/unit/<int:unit_id>")
def api_unit(unit_id: int):
    if not _ensure_models_loaded():
        return _models_missing()
    holdout = _state["holdout"]
    if holdout is None:
        return jsonify({"error": "demo holdout not available"}), 404
    unit_df = holdout[holdout["unit_id"] == unit_id].copy()
    if unit_df.empty:
        return jsonify({"error": f"unit {unit_id} not in holdout"}), 404

    predictor: Predictor = _state["predictor"]
    preds = predictor.predict_dataframe(unit_df)

    from src.time_series_forecasting import forecast_rul_for_unit
    forecast = forecast_rul_for_unit(preds["predicted_rul"], horizon=20).tolist()

    sensor_traces = {col: unit_df[col].tolist() for col in SENSOR_COLUMNS[:5]}
    decomposition = decompose_sensor(unit_df["sensor_3"])

    return jsonify(
        {
            "unit_id": unit_id,
            "cycles": unit_df["cycle"].tolist(),
            "actual_rul": unit_df["RUL"].tolist() if "RUL" in unit_df else None,
            "predicted_rul": preds["predicted_rul"].tolist(),
            "failure_risk": preds["failure_risk"].tolist(),
            "risk_band_latest": preds["risk_band"].iloc[-1],
            "forecast_rul_next_20": forecast,
            "sensor_traces": sensor_traces,
            "decomposition_sensor_3": decomposition,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not _ensure_models_loaded():
        return _models_missing()
    payload = request.get_json(force=True, silent=True) or {}

    history = payload.get("history")
    if history:
        df = pd.DataFrame(history)
        df["unit_id"] = df.get("unit_id", "adhoc")
        if "cycle" not in df:
            df["cycle"] = np.arange(1, len(df) + 1)
        result = _state["predictor"].predict_from_history(df)
        return jsonify(result)

    sensors = payload.get("sensors")
    if not sensors or not isinstance(sensors, dict):
        return jsonify({"error": "expected JSON {sensors: {sensor_X: value, ...}, cycle: int}"}), 400
    cycle = int(payload.get("cycle", 100))

    # Synthesise a 30-cycle history converging on the supplied snapshot so
    # rolling/lag/EMA features have enough context to populate.
    op_defaults = {"op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 0.0}
    history_rows = []
    for offset in range(30, 0, -1):
        row = {"unit_id": "adhoc", "cycle": cycle - offset + 1, **op_defaults}
        for col in SENSOR_COLUMNS:
            base = float(sensors.get(col, 0.0))
            row[col] = base * (0.95 + 0.05 * (1 - offset / 30))
        history_rows.append(row)
    df = pd.DataFrame(history_rows)
    result = _state["predictor"].predict_from_history(df)
    return jsonify(result)


def _models_missing():
    return (
        jsonify(
            {
                "error": "Models not trained yet.",
                "fix": "Run `python -m src.train` from the project root.",
            }
        ),
        503,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
