# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Predictive-maintenance platform: takes turbofan-style sensor telemetry, predicts Remaining Useful Life (RUL) and binary failure risk, projects RUL forward with ARIMA, and serves everything through a Flask API plus an interactive demo dashboard.

The dataset is synthetic (`data/generate_synthetic_data.py` mimics NASA C-MAPSS) so the whole pipeline runs offline.

## Common commands

```bash
make install         # pip install -r requirements.txt
make train           # src.train: generates data if missing, trains models, writes metrics.json
make train-fast      # --units 60 --skip-cv (~20s)
make serve           # Flask dev server on :8000
make serve-prod      # gunicorn, 2 workers, 4 threads
make test            # pytest -q
make docker          # build and run container
make survival        # Rscript src/survival_analysis.R → reports/

# Single test:
pytest tests/test_ensemble_model.py::test_arima_forecast_is_finite_and_clipped -q

# Regenerate data without training:
python data/generate_synthetic_data.py --units 100
```

## Architecture: what reads what

The pipeline is a 5-stage flow. Each stage is one module; all share `src/config.py` for paths and constants:

1. `data/generate_synthetic_data.py` writes `data/turbofan_synthetic.csv`. Schema: `unit_id, cycle, op_setting_1..3, sensor_2..21, RUL`.
2. `src/feature_engineering.py::build_features` groups by `unit_id` and emits rolling mean/std, lag, EMA, and cycles-since-start features. No transform crosses unit boundaries (unit tests guard this).
3. `src/ensemble_model.py` holds `RegressionEnsemble` (RF + XGB + LGBM averaging → RUL) and `ClassificationEnsemble` (same trio → failure-within-30 probability). Both expose `feature_importances_` from the underlying trees.
4. `src/train.py` is the entry point. It performs a `GroupShuffleSplit` by `unit_id`, caps RUL at 125 (standard C-MAPSS practice), trains both ensembles, runs 5-fold `GroupKFold` CV, and persists `ensemble_rul.joblib`, `ensemble_failure_clf.joblib`, `feature_scaler.joblib`, `metrics.json`, `metadata.json`, plus `demo_holdout.csv` (the sample the dashboard reads).
5. `app/app.py` lazy-loads the artefacts on first request via `src/predict.Predictor`. `/api/predict` synthesises a 30-cycle pseudo-history from a single sensor snapshot so feature engineering can produce rolling statistics.

`src/time_series_forecasting.py` is shared by `predict.py` and the `/api/unit/<id>` endpoint. ARIMA(2,1,2) with a linear-extrapolation fallback for short series. LSTM is optional and only defined when TensorFlow imports successfully.

`src/survival_analysis.R` is an independent side path: it reads the same CSV but produces R-only outputs under `reports/`. The Flask app does not depend on it.

## Conventions

- Never split data without grouping by `unit_id`. All splits and CV in this repo use `GroupShuffleSplit` or `GroupKFold`. Row-level splits would leak cycles from the same engine into both folds.
- RUL is capped at 125 before training the regressor. Predictions in the demo are clipped to `[0, 125]`.
- Feature columns are persisted with the scaler. `feature_scaler.joblib` stores `{"scaler": ..., "feature_columns": [...]}`. `Predictor.predict_dataframe` reorders incoming columns to match. Do not bypass the scaler.
- The demo holdout file (`models/demo_holdout.csv`) is the only data the dashboard sees. If feature engineering changes, retrain to regenerate it.
- Optional deps (`xgboost`, `lightgbm`, `tensorflow`) are wrapped in try/except. Code paths still work without them, just with fewer ensemble members.

## Front-end

`app/templates/*.html` plus `app/static/{css,js}/*` is a no-build vanilla setup: Inter and JetBrains Mono via Google Fonts, Plotly.js from CDN. Keep it dependency-free; adding npm tooling would break the clone-and-run workflow.

## When things look broken

- `/api/fleet` returns 503: models are not trained. Run `python -m src.train`.
- ARIMA convergence warnings during training are silenced via `warnings.catch_warnings()`. Expected on short series; the forecaster falls back to linear extrapolation.
- Changes to `SENSOR_COLUMNS` in `config.py` require regenerating the dataset and retraining. Saved models embed the column order.
