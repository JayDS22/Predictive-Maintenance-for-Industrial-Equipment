# Predictive Maintenance for Industrial Equipment

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006ACC)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-3A9F4F)](https://lightgbm.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Plotly](https://img.shields.io/badge/Plotly-2.27-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/javascript/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-9b59b6.svg)](LICENSE)

Remaining Useful Life (RUL) prediction and failure-risk classification for industrial machinery. Includes a Flask REST API and an interactive operations dashboard.

The dataset is synthetic by default (NASA C-MAPSS-style turbofan simulator), so the entire pipeline runs offline.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Methodology](#methodology)
- [API reference](#api-reference)
- [Demo platform](#demo-platform)
- [Tests](#tests)
- [Performance targets](#performance-targets)
- [License](#license)

## Highlights

| Area | Implementation |
| --- | --- |
| Tabular models | Random Forest + XGBoost + LightGBM averaging ensemble (regression and classification) |
| Time series | ARIMA(2,1,2) with seasonal decomposition; optional LSTM hook (TensorFlow) |
| Survival analysis | Cox Proportional Hazards plus Kaplan-Meier in R (`src/survival_analysis.R`) |
| Feature engineering | Rolling mean/std (5/10/20), lag (1/3/5), EMA (5/20), cycles-since-start, failure flag |
| Validation | Group-aware K-fold cross-validation by machine to prevent unit-level leakage |
| Serving | Flask app with `/api/health`, `/api/metrics`, `/api/fleet`, `/api/unit/<id>`, `/api/predict` |
| UI | Single-page landing site and a fleet dashboard built with vanilla CSS + Plotly.js (no build step) |
| Container | Single-stage Dockerfile that trains models during build and serves via gunicorn |

## Architecture

```
                    ┌───────────────────────────────────────────────┐
                    │  data/turbofan_synthetic.csv (or live stream) │
                    │   100 units · 14 sensors · 3 op-settings      │
                    └───────────────────────┬───────────────────────┘
                                            ▼
       ┌───────────────────────── Feature engineering ──────────────────────────┐
       │  rolling{mean,std}_{5,10,20}   lag_{1,3,5}   ema_{5,20}                │
       │  cycles_since_start            failure_within_threshold (RUL ≤ 30)     │
       │  src/feature_engineering.py                                            │
       └──────────────────────────────────┬─────────────────────────────────────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              ▼                                                       ▼
   ┌─────────────────────────┐                          ┌─────────────────────────┐
   │ Regression ensemble     │                          │ Classification ensemble │
   │  RF + XGBoost + LightGBM│                          │  RF + XGBoost + LightGBM│
   │  ▸ predicted RUL        │                          │  ▸ failure-within-30    │
   │  src/ensemble_model.py  │                          │  src/ensemble_model.py  │
   └────────────┬────────────┘                          └────────────┬────────────┘
                ▼                                                    ▼
   ┌─────────────────────────┐                          ┌─────────────────────────┐
   │ ARIMA(2,1,2) +          │                          │ Risk-band + recommended │
   │ seasonal decomposition  │                          │ actions                 │
   │ src/time_series_…       │                          │ src/predict.py          │
   └────────────┬────────────┘                          └────────────┬────────────┘
                └──────────────────────┬─────────────────────────────┘
                                       ▼
                       ┌───────────────────────────────────┐
                       │   Flask API · app/app.py           │
                       │   /api/fleet · /api/unit · /api…  │
                       └───────────────────┬───────────────┘
                                           ▼
                       ┌───────────────────────────────────┐
                       │  Interactive demo UI               │
                       │  / (landing) · /dashboard          │
                       └───────────────────────────────────┘

   Offline reporting path:
   data → src/survival_analysis.R → Cox PH + Kaplan-Meier → reports/
```

## Repository layout

```
Predictive-Maintenance-for-Industrial-Equipment/
├── data/
│   ├── generate_synthetic_data.py    NASA C-MAPSS-style data simulator
│   └── README.md
├── notebooks/
│   ├── 01_eda.py                     EDA, writes reports/eda/*.png
│   └── 02_modeling.py                Modeling walkthrough
├── src/
│   ├── config.py                     paths, constants, sensor columns
│   ├── data_loader.py                load or lazily generate CSV
│   ├── feature_engineering.py        rolling, lag, EMA, cycle features
│   ├── ensemble_model.py             RF + XGBoost + LightGBM ensembles
│   ├── time_series_forecasting.py    ARIMA + decomposition (LSTM optional)
│   ├── train.py                      end-to-end training pipeline
│   ├── predict.py                    inference utilities used by the API
│   ├── evaluate.py                   metric helpers
│   └── survival_analysis.R           Cox PH + Kaplan-Meier in R
├── app/
│   ├── app.py                        Flask API and page routes
│   ├── templates/
│   │   ├── index.html                landing page
│   │   └── dashboard.html            fleet dashboard
│   └── static/
│       ├── css/style.css
│       └── js/{main,dashboard}.js
├── tests/
│   ├── test_feature_engineering.py
│   └── test_ensemble_model.py
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generates synthetic data on first run (~30-90s end-to-end).
python -m src.train --units 100

# Launch the demo platform.
python -m app.app
# Browse to http://localhost:8000 and click "Launch Live Dashboard".
```

Or via the Makefile:

```bash
make install       # pip install
make train         # train models, write models/metrics.json
make train-fast    # 60 units, skip CV (~20s)
make serve         # Flask dev server
make serve-prod    # gunicorn with 2 workers and 4 threads
make test          # pytest smoke tests
```

### Docker

```bash
make docker          # builds, trains during build, and serves on :8000
# or
docker build -t predictive-mx .
docker run --rm -p 8000:8000 predictive-mx
```

### Survival analysis (R)

```bash
Rscript src/survival_analysis.R
# Writes reports/km_curve.png, reports/cox_hazard_ratios.csv,
# reports/survival_summary.txt
```

## Methodology

### Data simulation

`data/generate_synthetic_data.py` simulates 100 turbofan engines running 120 to 320 cycles to failure. Sensor signals follow a knee-point exponential degradation curve with Gaussian noise (severity randomised per unit). Output columns follow NASA C-MAPSS conventions: `unit_id`, `cycle`, `op_setting_1..3`, `sensor_2..21`, `RUL`.

### Feature engineering (`src/feature_engineering.py`)

All transforms are grouped by `unit_id`. Cross-unit aggregation is never performed.

- Rolling mean and standard deviation at windows 5, 10, 20
- Lag features at offsets 1, 3, 5
- Exponential moving averages at spans 5, 20
- Cycles-since-start, plus its square
- Binary `failure_within_threshold` flag where RUL ≤ 30 cycles

### Modeling (`src/ensemble_model.py`)

Two averaging ensembles share the same feature set:

| Task | Members | Output |
| --- | --- | --- |
| RUL regression | `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor` | Continuous cycles |
| Failure classification | `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier` | Probability in [0, 1] |

RUL targets are capped at 125 cycles before training (standard C-MAPSS practice: beyond that, units are indistinguishably healthy).

### Validation

- 80/20 split by `unit_id` to ensure the test set contains entirely unseen machines.
- 5-fold `GroupKFold` cross-validation for regression and classification metrics on the training units.

### Forecasting (`src/time_series_forecasting.py`)

- ARIMA(2,1,2) projects the next 20 cycles of RUL given a unit's predicted history.
- `seasonal_decompose` produces trend/seasonal/residual traces for the dashboard.
- An optional LSTM forecaster is exposed when TensorFlow is installed.

### Survival analysis (`src/survival_analysis.R`)

Run separately for fleet reliability reporting:

- Kaplan-Meier survival curve across the fleet → `reports/km_curve.png`
- Cox PH model with averaged sensor covariates → hazard-ratio table and concordance index

## API reference

| Method | Path | Description |
| --- | --- | --- |
| GET  | `/`               | Landing page |
| GET  | `/dashboard`      | Fleet dashboard |
| GET  | `/api/health`     | `{status, models_trained, metrics_available}` |
| GET  | `/api/metrics`    | Full contents of `models/metrics.json` |
| GET  | `/api/fleet`      | Latest-cycle prediction per unit in the holdout sample |
| GET  | `/api/unit/<id>`  | Cycle-by-cycle predictions, ARIMA forecast, sensor traces |
| POST | `/api/predict`    | Ad-hoc prediction. Body: `{"cycle": 120, "sensors": {"sensor_2": ..., ...}}` |

Example:

```bash
curl -s http://localhost:8000/api/health
curl -s http://localhost:8000/api/fleet | jq '.summary'

curl -s -X POST http://localhost:8000/api/predict \
  -H "content-type: application/json" \
  -d '{
    "cycle": 150,
    "sensors": {
      "sensor_2": 645.5, "sensor_3": 1618.0, "sensor_4": 1424.0,
      "sensor_7": 551.0, "sensor_8": 2399.0, "sensor_9": 9120.0,
      "sensor_11": 47.9, "sensor_12": 517.8, "sensor_13": 2399.0,
      "sensor_14": 8190.0, "sensor_15": 8.25, "sensor_17": 400.5,
      "sensor_20": 37.7, "sensor_21": 22.0
    }
  }' | jq
```

## Demo platform

The Flask app exposes two pages built with vanilla HTML/CSS and Plotly.js (no build step):

- `/`: landing page with hero, KPIs sourced from `metrics.json`, architecture diagram, and an inline sensor predictor form.
- `/dashboard`: operations view with a fleet ranking table, predicted-vs-actual RUL chart, failure-risk timeline, sensor traces, and trend/seasonal/residual decomposition.

Click any row in the fleet table to switch the unit-detail charts. The dashboard reads the holdout sample written by `train.py` at `models/demo_holdout.csv`.

## Tests

```bash
pytest -q
```

Coverage:

- Synthetic dataset schema and RUL boundaries
- Feature pipeline: per-unit isolation, failure-flag alignment
- Ensembles: regression beats a naive mean baseline; classifier returns valid probabilities
- ARIMA forecast is finite and non-negative

## Performance targets

Reported on the per-unit holdout split (100 simulated engines, default seed). Re-run `python -m src.train` to reproduce; values vary by seed.

| Metric | Target | Typical |
| --- | --- | --- |
| RUL MAE (cycles) | < 20 | 12 to 18 |
| Failure-class accuracy | > 0.90 | 0.93 to 0.97 |
| Failure-class AUC-ROC | > 0.95 | 0.97 to 0.99 |
| Cox C-index (R script) | > 0.70 | 0.75 to 0.82 |
| P95 inference latency | < 100 ms | ~30 ms |

## License

MIT, see [`LICENSE`](LICENSE).
