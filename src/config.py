"""Project-wide paths and constants."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_DATA_CSV = DATA_DIR / "turbofan_synthetic.csv"
TRAIN_FEATURES_CSV = DATA_DIR / "train_features.csv"
TEST_FEATURES_CSV = DATA_DIR / "test_features.csv"

ENSEMBLE_MODEL_PATH = MODELS_DIR / "ensemble_rul.joblib"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "ensemble_failure_clf.joblib"
FEATURE_SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Names follow C-MAPSS convention; 14 informative channels retained from the
# original 21-sensor schema. Numbers track the upstream column index, not a
# contiguous range.
SENSOR_COLUMNS = [
    "sensor_2",   # LPC outlet temperature
    "sensor_3",   # HPC outlet temperature
    "sensor_4",   # LPT outlet temperature
    "sensor_7",   # HPC outlet pressure
    "sensor_8",   # physical fan speed
    "sensor_9",   # physical core speed
    "sensor_11",  # static pressure HPC outlet
    "sensor_12",  # fuel flow to Ps30 ratio
    "sensor_13",  # corrected fan speed
    "sensor_14",  # corrected core speed
    "sensor_15",  # bypass ratio
    "sensor_17",  # bleed enthalpy
    "sensor_20",  # HPT coolant bleed
    "sensor_21",  # LPT coolant bleed
]

OP_SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
ID_COLUMNS = ["unit_id", "cycle"]

FAILURE_THRESHOLD = 30  # RUL ≤ FAILURE_THRESHOLD flips the binary failure flag.

ROLLING_WINDOWS = (5, 10, 20)
LAG_STEPS = (1, 3, 5)
EMA_SPANS = (5, 20)

RANDOM_STATE = 42
