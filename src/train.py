"""End-to-end training pipeline.

Stages:
    1. Load or generate the turbofan dataset.
    2. Feature engineering (rolling, lag, EMA).
    3. Train RUL regression ensemble + failure classifier ensemble.
    4. Per-unit GroupKFold cross-validation.
    5. Persist models, scaler, feature importances, metrics, demo holdout.

Usage:
    python -m src.train --units 100
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .config import (
    CLASSIFIER_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
    FEATURE_SCALER_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    SENSOR_COLUMNS,
)
from .data_loader import load_dataset
from .ensemble_model import (
    ClassificationEnsemble,
    RegressionEnsemble,
    classification_metrics,
    grouped_cv_score,
    regression_metrics,
)
from .feature_engineering import build_features, feature_columns


def _split_by_unit(df: pd.DataFrame, test_size: float = 0.2):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["unit_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def run(units: int = 100, cv_splits: int = 5, skip_cv: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/5] Loading dataset...")
    df = load_dataset(units_if_missing=units)
    print(f"      rows={len(df):,}  units={df['unit_id'].nunique()}")

    print("[2/5] Engineering features...")
    feats = build_features(df, sensor_cols=SENSOR_COLUMNS)
    cols = feature_columns(feats)
    print(f"      n_features={len(cols)}")

    train_df, test_df = _split_by_unit(feats, test_size=0.2)

    scaler = StandardScaler().fit(train_df[cols])
    X_train = pd.DataFrame(scaler.transform(train_df[cols]), columns=cols, index=train_df.index)
    X_test = pd.DataFrame(scaler.transform(test_df[cols]), columns=cols, index=test_df.index)

    # Standard C-MAPSS practice: cap RUL beyond a healthy plateau.
    y_train_rul = train_df["RUL"].clip(upper=125)
    y_test_rul = test_df["RUL"].clip(upper=125)
    y_train_clf = train_df["failure_within_threshold"]
    y_test_clf = test_df["failure_within_threshold"]

    print("[3/5] Training RUL regression ensemble...")
    reg = RegressionEnsemble().fit(X_train, y_train_rul)
    reg_preds = reg.predict(X_test)
    reg_metrics = regression_metrics(y_test_rul.values, reg_preds)
    print(f"      MAE={reg_metrics['MAE']:.2f}  RMSE={reg_metrics['RMSE']:.2f}  MAPE={reg_metrics['MAPE']:.3f}")

    print("[4/5] Training failure-classification ensemble...")
    clf = ClassificationEnsemble().fit(X_train, y_train_clf)
    proba = clf.predict_proba(X_test)
    preds = (proba >= 0.5).astype(int)
    clf_metrics = classification_metrics(y_test_clf.values, preds, proba)
    print(f"      acc={clf_metrics['accuracy']:.3f}  f1={clf_metrics['f1']:.3f}  AUC={clf_metrics['auc_roc']:.3f}")

    cv_metrics: dict = {}
    if not skip_cv:
        print(f"[5/5] Group {cv_splits}-fold CV (per-unit)...")
        cv_metrics["regression_cv"] = grouped_cv_score(
            X_train, y_train_rul, train_df["unit_id"], n_splits=cv_splits
        )
        cv_metrics["classification_cv"] = grouped_cv_score(
            X_train, y_train_clf, train_df["unit_id"], n_splits=cv_splits, is_classifier=True
        )

    importance = reg.feature_importance(top_n=20).to_dict(orient="records")

    print("Saving artifacts...")
    joblib.dump(reg, ENSEMBLE_MODEL_PATH)
    joblib.dump(clf, CLASSIFIER_MODEL_PATH)
    joblib.dump({"scaler": scaler, "feature_columns": cols}, FEATURE_SCALER_PATH)

    metrics = {
        "test_regression": reg_metrics,
        "test_classification": clf_metrics,
        "cross_validation": cv_metrics,
        "feature_importance_top20": importance,
        "training_time_seconds": round(time.time() - t0, 2),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_features": len(cols),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    METADATA_PATH.write_text(
        json.dumps(
            {
                "feature_columns": cols,
                "sensor_columns": SENSOR_COLUMNS,
                "rul_cap": 125,
                "failure_threshold_cycles": int(feats["failure_within_threshold"].sum()),
            },
            indent=2,
        )
    )

    # Truncate each demo unit at a different fraction of its life so the fleet
    # dashboard surfaces a mix of risk bands instead of every unit at failure.
    sample_units = test_df["unit_id"].drop_duplicates().head(6).tolist()
    truncate_fractions = [0.30, 0.50, 0.70, 0.85, 0.95, 1.00]
    demo_pieces = []
    for unit, frac in zip(sample_units, truncate_fractions):
        unit_df = test_df[test_df["unit_id"] == unit].sort_values("cycle")
        keep = max(20, int(len(unit_df) * frac))
        demo_pieces.append(unit_df.head(keep))
    pd.concat(demo_pieces, ignore_index=True).to_csv(MODELS_DIR / "demo_holdout.csv", index=False)

    print(f"Done in {metrics['training_time_seconds']}s. Metrics -> {METRICS_PATH}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train predictive maintenance models.")
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true", help="Skip cross-validation for faster runs.")
    args = parser.parse_args()
    run(units=args.units, cv_splits=args.cv_splits, skip_cv=args.skip_cv)


if __name__ == "__main__":
    main()
