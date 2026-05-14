"""RF + XGBoost + LightGBM averaging ensembles for RUL regression and binary
failure classification, plus shared metric and CV helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .config import RANDOM_STATE

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


@dataclass
class RegressionEnsemble:
    """Equal-weighted average of RF, XGBoost, LightGBM regressors."""

    n_estimators: int = 300
    max_depth: int = 8
    models: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegressionEnsemble":
        self.feature_names_ = list(X.columns)
        self.models["rf"] = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ).fit(X, y)
        if _HAS_XGB:
            self.models["xgb"] = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ).fit(X, y)
        if _HAS_LGB:
            self.models["lgb"] = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=-1,
                num_leaves=63,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.models.values()])
        return preds.mean(axis=1).clip(min=0)

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        imps = []
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imps.append(
                    pd.Series(model.feature_importances_, index=self.feature_names_, name=name)
                )
        if not imps:
            return pd.DataFrame(columns=["feature", "importance"])
        df = pd.concat(imps, axis=1).fillna(0)
        df["importance"] = df.mean(axis=1)
        return (
            df.reset_index().rename(columns={"index": "feature"})
            .sort_values("importance", ascending=False)
            .head(top_n)[["feature", "importance"]]
            .reset_index(drop=True)
        )


@dataclass
class ClassificationEnsemble:
    """Equal-weighted average of RF, XGBoost, LightGBM classifiers."""

    n_estimators: int = 300
    max_depth: int = 8
    models: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ClassificationEnsemble":
        self.feature_names_ = list(X.columns)
        self.models["rf"] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ).fit(X, y)
        if _HAS_XGB:
            self.models["xgb"] = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ).fit(X, y)
        if _HAS_LGB:
            self.models["lgb"] = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                num_leaves=63,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ).fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.column_stack(
            [m.predict_proba(X)[:, 1] for m in self.models.values()]
        )
        return probs.mean(axis=1)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE is undefined when y_true → 0; mask out small RUL near failure.
    mape_mask = y_true > 5
    mape = (
        float(mean_absolute_percentage_error(y_true[mape_mask], y_pred[mape_mask]))
        if mape_mask.any()
        else float("nan")
    )
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def grouped_cv_score(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    is_classifier: bool = False,
) -> dict:
    """Per-unit GroupKFold CV. Returns averaged metrics across folds."""
    splitter = GroupKFold(n_splits=n_splits)
    fold_scores: list[dict] = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if is_classifier:
            model = ClassificationEnsemble().fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)
            preds = (proba >= 0.5).astype(int)
            fold_scores.append(classification_metrics(y_te.values, preds, proba))
        else:
            model = RegressionEnsemble().fit(X_tr, y_tr)
            preds = model.predict(X_te)
            fold_scores.append(regression_metrics(y_te.values, preds))
    keys = fold_scores[0].keys()
    return {k: float(np.nanmean([s[k] for s in fold_scores])) for k in keys}
