"""Step-through modeling script.

Equivalent to an interactive notebook; run from the project root or step through
in VS Code with `# %%` cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR
from src.data_loader import load_dataset
from src.ensemble_model import (
    ClassificationEnsemble,
    RegressionEnsemble,
    classification_metrics,
    regression_metrics,
)
from src.feature_engineering import build_features, feature_columns

OUT = REPORTS_DIR / "modeling"
OUT.mkdir(parents=True, exist_ok=True)

df = load_dataset()
feats = build_features(df)
cols = feature_columns(feats)

# Per-unit split (no leakage).
units = feats["unit_id"].unique()
train_units = units[: int(0.8 * len(units))]
train = feats[feats["unit_id"].isin(train_units)]
test = feats[~feats["unit_id"].isin(train_units)]

reg = RegressionEnsemble().fit(train[cols], train["RUL"].clip(upper=125))
preds = reg.predict(test[cols])
print("Regression:", regression_metrics(test["RUL"].clip(upper=125).values, preds))

clf = ClassificationEnsemble().fit(train[cols], train["failure_within_threshold"])
proba = clf.predict_proba(test[cols])
clf_pred = (proba >= 0.5).astype(int)
print("Classification:", classification_metrics(
    test["failure_within_threshold"].values, clf_pred, proba
))

# Feature-importance bar chart.
imp = reg.feature_importance(top_n=20)
plt.figure(figsize=(9, 7))
sns.barplot(data=imp, y="feature", x="importance", color="#6366f1")
plt.title("Top 20 features, averaged across RF/XGBoost/LightGBM")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=140)
plt.close()
print(f"Wrote {OUT/'feature_importance.png'}")
