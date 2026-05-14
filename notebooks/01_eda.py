"""EDA for the synthetic turbofan dataset.

Writes three plots to reports/eda/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR, SENSOR_COLUMNS
from src.data_loader import load_dataset

OUT = REPORTS_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

df = load_dataset()
print(f"Loaded {len(df):,} rows · {df['unit_id'].nunique()} units")

life = df.groupby("unit_id")["cycle"].max()
plt.figure(figsize=(8, 4))
sns.histplot(life, bins=20, color="#6366f1")
plt.title("Engine life distribution")
plt.xlabel("Cycles to failure")
plt.tight_layout()
plt.savefig(OUT / "life_distribution.png", dpi=140)
plt.close()

sample_units = df["unit_id"].unique()[:6]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, sensor in zip(axes.flatten(), ["sensor_3", "sensor_4", "sensor_9", "sensor_14"]):
    for u in sample_units:
        sub = df[df["unit_id"] == u]
        ax.plot(sub["cycle"], sub[sensor], alpha=0.55)
    ax.set_title(sensor)
    ax.set_xlabel("cycle")
fig.suptitle("Sensor degradation across sample units")
plt.tight_layout()
plt.savefig(OUT / "sensor_degradation.png", dpi=140)
plt.close()

corr = df[SENSOR_COLUMNS + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
plt.figure(figsize=(8, 5))
corr.plot(kind="barh", color="#22d3ee")
plt.title("Sensor correlation with RUL")
plt.tight_layout()
plt.savefig(OUT / "sensor_rul_correlation.png", dpi=140)
plt.close()

print(f"EDA plots written to {OUT}")
