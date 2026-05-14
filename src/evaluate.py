"""Metric persistence helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def summarize_metrics(metrics: dict) -> pd.DataFrame:
    rows = []
    for section, payload in metrics.items():
        if isinstance(payload, dict):
            for k, v in payload.items():
                rows.append({"section": section, "metric": k, "value": v})
        else:
            rows.append({"section": "summary", "metric": section, "value": payload})
    return pd.DataFrame(rows)
