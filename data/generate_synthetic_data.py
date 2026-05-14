"""Synthetic NASA C-MAPSS-style turbofan degradation dataset.

Schema mirrors the original C-MAPSS files: unit_id, cycle, three operational
settings, 14 sensor channels, and a per-row Remaining Useful Life label.

Usage:
    python data/generate_synthetic_data.py --units 100 --out data/turbofan_synthetic.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OP_SETTING_COLUMNS, RAW_DATA_CSV, RANDOM_STATE, SENSOR_COLUMNS


def _degradation_curve(length: int, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Monotonic degradation in [0, 1] with a randomised knee point."""
    cycles = np.arange(length)
    knee = rng.uniform(0.55, 0.85) * length
    growth = rng.uniform(2.5, 5.0)
    # Clip the post-knee base before fractional exponent: np.where evaluates
    # both branches, so negative values pre-knee would emit a RuntimeWarning.
    post = np.clip((cycles - knee) / max(length - knee, 1), 0, None)
    curve = np.where(
        cycles < knee,
        0.05 * (cycles / max(knee, 1)),
        0.05 + (1 - 0.05) * post ** growth,
    )
    curve = np.clip(curve * severity, 0, 1)
    noise = rng.normal(0, 0.01, length).cumsum() * 0.1
    return np.clip(curve + noise, 0, 1)


def simulate_unit(unit_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """One engine: run-to-failure observation table."""
    life_cycles = int(rng.integers(120, 320))
    severity = rng.uniform(0.8, 1.2)
    degradation = _degradation_curve(life_cycles, severity, rng)

    op_settings = rng.normal(0, 0.02, (life_cycles, 3)) + rng.normal(0, 0.05, (1, 3))

    # Sensor baselines and per-cycle drift amplitudes match the C-MAPSS profile.
    baselines = np.array(
        [641.8, 1589.7, 1400.6, 554.4, 2388.0, 9046.2, 47.5, 521.7,
         2388.0, 8138.6, 8.4, 391.8, 38.8, 23.3]
    )
    direction = np.array([1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1])
    drift_amp = np.array([5.0, 30.0, 25.0, 4.0, 12.0, 80.0, 0.5, 6.0,
                          12.0, 70.0, 0.4, 10.0, 1.5, 1.5])

    sensors = baselines + np.outer(degradation, direction * drift_amp)
    sensors += rng.normal(0, drift_amp * 0.05, sensors.shape)

    cycles = np.arange(1, life_cycles + 1)
    rul = life_cycles - cycles

    df = pd.DataFrame(sensors, columns=SENSOR_COLUMNS)
    df.insert(0, "unit_id", unit_id)
    df.insert(1, "cycle", cycles)
    for i, col in enumerate(OP_SETTING_COLUMNS):
        df[col] = op_settings[:, i]
    df["RUL"] = rul
    return df


def generate(num_units: int = 100, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [simulate_unit(i + 1, rng) for i in range(num_units)]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", type=Path, default=RAW_DATA_CSV)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(num_units=args.units, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows across {df['unit_id'].nunique()} units to {args.out}")


if __name__ == "__main__":
    main()
