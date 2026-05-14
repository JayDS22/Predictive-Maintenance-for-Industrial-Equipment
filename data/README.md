# Data

A synthetic NASA C-MAPSS-style turbofan dataset is generated on demand so the
pipeline runs offline. Real C-MAPSS data can be dropped into this directory
using the same schema.

## Schema

| Column | Type | Description |
| --- | --- | --- |
| `unit_id` | int | Engine identifier (1..N) |
| `cycle` | int | Operational cycle (1 = new) |
| `op_setting_1..3` | float | Operational regime settings |
| `sensor_2..21` | float | 14 sensor channels (temperatures, pressures, fan/core speeds) |
| `RUL` | int | Remaining useful life in cycles (target) |

Generate with:

```bash
python data/generate_synthetic_data.py --units 100
```
