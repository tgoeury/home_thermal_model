# home_model — Indoor Thermal Modelling

PyTorch GRU model that predicts indoor temperature and humidity per room, and
recommends daily shutter/window schedules to keep each room within a comfort
temperature range.

---

## Overview

The system is built around two stacked models:

- **LimitedModel** — takes weather forecasts, solar position features, and
  shutter/window states as input. Used for daily planning because it works
  with forecast data alone (no live outdoor sensors required).
- **FullModel** — wraps a frozen `LimitedModel` and adds a residual correction
  from live outdoor façade sensors (temperature, humidity, luminosity per
  cardinal face). Used for real-time inference.

Training data is currently simulated with a toy thermal model. The pipeline is
designed to accept real CSV sensor data with no code changes — just drop files
into `data_raw/` and retrain.

---

## Architecture

### Multi-Resolution History Encoder

Rather than feeding 540 raw 2-minute steps (18 hours), the encoder compresses
the history to **108 steps** across three resolution bands:

| Band | Duration | Resolution | Steps |
|------|----------|------------|-------|
| Recent | 0–2 h | 2 min (native) | 60 |
| Mid | 2–6 h | 10 min | 24 |
| Old | 6–18 h | 30 min | 24 |

A `dt` channel (normalised time delta between steps) is appended to each
feature vector so the GRU knows the temporal spacing within each band.

### Model Stack

```
LimitedModel
  x_limited (batch, 108, n_limited_features)
    └─► MultiResolutionEncoder (GRU)
          └─► h  (batch, hidden_size)
                └─► RegressionHead (MLP)
                      └─► predictions (batch, n_targets)

FullModel
  x_limited ──► LimitedModel (frozen) ──► base_pred, h_limited
  x_outdoor ──► MultiResolutionEncoder ──► h_outdoor
  [h_limited, h_outdoor] ──► RegressionHead ──► correction
  output = base_pred + correction
```

---

## Project Structure

```
home_model/
├── config.py                  # All constants (paths, hyperparameters, comfort bounds)
├── home_model.py              # CLI entry point
├── train.py                   # Training loops + ONNX export
├── evaluate.py                # Metrics, sensitivity report, prediction plots
│
├── data/
│   ├── pipeline.py            # Data fusion, multi-resolution windowing, PyTorch Dataset
│   ├── solar.py               # NOAA simplified solar position (pure numpy)
│   └── sources/
│       ├── base.py            # Generic pivot CSV loader
│       ├── indoor_sensors.py  # Indoor sensors (pivot on sensor_id)
│       ├── outdoor_sensors.py # Outdoor sensors per façade (pivot on face)
│       ├── weather.py         # Observed weather + forecasts
│       ├── house_state.py     # Shutter/window states
│       └── simulated.py       # Toy thermal simulator (generates training data)
│
├── models/
│   ├── layers.py              # MultiResolutionEncoder, RegressionHead
│   ├── limited.py             # LimitedModel
│   └── full.py                # FullModel (residual correction on top of limited)
│
├── strategy/
│   ├── comfort.py             # Comfort cost (temperature range violations)
│   ├── schedule.py            # Random schedule generation and upsampling
│   ├── rollout.py             # Batch schedule evaluation via LimitedModel
│   ├── planner.py             # Random search over N candidate schedules
│   └── format.py             # Convert schedule → JSON/CSV (merged intervals)
│
├── docs/                      # Per-module markdown documentation
└── tests/                     # pytest unit tests (123 tests)
```

---

## Quickstart

### Requirements

```bash
pip install torch pandas numpy onnx onnxruntime
```

### 1 — Generate simulated data

```bash
python home_model.py simulate --days 90 --forecast-hours 24
```

Writes CSV files to `data_raw/` (indoor sensors, outdoor façade sensors,
weather observed + forecast, shutter/window states).

### 2 — Train models

```bash
# Train the planning model
python home_model.py train-limited --epochs 50

# Train the real-time correction model (requires limited checkpoint)
python home_model.py train-full --epochs 50
```

Both commands automatically export `.onnx` files to `checkpoints/`.

### 3 — Evaluate

```bash
python home_model.py evaluate --model limited
python home_model.py evaluate --model full
```

Prints MAE/RMSE per room and writes prediction plots to `evaluation/`.

### 4 — Generate a daily schedule

```bash
python home_model.py plan --n-candidates 500
```

Tests 500 random shutter/window schedules using `LimitedModel` and returns the
one with the lowest comfort cost. Writes JSON and CSV outputs to
`strategy/output/`.

### 5 — Export ONNX manually

```bash
python home_model.py export-onnx --model limited
python home_model.py export-onnx --model full
```

---

## Column Naming Convention

All columns in the merged feature table follow `<source>__<key>__<measure>`:

| Prefix | Description | Examples |
|--------|-------------|---------|
| `indoor__` | Indoor sensors (prediction targets) | `indoor__salon__temperature` |
| `outdoor__` | Outdoor sensors per façade | `outdoor__S__luminosity` |
| `weather__` | Weather (observed + forecast) | `weather__solar_irradiance` |
| `solar__` | Computed solar features | `solar__face_exposure__S` |
| `house__` | Shutter/window states | `house__salon__shutter` |

The separator `__` is defined in `config.COLUMN_SEP`.

---

## Adding New Sensors

The pipeline detects columns dynamically by prefix. To add a new indoor
sensor:

1. Place a CSV in `data_raw/indoor/` with columns
   `timestamp, sensor_id, temperature, humidity`.
2. Retrain `limited` then `full` from scratch (input dimensions change).

No code changes are needed.

---

## Configuration

Edit `config.py` to adapt the system to a real house:

```python
# House location (for solar position calculation)
LATITUDE = 48.8566    # TODO: set real latitude
LONGITUDE = 2.3522    # TODO: set real longitude
ELEVATION = 35.0      # metres above sea level
TIMEZONE = "Europe/Paris"

# Comfort temperature range (applies to all rooms unless overridden at plan time)
COMFORT_TEMP_MIN = 19.0  # °C
COMFORT_TEMP_MAX = 26.0  # °C
```

Per-room comfort ranges can also be passed at plan time:

```bash
python home_model.py plan --comfort-ranges '{"salon": [20, 25], "bureau": [18, 24]}'
```

---

## ONNX Export

After training, both models are exported in ONNX format:

| File | Inputs | Outputs |
|------|--------|---------|
| `checkpoints/limited.onnx` | `x_limited` | `predictions` |
| `checkpoints/full.onnx` | `x_limited`, `x_outdoor` | `predictions`, `base_pred`, `correction` |

All exports use dynamic batch axes and opset 17. Inference is validated
against PyTorch outputs with `atol=1e-5` in the test suite.

---

## Tests

```bash
pytest tests/
```

123 tests covering config, solar math, CSV sources, data pipeline,
simulated data generation, model shapes and freeze behaviour, ONNX
export/inference, and the full strategy pipeline.

---

## Design Decisions

- **Room orientation is latent** — `DEFAULT_ROOM_FACES` in `config.py` is only
  used by the simulator. The model learns room orientation implicitly from
  correlations with `solar__face_exposure__*` features.
- **Chronological train/val split** — never shuffled, to prevent future data
  leaking into training.
- **No autoregressive rollout for planning** — `LimitedModel` does not consume
  its own outputs, so all prediction steps across a planning horizon can be
  evaluated in a single batch.
- **Graceful degradation** — if outdoor façade sensors go offline, `LimitedModel`
  can be used alone without retraining.
- **Observed weather beats forecast** — `WeatherSource` gives priority to
  `kind=observed` rows over same-timestamp `kind=forecast` rows.
