"""Fixtures partagées entre tous les modules de tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajoute la racine du projet au path pour que les imports (config, data, ...) fonctionnent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


# ---------------------------------------------------------------------------
# DataFrames simulés minimalistes (pas besoin de fichiers CSV)
# ---------------------------------------------------------------------------

def _make_index(n_steps: int = 200) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n_steps, freq="2min", tz="UTC")


@pytest.fixture
def idx():
    return _make_index()


@pytest.fixture
def sample_weather_df():
    """DataFrame météo au format long (observé uniquement)."""
    n = 200
    idx = _make_index(n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": idx,
        "kind": "observed",
        "outdoor_temperature": 15.0 + rng.normal(0, 1, n),
        "solar_irradiance": np.clip(200 + rng.normal(0, 50, n), 0, None),
        "cloud_cover": np.clip(rng.normal(0.3, 0.1, n), 0, 1),
    })


@pytest.fixture
def sample_outdoor_df():
    """DataFrame capteurs extérieurs au format long (4 façades)."""
    n = 200
    idx = _make_index(n)
    rng = np.random.default_rng(1)
    frames = []
    for face in config.HOUSE_FACES:
        frames.append(pd.DataFrame({
            "timestamp": idx,
            "face": face,
            "temperature": 15.0 + rng.normal(0, 1, n),
            "humidity": 60.0 + rng.normal(0, 3, n),
            "luminosity": np.clip(300 + rng.normal(0, 50, n), 0, None),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sample_house_state_df():
    """DataFrame état volets/fenêtres au format long."""
    n = 200
    idx = _make_index(n)
    rng = np.random.default_rng(2)
    frames = []
    for room in config.DEFAULT_HOUSE_STATE_ROOMS:
        frames.append(pd.DataFrame({
            "timestamp": idx, "room": room, "type": "shutter",
            "state": rng.integers(0, 2, n),
        }))
        frames.append(pd.DataFrame({
            "timestamp": idx, "room": room, "type": "window",
            "state": rng.integers(0, 2, n),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sample_indoor_df():
    """DataFrame capteurs intérieurs au format long."""
    n = 200
    idx = _make_index(n)
    rng = np.random.default_rng(3)
    frames = []
    for room in config.DEFAULT_INDOOR_ROOMS:
        frames.append(pd.DataFrame({
            "timestamp": idx,
            "sensor_id": room,
            "temperature": 20.0 + rng.normal(0, 1, n),
            "humidity": 50.0 + rng.normal(0, 3, n),
        }))
    return pd.concat(frames, ignore_index=True)
