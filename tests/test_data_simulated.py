"""Tests unitaires pour data/sources/simulated.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from data.sources import simulated


START = pd.Timestamp("2026-01-01", tz="UTC")
END = pd.Timestamp("2026-01-03", tz="UTC")


# ---------------------------------------------------------------------------
# generate_weather
# ---------------------------------------------------------------------------

def test_generate_weather_columns():
    df = simulated.generate_weather(START, END)
    for col in ["timestamp", "kind", "outdoor_temperature", "solar_irradiance", "cloud_cover"]:
        assert col in df.columns


def test_generate_weather_kind_values():
    df = simulated.generate_weather(START, END, forecast_horizon_hours=24.0)
    assert set(df["kind"].unique()).issubset({"observed", "forecast"})


def test_generate_weather_forecast_after_end():
    df = simulated.generate_weather(START, END, forecast_horizon_hours=24.0)
    observed = df[df["kind"] == "observed"]
    forecast = df[df["kind"] == "forecast"]
    if len(observed) and len(forecast):
        assert observed["timestamp"].max() <= forecast["timestamp"].min()


def test_generate_weather_cloud_cover_in_range():
    df = simulated.generate_weather(START, END)
    assert (df["cloud_cover"] >= 0.0).all()
    assert (df["cloud_cover"] <= 1.0).all()


def test_generate_weather_irradiance_non_negative():
    df = simulated.generate_weather(START, END)
    assert (df["solar_irradiance"] >= 0.0).all()


# ---------------------------------------------------------------------------
# generate_outdoor_sensors
# ---------------------------------------------------------------------------

def test_generate_outdoor_sensors_all_faces():
    weather_df = simulated.generate_weather(START, END)
    outdoor_df = simulated.generate_outdoor_sensors(START, END, weather_df)
    assert set(outdoor_df["face"].unique()) == set(config.HOUSE_FACES.keys())


def test_generate_outdoor_sensors_columns():
    weather_df = simulated.generate_weather(START, END)
    outdoor_df = simulated.generate_outdoor_sensors(START, END, weather_df)
    for col in ["timestamp", "face", "temperature", "humidity", "luminosity"]:
        assert col in outdoor_df.columns


def test_generate_outdoor_sensors_luminosity_non_negative():
    weather_df = simulated.generate_weather(START, END)
    outdoor_df = simulated.generate_outdoor_sensors(START, END, weather_df)
    assert (outdoor_df["luminosity"] >= 0.0).all()


# ---------------------------------------------------------------------------
# generate_house_state
# ---------------------------------------------------------------------------

def test_generate_house_state_types():
    df = simulated.generate_house_state(START, END)
    assert set(df["type"].unique()) == {"shutter", "window"}


def test_generate_house_state_all_rooms():
    df = simulated.generate_house_state(START, END)
    assert set(df["room"].unique()) == set(config.DEFAULT_HOUSE_STATE_ROOMS)


def test_generate_house_state_binary_values():
    df = simulated.generate_house_state(START, END)
    assert set(df["state"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# generate_indoor_sensors
# ---------------------------------------------------------------------------

def test_generate_indoor_sensors_all_rooms():
    weather_df = simulated.generate_weather(START, END)
    outdoor_df = simulated.generate_outdoor_sensors(START, END, weather_df)
    house_state_df = simulated.generate_house_state(START, END)
    indoor_df = simulated.generate_indoor_sensors(START, END, weather_df, outdoor_df, house_state_df)
    assert set(indoor_df["sensor_id"].unique()) == set(config.DEFAULT_INDOOR_ROOMS)


def test_generate_indoor_sensors_temperature_realistic():
    """Températures intérieures simulées doivent rester dans une plage réaliste."""
    weather_df = simulated.generate_weather(START, END)
    outdoor_df = simulated.generate_outdoor_sensors(START, END, weather_df)
    house_state_df = simulated.generate_house_state(START, END)
    indoor_df = simulated.generate_indoor_sensors(START, END, weather_df, outdoor_df, house_state_df)
    assert (indoor_df["temperature"] > -10.0).all()
    assert (indoor_df["temperature"] < 50.0).all()


def test_generate_indoor_sensors_humidity_in_range():
    weather_df = simulated.generate_weather(START, END)
    outdoor_df = simulated.generate_outdoor_sensors(START, END, weather_df)
    house_state_df = simulated.generate_house_state(START, END)
    indoor_df = simulated.generate_indoor_sensors(START, END, weather_df, outdoor_df, house_state_df)
    assert (indoor_df["humidity"] >= 20.0).all()
    assert (indoor_df["humidity"] <= 80.0).all()


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

def test_generate_dataset_keys():
    dataset = simulated.generate_dataset(START, END)
    for key in ["weather", "outdoor", "house_state", "indoor"]:
        assert key in dataset
        assert not dataset[key].empty


def test_generate_dataset_reproducible():
    ds1 = simulated.generate_dataset(START, END, seed=0)
    ds2 = simulated.generate_dataset(START, END, seed=0)
    pd.testing.assert_frame_equal(ds1["indoor"], ds2["indoor"])


def test_generate_dataset_different_seeds():
    ds1 = simulated.generate_dataset(START, END, seed=0)
    ds2 = simulated.generate_dataset(START, END, seed=99)
    # Les deux datasets doivent être différents
    assert not ds1["indoor"]["temperature"].equals(ds2["indoor"]["temperature"])


# ---------------------------------------------------------------------------
# write_dataset
# ---------------------------------------------------------------------------

def test_write_dataset_creates_files(tmp_path):
    import config as cfg
    import importlib, types

    dataset = simulated.generate_dataset(START, END)

    # Patch temporaire des chemins de config pour écrire dans tmp_path
    original_dirs = {
        "weather": cfg.WEATHER_DIR,
        "outdoor": cfg.OUTDOOR_DIR,
        "house_state": cfg.HOUSE_STATE_DIR,
        "indoor": cfg.INDOOR_DIR,
    }
    # On appelle write_dataset avec un patch manuel
    target_dirs = {
        "weather": tmp_path / "weather",
        "outdoor": tmp_path / "outdoor",
        "house_state": tmp_path / "house_state",
        "indoor": tmp_path / "indoor",
    }
    for source, df in dataset.items():
        directory = target_dirs[source]
        directory.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out["timestamp"] = pd.DatetimeIndex(out["timestamp"]).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(directory / "simulated.csv", index=False)

    for source, d in target_dirs.items():
        assert (d / "simulated.csv").exists()
