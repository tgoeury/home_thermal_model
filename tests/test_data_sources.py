"""Tests unitaires pour data/sources/base.py et les sources concrètes."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from data.sources.base import CSVSource
from data.sources.house_state import HouseStateSource
from data.sources.indoor_sensors import IndoorSensorSource
from data.sources.outdoor_sensors import OutdoorSensorSource
from data.sources.weather import WeatherSource


# ---------------------------------------------------------------------------
# CSVSource — répertoire inexistant
# ---------------------------------------------------------------------------

def test_csv_source_missing_directory_returns_empty(tmp_path):
    src = CSVSource(tmp_path / "nonexistent", prefix="test", measures=["value"])
    assert src.load().empty


def test_csv_source_empty_directory_returns_empty(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    src = CSVSource(d, prefix="test", measures=["value"])
    assert src.load().empty


# ---------------------------------------------------------------------------
# CSVSource — sans pivot
# ---------------------------------------------------------------------------

def test_csv_source_load_no_pivot(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "timestamp,outdoor_temperature,solar_irradiance\n"
        "2026-01-01T00:00:00Z,10.0,0.0\n"
        "2026-01-01T00:02:00Z,10.1,0.0\n"
    )
    src = CSVSource(tmp_path, prefix="weather", measures=["outdoor_temperature", "solar_irradiance"])
    df = src.load()

    assert "weather__outdoor_temperature" in df.columns
    assert "weather__solar_irradiance" in df.columns
    assert len(df) == 2
    assert df.index.name == "timestamp"


def test_csv_source_no_pivot_dedup_keeps_last(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "timestamp,value\n"
        "2026-01-01T00:00:00Z,1.0\n"
        "2026-01-01T00:00:00Z,2.0\n"
    )
    src = CSVSource(tmp_path, prefix="x", measures=["value"])
    df = src.load()
    assert len(df) == 1
    assert df["x__value"].iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# CSVSource — avec pivot
# ---------------------------------------------------------------------------

def test_csv_source_load_with_pivot(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "timestamp,sensor_id,temperature,humidity\n"
        "2026-01-01T00:00:00Z,salon,20.0,50.0\n"
        "2026-01-01T00:00:00Z,bureau,18.0,45.0\n"
    )
    src = CSVSource(tmp_path, prefix="indoor", measures=["temperature", "humidity"], pivot_column="sensor_id")
    df = src.load()

    assert "indoor__salon__temperature" in df.columns
    assert "indoor__bureau__humidity" in df.columns
    assert len(df) == 1


def test_csv_source_multiple_files_concatenated(tmp_path):
    for i in range(3):
        (tmp_path / f"file{i}.csv").write_text(
            f"timestamp,value\n2026-01-0{i+1}T00:00:00Z,{float(i)}\n"
        )
    src = CSVSource(tmp_path, prefix="x", measures=["value"])
    df = src.load()
    assert len(df) == 3


# ---------------------------------------------------------------------------
# IndoorSensorSource
# ---------------------------------------------------------------------------

def test_indoor_sensor_source_column_names(tmp_path):
    (tmp_path / "sensors.csv").write_text(
        "timestamp,sensor_id,temperature,humidity\n"
        "2026-01-01T00:00:00Z,salon,20.0,50.0\n"
    )
    src = IndoorSensorSource(directory=tmp_path)
    df = src.load()
    assert "indoor__salon__temperature" in df.columns
    assert "indoor__salon__humidity" in df.columns


# ---------------------------------------------------------------------------
# OutdoorSensorSource
# ---------------------------------------------------------------------------

def test_outdoor_sensor_source_column_names(tmp_path):
    (tmp_path / "sensors.csv").write_text(
        "timestamp,face,temperature,humidity,luminosity\n"
        "2026-01-01T00:00:00Z,S,15.0,60.0,300.0\n"
    )
    src = OutdoorSensorSource(directory=tmp_path)
    df = src.load()
    assert "outdoor__S__temperature" in df.columns
    assert "outdoor__S__luminosity" in df.columns


# ---------------------------------------------------------------------------
# WeatherSource — priorité observed > forecast
# ---------------------------------------------------------------------------

def test_weather_source_observed_beats_forecast(tmp_path):
    (tmp_path / "weather.csv").write_text(
        "timestamp,kind,outdoor_temperature,solar_irradiance,cloud_cover\n"
        "2026-01-01T00:00:00Z,forecast,5.0,0.0,0.5\n"
        "2026-01-01T00:00:00Z,observed,10.0,0.0,0.3\n"
    )
    src = WeatherSource(directory=tmp_path)
    df = src.load()
    assert len(df) == 1
    assert df["weather__outdoor_temperature"].iloc[0] == pytest.approx(10.0)


def test_weather_source_kind_column_preserved(tmp_path):
    (tmp_path / "weather.csv").write_text(
        "timestamp,kind,outdoor_temperature,solar_irradiance,cloud_cover\n"
        "2026-01-01T00:00:00Z,observed,10.0,0.0,0.3\n"
    )
    src = WeatherSource(directory=tmp_path)
    df = src.load()
    assert "weather__kind" in df.columns
    assert df["weather__kind"].iloc[0] == "observed"


# ---------------------------------------------------------------------------
# HouseStateSource
# ---------------------------------------------------------------------------

def test_house_state_source_column_names(tmp_path):
    (tmp_path / "state.csv").write_text(
        "timestamp,room,type,state\n"
        "2026-01-01T00:00:00Z,salon,shutter,1\n"
        "2026-01-01T00:00:00Z,salon,window,0\n"
    )
    src = HouseStateSource(directory=tmp_path)
    df = src.load()
    assert "house__salon__shutter" in df.columns
    assert "house__salon__window" in df.columns
    assert df["house__salon__shutter"].iloc[0] == pytest.approx(1.0)
    assert df["house__salon__window"].iloc[0] == pytest.approx(0.0)
