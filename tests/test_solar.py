"""Tests unitaires pour data/solar.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from data.solar import compute_solar_features, face_exposure, solar_position

PARIS_LAT = 48.8566
PARIS_LON = 2.3522


def _ts(datetime_str: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(datetime_str, tz="UTC")])


# ---------------------------------------------------------------------------
# solar_position
# ---------------------------------------------------------------------------

def test_solar_position_positive_elevation_at_noon():
    # Midi UTC en été à Paris → soleil bien au-dessus de l'horizon.
    ts = _ts("2026-06-21T10:00:00Z")  # ~midi solaire à Paris (UTC+2 en été, donc ~10h UTC)
    pos = solar_position(ts, PARIS_LAT, PARIS_LON)
    assert pos["elevation_deg"].iloc[0] > 30.0


def test_solar_position_negative_elevation_at_midnight():
    ts = _ts("2026-06-21T00:00:00Z")
    pos = solar_position(ts, PARIS_LAT, PARIS_LON)
    assert pos["elevation_deg"].iloc[0] < 0.0


def test_solar_position_returns_correct_columns():
    ts = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    pos = solar_position(ts, PARIS_LAT, PARIS_LON)
    assert "elevation_deg" in pos.columns
    assert "azimuth_deg" in pos.columns
    assert len(pos) == 5


def test_solar_position_azimuth_range():
    ts = pd.date_range("2026-06-21", periods=24, freq="h", tz="UTC")
    pos = solar_position(ts, PARIS_LAT, PARIS_LON)
    # L'azimut doit toujours être dans [0°, 360°].
    assert (pos["azimuth_deg"] >= 0.0).all()
    assert (pos["azimuth_deg"] <= 360.0).all()


def test_solar_position_naive_timestamps_treated_as_utc():
    ts_naive = pd.DatetimeIndex(["2026-06-21T10:00:00"])
    ts_utc = pd.DatetimeIndex(["2026-06-21T10:00:00+00:00"])
    pos_naive = solar_position(ts_naive, PARIS_LAT, PARIS_LON)
    pos_utc = solar_position(ts_utc, PARIS_LAT, PARIS_LON)
    np.testing.assert_allclose(
        pos_naive["elevation_deg"].values,
        pos_utc["elevation_deg"].values,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# face_exposure
# ---------------------------------------------------------------------------

def test_face_exposure_zero_below_horizon():
    elevation = np.array([-5.0, -10.0, 0.0])
    azimuth = np.array([180.0, 90.0, 45.0])
    exposure = face_exposure(elevation, azimuth, face_azimuth_deg=180.0)
    np.testing.assert_array_equal(exposure, np.zeros(3))


def test_face_exposure_positive_when_sun_faces_facade():
    # Soleil plein sud, façade sud (azimut 180°), élévation = 45°
    elevation = np.array([45.0])
    azimuth = np.array([180.0])
    exposure = face_exposure(elevation, azimuth, face_azimuth_deg=180.0)
    assert exposure[0] > 0.0


def test_face_exposure_zero_when_sun_behind_facade():
    # Soleil au nord (azimut 0°), façade sud (180°) → pas d'exposition directe.
    elevation = np.array([30.0])
    azimuth = np.array([0.0])
    exposure = face_exposure(elevation, azimuth, face_azimuth_deg=180.0)
    assert exposure[0] == pytest.approx(0.0, abs=1e-6)


def test_face_exposure_non_negative():
    rng = np.random.default_rng(0)
    elevation = rng.uniform(-20, 80, 1000)
    azimuth = rng.uniform(0, 360, 1000)
    for face_az in [0.0, 90.0, 180.0, 270.0]:
        exposure = face_exposure(elevation, azimuth, face_azimuth_deg=face_az)
        assert (exposure >= 0.0).all()


# ---------------------------------------------------------------------------
# compute_solar_features
# ---------------------------------------------------------------------------

def test_compute_solar_features_columns():
    ts = pd.date_range("2026-06-21", periods=10, freq="2min", tz="UTC")
    features = compute_solar_features(ts)
    sep = config.COLUMN_SEP
    assert f"solar{sep}elevation" in features.columns
    assert f"solar{sep}azimuth" in features.columns
    for face in config.HOUSE_FACES:
        assert f"solar{sep}face_exposure{sep}{face}" in features.columns


def test_compute_solar_features_length():
    ts = pd.date_range("2026-06-21", periods=10, freq="2min", tz="UTC")
    features = compute_solar_features(ts)
    assert len(features) == 10


def test_compute_solar_features_exposure_non_negative():
    ts = pd.date_range("2026-06-21", periods=24 * 30, freq="h", tz="UTC")
    features = compute_solar_features(ts)
    sep = config.COLUMN_SEP
    for face in config.HOUSE_FACES:
        col = f"solar{sep}face_exposure{sep}{face}"
        assert (features[col] >= 0.0).all()
