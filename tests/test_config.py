"""Tests unitaires pour config.py."""

from __future__ import annotations

import config


def test_resolution_segments_for_full_window():
    segs = config.resolution_segments_for(config.HISTORY_HOURS)
    assert len(segs) == len(config.RESOLUTION_SEGMENTS)
    for got, expected in zip(segs, config.RESOLUTION_SEGMENTS):
        assert got["duration_minutes"] == expected["duration_minutes"]
        assert got["resolution_minutes"] == expected["resolution_minutes"]


def test_resolution_segments_for_short_window():
    segs = config.resolution_segments_for(1.0)
    total = sum(s["duration_minutes"] for s in segs)
    assert abs(total - 60.0) < 1e-6


def test_resolution_segments_for_zero():
    segs = config.resolution_segments_for(0.0)
    assert segs == []


def test_resolution_segments_for_partial():
    # 3h d'historique couvre le premier segment (2h) + le début du deuxième (10min résol.)
    segs = config.resolution_segments_for(3.0)
    total = sum(s["duration_minutes"] for s in segs)
    assert abs(total - 180.0) < 1e-6
    # La résolution du premier segment doit être 2 min (résolution native)
    assert segs[0]["resolution_minutes"] == 2


def test_history_hours_matches_segments():
    total_minutes = sum(s["duration_minutes"] for s in config.RESOLUTION_SEGMENTS)
    assert abs(config.HISTORY_HOURS - total_minutes / 60.0) < 1e-9


def test_column_sep_is_double_underscore():
    assert config.COLUMN_SEP == "__"


def test_comfort_bounds_ordered():
    assert config.COMFORT_TEMP_MIN < config.COMFORT_TEMP_MAX
