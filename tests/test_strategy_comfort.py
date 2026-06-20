"""Tests unitaires pour strategy/comfort.py."""

from __future__ import annotations

import numpy as np
import pytest

import config
from strategy.comfort import REASON_COOL, REASON_MAINTAIN, REASON_WARM, block_reasons, comfort_cost, room_bounds


# ---------------------------------------------------------------------------
# room_bounds
# ---------------------------------------------------------------------------

def test_room_bounds_fallback_to_config():
    t_min, t_max = room_bounds("salon", comfort_ranges=None)
    assert t_min == config.COMFORT_TEMP_MIN
    assert t_max == config.COMFORT_TEMP_MAX


def test_room_bounds_custom_range():
    ranges = {"salon": (18.0, 24.0)}
    t_min, t_max = room_bounds("salon", comfort_ranges=ranges)
    assert t_min == pytest.approx(18.0)
    assert t_max == pytest.approx(24.0)


def test_room_bounds_missing_room_falls_back():
    ranges = {"bureau": (18.0, 24.0)}
    t_min, t_max = room_bounds("salon", comfort_ranges=ranges)
    assert t_min == config.COMFORT_TEMP_MIN


# ---------------------------------------------------------------------------
# comfort_cost
# ---------------------------------------------------------------------------

def test_comfort_cost_zero_within_range():
    temps = np.array([[22.0, 20.0], [23.0, 21.0]])  # (n_points, n_rooms)
    cost = comfort_cost(temps)
    assert cost == pytest.approx(0.0)


def test_comfort_cost_positive_above_max():
    temps = np.array([[30.0]])  # 4°C au-dessus de 26°C
    cost = comfort_cost(temps)
    assert cost == pytest.approx(4.0)


def test_comfort_cost_positive_below_min():
    temps = np.array([[15.0]])  # 4°C en dessous de 19°C
    cost = comfort_cost(temps)
    assert cost == pytest.approx(4.0)


def test_comfort_cost_sum_over_all_points_and_rooms():
    temps = np.array([[30.0, 15.0]])  # +4°C et -4°C
    cost = comfort_cost(temps)
    assert cost == pytest.approx(8.0)


def test_comfort_cost_custom_ranges():
    rooms = ["salon", "bureau"]
    ranges = {"salon": (20.0, 25.0), "bureau": (18.0, 23.0)}
    temps = np.array([[26.0, 24.0]])  # salon +1°C, bureau +1°C
    cost = comfort_cost(temps, rooms=rooms, comfort_ranges=ranges)
    assert cost == pytest.approx(2.0)


def test_comfort_cost_at_boundary_is_zero():
    temps = np.array([[config.COMFORT_TEMP_MIN, config.COMFORT_TEMP_MAX]])
    cost = comfort_cost(temps)
    assert cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# block_reasons
# ---------------------------------------------------------------------------

def _make_eval_rows(n: int, now_idx: int = 0, step: int = 15) -> np.ndarray:
    return np.arange(now_idx + step, now_idx + step * (n + 1), step)


def test_block_reasons_maintain_when_in_range():
    temps_by_room = {"salon": np.full(4, 22.0)}
    eval_rows = _make_eval_rows(4, now_idx=0, step=15)
    reasons = block_reasons(temps_by_room, eval_rows, now_idx=0, n_blocks=1, block_hours=2.0,
                            sample_interval_minutes=2.0)
    assert reasons["salon"] == [REASON_MAINTAIN]


def test_block_reasons_cool_when_above_max():
    temps_by_room = {"salon": np.full(4, 30.0)}
    eval_rows = _make_eval_rows(4, now_idx=0, step=15)
    reasons = block_reasons(temps_by_room, eval_rows, now_idx=0, n_blocks=1, block_hours=2.0,
                            sample_interval_minutes=2.0)
    assert reasons["salon"] == [REASON_COOL]


def test_block_reasons_warm_when_below_min():
    temps_by_room = {"salon": np.full(4, 10.0)}
    eval_rows = _make_eval_rows(4, now_idx=0, step=15)
    reasons = block_reasons(temps_by_room, eval_rows, now_idx=0, n_blocks=1, block_hours=2.0,
                            sample_interval_minutes=2.0)
    assert reasons["salon"] == [REASON_WARM]


def test_block_reasons_multiple_blocks():
    # 2 créneaux de 2h, 4 points d'évaluation par créneau
    # 1er créneau : 22°C (maintenir), 2e créneau : 28°C (refroidir)
    temps_by_room = {"salon": np.array([22.0, 22.0, 22.0, 22.0, 28.0, 28.0, 28.0, 28.0])}
    # step = 15 pas * 2 min = 30 min → 1 créneau de 2h = 4 points d'évaluation
    eval_rows = np.arange(15, 15 * 9, 15)  # 8 points
    reasons = block_reasons(temps_by_room, eval_rows, now_idx=0, n_blocks=2, block_hours=2.0,
                            sample_interval_minutes=2.0)
    assert reasons["salon"][0] == REASON_MAINTAIN
    assert reasons["salon"][1] == REASON_COOL
