"""Tests unitaires pour strategy/format.py."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy.format import schedule_to_plan, write_plan_csv


ROOMS = ["salon", "bureau"]
STATE_TYPES = ["shutter", "window"]
NOW = pd.Timestamp("2026-01-01T08:00:00+00:00")
HORIZON_H = 4.0
BLOCK_H = 2.0


def _make_schedule(shutter_vals: list[int], window_vals: list[int]) -> dict:
    n = len(shutter_vals)
    return {
        (room, "shutter"): np.array(shutter_vals, dtype=np.float32)
        for room in ROOMS
    } | {
        (room, "window"): np.array(window_vals, dtype=np.float32)
        for room in ROOMS
    }


# ---------------------------------------------------------------------------
# schedule_to_plan — structure
# ---------------------------------------------------------------------------

def test_schedule_to_plan_has_required_keys():
    sched = _make_schedule([0, 1], [0, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    assert "generated_at" in plan
    assert "horizon_hours" in plan
    assert "rooms" in plan


def test_schedule_to_plan_horizon_hours():
    sched = _make_schedule([0, 1], [0, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    assert plan["horizon_hours"] == pytest.approx(HORIZON_H)


def test_schedule_to_plan_all_rooms_present():
    sched = _make_schedule([0, 1], [0, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    for room in ROOMS:
        assert room in plan["rooms"]


def test_schedule_to_plan_interval_keys():
    sched = _make_schedule([0, 1], [1, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    for room in ROOMS:
        for interval in plan["rooms"][room]:
            assert "from" in interval
            assert "to" in interval
            assert "shutter" in interval
            assert "window" in interval


def test_schedule_to_plan_state_labels():
    sched = _make_schedule([0, 1], [1, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    valid_labels = {"open", "closed"}
    for room in ROOMS:
        for interval in plan["rooms"][room]:
            assert interval["shutter"] in valid_labels
            assert interval["window"] in valid_labels


# ---------------------------------------------------------------------------
# schedule_to_plan — fusion des créneaux consécutifs identiques
# ---------------------------------------------------------------------------

def test_consecutive_identical_blocks_merged():
    # 2 créneaux identiques → 1 seul intervalle
    sched = _make_schedule([0, 0], [1, 1])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    for room in ROOMS:
        assert len(plan["rooms"][room]) == 1


def test_different_consecutive_blocks_not_merged():
    # 2 créneaux différents → 2 intervalles
    sched = _make_schedule([0, 1], [0, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    for room in ROOMS:
        assert len(plan["rooms"][room]) == 2


def test_intervals_span_full_horizon():
    sched = _make_schedule([0, 1], [1, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    for room in ROOMS:
        intervals = plan["rooms"][room]
        t_start = pd.Timestamp(intervals[0]["from"])
        t_end = pd.Timestamp(intervals[-1]["to"])
        duration = (t_end - t_start).total_seconds() / 3600.0
        assert abs(duration - HORIZON_H) < 1e-6


def test_schedule_to_plan_with_reasons():
    sched = _make_schedule([0, 1], [0, 1])
    reasons = {room: ["maintenir", "refroidir"] for room in ROOMS}
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H, reasons=reasons)
    for room in ROOMS:
        for interval in plan["rooms"][room]:
            assert "reason" in interval


# ---------------------------------------------------------------------------
# write_plan_csv
# ---------------------------------------------------------------------------

def test_write_plan_csv_creates_file(tmp_path):
    sched = _make_schedule([0, 1], [1, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    out_path = tmp_path / "plan.csv"
    write_plan_csv(plan, out_path, STATE_TYPES)
    assert out_path.exists()


def test_write_plan_csv_headers(tmp_path):
    sched = _make_schedule([0, 1], [1, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    out_path = tmp_path / "plan.csv"
    write_plan_csv(plan, out_path, STATE_TYPES)
    with open(out_path) as f:
        reader = csv.DictReader(f)
        assert "room" in reader.fieldnames
        assert "from" in reader.fieldnames
        assert "to" in reader.fieldnames
        assert "shutter" in reader.fieldnames
        assert "window" in reader.fieldnames


def test_write_plan_csv_row_count(tmp_path):
    sched = _make_schedule([0, 1], [1, 0])
    plan = schedule_to_plan(sched, ROOMS, STATE_TYPES, NOW, HORIZON_H, BLOCK_H)
    total_intervals = sum(len(v) for v in plan["rooms"].values())
    out_path = tmp_path / "plan.csv"
    write_plan_csv(plan, out_path, STATE_TYPES)
    with open(out_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == total_intervals
