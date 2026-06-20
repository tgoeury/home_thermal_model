"""Tests unitaires pour strategy/schedule.py."""

from __future__ import annotations

import numpy as np
import pytest

import config
from strategy.schedule import n_blocks, random_schedule, schedule_to_steps


ROOMS = ["salon", "bureau"]
TYPES = ["shutter", "window"]
HORIZON_H = 4.0
BLOCK_H = 2.0
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# n_blocks
# ---------------------------------------------------------------------------

def test_n_blocks_exact():
    assert n_blocks(4.0, 2.0) == 2


def test_n_blocks_rounding():
    assert n_blocks(3.0, 2.0) == 2  # round(1.5) = 2


def test_n_blocks_single():
    assert n_blocks(2.0, 2.0) == 1


# ---------------------------------------------------------------------------
# random_schedule
# ---------------------------------------------------------------------------

def test_random_schedule_keys():
    sched = random_schedule(ROOMS, TYPES, HORIZON_H, BLOCK_H, RNG)
    for room in ROOMS:
        for t in TYPES:
            assert (room, t) in sched


def test_random_schedule_values_binary():
    sched = random_schedule(ROOMS, TYPES, HORIZON_H, BLOCK_H, RNG)
    for v in sched.values():
        assert set(np.unique(v)).issubset({0.0, 1.0})


def test_random_schedule_values_length():
    sched = random_schedule(ROOMS, TYPES, HORIZON_H, BLOCK_H, RNG)
    expected = n_blocks(HORIZON_H, BLOCK_H)
    for v in sched.values():
        assert len(v) == expected


def test_random_schedule_different_between_calls():
    rng = np.random.default_rng(0)
    s1 = random_schedule(ROOMS, TYPES, 10.0, 2.0, rng)
    s2 = random_schedule(ROOMS, TYPES, 10.0, 2.0, rng)
    # Très peu probable que deux tirages aléatoires soient identiques sur 5 créneaux
    key = (ROOMS[0], TYPES[0])
    assert not np.array_equal(s1[key], s2[key])


# ---------------------------------------------------------------------------
# schedule_to_steps
# ---------------------------------------------------------------------------

def test_schedule_to_steps_length():
    sched = random_schedule(ROOMS, TYPES, HORIZON_H, BLOCK_H, RNG)
    steps_per_block = int(round(BLOCK_H * 60 / config.SAMPLE_INTERVAL_MINUTES))
    n_steps = n_blocks(HORIZON_H, BLOCK_H) * steps_per_block
    upsampled = schedule_to_steps(sched, n_steps, BLOCK_H)
    for v in upsampled.values():
        assert len(v) == n_steps


def test_schedule_to_steps_constant_within_block():
    sched = {("salon", "shutter"): np.array([0.0, 1.0])}
    steps_per_block = int(round(BLOCK_H * 60 / config.SAMPLE_INTERVAL_MINUTES))
    n_steps = 2 * steps_per_block
    upsampled = schedule_to_steps(sched, n_steps, BLOCK_H)
    v = upsampled[("salon", "shutter")]
    # Premier bloc : tous les pas valent 0.0
    np.testing.assert_array_equal(v[:steps_per_block], np.zeros(steps_per_block))
    # Second bloc : tous les pas valent 1.0
    np.testing.assert_array_equal(v[steps_per_block:], np.ones(steps_per_block))


def test_schedule_to_steps_padding_when_too_short():
    sched = {("salon", "shutter"): np.array([1.0])}  # 1 créneau = 60 pas
    steps_per_block = int(round(BLOCK_H * 60 / config.SAMPLE_INTERVAL_MINUTES))
    n_steps = steps_per_block + 10  # demande plus que 1 créneau
    upsampled = schedule_to_steps(sched, n_steps, BLOCK_H)
    v = upsampled[("salon", "shutter")]
    assert len(v) == n_steps
    # Le padding doit répéter la dernière valeur (1.0)
    assert v[-1] == pytest.approx(1.0)


def test_schedule_to_steps_truncation_when_too_long():
    steps_per_block = int(round(BLOCK_H * 60 / config.SAMPLE_INTERVAL_MINUTES))
    sched = {("salon", "shutter"): np.array([0.0, 1.0, 0.0])}  # 3 créneaux
    n_steps = steps_per_block  # demande moins que 3 créneaux
    upsampled = schedule_to_steps(sched, n_steps, BLOCK_H)
    v = upsampled[("salon", "shutter")]
    assert len(v) == n_steps
