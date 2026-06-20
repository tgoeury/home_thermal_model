"""Mise en forme du planning recommandé : un planning journalier par pièce,
sous forme d'intervalles volet/fenêtre (créneaux consécutifs identiques
fusionnés), exporté en JSON et/ou CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from strategy.schedule import n_blocks

_STATE_LABELS = {0: "closed", 1: "open"}


def schedule_to_plan(
    schedule: dict[tuple[str, str], "np.ndarray"],
    rooms: list[str],
    state_types: list[str],
    now: pd.Timestamp,
    horizon_hours: float,
    block_hours: float,
    reasons: dict[str, list[str]] | None = None,
) -> dict:
    """Convertit un planning par créneaux (sortie de
    `strategy.schedule.random_schedule`) en JSON par pièce.

    Les créneaux consécutifs ayant le même état pour tous les `state_types`
    (et, si fourni, la même `reason`) sont fusionnés en un seul intervalle
    ``{"from": ..., "to": ..., ..., "reason": ...}``.

    `reasons` : par pièce, une raison par créneau (cf.
    `strategy.comfort.block_reasons`) expliquant si l'état volet/fenêtre du
    créneau sert à maintenir la pièce dans la plage de confort, à la
    refroidir, ou à la réchauffer.
    """
    n = n_blocks(horizon_hours, block_hours)
    block_delta = pd.Timedelta(hours=block_hours)

    rooms_plan: dict[str, list[dict]] = {}
    for room in rooms:
        states = [tuple(schedule[(room, state_type)][k] for state_type in state_types) for k in range(n)]
        room_reasons = reasons[room] if reasons is not None else None

        def merge_key(k: int) -> tuple:
            return (states[k], room_reasons[k]) if room_reasons is not None else (states[k],)

        intervals: list[dict] = []
        block_start = 0
        for k in range(1, n + 1):
            if k < n and merge_key(k) == merge_key(block_start):
                continue
            interval = {
                "from": (now + block_start * block_delta).isoformat(),
                "to": (now + k * block_delta).isoformat(),
            }
            for state_type, value in zip(state_types, states[block_start]):
                interval[state_type] = _STATE_LABELS[int(value)]
            if room_reasons is not None:
                interval["reason"] = room_reasons[block_start]
            intervals.append(interval)
            block_start = k

        rooms_plan[room] = intervals

    return {
        "generated_at": now.isoformat(),
        "horizon_hours": horizon_hours,
        "rooms": rooms_plan,
    }


def write_plan_csv(plan: dict, path: Path, state_types: list[str]) -> None:
    """Écrit le planning au format CSV : une ligne par intervalle, colonnes
    `room, from, to`, une colonne par type d'état (`shutter`, `window`), puis
    toute colonne additionnelle présente dans les intervalles (ex.
    `reason`)."""
    all_intervals = [interval for intervals in plan["rooms"].values() for interval in intervals]
    extra_fields = [k for k in (all_intervals[0] if all_intervals else {}) if k not in ("from", "to", *state_types)]
    fieldnames = ["room", "from", "to", *state_types, *extra_fields]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for room, intervals in plan["rooms"].items():
            for interval in intervals:
                writer.writerow({"room": room, **interval})
