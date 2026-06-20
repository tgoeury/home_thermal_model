"""Représentation et génération de plannings volets/fenêtres candidats.

Un planning candidat associe à chaque (pièce, type=volet|fenêtre) une suite
de "créneaux" de durée `block_hours`, chacun valant 0 (fermé) ou 1 (ouvert).
La granularité par créneaux (plutôt que par pas natif de 2 min) limite
naturellement la fréquence des changements d'état sans avoir besoin d'une
pénalité dédiée.
"""

from __future__ import annotations

import numpy as np

import config


def n_blocks(horizon_hours: float, block_hours: float) -> int:
    return int(round(horizon_hours / block_hours))


def random_schedule(
    rooms: list[str],
    state_types: list[str],
    horizon_hours: float,
    block_hours: float,
    rng: np.random.Generator,
) -> dict[tuple[str, str], np.ndarray]:
    """Tire un planning candidat : pour chaque (pièce, type), une suite de
    `n_blocks(horizon_hours, block_hours)` valeurs 0/1 indépendantes."""
    n = n_blocks(horizon_hours, block_hours)
    return {
        (room, state_type): rng.integers(0, 2, size=n).astype(np.float32)
        for room in rooms
        for state_type in state_types
    }


def schedule_to_steps(
    schedule: dict[tuple[str, str], np.ndarray],
    n_steps: int,
    block_hours: float,
) -> dict[tuple[str, str], np.ndarray]:
    """Sur-échantillonne chaque planning par créneaux vers le pas natif
    (`config.SAMPLE_INTERVAL_MINUTES`), tronqué/complété à `n_steps`."""
    steps_per_block = int(round(block_hours * 60 / config.SAMPLE_INTERVAL_MINUTES))

    out: dict[tuple[str, str], np.ndarray] = {}
    for key, blocks in schedule.items():
        steps = np.repeat(blocks, steps_per_block)
        if len(steps) < n_steps:
            pad = np.full(n_steps - len(steps), steps[-1], dtype=steps.dtype)
            steps = np.concatenate([steps, pad])
        out[key] = steps[:n_steps]
    return out
