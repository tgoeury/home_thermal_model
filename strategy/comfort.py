"""Coût de confort pour un planning candidat.

La cible n'est pas une température précise mais une plage de confort
(``config.COMFORT_TEMP_MIN``/``COMFORT_TEMP_MAX``) : seuls les dépassements
de cette plage sont pénalisés.
"""

from __future__ import annotations

import numpy as np

import config

# Raisons possibles pour l'état volets/fenêtres d'un créneau, en fonction de
# la position de la température prédite par rapport à la plage de confort.
REASON_MAINTAIN = "maintenir"
REASON_COOL = "refroidir"
REASON_WARM = "rechauffer"


def room_bounds(room: str, comfort_ranges: dict[str, tuple[float, float]] | None) -> tuple[float, float]:
    """Plage de confort (min, max) pour `room`.

    Si `comfort_ranges` est fourni et contient `room`, utilise cette plage ;
    sinon retombe sur les bornes globales `config.COMFORT_TEMP_MIN/MAX`.
    """
    if comfort_ranges and room in comfort_ranges:
        t_min, t_max = comfort_ranges[room]
        return float(t_min), float(t_max)
    return config.COMFORT_TEMP_MIN, config.COMFORT_TEMP_MAX


def comfort_cost(
    temperatures: np.ndarray,
    rooms: list[str] | None = None,
    comfort_ranges: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Somme des dépassements (en °C) de la plage de confort.

    `temperatures` : tableau (n_points, n_pièces) de températures prédites,
    en unités réelles. `rooms` donne le nom de la pièce de chaque colonne ;
    si fourni avec `comfort_ranges`, chaque pièce est évaluée contre sa
    propre plage de confort plutôt que la plage globale.
    """
    if rooms is not None:
        t_min = np.array([room_bounds(room, comfort_ranges)[0] for room in rooms])
        t_max = np.array([room_bounds(room, comfort_ranges)[1] for room in rooms])
    else:
        t_min, t_max = config.COMFORT_TEMP_MIN, config.COMFORT_TEMP_MAX

    below = np.clip(t_min - temperatures, 0.0, None)
    above = np.clip(temperatures - t_max, 0.0, None)
    return float((below + above).sum())


def block_reasons(
    temperatures_by_room: dict[str, np.ndarray],
    eval_rows: np.ndarray,
    now_idx: int,
    n_blocks: int,
    block_hours: float,
    sample_interval_minutes: float = config.SAMPLE_INTERVAL_MINUTES,
    comfort_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, list[str]]:
    """Pour chaque pièce et chaque créneau, explique l'état volet/fenêtre
    choisi par la température prédite sur ce créneau, par rapport à la plage
    de confort de la pièce (`comfort_ranges`, ou les bornes globales par
    défaut) :

    - ``REASON_WARM`` si la pièce serait sous la plage de confort (il faut
      réchauffer) ;
    - ``REASON_COOL`` si elle serait au-dessus (il faut refroidir) ;
    - ``REASON_MAINTAIN`` sinon (le planning sert juste à rester dans la
      plage).
    """
    block_minutes = block_hours * 60.0
    minutes_from_now = (eval_rows - now_idx) * sample_interval_minutes
    block_indices = np.minimum((minutes_from_now // block_minutes).astype(int), n_blocks - 1)

    reasons: dict[str, list[str]] = {}
    for room, temperatures in temperatures_by_room.items():
        t_min, t_max = room_bounds(room, comfort_ranges)
        room_reasons = []
        for k in range(n_blocks):
            block_temps = temperatures[block_indices == k]
            if block_temps.size == 0:
                room_reasons.append(REASON_MAINTAIN)
            elif (block_temps < t_min).any():
                room_reasons.append(REASON_WARM)
            elif (block_temps > t_max).any():
                room_reasons.append(REASON_COOL)
            else:
                room_reasons.append(REASON_MAINTAIN)
        reasons[room] = room_reasons

    return reasons
