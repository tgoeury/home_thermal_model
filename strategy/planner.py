"""Recherche aléatoire du planning volets/fenêtres minimisant le coût de
confort sur l'horizon de planification, via rollout du modèle "limited"."""

from __future__ import annotations

import numpy as np

import config
import evaluate
from data.pipeline import build_feature_table
from strategy import comfort
from strategy.format import schedule_to_plan
from strategy.rollout import PlanningContext
from strategy.schedule import n_blocks, random_schedule, schedule_to_steps


def plan(
    n_candidates: int = config.PLANNING_N_CANDIDATES,
    horizon_hours: float = config.PLANNING_HORIZON_HOURS,
    block_hours: float = config.PLANNING_BLOCK_HOURS,
    eval_step_minutes: float = config.PLANNING_EVAL_STEP_MINUTES,
    seed: int = config.RANDOM_SEED,
    comfort_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """`comfort_ranges` : `{ pièce: (t_min, t_max) }` — plage de confort par
    pièce. Les pièces absentes retombent sur `config.COMFORT_TEMP_MIN/MAX`."""
    sess, checkpoint = evaluate.load_limited_onnx()

    table = build_feature_table()
    ctx = PlanningContext(table, checkpoint, horizon_hours=horizon_hours, eval_step_minutes=eval_step_minutes)

    rooms = config.DEFAULT_HOUSE_STATE_ROOMS
    state_types = config.HOUSE_STATE_TYPES
    rng = np.random.default_rng(seed)

    best_schedule: dict[tuple[str, str], np.ndarray] | None = None
    best_cost = float("inf")
    for _ in range(n_candidates):
        candidate = random_schedule(rooms, state_types, horizon_hours, block_hours, rng)
        steps = schedule_to_steps(candidate, ctx.horizon_steps_count, block_hours)
        cost = ctx.evaluate(sess, steps, comfort_ranges)
        if cost < best_cost:
            best_cost = cost
            best_schedule = candidate

    # Si l'horizon évalué (limité par la météo prévisionnelle disponible) est
    # plus court que l'horizon demandé, on tronque le planning rendu aux
    # créneaux effectivement évalués.
    effective_horizon_hours = ctx.horizon_steps_count * config.SAMPLE_INTERVAL_MINUTES / 60.0
    effective_n_blocks = min(n_blocks(horizon_hours, block_hours), n_blocks(effective_horizon_hours, block_hours))
    if effective_n_blocks < n_blocks(horizon_hours, block_hours):
        best_schedule = {key: values[:effective_n_blocks] for key, values in best_schedule.items()}

    best_steps = schedule_to_steps(best_schedule, ctx.horizon_steps_count, block_hours)
    temperatures_by_room = ctx.predict_temperatures(sess, best_steps)
    reasons = comfort.block_reasons(
        temperatures_by_room, ctx.eval_rows, ctx.now_idx, effective_n_blocks, block_hours,
        comfort_ranges=comfort_ranges,
    )

    result = schedule_to_plan(
        best_schedule,
        rooms,
        state_types,
        now=ctx.now_timestamp,
        horizon_hours=effective_n_blocks * block_hours,
        block_hours=block_hours,
        reasons=reasons,
    )
    result["best_cost"] = best_cost
    result["n_candidates"] = n_candidates
    return result
