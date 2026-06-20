"""Évaluation d'un planning candidat par rollout du modèle "limited".

Le modèle "limited" ne prend pas la température intérieure en entrée : sa
prédiction à l'instant t+1 ne dépend que de la fenêtre météo/solaire/
volets-fenêtres se terminant à t. Il n'y a donc pas besoin de rollout
auto-régressif — pour un planning candidat donné (qui ne modifie que les
colonnes ``house__*`` sur l'horizon futur), toutes les fenêtres de
prédiction de l'horizon peuvent être construites et évaluées en un seul
batch.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pandas as pd

import config
from data.pipeline import _select_columns, compute_window_offsets
from strategy import comfort


def find_now_index(table: pd.DataFrame) -> int:
    """Index de la dernière ligne météo "observée" : sépare l'historique
    connu du futur (prévisions météo + planning volets/fenêtres candidat)."""
    sep = config.COLUMN_SEP
    kind_col = f"weather{sep}kind"
    observed = (table[kind_col] == "observed").to_numpy()
    if not observed.any():
        raise ValueError("Aucune donnée météo 'observed' dans la table : impossible de situer 'maintenant'.")
    return int(np.flatnonzero(observed)[-1])


class PlanningContext:
    """Pré-calcule ce qui est commun à l'évaluation de tous les plannings
    candidats : fenêtre de données utile, offsets, indices des colonnes
    volets/fenêtres et points d'évaluation du coût de confort."""

    def __init__(
        self,
        table: pd.DataFrame,
        checkpoint: dict,
        horizon_hours: float = config.PLANNING_HORIZON_HOURS,
        eval_step_minutes: float = config.PLANNING_EVAL_STEP_MINUTES,
    ) -> None:
        sep = config.COLUMN_SEP

        self.limited_columns: list[str] = checkpoint["limited_columns"]
        table_limited_columns = _select_columns(
            table, (f"weather{sep}", f"solar{sep}", f"house{sep}"), exclude=(f"weather{sep}kind",)
        )
        if table_limited_columns != self.limited_columns:
            raise ValueError(
                "Les colonnes 'limited' de la table actuelle ne correspondent pas à celles du "
                "modèle entraîné (capteurs ajoutés/supprimés ?). Réentraînez le modèle 'limited'."
            )

        self.offsets = compute_window_offsets(config.resolution_segments_for(checkpoint["history_hours"]))
        self.horizon_steps = checkpoint["horizon_steps"]
        self.limited_stats = checkpoint["limited_stats"]
        self.target_stats = checkpoint["target_stats"]

        self.now_idx = find_now_index(table)

        max_offset = int(self.offsets.max())
        if self.now_idx - max_offset < 0:
            raise ValueError("Pas assez d'historique avant 'maintenant' pour construire la fenêtre du modèle.")

        max_future_steps = len(table) - 1 - self.now_idx
        requested_steps = int(round(horizon_hours * 60.0 / config.SAMPLE_INTERVAL_MINUTES))
        self.horizon_steps_count = min(requested_steps, max_future_steps)
        if self.horizon_steps_count <= 0:
            raise ValueError(
                "Aucune donnée future (prévisions météo) disponible après 'maintenant' : "
                "impossible de planifier."
            )

        self.lo = self.now_idx - max_offset
        self.hi = self.now_idx + self.horizon_steps_count
        self.window_array = table[self.limited_columns].to_numpy(dtype=np.float32)[self.lo : self.hi + 1]

        # Premier pas "futur" (juste après maintenant), en index local au sein de window_array.
        self.future_start_local = self.now_idx - self.lo + 1

        # Points d'évaluation du coût de confort (indices de ligne cibles, absolus).
        eval_step = max(1, int(round(eval_step_minutes / config.SAMPLE_INTERVAL_MINUTES)))
        eval_rows = np.arange(self.now_idx + eval_step, self.hi + 1, eval_step)
        if len(eval_rows) == 0:
            eval_rows = np.asarray([self.hi])

        i_array = eval_rows - self.horizon_steps
        rows_matrix = i_array[:, None] - self.offsets[None, :]
        self.local_rows_matrix = rows_matrix - self.lo
        if self.local_rows_matrix.min() < 0 or self.local_rows_matrix.max() > self.hi - self.lo:
            raise ValueError("Fenêtre de planification mal alignée (index hors limites).")

        # Indices des colonnes house__<room>__<type> au sein de limited_columns.
        self.house_state_cols: dict[tuple[str, str], int] = {}
        for room in config.DEFAULT_HOUSE_STATE_ROOMS:
            for state_type in config.HOUSE_STATE_TYPES:
                col_name = f"house{sep}{room}{sep}{state_type}"
                if col_name in self.limited_columns:
                    self.house_state_cols[(room, state_type)] = self.limited_columns.index(col_name)

        # Indices des cibles de température au sein de target_columns, et
        # pièce correspondante (pour le diagnostic chaud/froid par pièce).
        target_columns: list[str] = checkpoint["target_columns"]
        self.temperature_target_indices = [
            i for i, c in enumerate(target_columns) if c.endswith(f"{sep}temperature")
        ]
        self.temperature_target_rooms = [
            target_columns[i].split(sep)[1] for i in self.temperature_target_indices
        ]

        self.eval_rows = eval_rows
        self.now_timestamp = table.index[self.now_idx]

    def _predict(self, sess: ort.InferenceSession, schedule_steps: dict[tuple[str, str], np.ndarray]) -> np.ndarray:
        """Prédictions dénormalisées (n_eval_rows, n_targets) pour un planning
        candidat (déjà sur-échantillonné au pas natif, via
        `strategy.schedule.schedule_to_steps`)."""
        arr = self.window_array.copy()
        end = self.future_start_local + self.horizon_steps_count
        for key, col in self.house_state_cols.items():
            arr[self.future_start_local : end, col] = schedule_steps[key]

        arr_norm = self.limited_stats.transform(arr).astype(np.float32)
        batch = arr_norm[self.local_rows_matrix]

        pred = sess.run(["predictions"], {"x_limited": batch})[0]

        return self.target_stats.inverse_transform(pred)

    def evaluate(
        self,
        sess: ort.InferenceSession,
        schedule_steps: dict[tuple[str, str], np.ndarray],
        comfort_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> float:
        """Coût de confort pour un planning candidat (plages par pièce si
        `comfort_ranges` est fourni, sinon plage globale `config.COMFORT_TEMP_*`)."""
        pred_real = self._predict(sess, schedule_steps)
        temperatures = pred_real[:, self.temperature_target_indices]
        return comfort.comfort_cost(temperatures, self.temperature_target_rooms, comfort_ranges)

    def predict_temperatures(
        self, sess: ort.InferenceSession, schedule_steps: dict[tuple[str, str], np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Températures prédites par pièce (n_eval_rows,) pour un planning
        candidat, en unités réelles."""
        pred_real = self._predict(sess, schedule_steps)
        return {
            room: pred_real[:, idx]
            for room, idx in zip(self.temperature_target_rooms, self.temperature_target_indices)
        }
