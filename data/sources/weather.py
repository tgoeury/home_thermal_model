"""Source : données météo (observées et prévisions).

CSV attendus dans `data_raw/weather/*.csv`, colonnes :
    timestamp, kind, outdoor_temperature, solar_irradiance, cloud_cover, ...

`kind` ∈ {observed, forecast}. Quand les deux sont disponibles pour un même
timestamp (une prévision passée et la mesure réelle qui a suivi), la valeur
`observed` est prioritaire. Les timestamps futurs n'ont en général qu'une
valeur `forecast` : c'est elle qui est utilisée par le modèle "limited" pour
planifier la journée à venir.

La colonne `weather__kind` est conservée en sortie pour que le pipeline
puisse, si besoin, distinguer passé observé / futur prévu.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from data.sources.base import CSVSource


class WeatherSource(CSVSource):
    # Priorité de fusion quand un timestamp a plusieurs lignes : la valeur
    # observée écrase la prévision correspondante.
    KIND_PRIORITY = {"forecast": 0, "observed": 1}

    def __init__(self, directory: Path = config.WEATHER_DIR):
        super().__init__(
            directory=directory,
            prefix="weather",
            measures=config.WEATHER_MEASURES,
            pivot_column=None,
        )

    def load(self) -> pd.DataFrame:
        raw = self.read_raw()
        if raw.empty:
            return pd.DataFrame()

        sep = config.COLUMN_SEP

        if "kind" not in raw.columns:
            raw["kind"] = "observed"
        raw["_priority"] = raw["kind"].map(self.KIND_PRIORITY).fillna(0)

        raw = raw.sort_values([self.timestamp_column, "_priority"])
        raw = raw.drop_duplicates(subset=self.timestamp_column, keep="last")
        raw = raw.set_index(self.timestamp_column).sort_index()

        out = raw[self.measures].copy()
        out.columns = [f"{self.prefix}{sep}{measure}" for measure in self.measures]
        out[f"{self.prefix}{sep}kind"] = raw["kind"]
        out.index.name = "timestamp"
        return out
