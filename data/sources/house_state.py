"""Source : état des volets et fenêtres (entrées pilotables).

CSV attendus dans `data_raw/house_state/*.csv`, colonnes :
    timestamp, room, type, state

`type` ∈ {shutter, window} (cf. config.HOUSE_STATE_TYPES), `state` ∈ {0, 1}
(fermé/ouvert). Chaque combinaison (room, type) devient une colonne
`house__<room>__<type>`.

Ces données sont simulées pour l'instant (pas de domotique existante), mais
le format est identique à ce qu'une vraie source produirait : aucune
modification de code ne sera nécessaire pour basculer sur des données
réelles.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from data.sources.base import CSVSource


class HouseStateSource(CSVSource):
    def __init__(self, directory: Path = config.HOUSE_STATE_DIR):
        super().__init__(
            directory=directory,
            prefix="house",
            measures=["state"],
            pivot_column=None,
        )

    def load(self) -> pd.DataFrame:
        raw = self.read_raw()
        if raw.empty:
            return pd.DataFrame()

        sep = config.COLUMN_SEP
        raw = raw.sort_values(self.timestamp_column)

        pivoted = raw.pivot_table(
            index=self.timestamp_column,
            columns=["room", "type"],
            values="state",
            aggfunc="last",
        )
        pivoted.columns = [f"{self.prefix}{sep}{room}{sep}{type_}" for room, type_ in pivoted.columns]
        pivoted = pivoted.sort_index()
        pivoted.index.name = "timestamp"
        return pivoted
