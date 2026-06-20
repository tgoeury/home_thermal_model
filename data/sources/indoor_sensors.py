"""Source : capteurs intérieurs température/humidité.

CSV attendus dans `data_raw/indoor/*.csv`, colonnes :
    timestamp, sensor_id, temperature, humidity

Chaque `sensor_id` distinct devient deux colonnes en sortie :
`indoor__<sensor_id>__temperature` et `indoor__<sensor_id>__humidity`.
Ce sont les features que les modèles doivent prédire (4 à N capteurs).
"""

from __future__ import annotations

from pathlib import Path

import config
from data.sources.base import CSVSource


class IndoorSensorSource(CSVSource):
    def __init__(self, directory: Path = config.INDOOR_DIR):
        super().__init__(
            directory=directory,
            prefix="indoor",
            measures=config.INDOOR_MEASURES,
            pivot_column="sensor_id",
        )
