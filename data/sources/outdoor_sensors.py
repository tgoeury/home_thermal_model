"""Source : capteurs extérieurs (température/humidité/luminosité par façade).

CSV attendus dans `data_raw/outdoor/*.csv`, colonnes :
    timestamp, face, temperature, humidity, luminosity

`face` ∈ {N, E, S, W} (cf. config.HOUSE_FACES). Chaque façade distincte
devient un groupe de colonnes `outdoor__<face>__<measure>`.
"""

from __future__ import annotations

from pathlib import Path

import config
from data.sources.base import CSVSource


class OutdoorSensorSource(CSVSource):
    def __init__(self, directory: Path = config.OUTDOOR_DIR):
        super().__init__(
            directory=directory,
            prefix="outdoor",
            measures=config.OUTDOOR_MEASURES,
            pivot_column="face",
        )
