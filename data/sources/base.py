"""Loader CSV générique et "pluggable".

Principe : chaque source de données est un répertoire (`data_raw/<source>/`)
contenant un nombre arbitraire de fichiers CSV au format long. Ajouter,
retirer ou faire grossir un fichier ne nécessite aucune modification de
code. De même, une nouvelle valeur de la colonne pivot (nouveau capteur,
nouvelle façade, nouvelle pièce) crée automatiquement de nouvelles colonnes
dans la table large produite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

import config


class CSVSource:
    """Charge un ensemble de CSV "longs" et les met en forme "large".

    Format CSV attendu :
      - `timestamp` : horodatage parseable (ISO 8601 recommandé)
      - éventuellement une colonne pivot (ex: `sensor_id`, `face`) dont
        chaque valeur distincte devient un groupe de colonnes
      - une ou plusieurs colonnes de mesure (ex: `temperature`, `humidity`)

    Convention de nommage des colonnes de sortie :
      - avec pivot   : `<prefix>__<pivot_value>__<measure>`
      - sans pivot   : `<prefix>__<measure>`
    """

    def __init__(
        self,
        directory: Path,
        prefix: str,
        measures: Iterable[str],
        pivot_column: Optional[str] = None,
        file_pattern: str = "*.csv",
        timestamp_column: str = "timestamp",
    ) -> None:
        self.directory = Path(directory)
        self.prefix = prefix
        self.measures = list(measures)
        self.pivot_column = pivot_column
        self.file_pattern = file_pattern
        self.timestamp_column = timestamp_column

    def list_files(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(self.directory.glob(self.file_pattern))

    def read_raw(self) -> pd.DataFrame:
        """Concatène tous les fichiers du répertoire, sans mise en forme."""
        files = self.list_files()
        if not files:
            return pd.DataFrame()

        frames = [pd.read_csv(f) for f in files]
        raw = pd.concat(frames, ignore_index=True)
        raw[self.timestamp_column] = pd.to_datetime(raw[self.timestamp_column], utc=True)
        return raw

    def load(self) -> pd.DataFrame:
        """Retourne une table large indexée par timestamp (UTC, triée)."""
        raw = self.read_raw()
        if raw.empty:
            return pd.DataFrame()

        sep = config.COLUMN_SEP
        raw = raw.sort_values(self.timestamp_column)

        if self.pivot_column:
            pivoted = raw.pivot_table(
                index=self.timestamp_column,
                columns=self.pivot_column,
                values=self.measures,
                aggfunc="last",
            )
            pivoted.columns = [
                f"{self.prefix}{sep}{pivot_value}{sep}{measure}"
                for measure, pivot_value in pivoted.columns
            ]
        else:
            pivoted = raw.drop_duplicates(subset=self.timestamp_column, keep="last")
            pivoted = pivoted.set_index(self.timestamp_column)[self.measures]
            pivoted.columns = [f"{self.prefix}{sep}{measure}" for measure in self.measures]

        pivoted = pivoted.sort_index()
        pivoted.index.name = "timestamp"
        return pivoted
