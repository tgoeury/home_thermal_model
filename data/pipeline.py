"""Alignement, fusion et fenêtrage des différentes sources de données.

Étapes :
  1. Charger chaque source (`data/sources/*`) → table large indexée par
     timestamp, à sa résolution native.
  2. Construire une grille temporelle commune à `config.SAMPLE_INTERVAL_MINUTES`
     couvrant l'union de toutes les sources (la météo peut s'étendre dans le
     futur via les prévisions : c'est volontaire, ça permet au modèle
     "limited" de planifier la journée).
  3. Reprojeter chaque source sur cette grille (interpolation temporelle pour
     les sources basse fréquence comme la météo, ffill/bfill pour les
     sources déjà natives) et ajouter les features solaires.
  4. Construire un `Dataset` PyTorch qui découpe la table fusionnée en
     fenêtres d'historique multi-résolution (cf. `config.RESOLUTION_SEGMENTS`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from data import solar
from data.sources.house_state import HouseStateSource
from data.sources.indoor_sensors import IndoorSensorSource
from data.sources.outdoor_sensors import OutdoorSensorSource
from data.sources.weather import WeatherSource


def load_raw_sources() -> dict[str, pd.DataFrame]:
    """Charge chaque source disponible. Une source vide (aucun fichier) est
    simplement absente du résultat — le reste du pipeline doit fonctionner
    avec un sous-ensemble de sources (dégradation gracieuse)."""
    sources = {
        "indoor": IndoorSensorSource().load(),
        "outdoor": OutdoorSensorSource().load(),
        "weather": WeatherSource().load(),
        "house_state": HouseStateSource().load(),
    }
    return {name: df for name, df in sources.items() if not df.empty}


def _build_grid(sources: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    starts = [df.index.min() for df in sources.values()]
    ends = [df.index.max() for df in sources.values()]
    freq = f"{config.SAMPLE_INTERVAL_MINUTES}min"
    return pd.date_range(min(starts), max(ends), freq=freq, tz="UTC")


def build_feature_table(sources: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Fusionne toutes les sources sur une grille temporelle commune.

    Les colonnes météo sont interpolées dans le temps (basse fréquence ->
    grille native). Les autres sources, déjà proches de la résolution
    native, sont alignées par ffill/bfill (tolère de petits trous de
    transmission). Les features solaires sont calculées directement sur la
    grille.
    """
    if sources is None:
        sources = load_raw_sources()
    if not sources:
        raise ValueError("Aucune source de données disponible dans data_raw/.")

    grid = _build_grid(sources)
    parts = []

    weather = sources.get("weather")
    if weather is not None:
        weather_on_grid = weather.reindex(weather.index.union(grid)).sort_index()
        kind_col = f"weather{config.COLUMN_SEP}kind"
        numeric_cols = [c for c in weather_on_grid.columns if c != kind_col]
        weather_on_grid[numeric_cols] = weather_on_grid[numeric_cols].interpolate(method="time")
        if kind_col in weather_on_grid.columns:
            weather_on_grid[kind_col] = weather_on_grid[kind_col].ffill().bfill()
        weather_on_grid = weather_on_grid.reindex(grid)
        weather_on_grid[numeric_cols] = weather_on_grid[numeric_cols].ffill().bfill()
        parts.append(weather_on_grid)

    for name in ("indoor", "outdoor", "house_state"):
        df = sources.get(name)
        if df is None:
            continue
        aligned = df.reindex(grid).ffill().bfill()
        parts.append(aligned)

    solar_features = solar.compute_solar_features(grid)
    parts.append(solar_features)

    table = pd.concat(parts, axis=1)
    table.index.name = "timestamp"
    return table


# ---------------------------------------------------------------------------
# Fenêtrage multi-résolution
# ---------------------------------------------------------------------------

def compute_window_offsets(
    resolution_segments: list[dict] | None = None,
    sample_interval_minutes: int = config.SAMPLE_INTERVAL_MINUTES,
) -> np.ndarray:
    """Décalages (en pas de temps) des points échantillonnés dans la fenêtre
    d'historique, triés du plus ancien au plus récent (le dernier vaut 0,
    c'est l'instant courant).

    Avec les segments par défaut : 24 points à 30 min (6h-18h), 24 points à
    10 min (2h-6h), 60 points à 2 min (0h-2h) -> 108 points pour 18h
    d'historique au lieu de 540.
    """
    if resolution_segments is None:
        resolution_segments = config.RESOLUTION_SEGMENTS

    minutes_ago: list[float] = []
    cursor = 0.0
    for seg in resolution_segments:
        duration = seg["duration_minutes"]
        resolution = seg["resolution_minutes"]
        n_points = int(round(duration / resolution))
        for k in range(n_points):
            minutes_ago.append(cursor + k * resolution)
        cursor += duration

    minutes_ago.sort(reverse=True)
    steps_ago = [round(m / sample_interval_minutes) for m in minutes_ago]
    return np.asarray(steps_ago, dtype=np.int64)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@dataclass
class FeatureStats:
    """Moyenne/écart-type par colonne, pour standardiser features et cibles."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return (array * self.std + self.mean).astype(np.float32)


def compute_stats(array: np.ndarray) -> FeatureStats:
    mean = np.nanmean(array, axis=0)
    std = np.nanstd(array, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return FeatureStats(mean.astype(np.float32), std.astype(np.float32))


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

def _select_columns(table: pd.DataFrame, prefixes: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    return [c for c in table.columns if c.startswith(prefixes) and c not in exclude]


class HouseDataset(Dataset):
    """Découpe la table fusionnée en fenêtres d'historique multi-résolution.

    Chaque échantillon fournit :
      - `x_limited` : (n_steps, n_limited_features) — météo + solaire +
        état volets/fenêtres (entrées du modèle "limited", toujours
        disponibles, y compris en planification via la météo prévisionnelle).
      - `x_outdoor` : (n_steps, n_outdoor_features) — capteurs des 4 façades
        (entrée additionnelle du modèle "full"), absent si la source
        `outdoor` n'est pas disponible.
      - `y` : (n_targets,) — températures/humidités intérieures à
        t + horizon (les "N features" à prédire, détectées dynamiquement).
    """

    def __init__(
        self,
        table: pd.DataFrame,
        history_hours: float = config.HISTORY_HOURS,
        horizon_steps: int = config.PREDICTION_HORIZON_STEPS,
    ) -> None:
        self.table = table
        self.offsets = compute_window_offsets(config.resolution_segments_for(history_hours))
        self.horizon_steps = horizon_steps

        sep = config.COLUMN_SEP
        self.target_columns = _select_columns(table, (f"indoor{sep}",))
        self.limited_columns = _select_columns(
            table, (f"weather{sep}", f"solar{sep}", f"house{sep}"), exclude=(f"weather{sep}kind",)
        )
        self.outdoor_columns = _select_columns(table, (f"outdoor{sep}",))

        if not self.target_columns:
            raise ValueError("Aucune colonne indoor__* trouvée : pas de cible à prédire.")
        if not self.limited_columns:
            raise ValueError("Aucune colonne weather__*/solar__*/house__* trouvée pour le modèle limited.")

        self._limited = table[self.limited_columns].to_numpy(dtype=np.float32)
        self._target = table[self.target_columns].to_numpy(dtype=np.float32)
        self._outdoor = table[self.outdoor_columns].to_numpy(dtype=np.float32) if self.outdoor_columns else None

        max_offset = int(self.offsets.max())
        n = len(table)
        candidates = np.arange(max_offset, n - horizon_steps)

        target_valid = ~np.isnan(self._target[candidates + horizon_steps]).any(axis=1)
        self.valid_indices = candidates[target_valid]

        self.limited_stats: FeatureStats | None = None
        self.outdoor_stats: FeatureStats | None = None
        self.target_stats: FeatureStats | None = None

    def chronological_split(self, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        """Découpe `valid_indices` en deux blocs temporels contigus (train/val).

        Découpage chronologique (pas aléatoire) pour éviter toute fuite
        d'information du futur vers le passé, essentiel en série temporelle.
        """
        split = int(len(self) * train_fraction)
        return np.arange(0, split), np.arange(split, len(self))

    def compute_normalization_stats(self, up_to_sample: int) -> dict[str, FeatureStats | None]:
        """Calcule moyenne/écart-type sur les lignes de la table couvertes par
        les `up_to_sample` premiers échantillons (typiquement la portion
        d'entraînement)."""
        split_row = int(self.valid_indices[up_to_sample - 1]) + 1
        return {
            "limited": compute_stats(self._limited[:split_row]),
            "target": compute_stats(self._target[:split_row]),
            "outdoor": compute_stats(self._outdoor[:split_row]) if self._outdoor is not None else None,
        }

    def apply_normalization(self, stats: dict[str, FeatureStats | None]) -> None:
        """Applique des stats de normalisation (calculées via
        `compute_normalization_stats`, éventuellement réutilisées d'un
        entraînement précédent pour rester cohérent avec un modèle gelé)."""
        self.limited_stats = stats["limited"]
        self.target_stats = stats["target"]
        self.outdoor_stats = stats.get("outdoor")

        self._limited = self.limited_stats.transform(self._limited)
        self._target = self.target_stats.transform(self._target)
        if self.outdoor_stats is not None and self._outdoor is not None:
            self._outdoor = self.outdoor_stats.transform(self._outdoor)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = int(self.valid_indices[idx])
        rows = i - self.offsets  # du plus ancien au plus récent

        sample = {
            "x_limited": torch.from_numpy(self._limited[rows]),
            "y": torch.from_numpy(self._target[i + self.horizon_steps]),
        }
        if self._outdoor is not None:
            sample["x_outdoor"] = torch.from_numpy(self._outdoor[rows])
        return sample

    @property
    def n_targets(self) -> int:
        return len(self.target_columns)

    @property
    def n_limited_features(self) -> int:
        return len(self.limited_columns)

    @property
    def n_outdoor_features(self) -> int:
        return len(self.outdoor_columns)
