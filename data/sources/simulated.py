"""Générateurs de données simulées, au même format que les sources réelles.

Tant que certains capteurs (façades extérieures, météo haute fréquence,
volets/fenêtres connectés) ne sont pas encore en place, ce module produit
des CSV directement dans `data_raw/<source>/`, avec exactement le même
schéma que les sources réelles (cf. `data/sources/{indoor,outdoor,weather,
house_state}.py`). Le jour où un vrai capteur est branché, il suffit
d'ajouter son fichier CSV à côté (ou de retirer le fichier simulé
correspondant) : aucun changement de code n'est nécessaire.

Le modèle physique "jouet" utilisé pour générer les températures intérieures
(perte thermique vers l'extérieur, gain solaire direct si volet ouvert,
ventilation si fenêtre ouverte) sert aussi de base de comparaison : un modèle
appris qui ne retrouve pas au moins ces effets sur ce jeu de données aurait
un problème.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from data import solar

# ---------------------------------------------------------------------------
# Constantes du modèle thermique jouet (par pas de SAMPLE_INTERVAL_MINUTES)
# ---------------------------------------------------------------------------

THERMAL_LOSS_RATE = 0.005      # fraction de l'écart T_in - T_out perdue par pas (isolation, cte de temps ~6-7h)
SOLAR_GAIN_FACTOR = 4e-5       # °C gagnés par pas, par unité d'irradiance (W/m2) * exposition, volet ouvert
VENTILATION_RATE = 0.05        # fraction de l'écart T_in - T_out équilibrée par pas si fenêtre ouverte
HUMIDITY_VENT_RATE = 0.03      # idem pour l'humidité
INDOOR_TEMP_NOISE_STD = 0.02
INDOOR_HUM_NOISE_STD = 0.1


def _time_index(start: pd.Timestamp, end: pd.Timestamp, freq_minutes: float) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=f"{freq_minutes}min", tz="UTC")


def _interp_to(target_idx: pd.DatetimeIndex, source_idx: pd.DatetimeIndex, values: np.ndarray) -> np.ndarray:
    """Interpolation linéaire d'une série temporelle sur une nouvelle grille."""
    target_num = target_idx.to_numpy(dtype="int64").astype(float)
    source_num = source_idx.to_numpy(dtype="int64").astype(float)
    return np.interp(target_num, source_num, values)


# ---------------------------------------------------------------------------
# Météo (résolution "API", basse fréquence, observée + prévision)
# ---------------------------------------------------------------------------

def generate_weather(
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_horizon_hours: float = 24.0,
    freq_minutes: float = 30.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Données météo au format long (timestamp, kind, <mesures>).

    `kind` vaut "observed" jusqu'à `end`, puis "forecast" sur l'horizon
    `forecast_horizon_hours` (simule des prévisions disponibles à l'avance,
    utilisées par le modèle "limited" pour la planification journalière).
    """
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    full_end = end + pd.Timedelta(hours=forecast_horizon_hours)
    idx = _time_index(start, full_end, freq_minutes)

    day_frac = (idx.hour + idx.minute / 60.0) / 24.0
    day_of_year = idx.dayofyear.to_numpy(dtype=float)

    seasonal = 10.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    daily = 5.0 * np.sin(2 * np.pi * (day_frac - 0.3))
    noise = rng.normal(0, 0.5, size=len(idx))
    outdoor_temperature = 12.0 + seasonal + daily + noise

    pos = solar.solar_position(idx, config.LATITUDE, config.LONGITUDE)
    elevation = pos["elevation_deg"].to_numpy()

    cloud_cover = np.clip(rng.normal(0.3, 0.2, size=len(idx)), 0.0, 1.0)
    cloud_cover = pd.Series(cloud_cover).rolling(6, min_periods=1, center=True).mean().to_numpy()

    clear_sky_irradiance = np.clip(np.sin(np.radians(elevation)), 0.0, None) * 1000.0
    solar_irradiance = clear_sky_irradiance * (1.0 - 0.75 * cloud_cover)

    kind = np.where(idx <= end, "observed", "forecast")

    return pd.DataFrame(
        {
            "timestamp": idx,
            "kind": kind,
            "outdoor_temperature": outdoor_temperature,
            "solar_irradiance": solar_irradiance,
            "cloud_cover": cloud_cover,
        }
    )


# ---------------------------------------------------------------------------
# Capteurs extérieurs (4 façades, résolution native)
# ---------------------------------------------------------------------------

def generate_outdoor_sensors(
    start: pd.Timestamp,
    end: pd.Timestamp,
    weather_df: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Capteurs extérieurs par façade au format long (timestamp, face, ...)."""
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    idx = _time_index(start, end, config.SAMPLE_INTERVAL_MINUTES)

    weather_idx = pd.DatetimeIndex(weather_df["timestamp"])
    outdoor_temp = _interp_to(idx, weather_idx, weather_df["outdoor_temperature"].to_numpy())
    solar_irr = _interp_to(idx, weather_idx, weather_df["solar_irradiance"].to_numpy())

    pos = solar.solar_position(idx, config.LATITUDE, config.LONGITUDE)
    elevation = pos["elevation_deg"].to_numpy()
    azimuth = pos["azimuth_deg"].to_numpy()

    frames = []
    for face, face_azimuth in config.HOUSE_FACES.items():
        exposure = solar.face_exposure(elevation, azimuth, face_azimuth)

        temperature = outdoor_temp + 2.0 * exposure * (solar_irr / 1000.0) + rng.normal(0, 0.2, size=len(idx))
        humidity = np.clip(60.0 - 0.6 * (temperature - 12.0) + rng.normal(0, 1.0, size=len(idx)), 20.0, 95.0)
        luminosity = np.clip(
            solar_irr * (0.5 + 0.5 * exposure) * 1.2 + rng.normal(0, 20.0, size=len(idx)), 0.0, None
        )

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": idx,
                    "face": face,
                    "temperature": temperature,
                    "humidity": humidity,
                    "luminosity": luminosity,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# État volets / fenêtres (entrées pilotables, résolution native)
# ---------------------------------------------------------------------------

def generate_house_state(
    start: pd.Timestamp,
    end: pd.Timestamp,
    rooms: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """État volets/fenêtres au format long (timestamp, room, type, state)."""
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    rooms = rooms or config.DEFAULT_HOUSE_STATE_ROOMS
    idx = _time_index(start, end, config.SAMPLE_INTERVAL_MINUTES)

    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0

    frames = []
    for room in rooms:
        # Volets : ouverts en journée (7h-22h), fermés la nuit, avec un peu
        # de bruit sur les horaires d'un jour à l'autre.
        offset = rng.normal(0, 0.5)
        shutter_open = ((hour >= 7 + offset) & (hour <= 22 + offset)).astype(int)

        # Fenêtres : majoritairement fermées, ouvertures ponctuelles l'après-midi
        # (aération), tirées aléatoirement par "journée".
        n_days = max(1, int(np.ceil((end - start) / pd.Timedelta(days=1))) + 1)
        window_open = np.zeros(len(idx), dtype=int)
        for day in range(n_days):
            if rng.random() < 0.6:
                day_start = start.normalize() + pd.Timedelta(days=day)
                open_hour = rng.uniform(10, 18)
                duration_h = rng.uniform(0.25, 1.5)
                mask = (idx >= day_start + pd.Timedelta(hours=open_hour)) & (
                    idx < day_start + pd.Timedelta(hours=open_hour + duration_h)
                )
                window_open[mask] = 1

        frames.append(pd.DataFrame({"timestamp": idx, "room": room, "type": "shutter", "state": shutter_open}))
        frames.append(pd.DataFrame({"timestamp": idx, "room": room, "type": "window", "state": window_open}))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Capteurs intérieurs (résolution native, simulés via un modèle thermique jouet)
# ---------------------------------------------------------------------------

def generate_indoor_sensors(
    start: pd.Timestamp,
    end: pd.Timestamp,
    weather_df: pd.DataFrame,
    outdoor_df: pd.DataFrame,
    house_state_df: pd.DataFrame,
    rooms: list[str] | None = None,
    room_faces: dict[str, str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Capteurs intérieurs au format long (timestamp, sensor_id, ...).

    Modèle jouet, par pas de temps et par pièce :
      T_in[t+1] = T_in[t]
                  - THERMAL_LOSS_RATE   * (T_in[t] - T_out[t])
                  + SOLAR_GAIN_FACTOR   * irradiance[t] * exposure_face[t] * shutter_open[t]
                  + VENTILATION_RATE    * (T_out[t] - T_in[t]) * window_open[t]
                  + bruit
    """
    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    rooms = rooms or config.DEFAULT_INDOOR_ROOMS
    room_faces = room_faces or config.DEFAULT_ROOM_FACES
    idx = _time_index(start, end, config.SAMPLE_INTERVAL_MINUTES)
    n = len(idx)

    weather_idx = pd.DatetimeIndex(weather_df["timestamp"])
    outdoor_temp = _interp_to(idx, weather_idx, weather_df["outdoor_temperature"].to_numpy())
    solar_irr = _interp_to(idx, weather_idx, weather_df["solar_irradiance"].to_numpy())

    pos = solar.solar_position(idx, config.LATITUDE, config.LONGITUDE)
    elevation = pos["elevation_deg"].to_numpy()
    azimuth = pos["azimuth_deg"].to_numpy()

    outdoor_humidity = (
        outdoor_df.loc[outdoor_df["face"] == next(iter(config.HOUSE_FACES)), "humidity"]
        .to_numpy()
    )

    frames = []
    for room in rooms:
        face = room_faces.get(room, "S")
        exposure = solar.face_exposure(elevation, azimuth, config.HOUSE_FACES[face])

        shutter = house_state_df.query("room == @room and type == 'shutter'").set_index("timestamp")["state"]
        window = house_state_df.query("room == @room and type == 'window'").set_index("timestamp")["state"]
        shutter_open = shutter.reindex(idx).to_numpy(dtype=float)
        window_open = window.reindex(idx).to_numpy(dtype=float)

        temperature = np.empty(n)
        humidity = np.empty(n)
        temperature[0] = outdoor_temp[0] + rng.normal(0, 0.5)
        humidity[0] = 50.0 + rng.normal(0, 2.0)

        temp_noise = rng.normal(0, INDOOR_TEMP_NOISE_STD, size=n)
        hum_noise = rng.normal(0, INDOOR_HUM_NOISE_STD, size=n)

        for t in range(n - 1):
            loss = THERMAL_LOSS_RATE * (temperature[t] - outdoor_temp[t])
            gain = SOLAR_GAIN_FACTOR * solar_irr[t] * exposure[t] * shutter_open[t]
            vent = VENTILATION_RATE * (outdoor_temp[t] - temperature[t]) * window_open[t]
            temperature[t + 1] = temperature[t] - loss + gain + vent + temp_noise[t]

            hum_vent = HUMIDITY_VENT_RATE * (outdoor_humidity[t] - humidity[t]) * window_open[t]
            humidity[t + 1] = np.clip(humidity[t] + hum_vent + hum_noise[t], 20.0, 80.0)

        frames.append(
            pd.DataFrame({"timestamp": idx, "sensor_id": room, "temperature": temperature, "humidity": humidity})
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_dataset(
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_horizon_hours: float = 24.0,
    seed: int = config.RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Génère un jeu de données cohérent (météo, extérieur, volets/fenêtres, intérieur)."""
    rng = np.random.default_rng(seed)

    weather_df = generate_weather(start, end, forecast_horizon_hours=forecast_horizon_hours, rng=rng)
    outdoor_df = generate_outdoor_sensors(start, end, weather_df, rng=rng)
    house_state_df = generate_house_state(start, end, rng=rng)
    indoor_df = generate_indoor_sensors(start, end, weather_df, outdoor_df, house_state_df, rng=rng)

    return {
        "weather": weather_df,
        "outdoor": outdoor_df,
        "house_state": house_state_df,
        "indoor": indoor_df,
    }


def write_dataset(dataset: dict[str, pd.DataFrame], filename: str = "simulated.csv") -> None:
    """Écrit chaque table dans `data_raw/<source>/<filename>`."""
    target_dirs = {
        "weather": config.WEATHER_DIR,
        "outdoor": config.OUTDOOR_DIR,
        "house_state": config.HOUSE_STATE_DIR,
        "indoor": config.INDOOR_DIR,
    }
    for source, df in dataset.items():
        directory = target_dirs[source]
        directory.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        out["timestamp"] = pd.DatetimeIndex(out["timestamp"]).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(directory / filename, index=False)


def generate_and_write(
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_horizon_hours: float = 24.0,
    seed: int = config.RANDOM_SEED,
    filename: str = "simulated.csv",
) -> dict[str, pd.DataFrame]:
    dataset = generate_dataset(start, end, forecast_horizon_hours=forecast_horizon_hours, seed=seed)
    write_dataset(dataset, filename=filename)
    return dataset
