"""Calcul de la position du soleil et de l'exposition des façades.

Implémentation vectorisée (numpy/pandas) de l'algorithme solaire simplifié
de la NOAA (https://gml.noaa.gov/grad/solcalc/solareqns.PDF). Volontairement
indépendante de toute librairie externe (astral, pvlib, ...) pour rester
légère sur un environnement de prod réduit, tout en restant précise à
quelques dixièmes de degré, largement suffisant pour estimer l'exposition
des façades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def solar_position(timestamps: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.DataFrame:
    """Retourne l'élévation et l'azimut du soleil (en degrés) pour chaque timestamp.

    `timestamps` peut être naïf (considéré comme UTC) ou timezone-aware.
    """
    if timestamps.tz is None:
        ts_utc = timestamps.tz_localize("UTC")
    else:
        ts_utc = timestamps.tz_convert("UTC")

    day_of_year = ts_utc.dayofyear.to_numpy(dtype=float)
    hour_utc = (ts_utc.hour + ts_utc.minute / 60.0 + ts_utc.second / 3600.0).to_numpy(dtype=float)
    n_days = np.where(ts_utc.is_leap_year, 366.0, 365.0)

    gamma = 2 * np.pi / n_days * (day_of_year - 1 + (hour_utc - 12) / 24)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl_rad = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    # Temps solaire vrai en minutes (longitude positive vers l'est, UTC = offset 0)
    time_offset = eqtime + 4 * longitude
    true_solar_time = hour_utc * 60 + time_offset
    hour_angle_deg = (true_solar_time / 4) - 180

    lat_rad = np.radians(latitude)
    ha_rad = np.radians(hour_angle_deg)

    cos_zenith = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith_rad = np.arccos(cos_zenith)
    elevation_deg = 90.0 - np.degrees(zenith_rad)

    sin_zenith = np.sin(zenith_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_azimuth = (np.sin(decl_rad) - np.sin(lat_rad) * cos_zenith) / (np.cos(lat_rad) * sin_zenith)
    cos_azimuth = np.nan_to_num(cos_azimuth, nan=1.0)
    cos_azimuth = np.clip(cos_azimuth, -1.0, 1.0)
    azimuth_deg = np.degrees(np.arccos(cos_azimuth))
    azimuth_deg = np.where(hour_angle_deg > 0, 360.0 - azimuth_deg, azimuth_deg)

    return pd.DataFrame(
        {"elevation_deg": elevation_deg, "azimuth_deg": azimuth_deg},
        index=timestamps,
    )


def face_exposure(elevation_deg: np.ndarray, azimuth_deg: np.ndarray, face_azimuth_deg: float) -> np.ndarray:
    """Proxy d'exposition directe d'une façade verticale au rayonnement solaire.

    Vaut 0 quand le soleil est sous l'horizon ou derrière la façade, et
    augmente avec le cosinus de l'angle d'incidence sinon (1 = soleil pile
    face à la façade, au zénith).
    """
    elevation_rad = np.radians(elevation_deg)
    azimuth_rad = np.radians(azimuth_deg)
    face_rad = np.radians(face_azimuth_deg)

    cos_incidence = np.cos(elevation_rad) * np.cos(azimuth_rad - face_rad)
    exposure = np.clip(cos_incidence, 0.0, None)
    return np.where(elevation_deg > 0, exposure, 0.0)


def compute_solar_features(
    timestamps: pd.DatetimeIndex,
    latitude: float = config.LATITUDE,
    longitude: float = config.LONGITUDE,
    house_faces: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Construit les features solaires alignées sur `timestamps`.

    Colonnes produites : `solar__elevation`, `solar__azimuth`,
    `solar__face_exposure__<face>` pour chaque façade de `house_faces`.
    """
    if house_faces is None:
        house_faces = config.HOUSE_FACES

    pos = solar_position(timestamps, latitude, longitude)

    sep = config.COLUMN_SEP
    out = pd.DataFrame(index=timestamps)
    out[f"solar{sep}elevation"] = pos["elevation_deg"].to_numpy()
    out[f"solar{sep}azimuth"] = pos["azimuth_deg"].to_numpy()

    for face, face_azimuth in house_faces.items():
        out[f"solar{sep}face_exposure{sep}{face}"] = face_exposure(
            pos["elevation_deg"].to_numpy(), pos["azimuth_deg"].to_numpy(), face_azimuth
        )

    return out
