# data/ — Sources de données et pipeline

---

## Sources de données (`data/sources/`)

### `base.py` — `CSVSource`

Loader CSV générique. Lit tous les fichiers `.csv` d'un répertoire, les
concatène, puis pivote les lignes vers un format **large** indexé par
timestamp.

**Format CSV attendu (format long) :**
```
timestamp, [colonne_pivot], mesure1, mesure2, ...
```

**Colonnes de sortie :**
- avec pivot : `<prefix>__<valeur_pivot>__<mesure>`
- sans pivot : `<prefix>__<mesure>`

**Méthodes :**

| Méthode | Description |
|---|---|
| `list_files()` | Liste les fichiers CSV du répertoire |
| `read_raw()` | Concatène tous les fichiers sans mise en forme |
| `load()` | Retourne la table large indexée par timestamp UTC |

Ajouter un nouveau fichier CSV au répertoire, ou une nouvelle valeur de pivot
(nouveau `sensor_id`, nouvelle `face`), crée automatiquement de nouvelles
colonnes — **aucune modification de code** n'est nécessaire.

---

### `indoor_sensors.py` — `IndoorSensorSource`

```
data_raw/indoor/*.csv   →   indoor__<sensor_id>__temperature
                            indoor__<sensor_id>__humidity
```

CSV attendu : `timestamp, sensor_id, temperature, humidity`

Ce sont les **colonnes cibles** que les modèles doivent prédire.

---

### `outdoor_sensors.py` — `OutdoorSensorSource`

```
data_raw/outdoor/*.csv  →   outdoor__<face>__temperature
                            outdoor__<face>__humidity
                            outdoor__<face>__luminosity
```

CSV attendu : `timestamp, face, temperature, humidity, luminosity`

`face` ∈ {N, E, S, W}. Ces colonnes sont les entrées additionnelles du
modèle `full`.

---

### `weather.py` — `WeatherSource`

```
data_raw/weather/*.csv  →   weather__outdoor_temperature
                            weather__solar_irradiance
                            weather__cloud_cover
                            weather__kind
```

CSV attendu : `timestamp, kind, outdoor_temperature, solar_irradiance, cloud_cover`

`kind` ∈ {`observed`, `forecast`}. Quand un même timestamp a les deux, la
valeur `observed` prime (supprime la prévision correspondante). Les timestamps
futurs n'ont que `forecast` : c'est ce que le modèle `limited` utilise pour
planifier la journée à venir.

La colonne `weather__kind` est conservée pour que `rollout.py` puisse
identifier la frontière passé/futur.

---

### `house_state.py` — `HouseStateSource`

```
data_raw/house_state/*.csv  →   house__<room>__shutter
                                house__<room>__window
```

CSV attendu : `timestamp, room, type, state`

`type` ∈ {`shutter`, `window`}, `state` ∈ {0, 1} (fermé/ouvert).
Ce sont des **entrées pilotables** : le module `strategy/` peut les remplacer
par un planning candidat pour simuler "et si on fermait les volets à 14h ?".

---

### `simulated.py` — Générateur de données simulées

Produit des CSV dans `data_raw/<source>/simulated.csv` avec exactement le même
schéma que les sources réelles.

**Modèle thermique jouet (par pas de 2 min, par pièce) :**

```
T_in[t+1] = T_in[t]
           - THERMAL_LOSS_RATE  * (T_in[t] - T_out[t])     # pertes vers extérieur
           + SOLAR_GAIN_FACTOR  * irradiance[t] * expo[t] * shutter_open[t]  # gain solaire
           + VENTILATION_RATE   * (T_out[t] - T_in[t]) * window_open[t]     # ventilation
           + bruit
```

| Constante | Valeur | Interprétation |
|---|---|---|
| `THERMAL_LOSS_RATE` | 0.005 | Constante de temps thermique ≈ 6–7h |
| `SOLAR_GAIN_FACTOR` | 4e-5 | °C / (W/m²) / pas, volet ouvert |
| `VENTILATION_RATE` | 0.05 | Fraction de l'écart T_in/T_out équilibrée par pas |
| `HUMIDITY_VENT_RATE` | 0.03 | Idem pour l'humidité |

**Fonctions principales :**

| Fonction | Description |
|---|---|
| `generate_weather(start, end, ...)` | Météo basse fréquence (observée + prévisions) |
| `generate_outdoor_sensors(start, end, weather_df, ...)` | Capteurs par façade |
| `generate_house_state(start, end, ...)` | Volets ouverts 7h–22h, fenêtres aléatoires |
| `generate_indoor_sensors(start, end, weather_df, outdoor_df, house_state_df, ...)` | Modèle thermique |
| `generate_dataset(start, end, ...)` | Orchestre les 4 fonctions ci-dessus |
| `write_dataset(dataset, filename)` | Écrit les CSV dans `data_raw/` |
| `generate_and_write(start, end, ...)` | `generate_dataset` + `write_dataset` |

---

## Pipeline de données (`data/pipeline.py`)

### `load_raw_sources()`

Charge les 4 sources (indoor, outdoor, weather, house_state). Une source
absente (aucun fichier dans son répertoire) est simplement omise — le reste
du pipeline continue sans elle.

### `build_feature_table(sources=None)`

Fusionne toutes les sources sur une grille temporelle commune à
`SAMPLE_INTERVAL_MINUTES` :

1. Grille = union de tous les timestamps sources (météo incluse → s'étend dans
   le futur via les prévisions).
2. Météo : interpolation temporelle (source basse fréquence).
3. Autres sources : `ffill().bfill()` (tolère de petits trous de transmission).
4. Features solaires calculées directement sur la grille.

### `compute_window_offsets(resolution_segments)`

Traduit `RESOLUTION_SEGMENTS` en un tableau de décalages (en pas de temps),
triés du plus ancien (`offsets.max()`) au plus récent (`0`). 108 valeurs avec
les segments par défaut.

### `FeatureStats`

Dataclass `(mean, std)` pour la normalisation z-score par colonne.

| Méthode | Description |
|---|---|
| `transform(array)` | Standardise : `(array - mean) / std` |
| `inverse_transform(array)` | Dénormalise : `array * std + mean` |

`std` est mis à `1.0` pour les colonnes constantes (évite la division par 0).

### `HouseDataset`

Dataset PyTorch. Chaque échantillon contient :

| Clé | Forme | Description |
|---|---|---|
| `x_limited` | `(n_steps, n_limited_features)` | Météo + solaire + volets/fenêtres |
| `x_outdoor` | `(n_steps, n_outdoor_features)` | Capteurs façades (si disponibles) |
| `y` | `(n_targets,)` | Températures/humidités intérieures à `t + horizon` |

**Méthodes :**

| Méthode | Description |
|---|---|
| `chronological_split(train_fraction)` | Split train/val temporellement (jamais aléatoire) |
| `compute_normalization_stats(up_to_sample)` | Calcule mean/std sur la portion train |
| `apply_normalization(stats)` | Applique les stats (peut réutiliser les stats d'un checkpoint) |

---

## Features solaires (`data/solar.py`)

Implémentation vectorisée de l'algorithme NOAA (numpy pur, sans dépendance
externe).

| Fonction | Description |
|---|---|
| `solar_position(timestamps, lat, lon)` | Élévation et azimut du soleil (degrés) |
| `face_exposure(elevation, azimuth, face_azimuth)` | Proxy d'exposition directe d'une façade (0–1) |
| `compute_solar_features(timestamps, ...)` | Construit les colonnes `solar__*` pour la table |

**Colonnes produites par `compute_solar_features` :**
- `solar__elevation` : élévation du soleil (degrés, négatif la nuit)
- `solar__azimuth` : azimut du soleil (degrés, 0=Nord)
- `solar__face_exposure__N/E/S/W` : exposition directe de chaque façade (0 si nuit ou soleil derrière)
