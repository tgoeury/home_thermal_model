# config.py — Configuration centrale

Toutes les constantes du projet sont regroupées ici. Ne jamais éparpiller de
valeurs en dur dans les autres modules.

---

## Chemins

| Constante | Valeur par défaut | Description |
|---|---|---|
| `PROJECT_ROOT` | répertoire de `config.py` | Racine du projet |
| `DATA_RAW_DIR` | `PROJECT_ROOT/data_raw` | CSV bruts par source |
| `INDOOR_DIR` | `DATA_RAW_DIR/indoor` | Capteurs intérieurs |
| `OUTDOOR_DIR` | `DATA_RAW_DIR/outdoor` | Capteurs extérieurs par façade |
| `WEATHER_DIR` | `DATA_RAW_DIR/weather` | Données météo |
| `HOUSE_STATE_DIR` | `DATA_RAW_DIR/house_state` | État volets/fenêtres |
| `CHECKPOINT_DIR` | `PROJECT_ROOT/checkpoints` | Modèles entraînés (.pt, .onnx) |
| `STRATEGY_OUTPUT_DIR` | `PROJECT_ROOT/strategy/output` | Plannings générés (JSON) |

---

## Localisation

```python
LATITUDE = 48.8566   # TODO: remplacer par la latitude réelle
LONGITUDE = 2.3522   # TODO: remplacer par la longitude réelle
ELEVATION = 35.0     # mètres au-dessus du niveau de la mer
TIMEZONE = "Europe/Paris"
```

`HOUSE_FACES` : orientation des 4 façades en degrés (0=Nord, 90=Est, 180=Sud,
270=Ouest). Utilisé par `data/solar.py` pour calculer l'exposition directe de
chaque façade.

---

## Échantillonnage et fenêtre d'historique

| Constante | Valeur | Description |
|---|---|---|
| `SAMPLE_INTERVAL_MINUTES` | 2 | Pas de temps natif des capteurs (min) |
| `PREDICTION_HORIZON_STEPS` | 1 | Horizon de prédiction (pas → 2 min) |
| `HISTORY_HOURS` | 18.0 | Longueur totale de la fenêtre d'historique |

### RESOLUTION_SEGMENTS

Découpe multi-résolution de la fenêtre :

```python
RESOLUTION_SEGMENTS = [
    {"duration_minutes": 120,  "resolution_minutes": 2},   # 0–2h, résolution native
    {"duration_minutes": 240,  "resolution_minutes": 10},  # 2–6h
    {"duration_minutes": 720,  "resolution_minutes": 30},  # 6–18h
]
```

Résultat : 60 + 24 + 24 = **108 pas** pour 18h d'historique
(contre 540 en résolution native).

### `resolution_segments_for(history_hours)`

Tronque `RESOLUTION_SEGMENTS` pour une fenêtre plus courte, en conservant la
même structure multi-résolution. Utilisé lors du chargement des checkpoints
pour reconstruire exactement la même fenêtre que lors de l'entraînement.

---

## Convention de nommage des colonnes

```
<source>__<clé>__<mesure>    ex: indoor__salon__temperature
<source>__<mesure>           ex: weather__outdoor_temperature
```

| Constante | Valeurs |
|---|---|
| `COLUMN_SEP` | `"__"` |
| `INDOOR_MEASURES` | `["temperature", "humidity"]` |
| `OUTDOOR_MEASURES` | `["temperature", "humidity", "luminosity"]` |
| `HOUSE_STATE_TYPES` | `["shutter", "window"]` |
| `WEATHER_MEASURES` | `["outdoor_temperature", "solar_irradiance", "cloud_cover"]` |

---

## Capteurs par défaut (simulation)

| Constante | Valeur |
|---|---|
| `DEFAULT_INDOOR_ROOMS` | `["salon", "chambre1", "chambre2", "bureau"]` |
| `DEFAULT_HOUSE_STATE_ROOMS` | `["salon", "chambre1", "chambre2", "bureau"]` |
| `DEFAULT_ROOM_FACES` | `{salon: S, chambre1: E, chambre2: W, bureau: N}` |

> `DEFAULT_ROOM_FACES` est utilisé **uniquement** par le générateur simulé
> pour que chaque pièce simulée réagisse à un côté réel de la maison. Il
> n'est **jamais** donné en feature aux modèles.

---

## Hyperparamètres des modèles

| Constante | Valeur | Description |
|---|---|---|
| `GRU_HIDDEN_SIZE` | 64 | Taille de l'état caché du GRU |
| `GRU_NUM_LAYERS` | 2 | Nombre de couches GRU empilées |
| `GRU_DROPOUT` | 0.1 | Dropout entre couches GRU |
| `FULL_CORRECTION_HIDDEN_SIZE` | 32 | Taille du GRU de correction dans `full` |
| `LEARNING_RATE` | 1e-3 | Taux d'apprentissage Adam |
| `WEIGHT_DECAY` | 1e-4 | Régularisation L2 Adam |
| `BATCH_SIZE` | 64 | Taille de batch |
| `NUM_EPOCHS` | 50 | Nombre max d'epochs |
| `EARLY_STOPPING_PATIENCE` | 8 | Epochs sans amélioration val_loss avant arrêt |
| `RANDOM_SEED` | 42 | Graine aléatoire globale |

---

## Module strategy/

| Constante | Valeur | Description |
|---|---|---|
| `COMFORT_TEMP_MIN` | 19.0 °C | Borne basse de la plage de confort |
| `COMFORT_TEMP_MAX` | 26.0 °C | Borne haute de la plage de confort |
| `PLANNING_HORIZON_HOURS` | 24.0 | Durée du planning (heures) |
| `PLANNING_BLOCK_HOURS` | 2.0 | Durée d'un créneau volet/fenêtre |
| `PLANNING_EVAL_STEP_MINUTES` | 30.0 | Fréquence d'évaluation du coût de confort |
| `PLANNING_N_CANDIDATES` | 500 | Nombre de plannings aléatoires testés |
