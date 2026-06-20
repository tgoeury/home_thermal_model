# home_model — Modélisation thermique de la maison

Prédiction de température/humidité intérieure par pièce via des GRU PyTorch,
et recommandation de planning volets/fenêtres pour maintenir le confort
thermique.

---

## Structure du projet

```
home_model/
├── config.py                  # Toutes les constantes (chemins, hyperparamètres, confort)
├── home_model.py              # CLI principal (simulate / train-* / evaluate / plan)
├── train.py                   # Boucles d'entraînement (limited, full, early stopping)
├── evaluate.py                # Métriques, test de sensibilité, graphiques
│
├── data/
│   ├── pipeline.py            # Fusion des sources, fenêtrage multi-résolution, Dataset PyTorch
│   ├── solar.py               # Position solaire NOAA (vectorisé numpy, sans dépendance externe)
│   └── sources/
│       ├── base.py            # CSVSource : loader CSV générique pivot → format large
│       ├── indoor_sensors.py  # Capteurs intérieurs (sensor_id pivot)
│       ├── outdoor_sensors.py # Capteurs extérieurs par façade (face pivot)
│       ├── weather.py         # Météo observée + prévisions (kind = observed|forecast)
│       ├── house_state.py     # État volets/fenêtres (room + type pivot)
│       └── simulated.py       # Générateur de données simulées (modèle thermique jouet)
│
├── models/
│   ├── layers.py              # MultiResolutionEncoder (GRU + canal dt), RegressionHead
│   ├── limited.py             # LimitedModel : météo+solaire+volets → températures
│   └── full.py                # FullModel : limited (gelé) + correction outdoor
│
├── strategy/
│   ├── comfort.py             # Coût de confort (dépassements de plage de temp)
│   ├── schedule.py            # Génération et sur-échantillonnage de plannings candidats
│   ├── rollout.py             # PlanningContext : évaluation batch de plannings via limited
│   ├── planner.py             # Recherche aléatoire du meilleur planning (N candidats)
│   └── format.py             # Conversion planning → JSON/CSV (intervalles fusionnés)
│
├── docs/                      # Documentation markdown par module
├── tests/                     # Tests unitaires pytest
├── checkpoints/               # Checkpoints PyTorch (.pt) et ONNX (.onnx) — ignoré par git
├── data_raw/                  # CSV bruts par source — ignoré par git
└── evaluation/                # Graphiques générés — ignoré par git
```

---

## Workflow de développement

```bash
# 1. Générer les données simulées (N jours)
python home_model.py simulate --days 90 --forecast-hours 24

# 2. Entraîner le modèle "limited" (météo/solaire/volets uniquement)
python home_model.py train-limited --epochs 50

# 3. Entraîner le modèle "full" (nécessite limited entraîné)
python home_model.py train-full --epochs 50

# 4. Évaluer un modèle
python home_model.py evaluate --model limited
python home_model.py evaluate --model full

# 5. Générer un planning journalier volets/fenêtres
python home_model.py plan --n-candidates 500

# 6. Tests unitaires
pytest tests/
```

---

## Conventions de nommage des colonnes

Toutes les colonnes de la table fusionnée suivent `<source>__<clé>__<mesure>` :

| Préfixe | Description | Exemples |
|---|---|---|
| `indoor__` | Capteurs intérieurs (cibles à prédire) | `indoor__salon__temperature` |
| `outdoor__` | Capteurs extérieurs par façade | `outdoor__S__luminosity` |
| `weather__` | Météo (observée + prévisions) | `weather__solar_irradiance` |
| `solar__` | Features solaires calculées | `solar__face_exposure__S` |
| `house__` | État volets/fenêtres | `house__salon__shutter` |

Le séparateur `__` est défini dans `config.COLUMN_SEP`.

---

## Architecture des modèles

### LimitedModel
Utilisé pour la **planification journalière** — ne dépend pas des capteurs
façades, donc exploitable avec les prévisions météo seules.

```
x_limited (batch, n_steps, n_limited_features)
  └─► MultiResolutionEncoder (GRU + canal dt)
        └─► h (batch, hidden_size)
              └─► RegressionHead (MLP 2 couches)
                    └─► prédictions (batch, n_targets)
```

### FullModel
Utilisé pour l'**inférence temps réel** — affine la prédiction de `limited`
grâce aux capteurs extérieurs façades.

```
x_limited → LimitedModel (gelé) → base_pred + h_limited
x_outdoor → MultiResolutionEncoder → h_outdoor
[h_limited, h_outdoor] → RegressionHead (correction)
sortie = base_pred + correction   (résiduel)
```

Le modèle `limited` est gelé (`freeze_limited=True`) par défaut : la
correction n'altère pas le modèle de planification.

### MultiResolutionEncoder
Fenêtre 18h d'historique compressée en 108 pas au lieu de 540 :
- 0–2h : résolution native 2 min (60 pts)
- 2–6h : 10 min (24 pts)
- 6–18h : 30 min (24 pts)

Un canal supplémentaire `dt` (écart de temps normalisé entre pas) est
concaténé aux features pour que le GRU connaisse la densité temporelle.

---

## Export ONNX

Après entraînement, les modèles sont aussi exportés en ONNX :
- `checkpoints/limited.onnx`
- `checkpoints/full.onnx`

Les exports sont produits automatiquement par `train-limited` et `train-full`.
Pour exporter manuellement :

```bash
python home_model.py export-onnx --model limited
python home_model.py export-onnx --model full
```

La fonction `export_onnx()` se trouve dans `train.py`.

---

## Ajout de nouveaux capteurs

Le pipeline détecte dynamiquement les colonnes par préfixe. Pour ajouter un
nouveau capteur intérieur :
1. Déposer un CSV dans `data_raw/indoor/` avec les colonnes
   `timestamp, sensor_id, temperature, humidity`.
2. Réentraîner `limited` puis `full` from scratch (les dimensions changent).

Aucune modification de code n'est nécessaire.

---

## Décisions de conception importantes

- **Pas d'orientation des pièces en feature** : `DEFAULT_ROOM_FACES` dans
  `config.py` n'est utilisé que par le générateur simulé. Le modèle apprend
  l'orientation de chaque pièce de façon latente via les corrélations avec
  `solar__face_exposure__*`.
- **Split chronologique** : le dataset est découpé train/val en blocs
  temporels contigus (jamais aléatoire) pour éviter toute fuite d'information
  du futur vers le passé.
- **Dégradation gracieuse** : si les capteurs façades tombent en panne, on
  peut retomber sur `limited` seul sans réentraînement.
- **Météo observée prime sur prévision** : dans `WeatherSource`, si un
  timestamp a deux lignes (une prévision passée + la mesure réelle), la valeur
  `observed` écrase la prévision.
- **Pas de rollout auto-régressif pour la planification** : `limited` ne prend
  pas sa propre sortie en entrée, donc toutes les fenêtres de prédiction d'un
  horizon peuvent être évaluées en un seul batch.

---

## Localisation (à adapter)

Dans `config.py`, remplacer les coordonnées par celles de la maison réelle :

```python
LATITUDE = 48.8566   # TODO: latitude réelle
LONGITUDE = 2.3522   # TODO: longitude réelle
ELEVATION = 35.0     # TODO: altitude réelle
TIMEZONE = "Europe/Paris"
```
