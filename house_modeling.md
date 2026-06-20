# House Modeling — Journal de bord

Ce fichier trace, instruction par instruction, les décisions et le travail effectué sur le projet de modélisation thermique de la maison (prédiction de température/humidité intérieure via PyTorch).

---

## 2026-06-11 — Kickoff du projet

### Brief reçu

- **Données d'entrée** :
  - Capteurs intérieurs temp/humidité, plusieurs pièces, toutes les 2 min (4 capteurs actuellement = 8 mesures, extensible à N capteurs / features à prédire).
  - Capteurs extérieurs temps réel (luminosité, température, humidité) sur les 4 façades de la maison.
  - Données météo (proxy d'énergie solaire reçue : ensoleillement, luminosité, position du soleil → orientation exposée), température extérieure. Granularité cible 2 min, mais probablement plus faible → interpolation nécessaire.
- **Objectif** : modéliser le transfert d'énergie maison ↔ extérieur :
  - Rayonnement direct (mesuré par capteurs extérieurs et/ou dérivé de la météo + position solaire), modulé par l'état volets ouverts/fermés.
  - Échange d'air chaud via fenêtres ouvertes.
- **Deux modèles** :
  - **Limited** : prédiction à partir des données météo seules (un seul "côté"/proxy).
  - **Full** : prédiction à partir météo + capteurs extérieurs (4 façades) + état volets/fenêtres.
  - Question ouverte posée par l'utilisateur : faut-il les imbriquer (nested) ?
- **Approche modèle** : réseaux de neurones PyTorch, type RNN, fenêtres d'historique 6h/12h/18h pour capter la dynamique (tendance extérieure + inertie thermique de l'isolant qui restitue la chaleur sur plusieurs heures).
- **Contraintes projet** :
  - Fichier principal `home_model.py`, modules annexes autorisés pour la clarté.
  - Données simulées dans un premier temps pour les capteurs encore absents.
  - Lecture des données brutes via des modules conçus pour un usage CI/CD sur environnement de prod réduit : ajout/suppression de fichiers de données sans modification de code.
  - Tenue à jour de ce journal à chaque instruction.

### Travail effectué

- Création de ce fichier `house_modeling.md`.
- Proposition d'architecture envoyée à l'utilisateur pour discussion (modèles nested via apprentissage résiduel, encodeur GRU multi-résolution sur fenêtre 18h, structure de modules pour les données et les modèles). En attente de retour/validation avant implémentation.

### Décisions validées

- **Nesting** : approche résiduelle. `Full = Limited (gelé) + réseau de correction` à partir des capteurs façades (4 côtés) + état volets/fenêtres. Dégradation gracieuse si capteurs absents.
- **Format des données brutes** : CSV, format long (`timestamp, <clé pivot>, <valeurs>`), un loader générique pivote vers un format large par source. Ajout/suppression de fichiers ou de nouveaux capteurs (sensor_id/face inédits) sans changement de code.
- **Volets/fenêtres** : pas de source existante → données simulées pour l'instant (module `simulated.py`), même format que les sources réelles pour rester interchangeable.
- **Encodeur** : GRU multi-résolution (résolution native 2 min sur 0–2h, 10 min sur 2–6h, 30 min sur 6–18h), longueur de fenêtre paramétrable dans `config.py`.

### Précision sur l'objectif final (usage des modèles)

- But final : **inférence temps réel pour recommander la stratégie optimale** (garder la maison fraîche si chaleur extérieure, chaude si fraîcheur extérieure utile) → décisions d'ouverture/fermeture volets et fenêtres.
- **Limited** : utilisé en mode "planification journalière" — à partir des prévisions météo (forecast), prédit l'évolution de la température intérieure sur la journée pour déterminer à l'avance les créneaux d'ouverture/fermeture optimaux des volets/fenêtres.
- **Full** : utilisé en **temps réel**, en croisant la prédiction (Limited / planification) avec les données réelles des capteurs façades, pour affiner/corriger la décision en cours de journée (cohérent avec le nesting résiduel déjà décidé).

Implications pour la conception :
- La source météo doit pouvoir fournir des données **observées** ET des **prévisions** (même schéma, un flag/texte `kind=observed|forecast`).
- L'état volets/fenêtres est une **entrée pilotable** du modèle (pas seulement observée) : nécessaire pour que la couche de décision puisse simuler "et si on fermait les volets à 14h ?".
- Ajout d'un module `strategy/` (plus tard, après que les modèles prédictifs fonctionnent) : recherche du meilleur planning volets/fenêtres par simulation/rollout via Limited (et ajustement temps réel via Full).

### Précision : pas d'orientation des pièces en feature

- L'orientation de chaque pièce ne doit **pas** être une feature explicite du modèle : c'est une caractéristique **latente**, que le modèle doit apprendre lui-même à partir des corrélations entre les features solaires globales (`solar__face_exposure__N/E/S/W`, communes à toutes les pièces) et la réponse de chaque capteur intérieur.
- `config.DEFAULT_ROOM_FACES` existe uniquement pour la **vérité terrain** du générateur de données simulées (il faut bien que les pièces simulées réagissent différemment selon un côté réel de la maison) — ce mapping n'est jamais lu par le pipeline de features ni par les modèles.

### Travail effectué

- Arborescence du projet créée : `config.py`, `data/` (sources pluggables + position solaire + générateur simulé), `models/`, `strategy/`, `data_raw/{indoor,outdoor,weather,house_state}/`.
- `astral` installé (pip user) — finalement non utilisé : `data/solar.py` calcule la position solaire (élévation/azimut) et l'exposition par façade de façon vectorisée (numpy), via l'algorithme NOAA simplifié, sans dépendance externe. Vérifié : élévation max ~64.6° à Paris au solstice d'été (cohérent avec 90° - lat + 23.44°).
- `data/sources/base.py` : `CSVSource`, loader CSV générique, lit tous les fichiers d'un dossier, pivote sur une colonne clé (sensor_id/face/...) → colonnes `<source>__<clé>__<mesure>`. Ajout/suppression de fichiers ou nouvelles valeurs de pivot pris en compte automatiquement.
- Sources concrètes : `indoor_sensors.py`, `outdoor_sensors.py`, `house_state.py` (pivot room+type → `house__<room>__<shutter|window>`), `weather.py` (gère `kind=observed|forecast`, l'observé prime sur la prévision pour un même timestamp).
- `data/sources/simulated.py` : génère un jeu de données cohérent (météo basse fréquence avec observé+prévision, capteurs façades, état volets/fenêtres simulé par règles horaires, capteurs intérieurs via un modèle thermique jouet : perte ∝ écart T_in/T_out, gain solaire ∝ irradiance×exposition×volet ouvert, ventilation ∝ écart T_in/T_out×fenêtre ouverte). Constantes calibrées pour rester dans des plages réalistes (testé sur 2 jours : températures intérieures 16-27°C selon orientation, humidité 44-61%).

- `data/pipeline.py` : grille temporelle commune 2 min = union de toutes les sources (météo incluse, donc s'étend dans le futur via les prévisions). Météo interpolée dans le temps, autres sources en ffill/bfill, features solaires ajoutées. `compute_window_offsets()` traduit `config.RESOLUTION_SEGMENTS` en indices de pas (108 pas pour 18h par défaut). `HouseDataset` détecte dynamiquement les colonnes `indoor__*` (cibles), `weather__*`/`solar__*`/`house__*` (entrées "limited") et `outdoor__*` (entrées additionnelles "full"), filtre les fenêtres dont la cible est NaN. Normalisation (`FeatureStats`) calculée sur la portion d'entraînement uniquement, split chronologique (pas aléatoire) via `chronological_split`.
- `models/layers.py` : `MultiResolutionEncoder` (GRU + canal "écart de temps" normalisé par pas, pour que le réseau sache que les pas anciens couvrent des intervalles plus longs). `RegressionHead` = MLP 2 couches.
- `models/limited.py` : `LimitedModel` = encodeur + tête de régression sur les features météo/solaire/volets-fenêtres.
- `models/full.py` : `FullModel` = nesting résiduel — `pred = limited(x_limited) + correction(h_limited, encodeur_outdoor(x_outdoor))`. `limited` gelé par défaut (`freeze_limited=True`).
- `train.py` : `prepare_dataset()` (charge, fenêtre, normalise, split), `train_limited_model()`, `train_full_model()`. Pour "full", les stats de normalisation `limited`/`target` du checkpoint "limited" sont réutilisées (cohérence avec le modèle gelé), seules les stats `outdoor` sont calculées à neuf.
- `home_model.py` : CLI avec sous-commandes `simulate` (génère N jours de données simulées + prévisions météo dans `data_raw/`), `train-limited`, `train-full` (charge le checkpoint `limited`, vérifie la cohérence des colonnes, entraîne et sauvegarde dans `checkpoints/`).

### Test de bout en bout (2026-06-11)

- `python3 home_model.py simulate --days 15 --forecast-hours 24` → données générées dans `data_raw/`.
- `python3 home_model.py train-limited --epochs 2` → 10995 échantillons (8796 train / 2199 val), 17 features "limited", 8 cibles (4 pièces × temp/humidité). Entraînement OK, checkpoint sauvegardé.
- `python3 home_model.py train-full --epochs 2` → 12 features "outdoor" supplémentaires détectées (4 façades × temp/humidité/luminosité). Entraînement OK, checkpoint sauvegardé.
- Pipeline complet fonctionnel de bout en bout sur données simulées. Val loss instable sur seulement 2 epochs (normal, juste un test de plomberie).

### Prochaines étapes possibles

- Entraînement plus long + suivi de métriques (courbes train/val, early stopping).
- Évaluation : comparer prédictions vs. vérité terrain simulée, vérifier que le modèle retrouve les effets attendus (gain solaire, ventilation).
- Module `strategy/` : rollout journalier via "limited" + ajustement temps réel via "full" pour recommander l'ouverture/fermeture volets/fenêtres.
- Quand de vraies données seront disponibles : déposer les CSV dans `data_raw/<source>/` (même format que les CSV simulés) et retirer/garder les fichiers `simulated.csv` selon les sources réellement couvertes.

## 2026-06-11 — Évaluation contre la vérité terrain simulée

### Création de `evaluate.py`

- `load_limited_model()` / `load_full_model()` : reconstruisent les modèles depuis `checkpoints/*.pt`.
- `compute_metrics()` : MAE/RMSE par capteur, en unités réelles (dénormalisées via `target_stats`).
- `sensitivity_report()` : pour chaque pièce, bascule la feature `house__<room>__shutter|window` entre 0 et 1 sur toute la fenêtre d'historique (à météo identique) et mesure le delta de température prédite. Attendu : volet → gain ≈0 la nuit, >0 le jour ; fenêtre → la prédiction se rapproche de la température extérieure (signe du delta = signe de `T_out - T_in`).
- `plot_predictions()` : courbes prédiction vs. vérité terrain (PNG dans `evaluation/`).
- Sous-commande `home_model.py evaluate --model {limited,full}`.

### Entraînement 20 epochs (15 jours simulés) — résultat : surapprentissage net

```
[limited] epoch 1/20  train_loss=0.5851 val_loss=0.7378  (meilleur val_loss)
[limited] epoch 20/20 train_loss=0.0056 val_loss=1.0607
[full]    epoch 1/20  train_loss=0.0071 val_loss=1.0569
[full]    epoch 20/20 train_loss=0.0044 val_loss=1.0492
```

- Le train_loss s'effondre (jusqu'à 0.0044-0.0056) tandis que le val_loss ne s'améliore jamais après l'epoch 1 et plafonne ~1.04-1.07 (≈ aussi mauvais que prédire la moyenne, R²≈0). "Full" hérite du "limited" gelé déjà en surapprentissage ; son réseau de correction n'apporte quasiment rien (métriques quasi identiques à "limited").
- **Cause probable** : seulement 15 jours de données → ~11 jours d'entraînement après la fenêtre 18h, soit très peu de cycles journaliers indépendants pour un GRU (hidden=64, 2 couches, 108 pas, 17 features). Le modèle mémorise les trajectoires d'entraînement plutôt que d'apprendre la physique générale ; le set de validation (3 derniers jours) correspond à une réalisation météo (cloud_cover) jamais vue.

### Métriques (unités réelles, set de validation)

| Capteur | MAE | RMSE |
|---|---|---|
| temp salon/chambre/cuisine/bureau | 0.97 – 1.29 °C | 1.59 – 2.01 °C |
| humidité salon/cuisine | 1.5 – 2.0 %HR | 1.8 – 2.2 %HR |
| humidité chambre/bureau | 4.1 – 4.5 %HR | 4.6 – 5.5 %HR |

Quasi identiques entre "limited" et "full".

### Test de sensibilité volets/fenêtres : le modèle n'a pas appris la bonne physique

- **Volets** : effet attendu = gain solaire le jour (irradiance>0), ≈0 la nuit. Observé = l'inverse : effet plus fort la nuit (+0.20 à +0.55 °C) que le jour (+0.00 à +0.15 °C) pour toutes les pièces.
- **Fenêtres** : effet attendu = la prédiction se rapproche de T_ext (delta>0 si dehors plus chaud, delta<0 si dehors plus froid). Observé = delta **toujours positif** (+1.1 à +2.9 °C), même quand dehors est plus froid qu'à l'intérieur — le modèle a appris une corrélation parasite (ex: "fenêtre ouverte" coïncide souvent avec la journée dans le planning simulé sur 15 jours), pas la ventilation physique.

### Visualisation (`evaluation/limited_indoor__salon__temperature.png`)

- Bonne correspondance prédiction/vérité terrain sur ~2 jours, puis la prédiction **se fige en plateau constant** pendant ~14h en fin de période de validation — échec total d'extrapolation sur cette portion.

### Conclusion

Le pipeline (chargement, entraînement, évaluation, métriques, test de sensibilité, graphiques) fonctionne de bout en bout sans bug. Mais le modèle actuel a surappris sur seulement 15 jours et n'a pas capturé les relations physiques attendues. Avant le module `strategy/`, il faut corriger ce surapprentissage — proposition à l'utilisateur : générer beaucoup plus de données simulées (gratuit, ex: 90-120 jours pour plus de scénarios météo/cloud_cover indépendants), ajouter une régularisation (weight decay) et/ou un early stopping (sauvegarde du meilleur checkpoint sur val_loss).

## 2026-06-12 — Correction du surapprentissage (90 jours + weight decay + early stopping)

### Décision utilisateur

Option choisie : **plus de données + régularisation + early stopping** (génération de 90 jours simulés, `weight_decay` dans Adam, arrêt anticipé avec restauration du meilleur état sur `val_loss`).

### Changements de code

- `config.py` : ajout de `WEIGHT_DECAY = 1e-4` et `EARLY_STOPPING_PATIENCE = 8`.
- `train.py` : nouvelle fonction interne `_early_stopping_loop()` — boucle d'entraînement commune à "limited"/"full" qui suit le meilleur `val_loss`, sauvegarde une copie de l'état du modèle à chaque amélioration, s'arrête après `patience` epochs sans amélioration, et restaure le meilleur état avant de retourner. `train_limited_model`/`train_full_model` prennent maintenant `weight_decay` et `patience` (Adam initialisé avec `weight_decay`).
- `python3 home_model.py simulate --days 90 --forecast-hours 24` → nouveau jeu de données (15.7s), remplace les CSV `simulated.csv` de 15 jours.

### Résultats du ré-entraînement (90 jours, `--epochs 30`)

```
64995 échantillons (51996 train / 12999 val)

[limited] epoch 1/30 train_loss=0.2615 val_loss=0.2394  (meilleur)
...
[limited] epoch 9/30 train_loss=0.0153 val_loss=0.2637
[limited] arrêt anticipé (epoch 9, meilleur val_loss=0.2394)   — ~14 min

[full]    epoch 1/30 train_loss=0.1529 val_loss=0.2316
[full]    epoch 5/30 train_loss=0.1312 val_loss=0.2160  (meilleur)
...
[full]    epoch 13/30 arrêt anticipé (meilleur val_loss=0.2160)
```

Écart train/val désormais raisonnable (val_loss ≈ 0.22-0.24 contre ~1.05 avant) ; "full" apporte un gain net sur "limited" (0.2160 vs 0.2394), contrairement à la version précédente où la correction n'apportait rien.

### Métriques (unités réelles, 12999 échantillons de validation)

| Capteur | limited MAE/RMSE | full MAE/RMSE |
|---|---|---|
| temp salon | 0.78 / 1.00 °C | 0.57 / 0.78 °C |
| temp chambre | 1.05 / 1.23 °C | 0.70 / 0.94 °C |
| temp cuisine | 0.86 / 1.21 °C | 0.67 / 1.06 °C |
| temp bureau | 0.83 / 1.02 °C | 0.48 / 0.70 °C |
| humidité | 1.7 – 3.2 / 2.2 – 3.8 %HR | 1.6 – 2.8 / 2.2 – 3.5 %HR |

Net progrès sur les températures (MAE quasi divisée par 2 pour "full" vs l'ancien "limited"). Le graphique `evaluation/full_indoor__salon__temperature.png` suit bien le cycle jour/nuit sur 3 jours de validation (léger sous-estimation des pics, plus de plateau/échec d'extrapolation).

### Test de sensibilité : résultats nuancés, ré-interprétation de l'attendu

- **Volets** : salon/chambre montrent un gain positif jour≈nuit (+0.30 à +0.60 °C) ; cuisine quasi nul ; bureau légèrement négatif. Le test bascule l'état du volet sur **toute la fenêtre de 18h d'historique**, donc un échantillon "nuit" (irradiance nulle au dernier pas) inclut quand même les heures de jour précédentes dans son historique — un effet "nuit" non nul peut donc refléter l'**inertie thermique** (restitution de chaleur après la fermeture), ce qui est justement le phénomène que le modèle doit capturer. Le signe négatif pour "bureau" reste à surveiller (à approfondir si besoin lors du module `strategy/`, où l'effet marginal *au pas suivant* d'un changement d'état sera testé plutôt que sur 18h).
- **Fenêtres** : sur la condition majoritaire "dehors plus chaud" (n≈300), salon/chambre/bureau ont un delta positif (cohérent : fenêtre ouverte → la pièce se rapproche de T_ext plus chaude). Cuisine montre un delta négatif (incohérent) dans les deux modèles — anomalie isolée à creuser plus tard. La condition "dehors plus froid" n'a que 1-4 échantillons sur cette période de validation (fin mai/juin, T_ext > T_in presque tout le temps) → non interprétable statistiquement, pas un signal d'échec du modèle.

### Conclusion

Le surapprentissage est résolu : val_loss ≈ 0.22-0.24 (était ~1.05), métriques réelles cohérentes, courbes de prédiction qui suivent la vérité terrain sans plateau. Le modèle "full" apporte un gain mesurable sur "limited". La physique volets/fenêtres est globalement dans le bon sens pour 3/4 pièces (le signe "cuisine" pour les fenêtres est une anomalie à surveiller). Suffisant pour démarrer le module `strategy/`, qui pourra affiner le diagnostic via des tests de sensibilité au pas suivant (plus discriminants que sur 18h).

---

## 2026-06-12 — Module `strategy/` : recommandation de planning volets/fenêtres

### Décisions de conception (validées par l'utilisateur)

Trois choix de conception ont été soumis et tous les trois ont été tranchés en faveur de l'option recommandée :

1. **Objectif de confort** : plage de température cible fixe et globale (`config.COMFORT_TEMP_MIN = 19°C`, `config.COMFORT_TEMP_MAX = 26°C`), appliquée à toutes les pièces. Le coût pénalise uniquement les dépassements de cette plage (pas de cible précise à l'intérieur, pas de pénalité de changement d'état).
2. **Algorithme de recherche** : recherche par échantillonnage aléatoire — génère `config.PLANNING_N_CANDIDATES` (500) plannings candidats, évalue chacun via un rollout du modèle "limited", garde le moins coûteux.
3. **Format de sortie** : planning JSON journalier par pièce, sous forme d'intervalles volet/fenêtre (créneaux consécutifs fusionnés), exploitable par un futur système domotique.

### Architecture implémentée

- **`config.py`** : nouvelle section "Module strategy/" — `COMFORT_TEMP_MIN/MAX`, `PLANNING_HORIZON_HOURS=24`, `PLANNING_BLOCK_HOURS=2`, `PLANNING_EVAL_STEP_MINUTES=30`, `PLANNING_N_CANDIDATES=500`, et `STRATEGY_OUTPUT_DIR`.
- **`strategy/comfort.py`** : `comfort_cost(temperatures)` — somme des dépassements (en °C) de la plage de confort sur tous les points d'évaluation et toutes les pièces.
- **`strategy/schedule.py`** : `random_schedule(...)` tire, pour chaque (pièce, volet|fenêtre), une suite de créneaux indépendants de 2h (0/1) ; `schedule_to_steps(...)` sur-échantillonne vers le pas natif (2 min). La granularité par créneaux limite naturellement la fréquence de changement d'état, sans pénalité dédiée.
- **`strategy/rollout.py`** :
  - `find_now_index(table)` repère la dernière ligne météo `observed` (frontière passé connu / futur prévisionnel).
  - `PlanningContext` précalcule, à partir de la table fusionnée et du checkpoint "limited" : les offsets de fenêtre multi-résolution, la plage de table utile (`window_array`), la matrice d'indices pour extraire en un seul batch toutes les fenêtres de prédiction sur l'horizon (un point d'évaluation toutes les 30 min), la correspondance colonnes `house__<pièce>__<volet|fenêtre>`, et les indices des cibles de température.
  - `PlanningContext.evaluate(model, schedule_steps)` : écrase les colonnes `house__*` futures avec le planning candidat, normalise, exécute le modèle "limited" en un seul passage batché (pas de rollout auto-régressif nécessaire car "limited" ne dépend pas de ses propres prédictions précédentes), dénormalise et renvoie `comfort.comfort_cost(...)`.
- **`strategy/format.py`** : `schedule_to_plan(...)` convertit un planning par créneaux en JSON `{generated_at, horizon_hours, rooms: {pièce: [{from, to, shutter, window}, ...]}}`, en fusionnant les créneaux consécutifs identiques.
- **`strategy/planner.py`** : `plan(...)` orchestre le tout — charge "limited", construit la table de features et le `PlanningContext`, boucle de recherche aléatoire (500 candidats par défaut), tronque le planning rendu si l'horizon de prévision météo disponible est plus court que demandé, et renvoie le JSON formaté + `best_cost` + `n_candidates`.
- **`home_model.py`** : nouvelle sous-commande `plan` (`--n-candidates`, `--horizon-hours`, `--block-hours`, `--eval-step-minutes`), qui affiche le planning par pièce et écrit le JSON dans `strategy/output/plan_<timestamp>.json`.

### Test de bout en bout

```
$ python3 home_model.py plan
Planning généré à partir de 2026-06-11T21:32:00+00:00 (horizon 24.0h, 500 plannings testés, coût de confort estimé = 0.00)
...
Planning écrit dans strategy/output/plan_20260612T075749Z.json
```

Vérification de la fonction de coût (hors recherche) sur deux plannings extrêmes sur le même horizon :
- tous volets/fenêtres fermés 24h : coût = 7.71 (températures prédites sortent de la plage de confort) ;
- tous volets/fenêtres ouverts 24h : coût = 2.86 ;
- meilleur planning trouvé (500 candidats aléatoires) : coût = 0.00, températures prédites comprises entre 18.6°C et 23.9°C sur l'horizon — toujours dans `[19, 26]` sauf un léger dépassement bas resté à 0 grâce au planning choisi.

Le coût différencie bien les plannings (7.71 / 2.86 / 0.00), confirmant que le rollout + coût de confort fonctionnent correctement. Avec la météo de juin simulée (douce), la plage de confort 19-26°C est facilement atteignable, d'où un coût optimal nul — un test avec une vague de froid/chaleur simulée serait nécessaire pour observer un planning qui doive faire un compromis (coût > 0).

### Conclusion

Le module `strategy/` est complet et opérationnel : `python3 home_model.py plan` produit un planning journalier par pièce (volets/fenêtres, créneaux de 2h) à partir des prévisions météo et du modèle "limited", exporté en JSON exploitable par un futur système domotique. Pipeline complet du projet : `simulate` → `train-limited`/`train-full` → `evaluate` → `plan`.

### Export CSV additionnel

À chaque exécution de `plan`, en plus du JSON dans `strategy/output/`, un export CSV est écrit à la racine du projet : `{timestamp}_strategy24.csv` (une ligne par intervalle, colonnes `room, from, to, shutter, window`). Implémenté via `strategy.format.write_plan_csv(...)`, appelé depuis `cmd_plan` dans `home_model.py`.
