# strategy/ — Recommandation de planning volets/fenêtres

Le module `strategy/` orchestre la **planification journalière** : à partir
des prévisions météo et du modèle `limited`, il cherche le planning
volets/fenêtres qui maintient toutes les pièces dans la plage de confort
thermique.

---

## Vue d'ensemble

```
planner.plan()
  ├── load_limited_model()           # checkpoint limited
  ├── build_feature_table()          # table fusionnée (avec prévisions météo)
  ├── PlanningContext(table, ...)    # pré-calcul : offsets, window_array, indices
  └── for _ in range(n_candidates):
        candidate = random_schedule(...)    # planning aléatoire (créneaux 2h)
        steps = schedule_to_steps(...)      # sur-échantillonné au pas 2min
        cost = ctx.evaluate(model, steps)   # rollout batché → comfort_cost
        → garde le meilleur
  └── schedule_to_plan(best_schedule, ...)  # JSON par pièce (intervalles fusionnés)
```

---

## `comfort.py`

### `room_bounds(room, comfort_ranges)`

Retourne `(t_min, t_max)` pour une pièce. Utilise `comfort_ranges[room]` si
fourni, sinon retombe sur `config.COMFORT_TEMP_MIN/MAX`.

### `comfort_cost(temperatures, rooms, comfort_ranges)`

Somme des dépassements (en °C) de la plage de confort sur tous les points
d'évaluation et toutes les pièces :

```
cost = Σ max(0, t_min - T) + max(0, T - t_max)   ∀ pièce, ∀ instant
```

Un coût de 0 signifie que toutes les pièces restent dans la plage de confort
sur tout l'horizon.

### `block_reasons(temperatures_by_room, eval_rows, now_idx, n_blocks, block_hours, ...)`

Pour chaque pièce et chaque créneau, détermine si l'état volet/fenêtre choisi
sert à :
- `"maintenir"` : températures dans la plage de confort ;
- `"refroidir"` : températures au-dessus de `t_max` ;
- `"rechauffer"` : températures en dessous de `t_min`.

---

## `schedule.py`

### `n_blocks(horizon_hours, block_hours)`

Nombre de créneaux dans l'horizon : `round(horizon_hours / block_hours)`.

### `random_schedule(rooms, state_types, horizon_hours, block_hours, rng)`

Tire un planning candidat : pour chaque `(pièce, type)`, une suite de
`n_blocks` valeurs 0/1 indépendantes (Bernoulli 0.5).

```python
{("salon", "shutter"): array([0, 1, 1, 0, ...]),
 ("salon", "window"):  array([1, 0, 0, 1, ...]),
 ...}
```

### `schedule_to_steps(schedule, n_steps, block_hours)`

Sur-échantillonne chaque planning par créneaux vers le **pas natif**
(`SAMPLE_INTERVAL_MINUTES = 2 min`), tronqué/complété à `n_steps`.

Exemple : un créneau de 2h = 60 pas de 2 min, tous avec la même valeur.

---

## `rollout.py`

### `find_now_index(table)`

Retourne l'index de la **dernière ligne météo `observed`** dans la table
fusionnée. C'est la frontière entre le passé connu et le futur prévisionnel.

### `PlanningContext`

Pré-calcule toutes les données communes à l'évaluation de N plannings
candidats (évite de tout recalculer à chaque itération) :

| Attribut | Description |
|---|---|
| `offsets` | Décalages multi-résolution (108 valeurs) |
| `now_idx` | Index de "maintenant" dans la table |
| `horizon_steps_count` | Nombre de pas futurs disponibles (limité par les prévisions météo) |
| `window_array` | Sous-tableau de la table sur la fenêtre utile |
| `local_rows_matrix` | Indices pour extraire tous les batch de prédiction en une fois |
| `house_state_cols` | Indices des colonnes `house__*` dans `limited_columns` |
| `temperature_target_indices` | Indices des cibles de température dans `target_columns` |

**`PlanningContext.evaluate(model, schedule_steps, comfort_ranges)`**

Évalue un planning candidat en un **seul passage batché** :
1. Copie `window_array`.
2. Écrase les colonnes `house__*` futures avec le planning candidat.
3. Normalise via `limited_stats`.
4. Extrait tous les batch via `local_rows_matrix` (une fenêtre par point
   d'évaluation, toutes les 30 min).
5. Passe le batch dans le modèle `limited`.
6. Dénormalise et calcule `comfort_cost`.

> Pas de rollout auto-régressif nécessaire : `limited` ne prend pas sa
> propre sortie en entrée, donc toutes les prédictions d'un horizon peuvent
> être évaluées simultanément.

---

## `format.py`

### `schedule_to_plan(schedule, rooms, state_types, now, horizon_hours, block_hours, reasons)`

Convertit un planning par créneaux en JSON lisible :

```json
{
  "generated_at": "2026-06-12T08:05:14+00:00",
  "horizon_hours": 24.0,
  "rooms": {
    "salon": [
      {"from": "...", "to": "...", "shutter": "open", "window": "closed", "reason": "maintenir"},
      ...
    ]
  }
}
```

Les créneaux consécutifs ayant le même état (et la même raison) sont
**fusionnés** en un seul intervalle.

### `write_plan_csv(plan, path, state_types)`

Écrit le planning au format CSV : une ligne par intervalle, colonnes
`room, from, to, shutter, window, reason`.

---

## `planner.py` — `plan(...)`

Point d'entrée principal. Paramètres :

| Paramètre | Défaut | Description |
|---|---|---|
| `n_candidates` | 500 | Nombre de plannings aléatoires testés |
| `horizon_hours` | 24.0 | Horizon de planification (h) |
| `block_hours` | 2.0 | Durée d'un créneau (h) |
| `eval_step_minutes` | 30.0 | Fréquence d'évaluation du coût de confort |
| `comfort_ranges` | None | `{"pièce": (t_min, t_max)}` par pièce |

Retourne un dictionnaire JSON prêt à écrire (inclut `best_cost` et
`n_candidates`).

**Tronquage de l'horizon :** si l'horizon de prévision météo disponible est
plus court que `horizon_hours`, le planning rendu est automatiquement tronqué
aux créneaux effectivement évalués.

---

## Commande CLI

```bash
python home_model.py plan \
  --n-candidates 500 \
  --horizon-hours 24 \
  --block-hours 2 \
  --eval-step-minutes 30 \
  --comfort-ranges '{"salon": [20, 25], "bureau": [18, 24]}'
```

Sorties :
- `strategy/output/plan_<timestamp>.json`
- `<timestamp>_strategy24.csv` (à la racine du projet)
