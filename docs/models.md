# models/ — Architecture des modèles

---

## `layers.py` — Briques partagées

### `MultiResolutionEncoder`

Encodeur GRU pour une fenêtre d'historique multi-résolution. Produit un vecteur
de contexte `(batch, hidden_size)` à partir d'une séquence temporelle.

**Particularité :** un canal `dt` (écart de temps entre pas, normalisé par
l'écart maximal) est concaténé aux features à chaque pas. Le GRU sait ainsi
que les pas anciens représentent des intervalles de temps plus longs — essentiel
pour une fenêtre multi-résolution où les pas ne sont pas équidistants.

```python
MultiResolutionEncoder(
    input_size: int,          # nombre de features d'entrée
    hidden_size: int = 64,    # config.GRU_HIDDEN_SIZE
    num_layers: int = 2,      # config.GRU_NUM_LAYERS
    dropout: float = 0.1,     # config.GRU_DROPOUT
    history_hours: float,     # détermine les offsets temporels via config.resolution_segments_for
)
```

**Forward :**
```
x: (batch, n_steps, input_size) → h_n[-1]: (batch, hidden_size)
```

Le buffer `dt` de forme `(1, n_steps, 1)` est pré-calculé à la construction
et enregistré comme buffer PyTorch (persistant dans le checkpoint).

---

### `RegressionHead`

MLP à 2 couches linéaires avec ReLU entre les deux.

```python
RegressionHead(
    input_size: int,
    output_size: int,
    hidden_size: int | None = None,  # = input_size si None
)
```

**Forward :**
```
x: (batch, input_size) → (batch, output_size)
```

---

## `limited.py` — `LimitedModel`

Prédit `n_targets` valeurs (températures + humidités intérieures) à partir des
features météo, solaires et état volets/fenêtres.

Utilisé pour la **planification journalière** : ne dépend que des données
disponibles à l'avance (météo prévisionnelle), ce qui permet de simuler des
scénarios "et si on ouvre le volet à 14h ?" sans capteurs en temps réel.

```python
LimitedModel(
    n_limited_features: int,   # détecté dynamiquement par HouseDataset
    n_targets: int,            # détecté dynamiquement (colonnes indoor__*)
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    history_hours: float,
)
```

**Forward :**
```
x_limited: (batch, n_steps, n_limited_features) → (batch, n_targets)
```

**Sauvegarde checkpoint (`checkpoints/limited.pt`) :**
```python
{
    "model_state": ...,
    "n_limited_features": int,
    "n_targets": int,
    "limited_columns": list[str],   # ordre des colonnes (doit rester stable)
    "target_columns": list[str],
    "limited_stats": FeatureStats,
    "target_stats": FeatureStats,
    "history_hours": float,
    "horizon_steps": int,
}
```

---

## `full.py` — `FullModel`

Modèle résiduel : `full(x) = limited(x_limited) + correction(h_limited, x_outdoor)`.

Utilisé pour l'**inférence temps réel** — affine la prédiction de `limited`
grâce aux données mesurées sur les façades.

```python
FullModel(
    limited_model: LimitedModel,    # modèle limited pré-entraîné (gelé par défaut)
    n_outdoor_features: int,
    n_targets: int,
    correction_hidden_size: int = 32,   # config.FULL_CORRECTION_HIDDEN_SIZE
    num_layers: int = 2,
    dropout: float = 0.1,
    history_hours: float,
    freeze_limited: bool = True,         # True = limited non ré-entraîné
)
```

**Forward :**
```
x_limited: (batch, n_steps, n_limited_features)
x_outdoor: (batch, n_steps, n_outdoor_features)
→ (final_pred, base_pred, correction)   # tout en (batch, n_targets)
```

`base_pred` et `correction` sont exposés pour le diagnostic (on peut vérifier
que la correction est petite par rapport à la prédiction de base).

**Avantage du freeze :** si les capteurs façades tombent en panne, on peut
retomber sur `limited` seul sans réentraînement (dégradation gracieuse).

**Sauvegarde checkpoint (`checkpoints/full.pt`) :**
Idem `limited.pt` + `n_outdoor_features`, `outdoor_columns`, `outdoor_stats`.

---

## Dimensions typiques (avec la config par défaut)

| Variable | Valeur typique |
|---|---|
| `n_steps` | 108 (18h, multi-résolution) |
| `n_limited_features` | 17 (3 météo + 6 solaires + 8 volets/fenêtres) |
| `n_outdoor_features` | 12 (4 façades × 3 mesures) |
| `n_targets` | 8 (4 pièces × température + humidité) |
| `hidden_size` (limited encoder) | 64 |
| `correction_hidden_size` (full) | 32 |

---

## Export ONNX

Les deux modèles sont exportés en ONNX après chaque entraînement.

- `checkpoints/limited.onnx` — entrée : `x_limited (batch, n_steps, n_limited_features)`
- `checkpoints/full.onnx` — entrées : `x_limited`, `x_outdoor`

Les sessions ONNX Runtime sont chargées via `onnxruntime.InferenceSession`.
Voir `train.py::export_onnx()` pour les détails d'export.
