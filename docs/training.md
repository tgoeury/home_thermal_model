# train.py — Entraînement des modèles

---

## `prepare_dataset(history_hours, horizon_steps, train_fraction, stats)`

Charge les données, construit le dataset fenêtré, normalise et découpe
chronologiquement.

```python
dataset, train_subset, val_subset = prepare_dataset(
    history_hours=18.0,    # longueur de la fenêtre d'historique
    horizon_steps=1,       # t+2min par défaut
    train_fraction=0.8,    # 80% train, 20% val (chronologiquement)
    stats=None,            # réutiliser des stats existantes (pour train-full)
)
```

**Normalisation :**
- Par défaut, les stats (mean/std) sont calculées sur la portion d'entraînement
  uniquement.
- Si `stats` est fourni (ex: stats `limited` + `target` du checkpoint `limited`),
  ces valeurs sont réutilisées telles quelles. Seules les stats `outdoor`
  (absentes de `limited`) sont recalculées à neuf. Cette réutilisation est
  essentielle pour rester cohérent avec le modèle `limited` gelé dans `full`.

---

## `train_limited_model(dataset, train_subset, val_subset, ...)`

Entraîne un `LimitedModel` avec arrêt anticipé.

```python
model = train_limited_model(
    dataset, train_subset, val_subset,
    num_epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    patience=8,
)
```

---

## `train_full_model(dataset, limited_model, train_subset, val_subset, ...)`

Entraîne un `FullModel` en gardant `limited_model` gelé. Seuls les paramètres
du `outdoor_encoder` et du `correction_head` sont mis à jour.

Lève `ValueError` si aucun capteur extérieur (`outdoor__*`) n'est disponible.

---

## `export_onnx(model, checkpoint, output_path)`

Exporte un modèle PyTorch en ONNX avec des axes dynamiques pour le batch.

```python
export_onnx(model, checkpoint, config.CHECKPOINT_DIR / "limited.onnx")
```

Les noms d'entrée/sortie ONNX correspondent aux clés du batch PyTorch :
- `limited` : `["x_limited"]` → `["predictions"]`
- `full` : `["x_limited", "x_outdoor"]` → `["predictions", "base_pred", "correction"]`

---

## Boucle interne `_early_stopping_loop`

Boucle d'entraînement commune aux deux modèles, avec :
- Sauvegarde de l'état du modèle au meilleur `val_loss`.
- Restauration de ce meilleur état avant de retourner (évite de renvoyer un
  modèle en surapprentissage si `val_loss` remonte après le minimum).
- Arrêt après `patience` epochs consécutives sans amélioration.

```
epoch 1 → val_loss=0.24  ✓ nouveau meilleur
epoch 2 → val_loss=0.26  (pas d'amélioration, compteur=1)
...
epoch 9 → arrêt anticipé, meilleur val_loss=0.24
```

---

## Workflow d'entraînement recommandé

```bash
# Générer les données (90 jours = ~65 000 échantillons après fenêtrage)
python home_model.py simulate --days 90

# Entraîner limited (early stopping typique : 8–15 epochs sur 90 jours)
python home_model.py train-limited --epochs 50

# Vérifier que les colonnes n'ont pas changé depuis limited, puis entraîner full
python home_model.py train-full --epochs 50
```

Si les colonnes `limited_columns` ou `target_columns` ont changé depuis le
dernier `train-limited` (nouveau capteur ajouté), `train-full` lève une erreur
explicite et demande de réentraîner `limited` d'abord.
