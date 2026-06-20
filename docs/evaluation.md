# evaluate.py — Évaluation des modèles

---

## Chargement des modèles

### `load_limited_model(checkpoint_path)`

Reconstruit un `LimitedModel` depuis `checkpoints/limited.pt` et le passe en
mode évaluation (`model.eval()`).

Retourne `(model, checkpoint)` — le dictionnaire `checkpoint` contient les
métadonnées nécessaires pour reconstruire exactement le même dataset
(colonnes, stats de normalisation, fenêtre d'historique).

### `load_full_model(checkpoint_path, limited_checkpoint_path)`

Reconstruit un `FullModel` en chargeant d'abord le `limited` depuis son propre
checkpoint, puis le `full`. Retourne `(model, checkpoint)`.

---

## `prepare_eval_dataset(checkpoint)`

Reconstruit le `HouseDataset` avec **exactement la même normalisation**
`limited`/`target` que lors de l'entraînement. Les stats sont injectées via
le paramètre `stats` de `train.prepare_dataset`.

Garantit que les prédictions dénormalisées sont comparables à la vérité
terrain.

---

## `compute_metrics(model, dataset, val_subset, is_full)`

Calcule MAE et RMSE par cible, **en unités réelles** (dénormalisées).

```python
report, preds_real, targets_real = compute_metrics(model, dataset, val_subset, is_full=False)
# report["indoor__salon__temperature"] == {"mae": 0.78, "rmse": 1.00}
```

Retourne aussi `preds_real` et `targets_real` sous forme numpy (utiles pour
les graphiques).

---

## `sensitivity_report(model, dataset, val_subset, is_full, n_samples=300)`

Test de sensibilité physique : pour chaque pièce et chaque type d'état
(volet/fenêtre), bascule la feature `house__<room>__<type>` de 0 (fermé) à 1
(ouvert) sur toute la fenêtre d'historique d'un échantillon, à météo
identique, et mesure le delta de température prédit.

**Effets attendus :**

| État | Condition | Delta attendu |
|---|---|---|
| Volet ouvert | Irradiance > 0 (jour) | > 0 (gain solaire) |
| Volet ouvert | Irradiance ≈ 0 (nuit) | ≈ 0 (inertie thermique possible) |
| Fenêtre ouverte | T_ext > T_int | > 0 (la pièce se réchauffe) |
| Fenêtre ouverte | T_ext < T_int | < 0 (la pièce se refroidit) |

> Note : le test bascule l'état sur **toute** la fenêtre de 18h. Un delta
> "nuit" non nul pour les volets peut refléter l'inertie thermique (chaleur
> accumulée pendant les heures de jour précédentes dans l'historique) — ce
> n'est pas forcément une anomalie.

---

## `plot_predictions(dataset, val_subset, preds_real, targets_real, model_name)`

Génère un graphique `prédiction vs. vérité terrain` par cible sur 3 jours de
validation, sauvegardé dans `evaluation/<model_name>_<col>.png`.

---

## `evaluate_model(model_name)`

Orchestre le tout : charge le modèle, reconstruit le dataset, calcule les
métriques, affiche le rapport de sensibilité et génère les graphiques.

```bash
python home_model.py evaluate --model limited
python home_model.py evaluate --model full
```

---

## Résultats de référence (90 jours simulés)

| Capteur | limited MAE | full MAE |
|---|---|---|
| temp salon | 0.78 °C | 0.57 °C |
| temp chambre1 | 1.05 °C | 0.70 °C |
| temp chambre2 | 0.86 °C | 0.67 °C |
| temp bureau | 0.83 °C | 0.48 °C |
| humidité | 1.7–3.2 %HR | 1.6–2.8 %HR |

`full` apporte un gain net sur `limited` grâce aux capteurs façades.
