"""Évaluation des modèles "limited" et "full" contre la vérité terrain simulée.

Comme `data/sources/simulated.py` génère les températures intérieures avec un
modèle physique jouet *connu* (perte thermique, gain solaire, ventilation),
on peut évaluer au-delà des métriques globales :

  - métriques (MAE/RMSE) par capteur, en unités réelles (°C, %HR)
  - test de sensibilité : à historique identique, ouvrir/fermer un volet ou
    une fenêtre doit faire bouger la prédiction dans le sens physiquement
    attendu (gain solaire >= 0 volet ouvert de jour, tendance vers la
    température extérieure fenêtre ouverte)
  - tracé prédiction vs réalité sur quelques jours
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import config
import train as train_module
from data.pipeline import HouseDataset
from models.full import FullModel
from models.limited import LimitedModel

EVAL_DIR = config.PROJECT_ROOT / "evaluation"


# ---------------------------------------------------------------------------
# Chargement des modèles entraînés
# ---------------------------------------------------------------------------

def load_limited_model(checkpoint_path: Path = config.CHECKPOINT_DIR / "limited.pt"):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = LimitedModel(
        n_limited_features=checkpoint["n_limited_features"],
        n_targets=checkpoint["n_targets"],
        history_hours=checkpoint["history_hours"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_full_model(
    checkpoint_path: Path = config.CHECKPOINT_DIR / "full.pt",
    limited_checkpoint_path: Path = config.CHECKPOINT_DIR / "limited.pt",
):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    limited_checkpoint = torch.load(limited_checkpoint_path, weights_only=False)

    limited_model = LimitedModel(
        n_limited_features=limited_checkpoint["n_limited_features"],
        n_targets=limited_checkpoint["n_targets"],
        history_hours=limited_checkpoint["history_hours"],
    )
    model = FullModel(
        limited_model,
        n_outdoor_features=checkpoint["n_outdoor_features"],
        n_targets=checkpoint["n_targets"],
        history_hours=checkpoint["history_hours"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Chargement des sessions ONNX Runtime (planification / inférence temps réel)
# ---------------------------------------------------------------------------
# Les métadonnées (stats de normalisation, colonnes, dimensions) sont
# embarquées dans le fichier .onnx lui-même (metadata_props) à l'export.
# Ces fonctions ne requièrent ni PyTorch ni le fichier .pt — uniquement
# onnxruntime et numpy, adaptés à un déploiement sur petit hardware (RPi…).

def _onnx_meta_to_checkpoint(sess) -> dict:
    """Reconstruit un dict checkpoint-like depuis les metadata_props ONNX."""
    import json
    import numpy as np
    from data.pipeline import FeatureStats

    raw = sess.get_modelmeta().custom_metadata_map.get("home_model_meta")
    if raw is None:
        raise ValueError(
            "Ce fichier ONNX ne contient pas de métadonnées 'home_model_meta'. "
            "Réexportez le modèle avec la version actuelle de train.export_onnx()."
        )
    meta = json.loads(raw)

    def _stats(d):
        if d is None:
            return None
        return FeatureStats(mean=np.array(d["mean"]), std=np.array(d["std"]))

    checkpoint: dict = {
        "n_limited_features": meta["n_limited_features"],
        "n_targets": meta["n_targets"],
        "history_hours": meta["history_hours"],
        "horizon_steps": meta["horizon_steps"],
        "limited_columns": meta["limited_columns"],
        "target_columns": meta["target_columns"],
        "limited_stats": _stats(meta["limited_stats"]),
        "target_stats": _stats(meta["target_stats"]),
    }
    if "n_outdoor_features" in meta:
        checkpoint["n_outdoor_features"] = meta["n_outdoor_features"]
        checkpoint["outdoor_columns"] = meta["outdoor_columns"]
        checkpoint["outdoor_stats"] = _stats(meta["outdoor_stats"])
    return checkpoint


def load_limited_onnx(onnx_path: Path = config.CHECKPOINT_DIR / "limited.onnx"):
    """Session ORT + métadonnées pour le modèle 'limited'.

    Aucune dépendance PyTorch — uniquement onnxruntime + numpy.
    Utilisé pour la planification volets/fenêtres.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    checkpoint = _onnx_meta_to_checkpoint(sess)
    return sess, checkpoint


def load_full_onnx(onnx_path: Path = config.CHECKPOINT_DIR / "full.onnx"):
    """Session ORT + métadonnées pour le modèle 'full'.

    Aucune dépendance PyTorch — uniquement onnxruntime + numpy.
    Utilisé pour l'inférence temps réel (x_limited + x_outdoor).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    checkpoint = _onnx_meta_to_checkpoint(sess)
    return sess, checkpoint


def prepare_eval_dataset(checkpoint: dict) -> tuple[HouseDataset, Subset, Subset]:
    """Reconstruit le dataset avec exactement la même normalisation
    `limited`/`target` que celle utilisée à l'entraînement."""
    stats = {"limited": checkpoint["limited_stats"], "target": checkpoint["target_stats"]}
    return train_module.prepare_dataset(
        history_hours=checkpoint["history_hours"],
        horizon_steps=checkpoint["horizon_steps"],
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def _predict(model, loader: DataLoader, is_full: bool) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_full:
                pred, _, _ = model(batch["x_limited"], batch["x_outdoor"])
            else:
                pred = model(batch["x_limited"])
            preds.append(pred.numpy())
            targets.append(batch["y"].numpy())
    return np.concatenate(preds), np.concatenate(targets)


def compute_metrics(
    model, dataset: HouseDataset, val_subset: Subset, is_full: bool, batch_size: int = config.BATCH_SIZE
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray]:
    """MAE/RMSE par cible, en unités réelles. Retourne aussi les prédictions
    et cibles dénormalisées (utiles pour les graphiques)."""
    loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    preds, targets = _predict(model, loader, is_full)

    preds_real = dataset.target_stats.inverse_transform(preds)
    targets_real = dataset.target_stats.inverse_transform(targets)

    errors = preds_real - targets_real
    mae = np.abs(errors).mean(axis=0)
    rmse = np.sqrt((errors**2).mean(axis=0))

    report = {
        col: {"mae": float(mae[i]), "rmse": float(rmse[i])} for i, col in enumerate(dataset.target_columns)
    }
    return report, preds_real, targets_real


# ---------------------------------------------------------------------------
# Test de sensibilité physique (volets / fenêtres)
# ---------------------------------------------------------------------------

def _normalized_value(stats, feature_idx: int, raw_value: float) -> float:
    return float((raw_value - stats.mean[feature_idx]) / stats.std[feature_idx])


def sensitivity_report(
    model,
    dataset: HouseDataset,
    val_subset: Subset,
    is_full: bool,
    n_samples: int = 300,
    seed: int = config.RANDOM_SEED,
) -> dict[tuple[str, str], dict[str, float]]:
    """Pour chaque pièce, mesure l'effet (volet/fenêtre fermé -> ouvert) sur
    la température prédite, à historique météo/solaire identique.

    Effets attendus :
      - volet : gain >= 0 de jour (irradiance > 0), ~0 la nuit.
      - fenêtre : la prédiction doit se rapprocher de la température
        extérieure (delta > 0 si dehors plus chaud que dedans, < 0 sinon).
    """
    rng = np.random.default_rng(seed)
    sep = config.COLUMN_SEP

    n = min(n_samples, len(val_subset))
    sample_positions = rng.choice(len(val_subset), size=n, replace=False)

    irradiance_col = f"weather{sep}solar_irradiance"
    outdoor_temp_col = f"weather{sep}outdoor_temperature"
    irradiance_idx = dataset.limited_columns.index(irradiance_col)
    outdoor_temp_idx = dataset.limited_columns.index(outdoor_temp_col)

    results: dict[tuple[str, str], dict[str, float]] = {}

    for room in config.DEFAULT_HOUSE_STATE_ROOMS:
        temp_col = f"indoor{sep}{room}{sep}temperature"
        if temp_col not in dataset.target_columns:
            continue
        target_idx = dataset.target_columns.index(temp_col)

        for state_type in config.HOUSE_STATE_TYPES:
            feature_col = f"house{sep}{room}{sep}{state_type}"
            if feature_col not in dataset.limited_columns:
                continue
            feature_idx = dataset.limited_columns.index(feature_col)

            closed_value = _normalized_value(dataset.limited_stats, feature_idx, 0.0)
            open_value = _normalized_value(dataset.limited_stats, feature_idx, 1.0)

            deltas = np.empty(n)
            irradiance_now = np.empty(n)
            temp_diff_now = np.empty(n)

            with torch.no_grad():
                for j, pos in enumerate(sample_positions):
                    sample = val_subset[int(pos)]
                    x_limited = sample["x_limited"]

                    x_closed = x_limited.clone()
                    x_closed[:, feature_idx] = closed_value
                    x_open = x_limited.clone()
                    x_open[:, feature_idx] = open_value

                    if is_full:
                        x_outdoor = sample["x_outdoor"].unsqueeze(0)
                        pred_closed, _, _ = model(x_closed.unsqueeze(0), x_outdoor)
                        pred_open, _, _ = model(x_open.unsqueeze(0), x_outdoor)
                    else:
                        pred_closed = model(x_closed.unsqueeze(0))
                        pred_open = model(x_open.unsqueeze(0))

                    pred_closed_real = dataset.target_stats.inverse_transform(pred_closed.numpy())
                    pred_open_real = dataset.target_stats.inverse_transform(pred_open.numpy())
                    deltas[j] = pred_open_real[0, target_idx] - pred_closed_real[0, target_idx]

                    last_step = x_limited[-1]
                    irr = last_step[irradiance_idx] * dataset.limited_stats.std[irradiance_idx] + (
                        dataset.limited_stats.mean[irradiance_idx]
                    )
                    out_t = last_step[outdoor_temp_idx] * dataset.limited_stats.std[outdoor_temp_idx] + (
                        dataset.limited_stats.mean[outdoor_temp_idx]
                    )
                    irradiance_now[j] = irr
                    row = dataset.valid_indices[int(pos)]
                    in_t_now = dataset.table[temp_col].iloc[row]
                    temp_diff_now[j] = out_t - in_t_now

            if state_type == "shutter":
                day_mask = irradiance_now > 50.0
                entry = {
                    "mean_delta_jour": float(deltas[day_mask].mean()) if day_mask.any() else float("nan"),
                    "mean_delta_nuit": float(deltas[~day_mask].mean()) if (~day_mask).any() else float("nan"),
                    "n_jour": int(day_mask.sum()),
                    "n_nuit": int((~day_mask).sum()),
                }
            else:
                warmer_mask = temp_diff_now > 0
                entry = {
                    "mean_delta_dehors_plus_chaud": float(deltas[warmer_mask].mean()) if warmer_mask.any() else float("nan"),
                    "mean_delta_dehors_plus_froid": float(deltas[~warmer_mask].mean()) if (~warmer_mask).any() else float("nan"),
                    "n_plus_chaud": int(warmer_mask.sum()),
                    "n_plus_froid": int((~warmer_mask).sum()),
                }

            results[(room, state_type)] = entry

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_predictions(
    dataset: HouseDataset,
    val_subset: Subset,
    preds_real: np.ndarray,
    targets_real: np.ndarray,
    model_name: str,
    n_days: float = 3.0,
    output_dir: Path = EVAL_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(n_days * 24 * 60 / config.SAMPLE_INTERVAL_MINUTES)
    n_steps = min(n_steps, len(preds_real))

    val_rows = dataset.valid_indices[np.asarray(val_subset.indices[:n_steps])]
    timestamps = dataset.table.index[val_rows + dataset.horizon_steps]

    written = []
    for i, col in enumerate(dataset.target_columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(timestamps, targets_real[:n_steps, i], label="simulé (vérité terrain)")
        ax.plot(timestamps, preds_real[:n_steps, i], label="prédiction", alpha=0.8)
        ax.set_title(f"{model_name} — {col}")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        path = output_dir / f"{model_name}_{col}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_model(model_name: str) -> None:
    if model_name == "limited":
        model, checkpoint = load_limited_model()
        is_full = False
    elif model_name == "full":
        model, checkpoint = load_full_model()
        is_full = True
    else:
        raise ValueError(f"Modèle inconnu : {model_name!r}")

    dataset, _, val_subset = prepare_eval_dataset(checkpoint)

    metrics, preds_real, targets_real = compute_metrics(model, dataset, val_subset, is_full)
    print(f"\n=== Métriques ({model_name}, {len(val_subset)} échantillons de validation) ===")
    for col, values in metrics.items():
        print(f"  {col:35s} MAE={values['mae']:.3f}  RMSE={values['rmse']:.3f}")

    print(f"\n=== Test de sensibilité volets/fenêtres ({model_name}) ===")
    sensitivity = sensitivity_report(model, dataset, val_subset, is_full)
    for (room, state_type), entry in sensitivity.items():
        if state_type == "shutter":
            print(
                f"  {room:10s} volet  : delta jour={entry['mean_delta_jour']:+.3f}°C "
                f"(n={entry['n_jour']}), delta nuit={entry['mean_delta_nuit']:+.3f}°C (n={entry['n_nuit']})"
            )
        else:
            print(
                f"  {room:10s} fenêtre: delta (dehors+chaud)={entry['mean_delta_dehors_plus_chaud']:+.3f}°C "
                f"(n={entry['n_plus_chaud']}), delta (dehors+froid)={entry['mean_delta_dehors_plus_froid']:+.3f}°C "
                f"(n={entry['n_plus_froid']})"
            )

    written = plot_predictions(dataset, val_subset, preds_real, targets_real, model_name=model_name)
    print(f"\nGraphiques écrits dans {EVAL_DIR} : {[p.name for p in written]}")
