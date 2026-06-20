"""Boucles d'entraînement pour les modèles "limited" et "full"."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import config
from data.pipeline import HouseDataset, build_feature_table
from models.full import FullModel
from models.limited import LimitedModel


def prepare_dataset(
    history_hours: float = config.HISTORY_HOURS,
    horizon_steps: int = config.PREDICTION_HORIZON_STEPS,
    train_fraction: float = 0.8,
    stats: dict | None = None,
) -> tuple[HouseDataset, Subset, Subset]:
    """Charge les données, construit le dataset fenêtré, normalise et découpe
    chronologiquement en train/validation.

    Par défaut, les statistiques de normalisation sont calculées sur la
    portion d'entraînement. Si `stats` est fourni (ex: stats `limited`/
    `target` sauvegardées lors de l'entraînement de "limited"), elles sont
    réutilisées telles quelles pour rester cohérentes avec un modèle gelé —
    seules les stats `outdoor` (absentes de "limited") sont alors calculées
    fraîchement si nécessaire.
    """
    table = build_feature_table()
    dataset = HouseDataset(table, history_hours=history_hours, horizon_steps=horizon_steps)

    train_idx, val_idx = dataset.chronological_split(train_fraction)
    if len(train_idx) == 0:
        raise ValueError("Pas assez de données pour constituer un jeu d'entraînement.")

    computed_stats = dataset.compute_normalization_stats(up_to_sample=len(train_idx))
    if stats is not None:
        for key in ("limited", "target"):
            if stats.get(key) is not None:
                computed_stats[key] = stats[key]
    dataset.apply_normalization(computed_stats)

    return dataset, Subset(dataset, train_idx), Subset(dataset, val_idx)


def _run_epoch(model, loader, optimizer, loss_fn, is_full: bool, train: bool) -> float:
    model.train(mode=train)
    total_loss = 0.0
    n_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            x_limited = batch["x_limited"]
            y = batch["y"]

            if is_full:
                pred, _, _ = model(x_limited, batch["x_outdoor"])
            else:
                pred = model(x_limited)

            loss = loss_fn(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = x_limited.shape[0]
            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples


def _early_stopping_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    is_full: bool,
    num_epochs: int,
    patience: int,
    tag: str,
) -> torch.nn.Module:
    """Boucle d'entraînement avec arrêt anticipé sur val_loss.

    Conserve le meilleur état du modèle (sur val_loss) et le restaure avant
    de retourner, pour éviter de renvoyer un modèle surentraîné si val_loss
    remonte après l'epoch où il était minimal.
    """
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, loss_fn, is_full=is_full, train=True)
        val_loss = _run_epoch(model, val_loader, optimizer, loss_fn, is_full=is_full, train=False)
        print(f"[{tag}] epoch {epoch + 1}/{num_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[{tag}] arrêt anticipé (epoch {epoch + 1}, meilleur val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model


def train_limited_model(
    dataset: HouseDataset,
    train_subset: Subset,
    val_subset: Subset,
    num_epochs: int = config.NUM_EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    batch_size: int = config.BATCH_SIZE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
) -> LimitedModel:
    model = LimitedModel(n_limited_features=dataset.n_limited_features, n_targets=dataset.n_targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return _early_stopping_loop(
        model, train_loader, val_loader, optimizer, loss_fn, is_full=False,
        num_epochs=num_epochs, patience=patience, tag="limited",
    )


def train_full_model(
    dataset: HouseDataset,
    limited_model: LimitedModel,
    train_subset: Subset,
    val_subset: Subset,
    num_epochs: int = config.NUM_EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    batch_size: int = config.BATCH_SIZE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
) -> FullModel:
    if dataset.n_outdoor_features == 0:
        raise ValueError("Aucun capteur extérieur (outdoor__*) disponible : impossible d'entraîner 'full'.")

    model = FullModel(
        limited_model,
        n_outdoor_features=dataset.n_outdoor_features,
        n_targets=dataset.n_targets,
    )
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
    )
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return _early_stopping_loop(
        model, train_loader, val_loader, optimizer, loss_fn, is_full=True,
        num_epochs=num_epochs, patience=patience, tag="full",
    )


def export_onnx(model: torch.nn.Module, checkpoint: dict, output_path: Path) -> None:
    """Exporte un modèle PyTorch entraîné en ONNX.

    Les noms d'entrée/sortie correspondent aux clés du batch PyTorch :
    - limited : ["x_limited"] → ["predictions"]
    - full    : ["x_limited", "x_outdoor"] → ["predictions", "base_pred", "correction"]
    """
    model.eval()
    from data.pipeline import compute_window_offsets
    n_steps = len(compute_window_offsets(config.resolution_segments_for(checkpoint["history_hours"])))

    is_full = "n_outdoor_features" in checkpoint

    dummy_limited = torch.zeros(1, n_steps, checkpoint["n_limited_features"])
    dynamic_axes = {"x_limited": {0: "batch"}, "predictions": {0: "batch"}}

    export_kwargs = dict(opset_version=17, dynamo=False)

    if is_full:
        dummy_outdoor = torch.zeros(1, n_steps, checkpoint["n_outdoor_features"])
        dynamic_axes.update({
            "x_outdoor": {0: "batch"},
            "base_pred": {0: "batch"},
            "correction": {0: "batch"},
        })
        torch.onnx.export(
            model,
            (dummy_limited, dummy_outdoor),
            str(output_path),
            input_names=["x_limited", "x_outdoor"],
            output_names=["predictions", "base_pred", "correction"],
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )
    else:
        torch.onnx.export(
            model,
            dummy_limited,
            str(output_path),
            input_names=["x_limited"],
            output_names=["predictions"],
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )
