"""Point d'entrée principal du projet de modélisation thermique de la maison.

Sous-commandes :
  simulate      Génère un jeu de données simulées dans data_raw/
  train-limited Entraîne le modèle "limited" (météo + solaire + volets/fenêtres)
  train-full    Entraîne le modèle "full" (limited gelé + capteurs façades)
  evaluate      Évalue un modèle entraîné contre la vérité terrain simulée
  plan          Recommande un planning volets/fenêtres sur 24h (modèle "limited")
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

import config
import evaluate as evaluate_module
import train as train_module
from data.sources import simulated
from models.full import FullModel
from models.limited import LimitedModel
from strategy import format as strategy_format
from strategy import planner as strategy_planner

LIMITED_CHECKPOINT = config.CHECKPOINT_DIR / "limited.pt"
FULL_CHECKPOINT = config.CHECKPOINT_DIR / "full.pt"


def cmd_simulate(args: argparse.Namespace) -> None:
    end = pd.Timestamp.now(tz="UTC").floor(f"{config.SAMPLE_INTERVAL_MINUTES}min")
    start = end - pd.Timedelta(days=args.days)
    simulated.generate_and_write(start, end, forecast_horizon_hours=args.forecast_hours)
    print(
        f"Données simulées écrites dans {config.DATA_RAW_DIR} : {start} -> {end} "
        f"(+{args.forecast_hours}h de prévisions météo)."
    )


def cmd_train_limited(args: argparse.Namespace) -> None:
    dataset, train_subset, val_subset = train_module.prepare_dataset(history_hours=args.history_hours)
    print(
        f"{len(dataset)} échantillons ({len(train_subset)} train / {len(val_subset)} val), "
        f"{dataset.n_limited_features} features 'limited', {dataset.n_targets} cibles."
    )

    model = train_module.train_limited_model(dataset, train_subset, val_subset, num_epochs=args.epochs)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "n_limited_features": dataset.n_limited_features,
        "n_targets": dataset.n_targets,
        "limited_columns": dataset.limited_columns,
        "target_columns": dataset.target_columns,
        "limited_stats": dataset.limited_stats,
        "target_stats": dataset.target_stats,
        "history_hours": args.history_hours,
        "horizon_steps": config.PREDICTION_HORIZON_STEPS,
    }
    torch.save(checkpoint, LIMITED_CHECKPOINT)
    print(f"Modèle 'limited' sauvegardé dans {LIMITED_CHECKPOINT}")

    onnx_path = config.CHECKPOINT_DIR / "limited.onnx"
    train_module.export_onnx(model, checkpoint, onnx_path)
    print(f"Modèle 'limited' exporté en ONNX dans {onnx_path}")


def cmd_train_full(args: argparse.Namespace) -> None:
    if not LIMITED_CHECKPOINT.exists():
        raise SystemExit(f"Modèle 'limited' introuvable ({LIMITED_CHECKPOINT}). Lancez d'abord 'train-limited'.")

    checkpoint = torch.load(LIMITED_CHECKPOINT, weights_only=False)
    history_hours = checkpoint["history_hours"]

    stats = {"limited": checkpoint["limited_stats"], "target": checkpoint["target_stats"]}
    dataset, train_subset, val_subset = train_module.prepare_dataset(history_hours=history_hours, stats=stats)
    print(
        f"{len(dataset)} échantillons ({len(train_subset)} train / {len(val_subset)} val), "
        f"{dataset.n_limited_features} features 'limited', {dataset.n_outdoor_features} features 'outdoor', "
        f"{dataset.n_targets} cibles."
    )

    if dataset.limited_columns != checkpoint["limited_columns"] or dataset.target_columns != checkpoint["target_columns"]:
        raise SystemExit(
            "Les colonnes du dataset ont changé depuis l'entraînement de 'limited' "
            "(nouveaux capteurs ?). Réentraînez 'limited' avant 'full'."
        )

    limited_model = LimitedModel(
        n_limited_features=checkpoint["n_limited_features"],
        n_targets=checkpoint["n_targets"],
        history_hours=history_hours,
    )
    limited_model.load_state_dict(checkpoint["model_state"])

    model = train_module.train_full_model(dataset, limited_model, train_subset, val_subset, num_epochs=args.epochs)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    full_checkpoint = {
        "model_state": model.state_dict(),
        "n_limited_features": dataset.n_limited_features,
        "n_outdoor_features": dataset.n_outdoor_features,
        "n_targets": dataset.n_targets,
        "limited_columns": dataset.limited_columns,
        "outdoor_columns": dataset.outdoor_columns,
        "target_columns": dataset.target_columns,
        "limited_stats": dataset.limited_stats,
        "outdoor_stats": dataset.outdoor_stats,
        "target_stats": dataset.target_stats,
        "history_hours": history_hours,
        "horizon_steps": config.PREDICTION_HORIZON_STEPS,
    }
    torch.save(full_checkpoint, FULL_CHECKPOINT)
    print(f"Modèle 'full' sauvegardé dans {FULL_CHECKPOINT}")

    onnx_path = config.CHECKPOINT_DIR / "full.onnx"
    train_module.export_onnx(model, full_checkpoint, onnx_path)
    print(f"Modèle 'full' exporté en ONNX dans {onnx_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    evaluate_module.evaluate_model(args.model)


def cmd_export_onnx(args: argparse.Namespace) -> None:
    if args.model == "limited":
        model, checkpoint = evaluate_module.load_limited_model()
        onnx_path = config.CHECKPOINT_DIR / "limited.onnx"
    elif args.model == "full":
        model, checkpoint = evaluate_module.load_full_model()
        onnx_path = config.CHECKPOINT_DIR / "full.onnx"
    else:
        raise SystemExit(f"Modèle inconnu : {args.model!r}")

    train_module.export_onnx(model, checkpoint, onnx_path)
    print(f"Modèle '{args.model}' exporté en ONNX dans {onnx_path}")


def cmd_plan(args: argparse.Namespace) -> None:
    comfort_ranges = None
    if args.comfort_ranges:
        raw = json.loads(args.comfort_ranges)
        comfort_ranges = {room: (float(bounds[0]), float(bounds[1])) for room, bounds in raw.items()}

    result = strategy_planner.plan(
        n_candidates=args.n_candidates,
        horizon_hours=args.horizon_hours,
        block_hours=args.block_hours,
        eval_step_minutes=args.eval_step_minutes,
        comfort_ranges=comfort_ranges,
    )

    print(
        f"Planning généré à partir de {result['generated_at']} "
        f"(horizon {result['horizon_hours']:.1f}h, {result['n_candidates']} plannings testés, "
        f"coût de confort estimé = {result['best_cost']:.2f})"
    )
    for room, intervals in result["rooms"].items():
        print(f"\n{room} :")
        for interval in intervals:
            print(
                f"  {interval['from']} -> {interval['to']} : "
                f"volet={interval['shutter']}, fenêtre={interval['window']}"
            )

    config.STRATEGY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    output_path = config.STRATEGY_OUTPUT_DIR / f"plan_{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nPlanning écrit dans {output_path}")

    csv_path = config.PROJECT_ROOT / f"{timestamp}_strategy24.csv"
    strategy_format.write_plan_csv(result, csv_path, config.HOUSE_STATE_TYPES)
    print(f"Planning écrit dans {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modélisation thermique de la maison")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim = subparsers.add_parser("simulate", help="Générer des données simulées dans data_raw/")
    sim.add_argument("--days", type=int, default=30, help="Nombre de jours de données à générer")
    sim.add_argument("--forecast-hours", type=float, default=24.0, help="Horizon de prévision météo simulé")
    sim.set_defaults(func=cmd_simulate)

    train_limited = subparsers.add_parser("train-limited", help="Entraîner le modèle 'limited'")
    train_limited.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    train_limited.add_argument("--history-hours", type=float, default=config.HISTORY_HOURS)
    train_limited.set_defaults(func=cmd_train_limited)

    train_full = subparsers.add_parser("train-full", help="Entraîner le modèle 'full' (nécessite 'limited')")
    train_full.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    train_full.set_defaults(func=cmd_train_full)

    evaluate = subparsers.add_parser("evaluate", help="Évaluer un modèle entraîné contre la vérité terrain simulée")
    evaluate.add_argument("--model", choices=["limited", "full"], default="limited")
    evaluate.set_defaults(func=cmd_evaluate)

    export_onnx = subparsers.add_parser("export-onnx", help="Exporter un modèle entraîné en ONNX")
    export_onnx.add_argument("--model", choices=["limited", "full"], default="limited")
    export_onnx.set_defaults(func=cmd_export_onnx)

    plan = subparsers.add_parser("plan", help="Recommander un planning volets/fenêtres (modèle 'limited')")
    plan.add_argument("--n-candidates", type=int, default=config.PLANNING_N_CANDIDATES)
    plan.add_argument("--horizon-hours", type=float, default=config.PLANNING_HORIZON_HOURS)
    plan.add_argument("--block-hours", type=float, default=config.PLANNING_BLOCK_HOURS)
    plan.add_argument("--eval-step-minutes", type=float, default=config.PLANNING_EVAL_STEP_MINUTES)
    plan.add_argument(
        "--comfort-ranges", type=str, default=None,
        help='JSON {"piece": [t_min, t_max], ...} — plage de confort par pièce '
             '(retombe sur COMFORT_TEMP_MIN/MAX pour les pièces absentes)',
    )
    plan.set_defaults(func=cmd_plan)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
