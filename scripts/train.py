"""Full training pipeline for Proctoring CV object detector.

Supports scratch initialization from YAML, environment diagnostics, preflight checks,
checkpoint saving, and Google Drive artifact synchronization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import torch

from proctoring_cv.config import load_config, save_config, AppConfig
from proctoring_cv.detector import build_detector
from proctoring_cv.environment import save_environment_snapshot, recommend_batch_size
from proctoring_cv.logging_utils import setup_logger
from proctoring_cv.reproducibility import set_seed, get_git_metadata, compute_model_parameter_hash
from proctoring_cv.checkpoint_manager import CheckpointManager
from proctoring_cv.drive_sync import DriveSyncManager
from scripts.validate_dataset import validate_yolo_dataset


def run_training(
    config_path: Path | str,
    experiment_id: Optional[str] = None,
    drive_root: Optional[str] = None,
    mode: str = "full",
    epochs_override: Optional[int] = None,
) -> int:
    """Executes object detector training with reproducibility, integrity checks, and drive sync."""
    logger = setup_logger("train")

    overrides = {}
    if experiment_id:
        overrides["experiment_id"] = experiment_id
    if drive_root:
        overrides["drive_root"] = drive_root
    if epochs_override is not None:
        overrides["training.epochs"] = epochs_override

    config: AppConfig = load_config(config_path, cli_overrides=overrides)
    set_seed(config.training.seed, deterministic=config.training.deterministic)

    logger.info(f"=== Starting Training for Experiment: {config.experiment_id} ===")
    logger.info(f"Model Architecture: {config.model.architecture} (Mode: {config.model.initialization_method})")
    logger.info(f"Pretrained flag: {config.model.pretrained}")

    # Set up experiment directories locally & on Drive
    exp_dir = Path("runs") / "experiments" / config.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    metrics_dir = exp_dir / "metrics"
    for d in (checkpoints_dir, logs_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Environment snapshot
    env_file = exp_dir / "environment.json"
    env_info = save_environment_snapshot(env_file)
    logger.info(f"Environment: PyTorch {env_info['torch_version']}, CUDA available: {env_info['cuda_available']}")

    # 2. Git metadata
    git_meta = get_git_metadata()
    with open(exp_dir / "git_metadata.json", "w", encoding="utf-8") as f:
        json.dump(git_meta, f, indent=2)

    # 3. Dataset Preflight & Validation
    dataset_root = Path(config.dataset.root)
    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.is_file():
        # Auto-create if folder structure exists
        dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        with open(dataset_yaml, "w", encoding="utf-8") as f:
            yaml.dump({
                "path": str(dataset_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": config.dataset.names,
                "nc": config.dataset.nc,
            }, f)

    logger.info(f"Validating dataset at: {dataset_root}")
    is_valid, val_report = validate_yolo_dataset(dataset_root, dry_run=False)
    if not is_valid and mode == "full":
        logger.error("Dataset validation failed prior to training. Halting run.")
        return 1

    # 4. Save Config snapshot
    save_config(config, exp_dir / "config.yaml")

    # 5. Build detector model in scratch mode
    logger.info(f"Constructing scratch YOLO model from {config.model.architecture}...")
    model = build_detector(config, mode="scratch")

    # 6. Verify and record random initialization proof
    init_hash = compute_model_parameter_hash(model.model)
    init_proof = {
        "architecture": config.model.architecture,
        "initialization_method": config.model.initialization_method,
        "seed": config.training.seed,
        "parameter_hash": init_hash,
        "pretrained": config.model.pretrained,
    }
    with open(exp_dir / "initialization_proof.json", "w", encoding="utf-8") as f:
        json.dump(init_proof, f, indent=2)
    logger.info(f"Initialization proof recorded. Parameter SHA-256: {init_hash[:16]}...")

    # 7. Determine safe batch size & device
    device = config.training.device
    if not torch.cuda.is_available() and device != "cpu":
        logger.warning("CUDA device requested but unavailable; falling back to CPU.")
        device = "cpu"

    batch_size = config.training.batch_size
    if device != "cpu":
        safe_batch = recommend_batch_size(config.training.imgsz)
        batch_size = min(batch_size, safe_batch)
    logger.info(f"Selected device: {device}, Batch size: {batch_size}, Image size: {config.training.imgsz}")

    # 8. Set up Checkpoint and Drive Sync Managers
    checkpoint_mgr = CheckpointManager(
        output_dir=checkpoints_dir,
        experiment_id=config.experiment_id,
        keep_periodic=config.training.keep_periodic_checkpoints,
    )
    drive_mgr = DriveSyncManager(drive_root=config.drive_root)

    # 9. Train Model
    logger.info(f"Starting training for {config.training.epochs} epochs...")
    try:
        train_results = model.train(
            data=str(dataset_yaml),
            epochs=config.training.epochs,
            imgsz=config.training.imgsz,
            batch=batch_size,
            optimizer=config.training.optimizer,
            lr0=config.training.lr0,
            lrf=config.training.lrf,
            weight_decay=config.training.weight_decay,
            seed=config.training.seed,
            deterministic=config.training.deterministic,
            amp=config.training.amp if device != "cpu" else False,
            workers=config.training.workers,
            device=device,
            project=str(exp_dir / "train_runs"),
            name="train",
            exist_ok=True,
            val=True,
            plots=True,
        )

        logger.info("Training completed successfully!")

        # 10. Sync artifacts to Google Drive
        drive_dest = drive_mgr.sync_experiment_directory(
            local_exp_dir=exp_dir,
            experiment_id=config.experiment_id,
        )
        logger.info(f"Durable artifacts synchronized to Drive destination: {drive_dest}")
        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Proctoring Object Detector")
    parser.add_argument("--config", type=str, default="configs/experiments/yolo11n_scratch_coco.yaml")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--drive-root", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "dry-run"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    sys.exit(run_training(
        config_path=args.config,
        experiment_id=args.experiment_id,
        drive_root=args.drive_root,
        mode=args.mode,
        epochs_override=args.epochs,
    ))


if __name__ == "__main__":
    main()
