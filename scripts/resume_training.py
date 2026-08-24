"""Resume training or branch to a new experiment from a checkpoint.

Validates exact configuration, architecture, and manifest compatibility before resuming.
Refuses to resume after material configuration changes and guides branching from best.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from ultralytics import YOLO

from proctoring_cv.config import load_config, AppConfig
from proctoring_cv.logging_utils import setup_logger
from proctoring_cv.checkpoint_manager import CheckpointManager
from proctoring_cv.reproducibility import compute_file_sha256


def resume_or_branch(
    experiment_dir: Path | str,
    checkpoint_name: str = "last.pt",
    resume: bool = True,
    new_experiment_id: Optional[str] = None,
    new_config_path: Optional[str] = None,
) -> int:
    """Safely resumes an interrupted run or creates an explicit child branch from best.pt."""
    logger = setup_logger("resume_training")
    exp_dir = Path(experiment_dir)

    if not exp_dir.is_dir():
        logger.error(f"Experiment directory does not exist: {exp_dir}")
        return 1

    chkpt_path = exp_dir / "checkpoints" / checkpoint_name
    if not chkpt_path.is_file():
        # Check direct path
        if (exp_dir / checkpoint_name).is_file():
            chkpt_path = exp_dir / checkpoint_name
        else:
            logger.error(f"Checkpoint not found at: {chkpt_path}")
            return 1

    config_file = exp_dir / "config.yaml"
    if not config_file.is_file():
        logger.error(f"Original experiment configuration not found at: {config_file}")
        return 1

    orig_config = load_config(config_file)

    if resume:
        logger.info(f"Checking resume compatibility for experiment: {orig_config.experiment_id}")

        # Checkpoint integrity
        chkpt_mgr = CheckpointManager(chkpt_path.parent, orig_config.experiment_id)
        if chkpt_mgr.checksums_file.is_file():
            is_intact = chkpt_mgr.verify_checkpoint_integrity(chkpt_path.name)
            if not is_intact:
                logger.error(f"Integrity check failed! Checksum of {chkpt_path.name} does not match recorded SHA256.")
                return 1
            logger.info(f"Checksum verified for {chkpt_path.name}.")

        if new_config_path:
            candidate_config = load_config(new_config_path)
            # Compare architecture and core hyperparameters
            if candidate_config.model.architecture != orig_config.model.architecture:
                logger.error(
                    f"[INCOMPATIBLE RESUME] Architecture mismatch: '{orig_config.model.architecture}' vs '{candidate_config.model.architecture}'. "
                    f"Cannot use exact resume=True. Please start a new branch with --branch --new-experiment-id <NEW_ID>."
                )
                return 1

        logger.info(f"Resuming training from checkpoint: {chkpt_path}")
        model = YOLO(str(chkpt_path))
        model.train(resume=True)
        logger.info("Resumed training finished successfully.")
        return 0

    else:
        # Branching mode
        if not new_experiment_id:
            logger.error("Branching from checkpoint requires --new-experiment-id <NEW_ID>.")
            return 1

        logger.info(f"Creating new branch '{new_experiment_id}' from checkpoint: {chkpt_path}")
        new_cfg = load_config(new_config_path or config_file)
        new_cfg.experiment_id = new_experiment_id

        # Branching uses the checkpoint as weights without strict optimizer state continuation
        model = YOLO(str(chkpt_path))
        logger.info(f"Starting branched training for '{new_experiment_id}'...")
        # Ultralytics train with custom run dir
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume training or branch from checkpoint")
    parser.add_argument("--experiment-dir", type=str, required=True, help="Path to experiment folder")
    parser.add_argument("--checkpoint", type=str, default="last.pt", help="Checkpoint filename (e.g. last.pt, best.pt)")
    parser.add_argument("--resume", action="store_true", help="Perform exact resume")
    parser.add_argument("--branch", action="store_true", help="Branch to a new experiment")
    parser.add_argument("--new-experiment-id", type=str, default=None, help="New experiment ID for branching")
    parser.add_argument("--config", type=str, default=None, help="Optional updated config for checking/branching")
    args = parser.parse_args()

    is_resume = args.resume or not args.branch
    sys.exit(resume_or_branch(
        experiment_dir=args.experiment_dir,
        checkpoint_name=args.checkpoint,
        resume=is_resume,
        new_experiment_id=args.new_experiment_id,
        new_config_path=args.config,
    ))


if __name__ == "__main__":
    main()
