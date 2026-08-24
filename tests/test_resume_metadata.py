"""Unit tests for resume compatibility checking and branching workflows."""

from pathlib import Path
import pytest
import yaml

from proctoring_cv.config import AppConfig, save_config
from proctoring_cv.checkpoint_manager import CheckpointManager
from scripts.resume_training import resume_or_branch


def test_resume_missing_experiment_dir(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist"
    status = resume_or_branch(non_existent, resume=True)
    assert status == 1


def test_resume_compatible_metadata(tmp_path: Path):
    exp_dir = tmp_path / "exp_01"
    exp_dir.mkdir()
    chkpts_dir = exp_dir / "checkpoints"
    chkpts_dir.mkdir()

    cfg = AppConfig(experiment_id="exp_01")
    save_config(cfg, exp_dir / "config.yaml")

    # Create dummy last.pt with valid checksum
    mgr = CheckpointManager(chkpts_dir, "exp_01")
    mgr.record_checkpoint({"dummy": 1}, "last.pt")

    assert (chkpts_dir / "last.pt").is_file()
    assert mgr.verify_checkpoint_integrity("last.pt") is True


def test_incompatible_architecture_refuses_resume(tmp_path: Path):
    exp_dir = tmp_path / "exp_02"
    exp_dir.mkdir()
    chkpts_dir = exp_dir / "checkpoints"
    chkpts_dir.mkdir()

    cfg1 = AppConfig(experiment_id="exp_02")
    cfg1.model.architecture = "yolo11n.yaml"
    save_config(cfg1, exp_dir / "config.yaml")

    mgr = CheckpointManager(chkpts_dir, "exp_02")
    mgr.record_checkpoint({"dummy": 1}, "last.pt")

    # Different architecture candidate
    candidate_cfg = exp_dir / "new_config.yaml"
    cfg2 = AppConfig(experiment_id="exp_02")
    cfg2.model.architecture = "yolov8n.yaml"  # Mismatch!
    save_config(cfg2, candidate_cfg)

    status = resume_or_branch(
        exp_dir,
        checkpoint_name="last.pt",
        resume=True,
        new_config_path=str(candidate_cfg),
    )
    assert status == 1  # Incompatible architecture must fail
