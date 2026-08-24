"""Unit tests for CheckpointManager: atomicity, rotation, metadata, and checksum verification."""

from pathlib import Path
import pytest
import torch

from proctoring_cv.checkpoint_manager import CheckpointManager
from proctoring_cv.reproducibility import compute_file_sha256


def test_checkpoint_atomic_save_and_checksum(tmp_path: Path):
    chkpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(chkpt_dir, experiment_id="exp_test_01", keep_periodic=2)

    # Save a dummy state dict
    dummy_state = {"epoch": 1, "weights": [1, 2, 3]}
    chkpt_file = mgr.record_checkpoint(
        dummy_state,
        "epoch_001.pt",
        metadata={"epoch": 1, "mAP50": 0.72},
        is_periodic=True,
    )

    assert chkpt_file.is_file()
    assert (chkpt_dir / "epoch_001.pt.json").is_file()
    assert mgr.checksums_file.is_file()

    # Integrity verification
    assert mgr.verify_checkpoint_integrity("epoch_001.pt") is True


def test_corrupted_checkpoint_fails_integrity(tmp_path: Path):
    chkpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(chkpt_dir, experiment_id="exp_test_02", keep_periodic=2)

    dummy_state = {"epoch": 1}
    mgr.record_checkpoint(dummy_state, "last.pt", is_periodic=False)

    # Verify initial integrity
    assert mgr.verify_checkpoint_integrity("last.pt") is True

    # Mutate the file
    with open(chkpt_dir / "last.pt", "ab") as f:
        f.write(b"corrupting_bytes")

    # Verify integrity failure
    assert mgr.verify_checkpoint_integrity("last.pt") is False


def test_checkpoint_rotation(tmp_path: Path):
    chkpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(chkpt_dir, experiment_id="exp_test_03", keep_periodic=2)

    # Save 4 periodic checkpoints
    for i in range(1, 5):
        mgr.record_checkpoint(
            {"epoch": i},
            f"epoch_{i:03d}.pt",
            metadata={"epoch": i},
            is_periodic=True,
        )

    # Only the last 2 periodic checkpoints should remain
    assert not (chkpt_dir / "epoch_001.pt").exists()
    assert not (chkpt_dir / "epoch_002.pt").exists()
    assert (chkpt_dir / "epoch_003.pt").exists()
    assert (chkpt_dir / "epoch_004.pt").exists()
