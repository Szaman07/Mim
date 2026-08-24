"""Unit tests for configuration loading, validation, environment probes, and reproducibility."""

import os
from pathlib import Path
import pytest
from proctoring_cv.config import load_config, save_config, AppConfig, DatasetConfig, ModelConfig
from proctoring_cv.environment import get_environment_snapshot, recommend_batch_size
from proctoring_cv.reproducibility import set_seed, get_git_metadata, compute_file_sha256
from proctoring_cv.schemas import sanitize_no_forbidden_labels, ObservableEventType, DiagnosticCode


def test_default_config_validity():
    config = load_config()
    assert config.dataset.names[0] == "person"
    assert config.dataset.names[1] == "cellphone"
    assert config.dataset.nc == 2
    assert config.model.pretrained is False
    assert config.model.initialization_method == "random_from_yaml"
    sha = config.compute_sha256()
    assert len(sha) == 64


def test_invalid_class_mapping_fails():
    with pytest.raises(Exception):
        DatasetConfig(names={0: "person", 1: "laptop"})


def test_config_load_from_yaml(tmp_path: Path):
    yaml_file = tmp_path / "test_exp.yaml"
    config = AppConfig(experiment_id="test_exp_123")
    save_config(config, yaml_file)

    loaded = load_config(yaml_file)
    assert loaded.experiment_id == "test_exp_123"


def test_cli_overrides():
    overrides = {
        "experiment_id": "override_exp",
        "training.epochs": 25,
        "dataset.nc": 2,
    }
    config = load_config(cli_overrides=overrides)
    assert config.experiment_id == "override_exp"
    assert config.training.epochs == 25


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PROCTORING_EXPERIMENT_ID", "env_exp_456")
    monkeypatch.setenv("PROCTORING_DRIVE_ROOT", "/custom/drive")
    config = load_config()
    assert config.experiment_id == "env_exp_456"
    assert config.drive_root == "/custom/drive"


def test_environment_snapshot():
    snapshot = get_environment_snapshot()
    assert "python_version" in snapshot
    assert "platform" in snapshot
    assert "torch_version" in snapshot
    assert "cuda_available" in snapshot
    assert isinstance(snapshot["gpu_devices"], list)


def test_recommend_batch_size():
    batch = recommend_batch_size(640)
    assert batch in (4, 8, 16, 32)


def test_reproducibility_seeding():
    set_seed(123, deterministic=True)
    import random
    import numpy as np
    import torch

    val_py = random.random()
    val_np = np.random.rand()
    val_th = torch.rand(1).item()

    # Re-seed and check equality
    set_seed(123, deterministic=True)
    assert random.random() == val_py
    assert np.random.rand() == val_np
    assert torch.rand(1).item() == val_th


def test_git_metadata():
    meta = get_git_metadata()
    assert "commit" in meta
    assert "branch" in meta
    assert "is_dirty" in meta


def test_forbidden_label_sanitizer():
    # Valid data
    sanitize_no_forbidden_labels({"event": "PHONE_DETECTED", "diagnostics": {"state": "OK"}})

    # Forbidden cheating judgment keyword
    with pytest.raises(ValueError, match="Forbidden label"):
        sanitize_no_forbidden_labels({"result": "CHEATING_TRUE"})

    with pytest.raises(ValueError, match="Forbidden label"):
        sanitize_no_forbidden_labels({"status": "suspicious_person"})
