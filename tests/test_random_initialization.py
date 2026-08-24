"""Unit tests verifying random initialization enforcement and detector prediction normalization."""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from proctoring_cv.config import load_config, AppConfig, ModelConfig
from proctoring_cv.detector import build_detector, normalize_detections, verify_random_initialization
from proctoring_cv.reproducibility import compute_model_parameter_hash, set_seed


def test_scratch_mode_rejects_pt_weights():
    # Attempting to load .pt in scratch mode MUST raise ValueError
    bad_config = ModelConfig(
        architecture="yolo11n.pt",
        initialization_method="random_from_yaml",
        pretrained=False,
    )
    with pytest.raises(ValueError, match="SCRATCH MODE VIOLATION"):
        build_detector(bad_config, mode="scratch")


def test_scratch_mode_rejects_pretrained_true():
    bad_config = ModelConfig(
        architecture="yolo11n.yaml",
        initialization_method="random_from_yaml",
        pretrained=True,
    )
    with pytest.raises(ValueError, match="SCRATCH MODE VIOLATION"):
        build_detector(bad_config, mode="scratch")


def test_random_initialization_verification():
    # Verify that different seeds generate distinct parameter hashes
    verification = verify_random_initialization("yolo11n.yaml", seed_a=42, seed_b=999)
    assert verification["is_random_initialization_proven"] is True
    assert verification["hash_seed_a"] != verification["hash_seed_b"]


def test_detection_normalization():
    # Create synthetic raw detection structures
    class DummyBoxes:
        def __init__(self):
            import torch
            self.xyxy = torch.tensor([[100.0, 100.0, 300.0, 400.0], [50.0, 50.0, 150.0, 200.0]])
            self.conf = torch.tensor([0.85, 0.92])
            self.cls = torch.tensor([0.0, 1.0])

        def __len__(self):
            return 2

    class DummyResult:
        def __init__(self):
            self.boxes = DummyBoxes()
            self.orig_shape = (480, 640)

    detections = normalize_detections(DummyResult(), timestamp=10.5, model_id="test_detector")
    assert len(detections) == 2

    d_person = detections[0]
    assert d_person.class_id == 0
    assert d_person.class_name == "person"
    assert d_person.confidence == pytest.approx(0.85, 0.01)
    assert d_person.timestamp == 10.5

    d_phone = detections[1]
    assert d_phone.class_id == 1
    assert d_phone.class_name == "cellphone"
    assert d_phone.confidence == pytest.approx(0.92, 0.01)
