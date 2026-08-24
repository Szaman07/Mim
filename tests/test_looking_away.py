"""Unit tests for LookingAwayEstimator: pose estimation, calibration, smoothing, and diagnostics."""

import numpy as np
import pytest
from proctoring_cv.config import LookingAwayConfig
from proctoring_cv.looking_away import LookingAwayEstimator
from proctoring_cv.schemas import DiagnosticCode


def test_looking_away_estimator_initialization():
    estimator = LookingAwayEstimator()
    assert estimator.is_calibrated is False
    assert estimator.baseline_yaw == 0.0
    assert estimator.baseline_pitch == 0.0


def test_pose_unavailable_on_blank_frame():
    estimator = LookingAwayEstimator()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = estimator.process_frame(blank_frame, timestamp=1.0)
    assert res.is_valid is False
    assert res.diagnostic == DiagnosticCode.POSE_UNAVAILABLE
    assert estimator.is_looking_away_instantaneous(res) is False


def test_geometric_pose_solve():
    estimator = LookingAwayEstimator()
    # Canonical landmark projections on 640x480 frame
    # Center = (320, 240)
    landmarks_2d = np.array([
        (320.0, 240.0),  # Nose
        (320.0, 350.0),  # Chin
        (250.0, 200.0),  # Left eye
        (390.0, 200.0),  # Right eye
        (270.0, 300.0),  # Left mouth
        (370.0, 300.0),  # Right mouth
    ], dtype=np.float64)

    pose = estimator.estimate_pose_from_landmarks_2d(landmarks_2d, 640, 480)
    assert pose is not None
    yaw, pitch, roll = pose
    assert isinstance(yaw, float)
    assert isinstance(pitch, float)
    assert isinstance(roll, float)


def test_calibration_and_deviation():
    config = LookingAwayConfig(calibration_frames=5, yaw_threshold_deg=25.0, pitch_threshold_deg=20.0)
    estimator = LookingAwayEstimator(config=config)

    # Supply 5 frontal calibration samples (yaw ~ 2.0 deg)
    samples = [(2.0, 1.0, 0.0) for _ in range(5)]
    estimator.calibrate(samples)
    assert estimator.is_calibrated is True
    assert pytest.approx(estimator.baseline_yaw, 0.1) == 2.0

    # Test deviation calculation
    landmarks_turned = np.array([
        (380.0, 240.0),  # Shifted nose -> turned head
        (360.0, 350.0),
        (310.0, 200.0),
        (430.0, 200.0),
        (330.0, 300.0),
        (410.0, 300.0),
    ], dtype=np.float64)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = estimator.process_frame(frame, timestamp=10.0, pre_extracted_landmarks=landmarks_turned)
    assert res.is_valid is True
    assert res.diagnostic == DiagnosticCode.OK
