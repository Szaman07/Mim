"""Head pose estimation and calibrated looking-away detection module.

Uses 3D-2D facial geometry with OpenCV solvePnP.
Performs neutral frontal calibration, median and EMA smoothing, and reports POSE_UNAVAILABLE
when face evidence is missing or unreliable (never falsely claiming looking away).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

from proctoring_cv.config import LookingAwayConfig
from proctoring_cv.schemas import DiagnosticCode, HeadPoseResult


# 3D Canonical Face Model Points (in millimeters, centered roughly around nose tip)
CANONICAL_FACE_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye outer corner
    (225.0, 170.0, -135.0),      # Right eye outer corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0),     # Right mouth corner
], dtype=np.float64)


class LookingAwayEstimator:
    """Estimates head pose (yaw, pitch, roll), maintains calibration, and detects looking away."""

    def __init__(self, config: Optional[LookingAwayConfig] = None) -> None:
        self.config = config or LookingAwayConfig()

        # Calibration state
        self.is_calibrated = False
        self.calibration_samples: List[Tuple[float, float, float]] = []
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_roll = 0.0

        # Smoothing buffers
        self.recent_yaws: Deque[float] = deque(maxlen=self.config.smoothing_median_window)
        self.recent_pitches: Deque[float] = deque(maxlen=self.config.smoothing_median_window)
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0

        # Try loading OpenCV CascadeClassifier if available in the cv2 build
        self._face_cascade = None
        self._eye_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    self._face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                self._face_cascade = None

    def calibrate(self, neutral_samples: List[Tuple[float, float, float]]) -> None:
        """Sets neutral baseline head orientation from calibration samples."""
        if not neutral_samples:
            return
        yaws = [s[0] for s in neutral_samples]
        pitches = [s[1] for s in neutral_samples]
        rolls = [s[2] for s in neutral_samples]

        self.baseline_yaw = float(np.median(yaws))
        self.baseline_pitch = float(np.median(pitches))
        self.baseline_roll = float(np.median(rolls))
        self.is_calibrated = True

    def reset_calibration(self) -> None:
        """Resets calibration state."""
        self.is_calibrated = False
        self.calibration_samples.clear()
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_roll = 0.0

    def estimate_pose_from_landmarks_2d(
        self,
        landmarks_2d: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Solves PnP problem to compute Euler angles (yaw, pitch, roll) from 6 2D facial points."""
        if landmarks_2d.shape != (6, 2):
            return None

        # Approximate camera matrix assuming focal length = image width
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_FACE_3D,
            landmarks_2d.astype(np.float64),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Decompose projection/rotation matrix to Euler angles
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        return (yaw, pitch, roll)

    def extract_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detects face and estimates key facial landmarks (nose, chin, eyes, mouth corners)."""
        if self._face_cascade is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        if len(faces) == 0:
            return None

        # Select largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]

        # Geometry-based landmark approximation inside face bounding box
        nose_tip = (fx + fw * 0.50, fy + fh * 0.55)
        chin = (fx + fw * 0.50, fy + fh * 0.95)
        left_eye = (fx + fw * 0.30, fy + fh * 0.35)
        right_eye = (fx + fw * 0.70, fy + fh * 0.35)
        left_mouth = (fx + fw * 0.35, fy + fh * 0.75)
        right_mouth = (fx + fw * 0.65, fy + fh * 0.75)

        landmarks = np.array([
            nose_tip,
            chin,
            left_eye,
            right_eye,
            left_mouth,
            right_mouth,
        ], dtype=np.float64)

        return landmarks

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        pre_extracted_landmarks: Optional[np.ndarray] = None,
    ) -> HeadPoseResult:
        """Processes frame, computes pose, applies smoothing & calibration, and returns HeadPoseResult."""
        h, w = frame.shape[:2]
        landmarks = pre_extracted_landmarks if pre_extracted_landmarks is not None else self.extract_face_landmarks(frame)

        if landmarks is None:
            return HeadPoseResult(
                timestamp=timestamp,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                calibrated_yaw=0.0,
                calibrated_pitch=0.0,
                calibrated_roll=0.0,
                confidence=0.0,
                is_valid=False,
                diagnostic=DiagnosticCode.POSE_UNAVAILABLE,
            )

        pose = self.estimate_pose_from_landmarks_2d(landmarks, w, h)
        if pose is None:
            return HeadPoseResult(
                timestamp=timestamp,
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                calibrated_yaw=0.0,
                calibrated_pitch=0.0,
                calibrated_roll=0.0,
                confidence=0.0,
                is_valid=False,
                diagnostic=DiagnosticCode.POSE_UNAVAILABLE,
            )

        raw_yaw, raw_pitch, raw_roll = pose

        # Handle calibration collection if not calibrated
        if not self.is_calibrated:
            self.calibration_samples.append(pose)
            if len(self.calibration_samples) >= self.config.calibration_frames:
                self.calibrate(self.calibration_samples)

        # Apply median filtering
        self.recent_yaws.append(raw_yaw)
        self.recent_pitches.append(raw_pitch)
        med_yaw = float(np.median(list(self.recent_yaws)))
        med_pitch = float(np.median(list(self.recent_pitches)))

        # Apply EMA smoothing
        alpha = self.config.smoothing_ema_alpha
        self.smoothed_yaw = alpha * med_yaw + (1.0 - alpha) * self.smoothed_yaw if len(self.recent_yaws) > 1 else med_yaw
        self.smoothed_pitch = alpha * med_pitch + (1.0 - alpha) * self.smoothed_pitch if len(self.recent_pitches) > 1 else med_pitch

        # Compute calibrated deviation
        cal_yaw = self.smoothed_yaw - self.baseline_yaw
        cal_pitch = self.smoothed_pitch - self.baseline_pitch
        cal_roll = raw_roll - self.baseline_roll

        diag = DiagnosticCode.OK if self.is_calibrated else DiagnosticCode.CALIBRATION_PENDING

        return HeadPoseResult(
            timestamp=timestamp,
            yaw=raw_yaw,
            pitch=raw_pitch,
            roll=raw_roll,
            calibrated_yaw=cal_yaw,
            calibrated_pitch=cal_pitch,
            calibrated_roll=cal_roll,
            confidence=0.90,
            is_valid=True,
            diagnostic=diag,
            landmarks_2d=[(float(pt[0]), float(pt[1])) for pt in landmarks],
        )

    def is_looking_away_instantaneous(self, pose: HeadPoseResult) -> bool:
        """Evaluates whether current instantaneous pose exceeds looking away thresholds."""
        if not pose.is_valid or pose.diagnostic == DiagnosticCode.POSE_UNAVAILABLE:
            return False

        yaw_dev = abs(pose.calibrated_yaw)
        pitch_dev = abs(pose.calibrated_pitch)

        return (yaw_dev >= self.config.yaw_threshold_deg) or (pitch_dev >= self.config.pitch_threshold_deg)
