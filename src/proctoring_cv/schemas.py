"""Data models and schemas for detections, poses, events, and configurations.

Observable events only:
1. PHONE_DETECTED
2. MULTIPLE_PERSONS
3. LOOKING_AWAY

Forbidden outputs: CHEATING, CHEATING_TRUE, SUSPICIOUS_PERSON, intent, or identity findings.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class ObservableEventType(str, Enum):
    PHONE_DETECTED = "PHONE_DETECTED"
    MULTIPLE_PERSONS = "MULTIPLE_PERSONS"
    LOOKING_AWAY = "LOOKING_AWAY"


class EventLifecycleState(str, Enum):
    INACTIVE = "INACTIVE"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    ENDING = "ENDING"


class EventRecordState(str, Enum):
    STARTED = "started"
    UPDATED = "updated"
    ENDED = "ended"


class DiagnosticCode(str, Enum):
    OK = "OK"
    POSE_UNAVAILABLE = "POSE_UNAVAILABLE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    FACE_NOT_FOUND = "FACE_NOT_FOUND"
    CALIBRATION_PENDING = "CALIBRATION_PENDING"
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    MULTI_PERSON_INTERFERENCE = "MULTI_PERSON_INTERFERENCE"


FORBIDDEN_OUTPUT_PATTERNS = {
    "cheating",
    "cheating_true",
    "suspicious_person",
    "dishonest",
    "fraud",
    "malpractice",
    "guilty",
    "intent",
}


def sanitize_no_forbidden_labels(data: Any) -> None:
    """Verifies that no forbidden judgment keywords exist in strings/keys."""
    if isinstance(data, str):
        val = data.lower().strip()
        for forbidden in FORBIDDEN_OUTPUT_PATTERNS:
            if forbidden in val:
                raise ValueError(f"Forbidden label/judgment detected in output: {data}")
    elif isinstance(data, dict):
        for k, v in data.items():
            sanitize_no_forbidden_labels(k)
            sanitize_no_forbidden_labels(v)
    elif isinstance(data, list):
        for item in data:
            sanitize_no_forbidden_labels(item)


class BoundingBox(BaseModel):
    """Normalized and pixel bounding box coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


class Detection(BaseModel):
    """Single object detection output from primary detector."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_norm_xywh: Tuple[float, float, float, float]
    timestamp: float
    model_id: str = "default_detector"

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        if v not in ("person", "cellphone"):
            raise ValueError(f"Invalid detection class: {v}. Must be 'person' or 'cellphone'")
        return v


class HeadPoseResult(BaseModel):
    """Head pose estimation output from auxiliary landmark component."""
    timestamp: float
    yaw: float
    pitch: float
    roll: float
    calibrated_yaw: float
    calibrated_pitch: float
    calibrated_roll: float
    confidence: float
    is_valid: bool
    diagnostic: DiagnosticCode = DiagnosticCode.OK
    landmarks_2d: Optional[List[Tuple[float, float]]] = None


class ConfidenceSummary(BaseModel):
    max: float = 0.0
    mean: float = 0.0
    min: float = 0.0


class EventRecord(BaseModel):
    """Structured event record output emitted to logs and consumers."""
    event_id: str
    event_type: ObservableEventType
    state: EventRecordState
    timestamp: str  # ISO-8601
    monotonic_seconds: float
    duration_seconds: float
    confidence_summary: ConfidenceSummary
    evidence: Dict[str, Any] = Field(default_factory=dict)
    model_id: str
    config_hash: str
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        # Enforce no forbidden output labels anywhere in the emitted record
        sanitize_no_forbidden_labels(self.model_dump())
