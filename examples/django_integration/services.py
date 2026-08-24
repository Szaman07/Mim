"""Proctoring Service Layer: manages model inference, session state machines, and event persistence."""

from __future__ import annotations

import base64
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from django.utils import timezone

from proctoring_cv.config import load_config, AppConfig
from proctoring_cv.detector import build_detector, predict_frame
from proctoring_cv.event_engine import EventEngine
from proctoring_cv.looking_away import LookingAwayEstimator
from proctoring_cv.schemas import Detection, HeadPoseResult, DiagnosticCode


class SessionEngineState:
    """Encapsulates per-student temporal state machines and calibration."""
    def __init__(self, session_id: str, config: AppConfig) -> None:
        self.session_id = session_id
        self.pose_estimator = LookingAwayEstimator(config=config.looking_away)
        self.event_engine = EventEngine(
            config=config.event_engine,
            model_id=config.experiment_id,
            config_hash=config.compute_sha256(),
        )
        self.prev_primary_box: Optional[Tuple[float, float, float, float]] = None
        self.start_monotonic = time.perf_counter()
        self.lock = threading.Lock()


class ProctoringService:
    """Singleton service loaded when Django initializes."""
    _instance: Optional[ProctoringService] = None
    _lock = threading.Lock()

    def __init__(self, config_path: str = "configs/runtime.yaml", checkpoint_path: Optional[str] = None) -> None:
        self.config: AppConfig = load_config(config_path)
        self.checkpoint_path = checkpoint_path or self.config.runtime.detector_checkpoint or "yolo11n.yaml"
        
        # Load detector model once in memory
        if str(self.checkpoint_path).endswith(".pt"):
            self.detector = build_detector(self.config, mode="inference")
        else:
            self.detector = build_detector(self.config, mode="scratch")

        self.sessions: Dict[str, SessionEngineState] = {}
        self.sessions_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config_path: str = "configs/runtime.yaml", checkpoint_path: Optional[str] = None) -> ProctoringService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config_path=config_path, checkpoint_path=checkpoint_path)
            return cls._instance

    def get_or_create_session_state(self, session_id: str) -> SessionEngineState:
        with self.sessions_lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionEngineState(session_id, self.config)
            return self.sessions[session_id]

    def cleanup_session(self, session_id: str) -> None:
        with self.sessions_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

    def decode_image_bytes(self, image_data: bytes | str) -> Optional[np.ndarray]:
        """Decodes raw binary bytes or base64 data string into OpenCV BGR numpy array."""
        try:
            if isinstance(image_data, str):
                if "," in image_data:
                    # Strip base64 header (e.g. data:image/jpeg;base64,...)
                    image_data = image_data.split(",", 1)[1]
                raw_bytes = base64.b64decode(image_data)
            else:
                raw_bytes = image_data

            nparr = np.frombuffer(raw_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    def process_frame(
        self,
        session_id: str,
        image_data: bytes | str,
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        """Processes a single webcam frame for an exam session."""
        frame = self.decode_image_bytes(image_data)
        if frame is None:
            return {"error": "Invalid image data", "diagnostic": "DECODE_FAILED"}

        session_state = self.get_or_create_session_state(session_id)

        with session_state.lock:
            now_mono = time.perf_counter()
            rel_timestamp = now_mono - session_state.start_monotonic

            # 1. Primary Detector Inference
            detections = predict_frame(
                self.detector,
                frame,
                timestamp=rel_timestamp,
                confidence_threshold=self.config.runtime.confidence_threshold,
                iou_threshold=self.config.runtime.iou_threshold,
                model_id=self.config.experiment_id,
            )

            # 2. Select Primary Person Box
            person_dets = [d for d in detections if d.class_id == 0]
            primary_box = None
            if person_dets:
                primary = max(person_dets, key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]))
                session_state.prev_primary_box = primary.bbox_xyxy
                primary_box = primary.bbox_xyxy

            # 3. Head Pose & Looking Away
            pose_result = session_state.pose_estimator.process_frame(frame, timestamp=rel_timestamp)
            is_looking_away = session_state.pose_estimator.is_looking_away_instantaneous(pose_result)

            # 4. Deterministic Event Engine
            emitted_events = session_state.event_engine.process_frame_observations(
                timestamp=rel_timestamp,
                detections=detections,
                head_pose=pose_result,
                is_looking_away=is_looking_away,
            )

            # 5. Optional Django ORM Persistence
            if persist_to_db and emitted_events:
                self._persist_events(session_id, emitted_events)

            # Build serializable payload for API / WebSocket
            return {
                "session_id": session_id,
                "timestamp": rel_timestamp,
                "person_count": len(person_dets),
                "phone_detected": any(d.class_id == 1 for d in detections),
                "pose": {
                    "valid": pose_result.is_valid,
                    "calibrated": pose_result.diagnostic == DiagnosticCode.OK,
                    "yaw_deg": round(pose_result.calibrated_yaw, 1),
                    "pitch_deg": round(pose_result.calibrated_pitch, 1),
                    "is_looking_away": is_looking_away,
                },
                "emitted_events": [ev.model_dump() for ev in emitted_events],
            }

    def _persist_events(self, session_id: str, events: List[Any]) -> None:
        """Persists emitted event records into Django database models."""
        try:
            from .models import ExamSession, ProctoringEvent
            session_obj = ExamSession.objects.filter(session_id=session_id).first()
            if not session_obj:
                return

            records_to_create = []
            for ev in events:
                records_to_create.append(ProctoringEvent(
                    event_id=ev.event_id,
                    session=session_obj,
                    event_type=ev.event_type.value,
                    state=ev.state.value,
                    duration_seconds=ev.duration_seconds,
                    confidence_max=ev.confidence_summary.max,
                    confidence_mean=ev.confidence_summary.mean,
                    evidence=ev.evidence,
                    diagnostics=ev.diagnostics,
                    model_id=ev.model_id,
                ))

            if records_to_create:
                ProctoringEvent.objects.bulk_create(records_to_create)
        except Exception:
            pass
