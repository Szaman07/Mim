"""Deterministic timestamp-driven Event Engine for observable proctoring events.

Observable Events:
1. PHONE_DETECTED
2. MULTIPLE_PERSONS
3. LOOKING_AWAY

Lifecycle states: INACTIVE -> CANDIDATE -> ACTIVE -> ENDING -> INACTIVE.
Enforces elapsed-seconds persistence, hysteresis, gap-merging, cooldowns, and uncertainty diagnostics.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from proctoring_cv.config import EventEngineConfig
from proctoring_cv.schemas import (
    ConfidenceSummary,
    DiagnosticCode,
    EventLifecycleState,
    EventRecord,
    EventRecordState,
    HeadPoseResult,
    ObservableEventType,
    Detection,
)


class EventStateMachine:
    """Finite State Machine managing temporal persistence and lifecycle for a single event type."""

    def __init__(
        self,
        event_type: ObservableEventType,
        start_persistence_sec: float,
        end_persistence_sec: float,
        cooldown_sec: float = 2.0,
        gap_merge_sec: float = 0.5,
    ) -> None:
        self.event_type = event_type
        self.start_persistence_sec = start_persistence_sec
        self.end_persistence_sec = end_persistence_sec
        self.cooldown_sec = cooldown_sec
        self.gap_merge_sec = gap_merge_sec

        # State variables
        self.lifecycle_state = EventLifecycleState.INACTIVE
        self.candidate_start_time: Optional[float] = None
        self.active_start_time: Optional[float] = None
        self.ending_start_time: Optional[float] = None
        self.last_ended_time: Optional[float] = None
        self.current_event_id: Optional[str] = None

        # Tracking evidence
        self.confidences: List[float] = []
        self.valid_frames_count = 0
        self.total_frames_in_interval = 0

    def reset(self) -> None:
        """Resets the state machine."""
        self.lifecycle_state = EventLifecycleState.INACTIVE
        self.candidate_start_time = None
        self.active_start_time = None
        self.ending_start_time = None
        self.current_event_id = None
        self.confidences.clear()
        self.valid_frames_count = 0
        self.total_frames_in_interval = 0

    def update(
        self,
        is_evidence_present: bool,
        confidence: float,
        timestamp: float,
        is_valid_evidence: bool = True,
    ) -> Optional[Tuple[EventRecordState, Dict[str, Any]]]:
        """Updates FSM with current frame evidence and returns event emission if lifecycle transition occurs."""
        self.total_frames_in_interval += 1
        if is_evidence_present and is_valid_evidence:
            self.valid_frames_count += 1
            self.confidences.append(confidence)

        # 1. INACTIVE State
        if self.lifecycle_state == EventLifecycleState.INACTIVE:
            # Check cooldown
            if self.last_ended_time is not None and (timestamp - self.last_ended_time) < self.cooldown_sec:
                return None

            if is_evidence_present and is_valid_evidence:
                self.lifecycle_state = EventLifecycleState.CANDIDATE
                self.candidate_start_time = timestamp
                self.valid_frames_count = 1
                self.total_frames_in_interval = 1
                self.confidences = [confidence]

        # 2. CANDIDATE State
        elif self.lifecycle_state == EventLifecycleState.CANDIDATE:
            if not is_evidence_present or not is_valid_evidence:
                # Evidence dropped before persistence reached
                self.reset()
            else:
                start_t = self.candidate_start_time if self.candidate_start_time is not None else timestamp
                elapsed = timestamp - start_t
                if elapsed >= self.start_persistence_sec:
                    # Promoted to ACTIVE
                    self.lifecycle_state = EventLifecycleState.ACTIVE
                    self.active_start_time = self.candidate_start_time
                    self.current_event_id = str(uuid.uuid4())
                    return (EventRecordState.STARTED, self._build_evidence_summary(timestamp))

        # 3. ACTIVE State
        elif self.lifecycle_state == EventLifecycleState.ACTIVE:
            if not is_evidence_present:
                # Evidence lost -> Enter ENDING state
                self.lifecycle_state = EventLifecycleState.ENDING
                self.ending_start_time = timestamp
            else:
                # Still active
                return (EventRecordState.UPDATED, self._build_evidence_summary(timestamp))

        # 4. ENDING State
        elif self.lifecycle_state == EventLifecycleState.ENDING:
            if is_evidence_present and is_valid_evidence:
                # Short gap merge: evidence returned within tolerance
                self.lifecycle_state = EventLifecycleState.ACTIVE
                self.ending_start_time = None
                return (EventRecordState.UPDATED, self._build_evidence_summary(timestamp))
            else:
                end_start_t = self.ending_start_time if self.ending_start_time is not None else timestamp
                elapsed_absent = timestamp - end_start_t
                if elapsed_absent >= self.end_persistence_sec:
                    # Formally emit ENDED record
                    summary = self._build_evidence_summary(timestamp)
                    self.last_ended_time = timestamp
                    self.lifecycle_state = EventLifecycleState.INACTIVE
                    self.candidate_start_time = None
                    self.active_start_time = None
                    self.ending_start_time = None
                    self.confidences.clear()
                    return (EventRecordState.ENDED, summary)

        return None

    def _build_evidence_summary(self, current_time: float) -> Dict[str, Any]:
        start = self.active_start_time if self.active_start_time is not None else current_time
        duration = max(0.0, current_time - start)
        max_c = max(self.confidences) if self.confidences else 0.0
        mean_c = sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
        min_c = min(self.confidences) if self.confidences else 0.0

        return {
            "event_id": self.current_event_id or str(uuid.uuid4()),
            "duration_seconds": round(duration, 3),
            "confidence_summary": ConfidenceSummary(
                max=round(max_c, 3),
                mean=round(mean_c, 3),
                min=round(min_c, 3),
            ),
            "valid_frames": self.valid_frames_count,
            "total_frames": self.total_frames_in_interval,
        }


class EventEngine:
    """Master event engine coordinating PHONE_DETECTED, MULTIPLE_PERSONS, and LOOKING_AWAY."""

    def __init__(
        self,
        config: Optional[EventEngineConfig] = None,
        model_id: str = "yolo11n_scratch",
        config_hash: str = "sha256_mock",
    ) -> None:
        self.config = config or EventEngineConfig()
        self.model_id = model_id
        self.config_hash = config_hash

        # Initialize FSMs for each observable event
        self.fsm_phone = EventStateMachine(
            ObservableEventType.PHONE_DETECTED,
            start_persistence_sec=self.config.phone_start_persistence_sec,
            end_persistence_sec=self.config.phone_end_persistence_sec,
            cooldown_sec=self.config.event_cooldown_sec,
            gap_merge_sec=self.config.gap_merge_tolerance_sec,
        )

        self.fsm_multi_person = EventStateMachine(
            ObservableEventType.MULTIPLE_PERSONS,
            start_persistence_sec=self.config.multi_person_start_persistence_sec,
            end_persistence_sec=self.config.multi_person_end_persistence_sec,
            cooldown_sec=self.config.event_cooldown_sec,
            gap_merge_sec=self.config.gap_merge_tolerance_sec,
        )

        self.fsm_looking_away = EventStateMachine(
            ObservableEventType.LOOKING_AWAY,
            start_persistence_sec=self.config.looking_away_start_persistence_sec,
            end_persistence_sec=self.config.looking_away_end_persistence_sec,
            cooldown_sec=self.config.event_cooldown_sec,
            gap_merge_sec=self.config.gap_merge_tolerance_sec,
        )

    def process_frame_observations(
        self,
        timestamp: float,
        detections: List[Detection],
        head_pose: HeadPoseResult,
        is_looking_away: bool,
    ) -> List[EventRecord]:
        """Evaluates detections and pose results for a single timestamped frame."""
        emitted_events: List[EventRecord] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. PHONE_DETECTED evaluation
        phone_dets = [d for d in detections if d.class_id == 1 and d.confidence >= self.config.phone_confidence_threshold]
        phone_present = len(phone_dets) > 0
        phone_conf = max([d.confidence for d in phone_dets]) if phone_present else 0.0

        phone_transition = self.fsm_phone.update(phone_present, phone_conf, timestamp)
        if phone_transition:
            state, summary = phone_transition
            emitted_events.append(EventRecord(
                event_id=summary["event_id"],
                event_type=ObservableEventType.PHONE_DETECTED,
                state=state,
                timestamp=now_iso,
                monotonic_seconds=timestamp,
                duration_seconds=summary["duration_seconds"],
                confidence_summary=summary["confidence_summary"],
                evidence={"valid_frames": summary["valid_frames"], "detected_phones_count": len(phone_dets)},
                model_id=self.model_id,
                config_hash=self.config_hash,
                diagnostics={"face_available": head_pose.is_valid},
            ))

        # 2. MULTIPLE_PERSONS evaluation
        person_dets = [d for d in detections if d.class_id == 0 and d.confidence >= self.config.person_confidence_threshold]
        multi_person_present = len(person_dets) >= 2
        multi_person_conf = sorted([d.confidence for d in person_dets], reverse=True)[1] if multi_person_present else 0.0

        multi_transition = self.fsm_multi_person.update(multi_person_present, multi_person_conf, timestamp)
        if multi_transition:
            state, summary = multi_transition
            emitted_events.append(EventRecord(
                event_id=summary["event_id"],
                event_type=ObservableEventType.MULTIPLE_PERSONS,
                state=state,
                timestamp=now_iso,
                monotonic_seconds=timestamp,
                duration_seconds=summary["duration_seconds"],
                confidence_summary=summary["confidence_summary"],
                evidence={"valid_frames": summary["valid_frames"], "person_count": len(person_dets)},
                model_id=self.model_id,
                config_hash=self.config_hash,
                diagnostics={"face_available": head_pose.is_valid},
            ))

        # 3. LOOKING_AWAY evaluation
        # Safeguard: If pose is unavailable, evidence is marked invalid so looking away is never falsely triggered
        pose_is_valid = head_pose.is_valid and head_pose.diagnostic != DiagnosticCode.POSE_UNAVAILABLE
        pose_conf = head_pose.confidence if pose_is_valid else 0.0

        away_transition = self.fsm_looking_away.update(
            is_evidence_present=is_looking_away,
            confidence=pose_conf,
            timestamp=timestamp,
            is_valid_evidence=pose_is_valid,
        )
        if away_transition:
            state, summary = away_transition
            emitted_events.append(EventRecord(
                event_id=summary["event_id"],
                event_type=ObservableEventType.LOOKING_AWAY,
                state=state,
                timestamp=now_iso,
                monotonic_seconds=timestamp,
                duration_seconds=summary["duration_seconds"],
                confidence_summary=summary["confidence_summary"],
                evidence={
                    "valid_frames": summary["valid_frames"],
                    "calibrated_yaw": round(head_pose.calibrated_yaw, 2),
                    "calibrated_pitch": round(head_pose.calibrated_pitch, 2),
                },
                model_id=self.model_id,
                config_hash=self.config_hash,
                diagnostics={
                    "face_available": pose_is_valid,
                    "diagnostic_code": head_pose.diagnostic.value,
                },
            ))

        return emitted_events
