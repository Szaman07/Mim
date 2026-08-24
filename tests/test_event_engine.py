"""Unit tests for the EventEngine: finite state logic, persistence, hysteresis, and diagnostics."""

import pytest
from proctoring_cv.config import EventEngineConfig
from proctoring_cv.event_engine import EventEngine, EventStateMachine
from proctoring_cv.schemas import (
    DiagnosticCode,
    EventRecordState,
    HeadPoseResult,
    ObservableEventType,
    Detection,
)


def test_phone_persistence_and_lifecycle():
    engine = EventEngine()
    pose_ok = HeadPoseResult(
        timestamp=0.0, yaw=0.0, pitch=0.0, roll=0.0,
        calibrated_yaw=0.0, calibrated_pitch=0.0, calibrated_roll=0.0,
        confidence=0.9, is_valid=True, diagnostic=DiagnosticCode.OK,
    )

    phone_det = [
        Detection(class_id=1, class_name="cellphone", confidence=0.85,
                  bbox_xyxy=(10, 10, 50, 50), bbox_norm_xywh=(0.1, 0.1, 0.1, 0.1), timestamp=0.0)
    ]

    # Frame at t=0.0s (phone appears -> enters candidate state, no emission yet)
    events_0 = engine.process_frame_observations(0.0, phone_det, pose_ok, False)
    assert len(events_0) == 0

    # Frame at t=0.2s (< 0.5s persistence threshold -> still candidate)
    events_02 = engine.process_frame_observations(0.2, phone_det, pose_ok, False)
    assert len(events_02) == 0

    # Frame at t=0.6s (>= 0.5s persistence threshold -> transitions to ACTIVE, emits started)
    events_06 = engine.process_frame_observations(0.6, phone_det, pose_ok, False)
    assert len(events_06) == 1
    assert events_06[0].event_type == ObservableEventType.PHONE_DETECTED
    assert events_06[0].state == EventRecordState.STARTED

    # Frame at t=1.0s (still present -> emits updated)
    events_10 = engine.process_frame_observations(1.0, phone_det, pose_ok, False)
    assert len(events_10) == 1
    assert events_10[0].state == EventRecordState.UPDATED

    # Phone removed at t=1.2s -> enters ENDING
    events_12 = engine.process_frame_observations(1.2, [], pose_ok, False)
    assert len(events_12) == 0

    # Frame at t=2.0s (0.8s elapsed without phone >= 0.75s end persistence -> emits ended)
    events_20 = engine.process_frame_observations(2.0, [], pose_ok, False)
    assert len(events_20) == 1
    assert events_20[0].state == EventRecordState.ENDED


def test_brief_glitch_does_not_trigger_event():
    engine = EventEngine()
    pose_ok = HeadPoseResult(
        timestamp=0.0, yaw=0.0, pitch=0.0, roll=0.0,
        calibrated_yaw=0.0, calibrated_pitch=0.0, calibrated_roll=0.0,
        confidence=0.9, is_valid=True, diagnostic=DiagnosticCode.OK,
    )
    phone_det = [
        Detection(class_id=1, class_name="cellphone", confidence=0.85,
                  bbox_xyxy=(10, 10, 50, 50), bbox_norm_xywh=(0.1, 0.1, 0.1, 0.1), timestamp=0.0)
    ]

    # Appears for only 0.1s then disappears
    engine.process_frame_observations(0.0, phone_det, pose_ok, False)
    events_next = engine.process_frame_observations(0.1, [], pose_ok, False)
    assert len(events_next) == 0


def test_pose_unavailable_never_triggers_looking_away():
    engine = EventEngine()
    # Face is missing / unavailable
    pose_unavail = HeadPoseResult(
        timestamp=0.0, yaw=0.0, pitch=0.0, roll=0.0,
        calibrated_yaw=0.0, calibrated_pitch=0.0, calibrated_roll=0.0,
        confidence=0.0, is_valid=False, diagnostic=DiagnosticCode.POSE_UNAVAILABLE,
    )

    # Even if away flag is requested, invalid pose must prevent LOOKING_AWAY event
    events_0 = engine.process_frame_observations(0.0, [], pose_unavail, is_looking_away=True)
    events_2 = engine.process_frame_observations(2.0, [], pose_unavail, is_looking_away=True)
    assert len(events_0) == 0
    assert len(events_2) == 0


def test_multiple_persons_event():
    engine = EventEngine()
    pose_ok = HeadPoseResult(
        timestamp=0.0, yaw=0.0, pitch=0.0, roll=0.0,
        calibrated_yaw=0.0, calibrated_pitch=0.0, calibrated_roll=0.0,
        confidence=0.9, is_valid=True, diagnostic=DiagnosticCode.OK,
    )
    two_persons = [
        Detection(class_id=0, class_name="person", confidence=0.90, bbox_xyxy=(10, 10, 50, 50), bbox_norm_xywh=(0.1, 0.1, 0.1, 0.1), timestamp=0.0),
        Detection(class_id=0, class_name="person", confidence=0.80, bbox_xyxy=(100, 10, 150, 50), bbox_norm_xywh=(0.3, 0.1, 0.1, 0.1), timestamp=0.0),
    ]

    engine.process_frame_observations(0.0, two_persons, pose_ok, False)
    events = engine.process_frame_observations(0.8, two_persons, pose_ok, False)
    assert len(events) == 1
    assert events[0].event_type == ObservableEventType.MULTIPLE_PERSONS
    assert events[0].state == EventRecordState.STARTED
