"""Unit tests asserting prohibition of cheating, identity, emotion, or intent labels."""

import pytest
from proctoring_cv.schemas import (
    EventRecord,
    ObservableEventType,
    EventRecordState,
    ConfidenceSummary,
    sanitize_no_forbidden_labels,
)


def test_permitted_event_types_only():
    permitted = {"PHONE_DETECTED", "MULTIPLE_PERSONS", "LOOKING_AWAY"}
    for ev in ObservableEventType:
        assert ev.value in permitted


def test_event_record_schema_sanitization():
    # Valid record
    rec = EventRecord(
        event_id="test_uuid",
        event_type=ObservableEventType.PHONE_DETECTED,
        state=EventRecordState.STARTED,
        timestamp="2026-08-25T00:00:00Z",
        monotonic_seconds=10.0,
        duration_seconds=0.5,
        confidence_summary=ConfidenceSummary(max=0.9, mean=0.85, min=0.8),
        model_id="test_model",
        config_hash="test_hash",
        diagnostics={"face_available": True},
    )
    assert rec.event_type.value == "PHONE_DETECTED"

    # Attempting to inject forbidden judgment in diagnostics or evidence must raise ValueError
    with pytest.raises(ValueError, match="Forbidden label"):
        EventRecord(
            event_id="test_uuid",
            event_type=ObservableEventType.PHONE_DETECTED,
            state=EventRecordState.STARTED,
            timestamp="2026-08-25T00:00:00Z",
            monotonic_seconds=10.0,
            duration_seconds=0.5,
            confidence_summary=ConfidenceSummary(),
            model_id="test_model",
            config_hash="test_hash",
            diagnostics={"verdict": "cheating_true"},
        )
