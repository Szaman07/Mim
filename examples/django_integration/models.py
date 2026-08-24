"""Django Models for Proctoring System."""

import uuid
from django.db import models
from django.utils import timezone


class ExamSession(models.Model):
    """Represents an active or completed student exam session."""
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    student_id = models.CharField(max_length=100, db_index=True)
    exam_title = models.CharField(max_length=255)
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"ExamSession {self.session_id} - Student: {self.student_id} ({'Active' if self.is_active else 'Closed'})"


class ProctoringEvent(models.Model):
    """Observable proctoring events: PHONE_DETECTED, MULTIPLE_PERSONS, LOOKING_AWAY."""

    EVENT_TYPES = [
        ("PHONE_DETECTED", "Mobile Phone Detected"),
        ("MULTIPLE_PERSONS", "Multiple Persons In Frame"),
        ("LOOKING_AWAY", "Head Turned Away"),
    ]

    RECORD_STATES = [
        ("started", "Started"),
        ("updated", "Updated"),
        ("ended", "Ended"),
    ]

    event_id = models.UUIDField(default=uuid.uuid4, db_index=True)
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name="events")
    event_type = models.CharField(max_length=32, choices=EVENT_TYPES, db_index=True)
    state = models.CharField(max_length=16, choices=RECORD_STATES)
    timestamp = models.DateTimeField(default=timezone.now)
    duration_seconds = models.FloatField(default=0.0)
    
    # Confidence Metrics
    confidence_max = models.FloatField(default=0.0)
    confidence_mean = models.FloatField(default=0.0)

    # Structured metadata
    evidence = models.JSONField(default=dict, blank=True)
    diagnostics = models.JSONField(default=dict, blank=True)
    model_id = models.CharField(max_length=128, default="yolo11n_scratch")

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["session", "event_type", "state"]),
        ]

    def __str__(self) -> str:
        return f"[{self.session.student_id}] {self.event_type} - {self.state} ({self.duration_seconds:.1f}s)"


class ProctoringDiagnostic(models.Model):
    """Periodic health and pose telemetry for exam audit trail."""
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name="diagnostics")
    timestamp = models.DateTimeField(default=timezone.now)
    face_available = models.BooleanField(default=True)
    calibrated = models.BooleanField(default=False)
    yaw_deg = models.FloatField(default=0.0)
    pitch_deg = models.FloatField(default=0.0)
    roll_deg = models.FloatField(default=0.0)
    diagnostic_code = models.CharField(max_length=64, default="OK")

    class Meta:
        ordering = ["-timestamp"]
