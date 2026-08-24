# SYSTEM_ARCHITECTURE.md

## Purpose and boundary

This system detects three visual conditions: a visible mobile phone, more than one visible person, and sustained head orientation away from the screen. It does not determine cheating, intent, identity, emotion, speech, mouth movement, or audio behavior.

> **Observable event:** a time-bounded, model-supported visual condition that is recorded with confidence and uncertainty. It is not a finding of misconduct.

## Component diagram

```text
Webcam frame
    |
    v
Frame adapter and timestamp source
    |
    +--> YOLO detector: person + cellphone
    |       |
    |       +--> person count --> MULTIPLE_PERSONS candidate
    |       +--> phone detections --> PHONE_DETECTED candidate
    |       +--> primary-person box
    |
    +--> face landmarker on primary person/full frame
            |
            +--> 3D landmarks / facial transform
                    |
                    +--> head pose: yaw, pitch, roll
                            |
                            +--> calibration + smoothing
                                    |
                                    +--> LOOKING_AWAY candidate

All candidates --> deterministic event engine --> event log + overlay/API
```

## Detector

The primary detector is the selected nano YOLO architecture, defaulting to YOLO11n from YAML with `pretrained=False`; YOLOv8n from YAML is an educational comparison and YOLO26n/P2 is a later small-object experiment. The detector has exactly two classes: `person` and `cellphone`. It returns class ID, class name, confidence, pixel box, frame timestamp, and model metadata.

The training code must instantiate the architecture from configuration, explicitly disable pretrained loading, verify random initialization, and prevent silent downloads. A pretrained auxiliary face-landmark component is permitted only because it solves a different subproblem and is explicitly separated from the main experiment.

## Primary-person selection

For looking-away analysis, select the largest valid person detection near the previous primary box, using IoU/center-distance continuity. If no prior target exists, select the largest person box. If two or more people are visible, keep the primary target for diagnostics but do not infer that another person’s pose is the examinee’s pose. If the face is missing, report pose unavailable rather than looking away.

## Looking-away module

MediaPipe Face Landmarker is the default auxiliary component. It supports video/live-stream processing, 3D facial landmarks, and facial transformation output.[1] Head pose is computed with a documented geometric method using canonical 3D points and 2D landmark points. OpenCV’s `solvePnP` family is suitable for the pose solve when camera intrinsics and coordinate conventions are explicitly documented.[2]

The module performs a short frontal calibration at session start. It stores robust baseline yaw/pitch medians, accepts an explicit recalibration command, and marks pose as unavailable if calibration fails. It computes deviations from baseline, applies a short median window and exponential moving average, and exposes raw and smoothed values for evaluation.

Initial thresholds are `abs(yaw-yaw0) >= 25°` or `abs(pitch-pitch0) >= 20°` for a candidate away state, with a lower end boundary around 15°. These values are starting points for validation, not universal definitions of looking away.

## Event engine

The event engine is deterministic and timestamp-based. It receives candidates, confidence, validity, and evidence metadata. Each event uses `INACTIVE`, `CANDIDATE`, `ACTIVE`, and `ENDING` states. Start and end persistence are measured in seconds so behavior is stable across camera frame rates.

| Event | Candidate | Start | End | Diagnostic safeguards |
|---|---|---|---|---|
| `PHONE_DETECTED` | At least one phone box above selected phone confidence | Present in 3/5 frames or approximately 0.5 s | Absent for approximately 0.75 s | NMS, minimum valid box, confidence threshold, merge nearby intervals. |
| `MULTIPLE_PERSONS` | At least two valid person boxes | Count ≥2 for approximately 0.75 s or 5/7 frames | Count ≤1 for 1.0 s | NMS and consistent person confidence; no identity claim. |
| `LOOKING_AWAY` | Valid calibrated yaw/pitch outside start boundary | Outside boundary for 1.5 s with pose valid in ≥70% of interval | Inside 15° boundary for 0.75 s | Calibration, confidence gating, smoothing, hysteresis, missing-face handling. |

The engine logs `started`, `updated`, and `ended` records. It merges short gaps shorter than the configured gap tolerance and applies cooldowns to avoid event spam. It never emits `CHEATING` or a proxy label such as `SUSPICIOUS_PERSON`.

## Event record schema

```json
{
  "event_id": "uuid",
  "event_type": "PHONE_DETECTED",
  "state": "started",
  "timestamp": "ISO-8601",
  "monotonic_seconds": 123.45,
  "duration_seconds": 0.0,
  "confidence_summary": {"max": 0.91, "mean": 0.86},
  "evidence": {"valid_frames": 3, "source": "detector"},
  "model_id": "exp_...",
  "config_hash": "sha256:...",
  "diagnostics": {"face_available": true}
}
```

Raw frames are not stored by default. Optional evidence snapshots require separate consent, retention, access-control, and institutional approval.

## Training/evaluation boundary

Training, validation, checkpointing, and evaluation are offline workflows. Webcam inference is a separate runtime workflow that loads a selected checkpoint and configuration. It must expose the checkpoint path and model metadata, measure latency, and show diagnostic states. An exported model is not accepted until its predictions are compared against the native checkpoint on a smoke-test set.

## Failure semantics

| Failure | Required behavior |
|---|---|
| Invalid dataset or label | Fail loudly before training; write a validation report. |
| Missing checkpoint | Refuse inference and explain the expected path. |
| Corrupt checkpoint | Fail integrity check; do not overwrite the last known good copy. |
| GPU unavailable | Fall back to CPU only for sanity/inference or stop with a clear training warning. |
| OOM | Save diagnostics, suggest lower batch/image size, and preserve the prior checkpoint. |
| Face unavailable | Emit `POSE_UNAVAILABLE`; do not emit `LOOKING_AWAY` solely from absence. |
| Multiple people | Emit the observable count event; do not assign identity or guilt. |
| Camera disconnect | End or pause the session explicitly and log the availability change. |

## References

[1]: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker "Google AI Edge MediaPipe Face Landmarker"
[2]: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html "OpenCV Camera Calibration and 3D Reconstruction"
