"""End-to-end realtime webcam inference with privacy-safe HUD overlay and structured JSONL logging.

Integrates:
- Primary Object Detector (person + cellphone)
- Primary-person tracking & selection
- Auxiliary Head Pose & Looking-Away Estimator
- Deterministic Event Engine
- Privacy-safe diagnostic HUD overlay (no raw frame uploads or cheating labels)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

from proctoring_cv.config import load_config, AppConfig
from proctoring_cv.detector import build_detector, predict_frame
from proctoring_cv.event_engine import EventEngine
from proctoring_cv.logging_utils import setup_logger, write_jsonl_event
from proctoring_cv.looking_away import LookingAwayEstimator
from proctoring_cv.schemas import Detection, HeadPoseResult, DiagnosticCode


def select_primary_person(
    detections: List[Detection],
    prev_primary_box: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Detection]:
    """Selects primary examinee person detection based on box area and temporal continuity."""
    person_dets = [d for d in detections if d.class_id == 0]
    if not person_dets:
        return None

    if prev_primary_box is None:
        # Default: select largest area person box
        return max(person_dets, key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]))

    # Find closest center and largest overlap to previous primary box
    prev_cx = (prev_primary_box[0] + prev_primary_box[2]) / 2.0
    prev_cy = (prev_primary_box[1] + prev_primary_box[3]) / 2.0

    best_score = -1e9
    best_cand = None

    for d in person_dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        score = area - (dist * 100.0)
        if score > best_score:
            best_score = score
            best_cand = d

    return best_cand


def draw_hud_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    primary_person: Optional[Detection],
    pose: HeadPoseResult,
    active_events: List[str],
    fps: float,
) -> np.ndarray:
    """Draws a clean, privacy-safe diagnostics overlay on the webcam frame."""
    hud = frame.copy()
    h, w = frame.shape[:2]

    # Draw Detections
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        is_primary = (primary_person and d == primary_person)

        if d.class_id == 0:
            color = (0, 220, 0) if is_primary else (180, 180, 180)
            label = f"EXAMINEE {d.confidence:.2f}" if is_primary else f"PERSON {d.confidence:.2f}"
        else:
            color = (0, 0, 255)
            label = f"PHONE {d.confidence:.2f}"

        cv2.rectangle(hud, (x1, y1), (x2, y2), color, 2)
        cv2.putText(hud, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Top Status Bar
    cv2.rectangle(hud, (0, 0), (w, 40), (25, 25, 25), -1)
    status_text = f"FPS: {fps:.1f} | Calibrated: {'YES' if pose.is_valid and pose.diagnostic == DiagnosticCode.OK else 'PENDING'}"
    cv2.putText(hud, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Pose metrics
    pose_text = f"Yaw: {pose.calibrated_yaw:+.1f} deg | Pitch: {pose.calibrated_pitch:+.1f} deg"
    cv2.putText(hud, pose_text, (w - 360, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 100), 1)

    # Active Observable Events Banner
    if active_events:
        banner_y = 65
        for ev in active_events:
            text = f"OBSERVATION: {ev}"
            cv2.rectangle(hud, (15, banner_y - 20), (380, banner_y + 10), (0, 0, 180), -1)
            cv2.putText(hud, text, (25, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            banner_y += 35

    return hud


def run_webcam_inference(
    config_path: Path | str = "configs/runtime.yaml",
    checkpoint_path: Optional[str] = None,
    webcam_index: int = 0,
) -> int:
    """Executes live webcam pipeline."""
    logger = setup_logger("infer_webcam")
    config: AppConfig = load_config(config_path)

    chkpt_path = checkpoint_path or config.runtime.detector_checkpoint or "yolo11n.yaml"
    logger.info(f"Loading detector from: {chkpt_path}")

    # Build or load detector
    if str(chkpt_path).endswith(".pt"):
        detector = build_detector(config, mode="inference")
    else:
        logger.info("Instantiating detector architecture (YAML mode)...")
        detector = build_detector(config, mode="scratch")

    pose_estimator = LookingAwayEstimator(config=config.looking_away)
    event_engine = EventEngine(
        config=config.event_engine,
        model_id=config.experiment_id,
        config_hash=config.compute_sha256(),
    )

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam index {webcam_index}")
        return 1

    logger.info("Webcam opened. Press 'q' to quit, 'c' to recalibrate.")
    prev_primary_box: Optional[Tuple[float, float, float, float]] = None
    log_file = Path(config.runtime.log_file)

    t_start = time.perf_counter()
    frame_count = 0
    fps = 0.0

    active_event_types: List[str] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame from webcam. Exiting loop.")
                break

            frame_count += 1
            now = time.perf_counter()
            current_timestamp = now - t_start

            # Calculate FPS
            if frame_count % 15 == 0:
                fps = 15.0 / (now - (t_start + (frame_count - 15) / 30.0))

            # 1. Run Detector
            detections = predict_frame(
                detector,
                frame,
                timestamp=current_timestamp,
                confidence_threshold=config.runtime.confidence_threshold,
                iou_threshold=config.runtime.iou_threshold,
                model_id=config.experiment_id,
            )

            # 2. Select Primary Person
            primary_person = select_primary_person(detections, prev_primary_box)
            if primary_person:
                prev_primary_box = primary_person.bbox_xyxy

            # 3. Head Pose & Looking-Away
            pose_result = pose_estimator.process_frame(frame, timestamp=current_timestamp)
            is_looking_away = pose_estimator.is_looking_away_instantaneous(pose_result)

            # 4. Event Engine
            emitted_events = event_engine.process_frame_observations(
                timestamp=current_timestamp,
                detections=detections,
                head_pose=pose_result,
                is_looking_away=is_looking_away,
            )

            # 5. Log structured event records
            for event in emitted_events:
                write_jsonl_event(log_file, event.model_dump())
                logger.info(f"[{current_timestamp:.2f}s] {event.event_type.value} -> {event.state.value} (dur={event.duration_seconds}s)")
                if event.state.value == "started" and event.event_type.value not in active_event_types:
                    active_event_types.append(event.event_type.value)
                elif event.state.value == "ended" and event.event_type.value in active_event_types:
                    active_event_types.remove(event.event_type.value)

            # 6. Display HUD
            if config.runtime.show_overlay:
                hud = draw_hud_overlay(frame, detections, primary_person, pose_result, active_event_types, fps)
                cv2.imshow("Proctoring CV - Live Stream", hud)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    logger.info("Recalibrating neutral head pose...")
                    pose_estimator.reset_calibration()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam inference session ended cleanly.")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam Live Proctoring Inference")
    parser.add_argument("--config", type=str, default="configs/runtime.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to detector checkpoint (.pt)")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()

    sys.exit(run_webcam_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        webcam_index=args.webcam,
    ))


if __name__ == "__main__":
    main()
