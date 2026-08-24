"""Event stream offline replay utility: runs synthetic or recorded frame events through EventEngine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from proctoring_cv.config import load_config
from proctoring_cv.event_engine import EventEngine
from proctoring_cv.logging_utils import setup_logger
from proctoring_cv.schemas import DiagnosticCode, HeadPoseResult, Detection


def generate_synthetic_scenario_stream(scenario: str = "sustained_phone", duration_sec: float = 10.0, fps: float = 10.0) -> List[Dict[str, Any]]:
    """Generates synthetic sequence of timestamped detections and pose inputs."""
    stream: List[Dict[str, Any]] = []
    total_frames = int(duration_sec * fps)
    dt = 1.0 / fps

    for i in range(total_frames):
        t = i * dt
        dets: List[Detection] = []
        is_away = False
        pose_valid = True
        diag = DiagnosticCode.OK

        # Primary person always present
        dets.append(Detection(
            class_id=0,
            class_name="person",
            confidence=0.88,
            bbox_xyxy=(100.0, 50.0, 500.0, 450.0),
            bbox_norm_xywh=(0.5, 0.5, 0.6, 0.8),
            timestamp=t,
        ))

        if scenario == "sustained_phone":
            # Phone visible between t=2.0s and t=7.0s
            if 2.0 <= t <= 7.0:
                dets.append(Detection(
                    class_id=1,
                    class_name="cellphone",
                    confidence=0.85,
                    bbox_xyxy=(350.0, 300.0, 420.0, 400.0),
                    bbox_norm_xywh=(0.6, 0.7, 0.1, 0.2),
                    timestamp=t,
                ))

        elif scenario == "multi_person":
            # Second person visible between t=1.5s and t=6.0s
            if 1.5 <= t <= 6.0:
                dets.append(Detection(
                    class_id=0,
                    class_name="person",
                    confidence=0.78,
                    bbox_xyxy=(450.0, 100.0, 600.0, 400.0),
                    bbox_norm_xywh=(0.8, 0.5, 0.2, 0.6),
                    timestamp=t,
                ))

        elif scenario == "sustained_looking_away":
            # Looking away between t=2.0s and t=8.0s (yaw=35 deg)
            if 2.0 <= t <= 8.0:
                is_away = True

        elif scenario == "face_unavailable":
            # Face unavailable between t=2.0s and t=5.0s
            if 2.0 <= t <= 5.0:
                pose_valid = False
                diag = DiagnosticCode.POSE_UNAVAILABLE

        pose = HeadPoseResult(
            timestamp=t,
            yaw=35.0 if is_away else 0.0,
            pitch=0.0,
            roll=0.0,
            calibrated_yaw=35.0 if is_away else 0.0,
            calibrated_pitch=0.0,
            calibrated_roll=0.0,
            confidence=0.9 if pose_valid else 0.0,
            is_valid=pose_valid,
            diagnostic=diag,
        )

        stream.append({
            "timestamp": t,
            "detections": dets,
            "pose": pose,
            "is_away": is_away,
        })

    return stream


def replay_event_stream(
    config_path: Path | str = "configs/runtime.yaml",
    scenario: str = "sustained_phone",
    output_log: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Runs a replay stream through the EventEngine and outputs emitted records."""
    logger = setup_logger("event_replay")
    config = load_config(config_path)

    engine = EventEngine(
        config=config.event_engine,
        model_id=config.experiment_id,
        config_hash=config.compute_sha256(),
    )

    stream = generate_synthetic_scenario_stream(scenario=scenario)
    logger.info(f"Replaying scenario '{scenario}' ({len(stream)} frames)...")

    emitted_records: List[Dict[str, Any]] = []

    for frame_data in stream:
        t = frame_data["timestamp"]
        dets = frame_data["detections"]
        pose = frame_data["pose"]
        is_away = frame_data["is_away"]

        events = engine.process_frame_observations(
            timestamp=t,
            detections=dets,
            head_pose=pose,
            is_looking_away=is_away,
        )

        for ev in events:
            rec_dict = ev.model_dump()
            emitted_records.append(rec_dict)
            logger.info(f"[{t:.2f}s] Event Emitted: {ev.event_type.value} -> state={ev.state.value}, dur={ev.duration_seconds}s")

    if output_log:
        out_p = Path(output_log)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for rec in emitted_records:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Emitted {len(emitted_records)} event records to {output_log}")

    return emitted_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Event Stream")
    parser.add_argument("--config", type=str, default="configs/runtime.yaml")
    parser.add_argument("--scenario", type=str, default="sustained_phone", choices=["sustained_phone", "multi_person", "sustained_looking_away", "face_unavailable"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    replay_event_stream(
        config_path=args.config,
        scenario=args.scenario,
        output_log=args.output,
    )


if __name__ == "__main__":
    main()
