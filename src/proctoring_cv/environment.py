"""Environment detection and system capability diagnostic probe."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict
import torch


def get_environment_snapshot() -> Dict[str, Any]:
    """Probes and returns complete runtime environment metadata."""
    env: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os_name": os.name,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_devices": [],
        "ultralytics_version": "unknown",
        "opencv_version": "unknown",
        "mediapipe_version": "unknown",
    }

    # Probe GPU device details
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            env["gpu_devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gib": round(props.total_memory / (1024**3), 2),
                "major": props.major,
                "minor": props.minor,
            })

    # Probe Ultralytics
    try:
        import ultralytics
        env["ultralytics_version"] = getattr(ultralytics, "__version__", "installed")
    except ImportError:
        pass

    # Probe OpenCV
    try:
        import cv2
        env["opencv_version"] = getattr(cv2, "__version__", "installed")
    except ImportError:
        pass

    # Probe MediaPipe
    try:
        import mediapipe
        env["mediapipe_version"] = getattr(mediapipe, "__version__", "installed")
    except ImportError:
        pass

    return env


def save_environment_snapshot(output_path: Path | str) -> Dict[str, Any]:
    """Probes environment and writes snapshot JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = get_environment_snapshot()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot


def recommend_batch_size(img_size: int = 640) -> int:
    """Calculates safe recommended batch size based on available VRAM."""
    if not torch.cuda.is_available():
        return 8  # Safe default on CPU
    vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gib >= 20.0:
        return 32 if img_size <= 640 else 16
    elif vram_gib >= 12.0:
        return 16 if img_size <= 640 else 8
    elif vram_gib >= 6.0:
        return 8 if img_size <= 640 else 4
    else:
        return 4
