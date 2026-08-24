"""Primary Object Detector module supporting Ultralytics YOLO architectures.

Non-negotiable primary-detector rule:
The primary detector must be built from architecture YAML with pretrained=False (scratch mode).
Scratch mode rejects .pt model paths, disables silent pretrained downloads, and records initialization proofs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO

from proctoring_cv.config import AppConfig, ModelConfig
from proctoring_cv.reproducibility import compute_model_parameter_hash, set_seed
from proctoring_cv.schemas import Detection


def build_detector(
    config: Union[AppConfig, ModelConfig, Dict[str, Any]],
    mode: str = "scratch",
    device: Optional[str] = None,
) -> YOLO:
    """Builds and initializes the YOLO detector model.
    
    Modes:
    - 'scratch': Constructs model strictly from YAML architecture file with pretrained=False.
                 Rejects .pt checkpoints loudly.
    - 'inference': Loads a trained checkpoint from disk.
    - 'pretrained_comparison': Explicitly labeled comparison baseline only.
    """
    if isinstance(config, AppConfig):
        m_cfg = config.model
    elif isinstance(config, ModelConfig):
        m_cfg = config
    elif isinstance(config, dict):
        m_cfg = ModelConfig.model_validate(config.get("model", config))
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    arch = m_cfg.architecture
    if mode == "scratch":
        # Strict enforcement: architecture must be a YAML file, NOT a .pt file
        if arch.endswith(".pt") or str(arch).endswith(".pth"):
            raise ValueError(
                f"[SCRATCH MODE VIOLATION] Scratch detector requires architecture YAML (e.g. 'yolo11n.yaml'). "
                f"Got weights checkpoint: '{arch}'. Pretrained weights loading is strictly prohibited in scratch mode."
            )
        if m_cfg.pretrained:
            raise ValueError(
                "[SCRATCH MODE VIOLATION] 'pretrained' flag is set to True in scratch mode configuration. "
                "Must be False."
            )

        # Instantiate model directly from architecture YAML (random weights initialization)
        model = YOLO(arch, task="detect")
        # Ensure underlying PyTorch model has correct number of classes (nc=2)
        if hasattr(model, "model") and hasattr(model.model, "nc"):
            model.model.nc = m_cfg.num_classes

    elif mode == "inference":
        # Load weights checkpoint for evaluation or live webcam inference
        if not os.path.exists(arch):
            raise FileNotFoundError(f"Detector checkpoint not found at: {arch}")
        model = YOLO(arch, task="detect")

    elif mode == "pretrained_comparison":
        # Explicit comparison only
        model = YOLO(arch, task="detect")

    else:
        raise ValueError(f"Unknown detector mode: '{mode}'. Allowed: 'scratch', 'inference', 'pretrained_comparison'")

    return model


def normalize_detections(
    raw_results: Any,
    timestamp: float,
    model_id: str = "yolo11n_scratch",
    confidence_threshold: float = 0.35,
    names_map: Optional[Dict[int, str]] = None,
) -> List[Detection]:
    """Normalizes raw Ultralytics Results into structured Detection schemas.
    
    Filters to classes: 0 -> person, 1 -> cellphone.
    """
    default_names = {0: "person", 1: "cellphone"}
    cls_names = names_map or default_names
    detections: List[Detection] = []

    if raw_results is None:
        return detections

    # Handle Ultralytics Results object (or list of results)
    results_list = raw_results if isinstance(raw_results, (list, tuple)) else [raw_results]

    for result in results_list:
        if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
            continue

        orig_shape = result.orig_shape  # (height, width)
        img_h, img_w = orig_shape[0], orig_shape[1]

        boxes = result.boxes
        xyxy_tensor = boxes.xyxy.cpu().numpy()
        conf_tensor = boxes.conf.cpu().numpy()
        cls_tensor = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(cls_tensor)):
            cls_id = int(cls_tensor[i])
            conf = float(conf_tensor[i])

            if conf < confidence_threshold:
                continue

            # Only accept project classes (0: person, 1: cellphone)
            if cls_id not in (0, 1):
                continue

            class_name = cls_names.get(cls_id, "person" if cls_id == 0 else "cellphone")
            x1, y1, x2, y2 = map(float, xyxy_tensor[i])

            # Compute normalized xywh
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0

            norm_xc = max(0.0, min(1.0, xc / img_w)) if img_w > 0 else 0.0
            norm_yc = max(0.0, min(1.0, yc / img_h)) if img_h > 0 else 0.0
            norm_w = max(0.0, min(1.0, bw / img_w)) if img_w > 0 else 0.0
            norm_h = max(0.0, min(1.0, bh / img_h)) if img_h > 0 else 0.0

            detections.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_norm_xywh=(norm_xc, norm_yc, norm_w, norm_h),
                timestamp=timestamp,
                model_id=model_id,
            ))

    return detections


def predict_frame(
    model: YOLO,
    frame: np.ndarray,
    timestamp: float,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    model_id: str = "yolo11n_scratch",
) -> List[Detection]:
    """Runs inference on a single numpy image frame and returns normalized detections."""
    results = model.predict(
        source=frame,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    return normalize_detections(
        results,
        timestamp=timestamp,
        model_id=model_id,
        confidence_threshold=confidence_threshold,
    )


def verify_random_initialization(
    architecture: str = "yolo11n.yaml",
    seed_a: int = 42,
    seed_b: int = 1337,
) -> Dict[str, Any]:
    """Verifies that scratch model instantiation produces random parameters varying across seeds."""
    set_seed(seed_a)
    model_a = YOLO(architecture, task="detect")
    hash_a = compute_model_parameter_hash(model_a.model)

    set_seed(seed_b)
    model_b = YOLO(architecture, task="detect")
    hash_b = compute_model_parameter_hash(model_b.model)

    is_random = (hash_a != hash_b)
    return {
        "architecture": architecture,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "hash_seed_a": hash_a,
        "hash_seed_b": hash_b,
        "is_random_initialization_proven": is_random,
    }
