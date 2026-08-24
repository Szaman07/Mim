"""Detector evaluation metrics: Precision, Recall, F1, mAP50, mAP50-95, Latency, and FPS."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union for two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_class_prf1(
    tp: int,
    fp: int,
    fn: int,
) -> Dict[str, float]:
    """Computes Precision, Recall, and F1 score from TP, FP, FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_detector_metrics(
    ground_truths: List[Dict[str, Any]],  # [{ "boxes": [[cls, x1, y1, x2, y2]], ... }]
    predictions: List[Dict[str, Any]],    # [{ "boxes": [[cls, conf, x1, y1, x2, y2]], ... }]
    iou_threshold: float = 0.50,
    classes: Tuple[int, ...] = (0, 1),
) -> Dict[str, Any]:
    """Evaluates detections against ground truths for specified classes."""
    class_stats: Dict[int, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}

    for gt_img, pred_img in zip(ground_truths, predictions):
        gt_boxes = gt_img.get("boxes", [])
        pred_boxes = sorted(pred_img.get("boxes", []), key=lambda b: b[1], reverse=True)  # Sort by conf

        matched_gt = set()

        for p in pred_boxes:
            p_cls = int(p[0])
            p_box = (p[2], p[3], p[4], p[5])

            if p_cls not in class_stats:
                continue

            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                g_cls = int(g[0])
                if g_cls != p_cls:
                    continue
                g_box = (g[1], g[2], g[3], g[4])
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                class_stats[p_cls]["tp"] += 1
                matched_gt.add(best_gt_idx)
            else:
                class_stats[p_cls]["fp"] += 1

        # Unmatched GT are false negatives
        for g_idx, g in enumerate(gt_boxes):
            g_cls = int(g[0])
            if g_cls in class_stats and g_idx not in matched_gt:
                class_stats[g_cls]["fn"] += 1

    per_class_results = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    class_names = {0: "person", 1: "cellphone"}
    for c in classes:
        st = class_stats[c]
        total_tp += st["tp"]
        total_fp += st["fp"]
        total_fn += st["fn"]
        c_name = class_names.get(c, str(c))
        per_class_results[c_name] = {
            "counts": st,
            **compute_class_prf1(st["tp"], st["fp"], st["fn"]),
        }

    overall_prf1 = compute_class_prf1(total_tp, total_fp, total_fn)

    return {
        "iou_threshold": iou_threshold,
        "overall": overall_prf1,
        "per_class": per_class_results,
    }


def measure_throughput_fps(
    model: Any,
    dummy_input: np.ndarray,
    warmup_runs: int = 10,
    test_runs: int = 50,
) -> Dict[str, float]:
    """Measures model inference latency (ms) and throughput (FPS)."""
    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(source=dummy_input, verbose=False)

    times = []
    for _ in range(test_runs):
        t0 = time.perf_counter()
        _ = model.predict(source=dummy_input, verbose=False)
        times.append(time.perf_counter() - t0)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    latency_ms = avg_time * 1000.0

    return {
        "average_latency_ms": round(latency_ms, 2),
        "fps": round(fps, 2),
        "min_latency_ms": round(min(times) * 1000.0, 2),
        "max_latency_ms": round(max(times) * 1000.0, 2),
    }
