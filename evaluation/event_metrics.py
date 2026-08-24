"""Event-level temporal metrics: interval precision/recall, onset delay, temporal IoU, and fragmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def compute_temporal_interval_iou(
    int1: Tuple[float, float],
    int2: Tuple[float, float],
) -> float:
    """Computes Intersection over Union for two 1D time intervals (start_sec, end_sec)."""
    start = max(int1[0], int2[0])
    end = min(int1[1], int2[1])
    inter = max(0.0, end - start)

    dur1 = max(0.0, int1[1] - int1[0])
    dur2 = max(0.0, int2[1] - int2[0])
    union = dur1 + dur2 - inter

    if union <= 0.0:
        return 0.0
    return inter / union


def evaluate_event_intervals(
    gt_intervals: List[Dict[str, Any]],    # [{ "event_type": "PHONE_DETECTED", "start": 5.0, "end": 12.0 }]
    pred_intervals: List[Dict[str, Any]],  # [{ "event_type": "PHONE_DETECTED", "start": 5.5, "end": 11.8 }]
    temporal_iou_threshold: float = 0.30,
    session_duration_hours: float = 1.0,
) -> Dict[str, Any]:
    """Computes event-level temporal metrics across ground-truth and predicted intervals."""
    tp, fp, fn = 0, 0, 0
    onset_delays: List[float] = []
    duration_biases: List[float] = []
    matched_gt = set()

    for p in pred_intervals:
        p_type = p["event_type"]
        p_span = (p["start"], p["end"])

        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g in enumerate(gt_intervals):
            if g_idx in matched_gt or g["event_type"] != p_type:
                continue
            g_span = (g["start"], g["end"])
            iou = compute_temporal_interval_iou(p_span, g_span)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= temporal_iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            g_match = gt_intervals[best_gt_idx]
            # Onset delay: how late did detection start after true event start
            onset_delays.append(max(0.0, p["start"] - g_match["start"]))
            # Duration bias: predicted duration - actual duration
            pred_dur = p["end"] - p["start"]
            gt_dur = g_match["end"] - g_match["start"]
            duration_biases.append(pred_dur - gt_dur)
        else:
            fp += 1

    fn = len(gt_intervals) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_onset_delay = sum(onset_delays) / len(onset_delays) if onset_delays else 0.0
    avg_dur_bias = sum(duration_biases) / len(duration_biases) if duration_biases else 0.0

    fp_per_hour = fp / session_duration_hours if session_duration_hours > 0 else fp

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
        "false_positives_per_hour": round(fp_per_hour, 2),
        "average_onset_delay_sec": round(avg_onset_delay, 3),
        "average_duration_bias_sec": round(avg_dur_bias, 3),
    }
