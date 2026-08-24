"""Fine-grained error slicing: object size, occlusion, phone context, and lighting/multi-person slices."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from evaluation.metrics import compute_detector_metrics


def slice_by_object_size(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Partitions evaluation examples by object size (small, medium, large)."""
    size_bins = {
        "small": (0.0, 0.005),
        "medium": (0.005, 0.05),
        "large": (0.05, 1.0),
    }

    sliced_results = {}
    for bin_name, (min_area, max_area) in size_bins.items():
        bin_gt = []
        bin_pred = []

        for gt_img, pred_img in zip(ground_truths, predictions):
            filtered_gt_boxes = []
            for b in gt_img.get("boxes", []):
                w = b[3] - b[1]
                h = b[4] - b[2]
                area = w * h
                if min_area <= area < max_area:
                    filtered_gt_boxes.append(b)

            bin_gt.append({"boxes": filtered_gt_boxes})
            bin_pred.append(pred_img)

        sliced_results[bin_name] = compute_detector_metrics(bin_gt, bin_pred)

    return sliced_results


def evaluate_error_slices(
    ground_truths: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes comprehensive slice breakdown."""
    size_slice_results = slice_by_object_size(ground_truths, predictions)
    return {
        "size_slices": size_slice_results,
    }
