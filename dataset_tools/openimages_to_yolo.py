"""Convert filtered Open Images V7 bounding boxes to normalized YOLO format."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


OPENIMAGES_CLASS_MAP = {
    "/m/01g317": 0,  # Person
    "Person": 0,
    "/m/050k8": 1,   # Mobile phone
    "Mobile phone": 1,
}


def convert_openimages_box_to_yolo(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Converts Open Images normalized coordinates (XMin, XMax, YMin, YMax) to YOLO (x_center, y_center, w, h)."""
    if xmax <= xmin or ymax <= ymin:
        return None

    # Clip to valid [0.0, 1.0] range
    xmin = max(0.0, min(1.0, xmin))
    xmax = max(0.0, min(1.0, xmax))
    ymin = max(0.0, min(1.0, ymin))
    ymax = max(0.0, min(1.0, ymax))

    w = xmax - xmin
    h = ymax - ymin
    if w <= 0.0 or h <= 0.0:
        return None

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    return (x_center, y_center, w, h)


def convert_openimages_csv_to_yolo(
    filtered_csv: Path | str,
    output_labels_dir: Path | str,
    class_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Reads filtered Open Images box CSV and writes YOLO label files."""
    csv_path = Path(filtered_csv)
    out_dir = Path(output_labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = class_map or OPENIMAGES_CLASS_MAP
    img_annotations: Dict[str, List[str]] = {}
    class_counts = {0: 0, 1: 0}
    invalid_boxes = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row.get("ImageID", "").strip()
            label = row.get("LabelName", "").strip()

            if not img_id or label not in mapping:
                continue

            target_cls = mapping[label]
            try:
                xmin = float(row["XMin"])
                xmax = float(row["XMax"])
                ymin = float(row["YMin"])
                ymax = float(row["YMax"])
            except (KeyError, ValueError):
                invalid_boxes += 1
                continue

            box = convert_openimages_box_to_yolo(xmin, xmax, ymin, ymax)
            if box is None:
                invalid_boxes += 1
                continue

            xc, yc, w, h = box
            line = f"{target_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(line)
            class_counts[target_cls] = class_counts.get(target_cls, 0) + 1

    # Write YOLO .txt files
    for img_id, lines in img_annotations.items():
        label_file = out_dir / f"{img_id}.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    report = {
        "source_csv": str(csv_path),
        "total_images_with_labels": len(img_annotations),
        "class_counts": class_counts,
        "invalid_boxes_dropped": invalid_boxes,
    }

    report_path = out_dir.parent / "openimages_conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
