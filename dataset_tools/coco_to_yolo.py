"""COCO annotation to YOLO format converter.

Maps:
- COCO 'person' (category_id 1 in official COCO 80-class) -> Project class 0
- COCO 'cell phone' (category_id 77 in official COCO 80-class) -> Project class 1
Supports arbitrary custom COCO category mappings via name lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


COCO_DEFAULT_NAME_MAP = {
    "person": 0,
    "cell phone": 1,
}


def convert_coco_box_to_yolo(
    bbox: List[float],
    img_w: int,
    img_h: int,
    clip_tolerance: float = 0.01,
) -> Optional[Tuple[float, float, float, float]]:
    """Converts COCO pixel bbox [x_min, y_min, width, height] to normalized YOLO [x_center, y_center, w, h].
    
    Returns None if the box is degenerate or completely invalid.
    """
    if len(bbox) != 4:
        return None

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    # Check bounds with tolerance
    x_max = x + w
    y_max = y + h

    if x < -clip_tolerance * img_w or y < -clip_tolerance * img_h:
        return None
    if x_max > (1.0 + clip_tolerance) * img_w or y_max > (1.0 + clip_tolerance) * img_h:
        return None

    # Clip safely to image dimensions
    x = max(0.0, min(float(x), float(img_w)))
    y = max(0.0, min(float(y), float(img_h)))
    x_max = max(0.0, min(float(x_max), float(img_w)))
    y_max = max(0.0, min(float(y_max), float(img_h)))

    w = x_max - x
    h = y_max - y
    if w <= 0 or h <= 0:
        return None

    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h

    # Ensure strictly in [0.0, 1.0]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return (x_center, y_center, norm_w, norm_h)


def convert_coco_dataset(
    coco_json_path: Path | str,
    output_labels_dir: Path | str,
    images_dir: Optional[Path | str] = None,
    name_map: Optional[Dict[str, int]] = None,
    filter_empty: bool = True,
) -> Dict[str, Any]:
    """Converts a COCO JSON dataset to normalized YOLO label files."""
    json_path = Path(coco_json_path)
    out_dir = Path(output_labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_map = name_map or COCO_DEFAULT_NAME_MAP

    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build category ID to target class ID mapping
    cat_id_to_target: Dict[int, int] = {}
    source_cat_names: Dict[int, str] = {}
    for cat in coco_data.get("categories", []):
        c_id = cat["id"]
        c_name = cat["name"].lower().strip()
        source_cat_names[c_id] = c_name
        if c_name in target_map:
            cat_id_to_target[c_id] = target_map[c_name]

    # Map image ID to metadata
    images: Dict[int, Dict[str, Any]] = {}
    for img in coco_data.get("images", []):
        images[img["id"]] = img

    # Group annotations by image
    img_annotations: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in images}
    dropped_category_counts: Dict[str, int] = {}
    invalid_box_count = 0

    for ann in coco_data.get("annotations", []):
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        if cat_id not in cat_id_to_target:
            cat_name = source_cat_names.get(cat_id, str(cat_id))
            dropped_category_counts[cat_name] = dropped_category_counts.get(cat_name, 0) + 1
            continue
        if img_id in img_annotations:
            img_annotations[img_id].append(ann)

    # Write YOLO label files
    converted_images = 0
    total_objects = {0: 0, 1: 0}

    for img_id, anns in img_annotations.items():
        if filter_empty and len(anns) == 0:
            continue

        img_info = images[img_id]
        img_filename = img_info["file_name"]
        stem = Path(img_filename).stem
        label_file = out_dir / f"{stem}.txt"

        img_w = img_info["width"]
        img_h = img_info["height"]

        lines: List[str] = []
        for ann in anns:
            target_cls = cat_id_to_target[ann["category_id"]]
            yolo_box = convert_coco_box_to_yolo(ann["bbox"], img_w, img_h)
            if yolo_box is None:
                invalid_box_count += 1
                continue

            x_c, y_c, nw, nh = yolo_box
            lines.append(f"{target_cls} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}")
            total_objects[target_cls] = total_objects.get(target_cls, 0) + 1

        if lines or not filter_empty:
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            converted_images += 1

    report = {
        "source_json": str(json_path),
        "total_source_images": len(images),
        "converted_images": converted_images,
        "class_counts": total_objects,
        "invalid_boxes_dropped": invalid_box_count,
        "dropped_non_target_categories": dropped_category_counts,
        "target_class_map": target_map,
    }

    report_path = out_dir.parent / "coco_conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
