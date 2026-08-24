"""Filter Open Images V7 metadata and annotations to create a capped acquisition manifest.

Filters target classes:
- Person (/m/01g317 or 'Person') -> Project class 0
- Mobile phone (/m/050k8 or 'Mobile phone') -> Project class 1
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


OPENIMAGES_TARGET_LABELS = {
    "/m/01g317": "Person",
    "/m/050k8": "Mobile phone",
}


def load_class_descriptions(descriptions_csv: Path | str) -> Dict[str, str]:
    """Loads Open Images class description CSV mapping LabelName (/m/...) to DisplayName."""
    path = Path(descriptions_csv)
    mapping: Dict[str, str] = {}
    if not path.is_file():
        return OPENIMAGES_TARGET_LABELS

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                mapping[row[0].strip()] = row[1].strip()
    return mapping


def filter_openimages_annotations(
    annotations_csv: Path | str,
    output_filtered_csv: Path | str,
    max_images_per_class: int = 10000,
    target_display_names: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Filters bounding boxes for target classes up to a specified image cap per class."""
    ann_path = Path(annotations_csv)
    out_path = Path(output_filtered_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target_map = target_display_names or OPENIMAGES_TARGET_LABELS
    # Inverse map from display name to label code
    name_to_code = {v.lower(): k for k, v in target_map.items()}

    images_per_class: Dict[str, Set[str]] = {k: set() for k in target_map}
    filtered_rows: List[Dict[str, str]] = []

    with open(ann_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_code = row.get("LabelName", "")
            img_id = row.get("ImageID", "")

            # Check if this label is in our target set
            matched_code = None
            if label_code in target_map:
                matched_code = label_code
            elif label_code.lower() in name_to_code:
                matched_code = name_to_code[label_code.lower()]

            if matched_code:
                if len(images_per_class[matched_code]) < max_images_per_class:
                    images_per_class[matched_code].add(img_id)
                    filtered_rows.append(row)
                elif img_id in images_per_class[matched_code]:
                    filtered_rows.append(row)

    # Write filtered rows
    if filtered_rows:
        fieldnames = list(filtered_rows[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)

    all_selected_images = set()
    for img_set in images_per_class.values():
        all_selected_images.update(img_set)

    summary = {
        "source_annotations": str(ann_path),
        "filtered_csv": str(out_path),
        "total_filtered_boxes": len(filtered_rows),
        "unique_images": len(all_selected_images),
        "counts_per_class": {target_map.get(k, k): len(v) for k, v in images_per_class.items()},
    }

    summary_file = out_path.parent / "openimages_filter_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
