"""Comprehensive dataset validator for YOLO format object detection datasets.

Fails loudly on invalid annotations, corrupt images, out-of-bound boxes, or cross-split leakage.
Generates JSON and CSV summary reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

# Ensure repository root is in sys.path for CLI execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

from proctoring_cv.config import load_config
from proctoring_cv.logging_utils import setup_logger
from dataset_tools.deduplicate import check_cross_split_leakage


def validate_yolo_dataset(
    dataset_root: Path | str,
    valid_classes: Tuple[int, ...] = (0, 1),
    dry_run: bool = False,
    sample_n: int = 0,
    check_leakage: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """Validates full dataset structure, image health, bounding box bounds, and split integrity."""
    logger = setup_logger("validate_dataset")
    root = Path(dataset_root)

    report: Dict[str, Any] = {
        "dataset_root": str(root),
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "splits": {},
        "overall_stats": {
            "total_images": 0,
            "total_boxes": 0,
            "class_counts": {c: 0 for c in valid_classes},
            "co_occurrence_count": 0,  # Images containing both person and phone
            "size_bins": {"small": 0, "medium": 0, "large": 0},
        },
    }

    if dry_run:
        logger.info(f"[DRY RUN] Validating dataset structure at {root} (sample={sample_n}).")
        return True, report

    splits = ["train", "val", "test"]
    split_images: Dict[str, List[Path]] = {}

    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.is_dir():
            if split in ("train", "val"):
                report["errors"].append(f"Missing required split image directory: {img_dir}")
                report["is_valid"] = False
            continue

        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
        if sample_n > 0:
            images = images[:sample_n]
        split_images[split] = images

        split_stat = {
            "images_count": len(images),
            "boxes_count": 0,
            "class_counts": {c: 0 for c in valid_classes},
            "corrupt_images": [],
            "malformed_labels": [],
            "out_of_bounds_boxes": [],
            "missing_label_files": 0,
        }

        for img_path in images:
            report["overall_stats"]["total_images"] += 1
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"

            # 1. Check Image readability and dimensions
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w <= 0 or h <= 0:
                        split_stat["corrupt_images"].append(str(img_path))
                        report["errors"].append(f"Corrupt image dimensions ({w}x{h}): {img_path}")
            except Exception as e:
                split_stat["corrupt_images"].append(str(img_path))
                report["errors"].append(f"Unreadable image {img_path}: {e}")
                continue

            # 2. Check Label File
            if not lbl_path.is_file():
                split_stat["missing_label_files"] += 1
                continue

            present_classes_in_img: Set[int] = set()

            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) != 5:
                            split_stat["malformed_labels"].append(f"{lbl_path}:{line_idx} - expected 5 items, got {len(parts)}")
                            report["errors"].append(f"Malformed label line in {lbl_path}:{line_idx}")
                            continue

                        try:
                            cls_id = int(parts[0])
                            xc = float(parts[1])
                            yc = float(parts[2])
                            bw = float(parts[3])
                            bh = float(parts[4])
                        except ValueError as e:
                            split_stat["malformed_labels"].append(f"{lbl_path}:{line_idx} - parse error {e}")
                            report["errors"].append(f"Non-numeric value in {lbl_path}:{line_idx}")
                            continue

                        # Check finite
                        if any(math.isnan(v) or math.isinf(v) for v in (xc, yc, bw, bh)):
                            report["errors"].append(f"Non-finite coordinates in {lbl_path}:{line_idx}")
                            continue

                        # Check valid class
                        if cls_id not in valid_classes:
                            report["errors"].append(f"Unknown class ID {cls_id} in {lbl_path}:{line_idx}")
                            continue

                        # Check bounding box bounds [0.0, 1.0] and positive dimensions
                        if bw <= 0 or bh <= 0 or xc < 0.0 or xc > 1.0 or yc < 0.0 or yc > 1.0:
                            split_stat["out_of_bounds_boxes"].append(f"{lbl_path}:{line_idx} - invalid box ({xc}, {yc}, {bw}, {bh})")
                            report["errors"].append(f"Out of bounds box in {lbl_path}:{line_idx}")
                            continue

                        # Stats update
                        split_stat["boxes_count"] += 1
                        report["overall_stats"]["total_boxes"] += 1
                        split_stat["class_counts"][cls_id] = split_stat["class_counts"].get(cls_id, 0) + 1
                        report["overall_stats"]["class_counts"][cls_id] = report["overall_stats"]["class_counts"].get(cls_id, 0) + 1
                        present_classes_in_img.add(cls_id)

                        # Area binning (normalized area)
                        box_area = bw * bh
                        if box_area < 0.005:
                            report["overall_stats"]["size_bins"]["small"] += 1
                        elif box_area < 0.05:
                            report["overall_stats"]["size_bins"]["medium"] += 1
                        else:
                            report["overall_stats"]["size_bins"]["large"] += 1

            except Exception as e:
                report["errors"].append(f"Failed to read label file {lbl_path}: {e}")

            if 0 in present_classes_in_img and 1 in present_classes_in_img:
                report["overall_stats"]["co_occurrence_count"] += 1

        split_stat["corrupt_images_count"] = len(split_stat["corrupt_images"])
        split_stat["malformed_labels_count"] = len(split_stat["malformed_labels"])
        split_stat["out_of_bounds_count"] = len(split_stat["out_of_bounds_boxes"])
        report["splits"][split] = split_stat

    # 3. Cross-Split Leakage Check
    if check_leakage and len(split_images) > 1:
        leak_report = check_cross_split_leakage(split_images)
        report["cross_split_leakage"] = leak_report
        if not leak_report["is_leak_free"]:
            report["errors"].append(f"Detected {leak_report['cross_split_leakages_count']} cross-split duplicates!")

    if report["errors"]:
        report["is_valid"] = False

    # Write summary reports
    json_path = root / "dataset_validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    csv_path = root / "dataset_validation_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["is_valid", report["is_valid"]])
        writer.writerow(["total_images", report["overall_stats"]["total_images"]])
        writer.writerow(["total_boxes", report["overall_stats"]["total_boxes"]])
        writer.writerow(["person_count (cls 0)", report["overall_stats"]["class_counts"].get(0, 0)])
        writer.writerow(["cellphone_count (cls 1)", report["overall_stats"]["class_counts"].get(1, 0)])
        writer.writerow(["co_occurrence_images", report["overall_stats"]["co_occurrence_count"]])
        writer.writerow(["total_errors", len(report["errors"])])

    logger.info(f"Dataset validation completed. Valid={report['is_valid']}. Errors={len(report['errors'])}")
    return report["is_valid"], report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO Proctoring Dataset")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--root", type=str, default=None, help="Explicit dataset root path")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--sample", type=int, default=0, help="Sample N images per split")
    args = parser.parse_args()

    if args.root:
        root_path = args.root
    else:
        config = load_config(args.config)
        root_path = config.dataset.root

    is_valid, _ = validate_yolo_dataset(root_path, dry_run=args.dry_run, sample_n=args.sample)
    if not is_valid and not args.dry_run:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
