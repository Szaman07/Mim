"""Dataset manifest generation and integrity verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image


def generate_manifest(
    images_dir: Path | str,
    labels_dir: Path | str,
    output_manifest_path: Path | str,
    split: str = "train",
    source: str = "coco2017_filtered",
    license_info: str = "CC BY 4.0",
) -> Dict[str, Any]:
    """Generates an immutable manifest JSON file for a dataset split."""
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    out_path = Path(output_manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    class_counts = {0: 0, 1: 0}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in valid_extensions])

    for img_path in image_files:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        # Read dimensions
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            continue

        # Compute image SHA256
        with open(img_path, "rb") as f:
            img_sha = hashlib.sha256(f.read()).hexdigest()

        # Parse labels
        boxes: List[Dict[str, Any]] = []
        if lbl_path.is_file():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                        boxes.append({
                            "class_id": cls_id,
                            "bbox_norm": [xc, yc, bw, bh],
                        })
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        entries.append({
            "image_id": stem,
            "split": split,
            "source": source,
            "relative_image_path": str(img_path.name),
            "width": w,
            "height": h,
            "sha256": img_sha,
            "license": license_info,
            "boxes": boxes,
        })

    # Manifest content without hash
    manifest_body = {
        "source": source,
        "split": split,
        "license": license_info,
        "total_images": len(entries),
        "total_boxes": sum(class_counts.values()),
        "class_counts": class_counts,
        "images": entries,
    }

    # Compute overall manifest SHA-256
    manifest_str = json.dumps(manifest_body, sort_keys=True)
    manifest_sha256 = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
    manifest_body["manifest_sha256"] = manifest_sha256

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest_body, f, indent=2)

    return manifest_body
