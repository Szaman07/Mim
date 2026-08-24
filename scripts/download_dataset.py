"""Automated dataset downloader and filter for Proctoring CV (Filtered COCO 2017 2-Class).

Downloads official COCO annotations and images for target classes (person=0, cellphone=1),
or downloads a fast sample subset for development and smoke testing.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proctoring_cv.config import load_config
from proctoring_cv.logging_utils import setup_logger
from dataset_tools.coco_to_yolo import convert_coco_dataset
from dataset_tools.manifest import generate_manifest


COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_and_extract_zip(url: str, extract_to: Path, logger) -> None:
    """Downloads a zip archive with progress reporting and extracts it."""
    logger.info(f"Downloading from: {url}")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    zip_path = extract_to / Path(url).name
    if not zip_path.is_file():
        urllib.request.urlretrieve(url, zip_path)
        logger.info(f"Download complete: {zip_path}")
    else:
        logger.info(f"Archive already exists locally: {zip_path}")

    logger.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info("Extraction complete.")


def create_dev_sample_dataset(target_dir: Path | str, num_samples: int = 20) -> Path:
    """Creates a fast, self-contained development dataset with realistic synthetic images."""
    import numpy as np
    from PIL import Image, ImageDraw

    dataset_root = Path(target_dir)
    splits = {"train": int(num_samples * 0.8), "val": int(num_samples * 0.2)}

    for split, count in splits.items():
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(max(2, count)):
            img_path = img_dir / f"sample_{split}_{i:03d}.jpg"
            lbl_path = lbl_dir / f"sample_{split}_{i:03d}.txt"

            rng = np.random.RandomState(i * 100 + (1 if split == "train" else 500))
            # Base noisy textured background
            arr = rng.randint(50, 220, size=(480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)

            # Draw person silhouette
            px1 = 120 + rng.randint(0, 80)
            py1 = 60 + rng.randint(0, 40)
            px2 = px1 + 280 + rng.randint(0, 50)
            py2 = 470
            draw.rectangle([px1, py1, px2, py2], fill=tuple(rng.randint(20, 100, size=3).tolist()))
            draw.ellipse([px1 + 80, py1 - 40, px1 + 200, py1 + 60], fill=(210, 175, 140)) # Face

            norm_pxc = (px1 + px2) / (2.0 * 640.0)
            norm_pyc = (py1 - 40 + py2) / (2.0 * 480.0)
            norm_pw = (px2 - px1) / 640.0
            norm_ph = (py2 - (py1 - 40)) / 480.0
            lines = [f"0 {norm_pxc:.6f} {norm_pyc:.6f} {norm_pw:.6f} {norm_ph:.6f}"]

            if i % 2 == 1:
                ph_x1 = px2 - 50
                ph_y1 = py1 + 140
                ph_x2 = ph_x1 + 45
                ph_y2 = ph_y1 + 80
                draw.rectangle([ph_x1, ph_y1, ph_x2, ph_y2], fill=(10, 10, 10), outline=(230, 230, 230), width=2)
                ph_xc = (ph_x1 + ph_x2) / (2.0 * 640.0)
                ph_yc = (ph_y1 + ph_y2) / (2.0 * 480.0)
                ph_w = (ph_x2 - ph_x1) / 640.0
                ph_h = (ph_y2 - ph_y1) / 480.0
                lines.append(f"1 {ph_xc:.6f} {ph_yc:.6f} {ph_w:.6f} {ph_h:.6f}")

            img.save(img_path, quality=90)
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    # Generate manifests
    for split in ("train", "val"):
        generate_manifest(
            images_dir=dataset_root / "images" / split,
            labels_dir=dataset_root / "labels" / split,
            output_manifest_path=dataset_root / "manifests" / f"{split}_manifest.json",
            split=split,
            source="proctoring_dev_sample",
        )

    # Generate dataset.yaml
    import yaml
    with open(dataset_root / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.dump({
            "path": str(dataset_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "person", 1: "cellphone"},
            "nc": 2,
        }, f)

    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Proctoring CV dataset")
    parser.add_argument("--source", type=str, default="sample", choices=["sample", "coco_val2017", "full_coco"], help="Dataset acquisition source")
    parser.add_argument("--output", type=str, default="data/datasets/coco2017_filtered_v1", help="Output dataset root directory")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of sample images for dev mode")
    args = parser.parse_args()

    logger = setup_logger("download_dataset")
    out_dir = Path(args.output)

    if args.source == "sample":
        logger.info(f"Generating synthetic development dataset ({args.sample_size} images) at: {out_dir}")
        create_dev_sample_dataset(out_dir, num_samples=args.sample_size)
        logger.info(f"Sample dataset ready at {out_dir}. You can now run training/testing immediately.")

    elif args.source in ("coco_val2017", "full_coco"):
        scratch = Path("scratch_download")
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Acquiring official COCO 2017 annotations...")
            download_and_extract_zip(COCO_ANNOTATIONS_URL, scratch, logger)
            download_and_extract_zip(COCO_VAL2017_URL, scratch, logger)

            val_json = scratch / "annotations" / "instances_val2017.json"
            val_images_src = scratch / "val2017"

            # Filter and convert
            out_labels_val = out_dir / "labels" / "val"
            out_images_val = out_dir / "images" / "val"
            out_images_val.mkdir(parents=True, exist_ok=True)

            logger.info("Filtering annotations for person and cellphone...")
            report = convert_coco_dataset(val_json, out_labels_val)
            logger.info(f"Converted {report['converted_images']} images containing target classes.")

            # Copy relevant images
            for lbl_file in out_labels_val.glob("*.txt"):
                img_name = f"{lbl_file.stem}.jpg"
                src_img = val_images_src / img_name
                if src_img.is_file():
                    shutil.copy2(src_img, out_images_val / img_name)

            # Create train split from a subset of val for small-scale baseline if requested
            out_labels_train = out_dir / "labels" / "train"
            out_images_train = out_dir / "images" / "train"
            out_labels_train.mkdir(parents=True, exist_ok=True)
            out_images_train.mkdir(parents=True, exist_ok=True)

            all_imgs = sorted(list(out_images_val.glob("*.jpg")))
            split_point = int(len(all_imgs) * 0.8)
            for img in all_imgs[:split_point]:
                shutil.move(img, out_images_train / img.name)
                shutil.move(out_labels_val / f"{img.stem}.txt", out_labels_train / f"{img.stem}.txt")

            # Build manifests and dataset.yaml
            for split in ("train", "val"):
                generate_manifest(
                    out_dir / "images" / split,
                    out_dir / "labels" / split,
                    out_dir / "manifests" / f"{split}_manifest.json",
                    split=split,
                    source="coco2017_filtered",
                )

            import yaml
            with open(out_dir / "dataset.yaml", "w", encoding="utf-8") as f:
                yaml.dump({
                    "path": str(out_dir.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "person", 1: "cellphone"},
                    "nc": 2,
                }, f)

            logger.info(f"COCO 2-Class Filtered dataset ready at: {out_dir.resolve()}")
        finally:
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
