"""Prepare dataset pipeline: download/filter manifests, convert annotations, and build dataset structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from proctoring_cv.config import load_config
from proctoring_cv.logging_utils import setup_logger
from dataset_tools.manifest import generate_manifest


def prepare_dataset(
    config_path: Path | str,
    dry_run: bool = False,
    sample_n: int = 0,
) -> int:
    """Prepares and validates dataset folders, manifests, and dataset.yaml."""
    logger = setup_logger("prepare_dataset")
    config = load_config(config_path)

    dataset_root = Path(config.dataset.root)
    logger.info(f"Preparing dataset '{config.dataset.name}' at: {dataset_root}")

    if dry_run:
        logger.info("[DRY RUN] Simulating dataset preparation workflow.")
        logger.info(f"Target classes: {config.dataset.names}")
        logger.info(f"Manifest paths: train={config.dataset.train_manifest}, val={config.dataset.val_manifest}")
        return 0

    # Ensure directories exist
    images_train = dataset_root / "images" / "train"
    images_val = dataset_root / "images" / "val"
    labels_train = dataset_root / "labels" / "train"
    labels_val = dataset_root / "labels" / "val"

    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    # Generate manifests if images exist
    if any(images_train.iterdir()):
        train_manifest = generate_manifest(
            images_train,
            labels_train,
            Path(config.dataset.train_manifest),
            split="train",
            source=config.dataset.name,
        )
        logger.info(f"Train manifest generated: {train_manifest['total_images']} images, SHA256={train_manifest['manifest_sha256'][:12]}")

    if any(images_val.iterdir()):
        val_manifest = generate_manifest(
            images_val,
            labels_val,
            Path(config.dataset.val_manifest),
            split="val",
            source=config.dataset.name,
        )
        logger.info(f"Val manifest generated: {val_manifest['total_images']} images, SHA256={val_manifest['manifest_sha256'][:12]}")

    # Generate Ultralytics-compatible dataset.yaml inside dataset root
    yolo_data_yaml = dataset_root / "dataset.yaml"
    yolo_data_dict = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": config.dataset.names,
        "nc": config.dataset.nc,
    }
    with open(yolo_data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yolo_data_dict, f, default_flow_style=False)

    logger.info(f"Ultralytics dataset YAML created at: {yolo_data_yaml}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Proctoring CV Dataset")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files")
    parser.add_argument("--sample", type=int, default=0, help="Sample N images for quick test")
    args = parser.parse_args()

    sys.exit(prepare_dataset(args.config, dry_run=args.dry_run, sample_n=args.sample))


if __name__ == "__main__":
    main()
