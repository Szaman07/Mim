"""Tiny overfit pipeline test: Trains a scratch YOLO detector on a tiny fixed subset to demonstrate convergence.

Acts as a pre-training gate. Failure blocks full training.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

from proctoring_cv.config import load_config
from proctoring_cv.detector import build_detector
from proctoring_cv.logging_utils import setup_logger
from proctoring_cv.reproducibility import set_seed


def create_synthetic_tiny_dataset(target_dir: Path) -> Path:
    """Creates a tiny synthetic 4-image dataset in YOLO format."""
    dataset_dir = target_dir / "tiny_dataset"
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            img_path = img_dir / f"tiny_{split}_{i}.jpg"
            lbl_path = lbl_dir / f"tiny_{split}_{i}.txt"

            # Create synthetic canvas with shapes
            img_arr = np.zeros((320, 320, 3), dtype=np.uint8)
            img_arr[50:250, 100:200] = [200, 150, 100]  # Person block
            img_arr[150:200, 120:160] = [30, 30, 200]   # Phone block
            Image.fromarray(img_arr).save(img_path)

            with open(lbl_path, "w", encoding="utf-8") as f:
                # 0: person, 1: cellphone
                f.write("0 0.468750 0.468750 0.312500 0.625000\n")
                f.write("1 0.437500 0.546875 0.125000 0.156250\n")

    yaml_path = dataset_dir / "dataset.yaml"
    data_dict = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person", 1: "cellphone"},
        "nc": 2,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_dict, f)

    return yaml_path


def run_tiny_overfit(
    config_path: Path | str = "configs/experiments/yolo11n_scratch_coco.yaml",
    epochs: int = 15,
    device: str = "cpu",
) -> bool:
    """Executes tiny overfit pipeline gate."""
    logger = setup_logger("tiny_overfit")
    config = load_config(config_path)
    set_seed(config.training.seed, deterministic=True)

    scratch_dir = Path("scratch_tiny_overfit")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Generating synthetic tiny dataset...")
        data_yaml = create_synthetic_tiny_dataset(scratch_dir)

        logger.info(f"Building scratch detector from architecture: {config.model.architecture}")
        model = build_detector(config, mode="scratch")

        logger.info(f"Starting {epochs} epochs overfit training on {device}...")
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=320,
            batch=2,
            device=device,
            project=str(scratch_dir / "runs"),
            name="tiny_run",
            exist_ok=True,
            verbose=False,
            plots=False,
            save=False,
            val=True,
        )

        logger.info("Tiny overfit test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Tiny overfit gate failed with error: {e}", exc_info=True)
        return False
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tiny Overfit Pre-training Gate")
    parser.add_argument("--config", type=str, default="configs/experiments/yolo11n_scratch_coco.yaml")
    parser.add_argument("--epochs", type=int, default=5, help="Number of test epochs (default: 5)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    args = parser.parse_args()

    success = run_tiny_overfit(args.config, epochs=args.epochs, device=args.device)
    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
