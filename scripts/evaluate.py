"""Evaluate object detector checkpoint on validation or locked test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

from proctoring_cv.config import load_config
from proctoring_cv.logging_utils import setup_logger
from proctoring_cv.reproducibility import compute_file_sha256


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    config_path: Path | str = "configs/experiments/yolo11n_scratch_coco.yaml",
    split: str = "val",
    output_dir: Optional[Path | str] = None,
) -> int:
    """Runs evaluation using Ultralytics validator on the specified dataset split."""
    logger = setup_logger("evaluate")
    chkpt = Path(checkpoint_path)

    if not chkpt.is_file():
        logger.error(f"Checkpoint file not found: {chkpt}")
        return 1

    config = load_config(config_path)
    dataset_root = Path(config.dataset.root)
    dataset_yaml = dataset_root / "dataset.yaml"

    if not dataset_yaml.is_file():
        logger.error(f"dataset.yaml not found at: {dataset_yaml}")
        return 1

    logger.info(f"=== Evaluating Checkpoint: {chkpt} on split: '{split}' ===")
    chkpt_sha = compute_file_sha256(chkpt)
    logger.info(f"Checkpoint SHA256: {chkpt_sha}")

    try:
        model = YOLO(str(chkpt))
        metrics = model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=config.training.imgsz,
            device=config.training.device if config.training.device == "cpu" else "0",
            plots=True,
            save_json=True,
        )

        results_dict = {
            "checkpoint": str(chkpt),
            "checkpoint_sha256": chkpt_sha,
            "split": split,
            "mAP50": round(float(metrics.box.map50), 4) if hasattr(metrics.box, "map50") else 0.0,
            "mAP50_95": round(float(metrics.box.map), 4) if hasattr(metrics.box, "map") else 0.0,
            "precision": round(float(metrics.box.mp), 4) if hasattr(metrics.box, "mp") else 0.0,
            "recall": round(float(metrics.box.mr), 4) if hasattr(metrics.box, "mr") else 0.0,
            "class_names": config.dataset.names,
        }

        out_path = Path(output_dir) if output_dir else chkpt.parent.parent / "results"
        out_path.mkdir(parents=True, exist_ok=True)
        res_file = out_path / f"eval_{split}_{chkpt.stem}.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Evaluation finished: mAP50={results_dict['mAP50']}, mAP50-95={results_dict['mAP50_95']}")
        logger.info(f"Results saved to: {res_file}")
        return 0

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Proctoring Detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, default="configs/experiments/yolo11n_scratch_coco.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    sys.exit(evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
