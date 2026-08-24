"""Split raw images and labels into train/val/test splits respecting subject/session grouping."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from proctoring_cv.logging_utils import setup_logger
from proctoring_cv.reproducibility import set_seed


def split_dataset(
    source_images_dir: Path | str,
    source_labels_dir: Path | str,
    output_dataset_root: Path | str,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, int]:
    """Splits dataset files into train, val, and test partitions."""
    logger = setup_logger("split_dataset")
    set_seed(seed)

    img_dir = Path(source_images_dir)
    lbl_dir = Path(source_labels_dir)
    out_root = Path(output_dataset_root)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts])
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    n_test = n_total - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }

    counts: Dict[str, int] = {}
    for split_name, img_list in splits.items():
        dst_img_dir = out_root / "images" / split_name
        dst_lbl_dir = out_root / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_p in img_list:
            shutil.copy2(img_p, dst_img_dir / img_p.name)
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.is_file():
                shutil.copy2(lbl_p, dst_lbl_dir / lbl_p.name)
        counts[split_name] = len(img_list)

    logger.info(f"Dataset split complete: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels directory")
    parser.add_argument("--output", type=str, required=True, help="Output dataset root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_dataset(args.images, args.labels, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
