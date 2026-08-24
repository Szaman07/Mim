"""Unit tests for dataset validation rules, error catching, and report generation."""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from scripts.validate_dataset import validate_yolo_dataset


def create_dummy_dataset(root: Path) -> None:
    """Helper to scaffold a minimal valid YOLO dataset."""
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            img_p = img_dir / f"img_{split}_{i}.jpg"
            lbl_p = lbl_dir / f"img_{split}_{i}.txt"

            # Create distinct image
            arr = np.full((100, 100, 3), 50 * (i + 1), dtype=np.uint8)
            Image.fromarray(arr).save(img_p)

            with open(lbl_p, "w", encoding="utf-8") as f:
                f.write("0 0.500000 0.500000 0.400000 0.400000\n")
                f.write("1 0.200000 0.200000 0.100000 0.100000\n")


def test_valid_dataset(tmp_path: Path):
    create_dummy_dataset(tmp_path)
    is_valid, report = validate_yolo_dataset(tmp_path, check_leakage=False)
    assert is_valid is True
    assert report["overall_stats"]["total_images"] == 4
    assert report["overall_stats"]["total_boxes"] == 8
    assert report["overall_stats"]["class_counts"][0] == 4
    assert report["overall_stats"]["class_counts"][1] == 4


def test_malformed_label_fails(tmp_path: Path):
    create_dummy_dataset(tmp_path)
    # Inject malformed line (only 3 numbers)
    bad_lbl = tmp_path / "labels" / "train" / "img_train_0.txt"
    with open(bad_lbl, "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5\n")

    is_valid, report = validate_yolo_dataset(tmp_path, check_leakage=False)
    assert is_valid is False
    assert any("Malformed label" in e for e in report["errors"])


def test_out_of_bounds_box_fails(tmp_path: Path):
    create_dummy_dataset(tmp_path)
    # Inject out of bounds coordinate (xc = 1.5)
    bad_lbl = tmp_path / "labels" / "train" / "img_train_0.txt"
    with open(bad_lbl, "w", encoding="utf-8") as f:
        f.write("0 1.500000 0.500000 0.400000 0.400000\n")

    is_valid, report = validate_yolo_dataset(tmp_path, check_leakage=False)
    assert is_valid is False
    assert any("Out of bounds" in e for e in report["errors"])


def test_unknown_class_fails(tmp_path: Path):
    create_dummy_dataset(tmp_path)
    # Inject unknown class 99
    bad_lbl = tmp_path / "labels" / "train" / "img_train_0.txt"
    with open(bad_lbl, "w", encoding="utf-8") as f:
        f.write("99 0.500000 0.500000 0.400000 0.400000\n")

    is_valid, report = validate_yolo_dataset(tmp_path, check_leakage=False)
    assert is_valid is False
    assert any("Unknown class ID 99" in e for e in report["errors"])


def test_corrupt_image_fails(tmp_path: Path):
    create_dummy_dataset(tmp_path)
    # Corrupt an image file with invalid bytes
    bad_img = tmp_path / "images" / "train" / "img_train_0.jpg"
    with open(bad_img, "wb") as f:
        f.write(b"not an image byte stream")

    is_valid, report = validate_yolo_dataset(tmp_path, check_leakage=False)
    assert is_valid is False
    assert any("Unreadable image" in e for e in report["errors"])
