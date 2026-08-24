"""Unit tests for dataset conversion tools (COCO to YOLO, Open Images to YOLO, deduplication, manifest)."""

import csv
import json
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from dataset_tools.coco_to_yolo import convert_coco_box_to_yolo, convert_coco_dataset
from dataset_tools.openimages_filter import filter_openimages_annotations
from dataset_tools.openimages_to_yolo import convert_openimages_box_to_yolo, convert_openimages_csv_to_yolo
from dataset_tools.deduplicate import compute_dhash, hamming_distance, find_duplicates, check_cross_split_leakage
from dataset_tools.manifest import generate_manifest


def test_coco_box_conversion_normal():
    # Box [x, y, w, h] in 1000x500 image: [100, 50, 200, 100]
    # x_center = (100 + 100) = 200 / 1000 = 0.20
    # y_center = (50 + 50) = 100 / 500 = 0.20
    # w = 200/1000 = 0.20, h = 100/500 = 0.20
    yolo_box = convert_coco_box_to_yolo([100, 50, 200, 100], 1000, 500)
    assert yolo_box is not None
    xc, yc, w, h = yolo_box
    assert pytest.approx(xc, 0.001) == 0.20
    assert pytest.approx(yc, 0.001) == 0.20
    assert pytest.approx(w, 0.001) == 0.20
    assert pytest.approx(h, 0.001) == 0.20


def test_coco_box_conversion_invalid():
    # Degenerate zero width/height
    assert convert_coco_box_to_yolo([10, 10, 0, 50], 100, 100) is None
    assert convert_coco_box_to_yolo([10, 10, 50, -5], 100, 100) is None
    # Way out of bounds
    assert convert_coco_box_to_yolo([-200, -200, 50, 50], 100, 100) is None


def test_coco_dataset_full_conversion(tmp_path: Path):
    coco_json = tmp_path / "sample_coco.json"
    data = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 800, "height": 600},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 77, "name": "cell phone"},
            {"id": 3, "name": "car"},
        ],
        "annotations": [
            {"id": 101, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 100]},
            {"id": 102, "image_id": 1, "category_id": 77, "bbox": [120, 120, 20, 30]},
            {"id": 103, "image_id": 2, "category_id": 3, "bbox": [200, 200, 100, 50]}, # non-target
        ]
    }
    with open(coco_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    out_labels = tmp_path / "labels"
    report = convert_coco_dataset(coco_json, out_labels)
    assert report["converted_images"] == 1
    assert report["class_counts"][0] == 1
    assert report["class_counts"][1] == 1
    assert (out_labels / "img1.txt").is_file()


def test_openimages_conversion(tmp_path: Path):
    csv_file = tmp_path / "oi_ann.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"])
        writer.writerow(["img_01", "freeform", "/m/01g317", "1", "0.1", "0.5", "0.2", "0.8", "0", "0", "0", "0", "0"])
        writer.writerow(["img_01", "freeform", "/m/050k8", "1", "0.6", "0.8", "0.6", "0.9", "0", "0", "0", "0", "0"])

    out_labels = tmp_path / "oi_labels"
    report = convert_openimages_csv_to_yolo(csv_file, out_labels)
    assert report["total_images_with_labels"] == 1
    assert report["class_counts"][0] == 1
    assert report["class_counts"][1] == 1
    assert (out_labels / "img_01.txt").is_file()


def test_deduplication_and_dhash(tmp_path: Path):
    img1_p = tmp_path / "img1.jpg"
    img2_p = tmp_path / "img2.jpg"
    img3_p = tmp_path / "img3.jpg"

    # Create distinct dummy images
    arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
    arr1[:50, :50] = 255
    Image.fromarray(arr1).save(img1_p)
    # Exact copy
    Image.fromarray(arr1).save(img2_p)
    # Completely different image
    arr3 = np.full((100, 100, 3), 128, dtype=np.uint8)
    arr3[20:80, 20:80] = 50
    Image.fromarray(arr3).save(img3_p)

    dup_report = find_duplicates([img1_p, img2_p, img3_p])
    assert dup_report["total_duplicates_found"] >= 1

    leak_report = check_cross_split_leakage({
        "train": [img1_p],
        "val": [img2_p],
        "test": [img3_p],
    })
    assert leak_report["is_leak_free"] is False
    assert leak_report["cross_split_leakages_count"] >= 1


def test_manifest_generation(tmp_path: Path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    img_p = img_dir / "sample.jpg"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_p)

    lbl_p = lbl_dir / "sample.txt"
    with open(lbl_p, "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n")

    manifest_p = tmp_path / "manifest.json"
    manifest = generate_manifest(img_dir, lbl_dir, manifest_p, split="train")
    assert manifest["total_images"] == 1
    assert manifest["total_boxes"] == 2
    assert manifest["class_counts"][0] == 1
    assert manifest["class_counts"][1] == 1
    assert "manifest_sha256" in manifest
    assert manifest_p.is_file()
