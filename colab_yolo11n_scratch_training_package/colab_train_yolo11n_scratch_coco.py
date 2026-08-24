#!/usr/bin/env python3
"""Colab-ready YOLO11n scratch-training workflow on a filtered COCO 2017 subset.

This utility intentionally constructs models from ``yolo11n.yaml`` rather than a
``.pt`` checkpoint. It records a hash of the initial parameters, filters COCO to
the project's two labels (person=0, cellphone=1), mirrors durable artifacts to
Google Drive, and supports a safe resume path. It does not run unless invoked.

Example:
  python scripts/colab_train_yolo11n_scratch_coco.py prepare --dataset-root /content/coco_filtered
  python scripts/colab_train_yolo11n_scratch_coco.py sanity --dataset-root /content/coco_filtered
  python scripts/colab_train_yolo11n_scratch_coco.py train --dataset-root /content/coco_filtered
  python scripts/colab_train_yolo11n_scratch_coco.py verify --dataset-root /content/coco_filtered
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # Keep --help and static checks available outside a GPU notebook.
    import yaml
    import torch
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - depends on the invoking environment
    yaml = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    YOLO = Any  # type: ignore[misc,assignment]


COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_BASE = "http://images.cocodataset.org"
CLASS_MAP = {1: 0, 77: 1}  # COCO person -> person; COCO cell phone -> cellphone
CLASS_NAMES = {0: "person", 1: "cellphone"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_training_runtime(require_yolo: bool = True) -> None:
    if yaml is None or (require_yolo and (torch is None or YOLO is Any)):
        raise RuntimeError(
            "Required runtime dependencies are unavailable. Install the project requirements and "
            "Ultralytics first, for example: pip install -e . pycocotools pillow"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_hash(model: YOLO) -> str:
    """Hash model tensors without serializing a pretrained checkpoint."""
    require_training_runtime()
    digest = hashlib.sha256()
    for key, tensor in sorted(model.model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_annotations(cache_dir: Path) -> Path:
    """Download the official COCO 2017 annotation archive once and return its folder."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "annotations_trainval2017.zip"
    annotation_dir = cache_dir / "annotations"
    required = [annotation_dir / "instances_train2017.json", annotation_dir / "instances_val2017.json"]
    if all(item.is_file() for item in required):
        return annotation_dir
    if not target.is_file():
        print(f"Downloading official COCO annotations to {target}")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, target)
    with zipfile.ZipFile(target) as archive:
        archive.extractall(cache_dir)
    if not all(item.is_file() for item in required):
        raise RuntimeError("Official COCO annotation files were not found after extraction.")
    return annotation_dir


def load_instances(annotation_file: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in payload["images"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if int(annotation["category_id"]) in CLASS_MAP and not annotation.get("iscrowd", 0):
            by_image[int(annotation["image_id"])].append(annotation)
    return images, by_image


def choose_ids(by_image: dict[int, list[dict[str, Any]]], limit: int, phone_fraction: float, seed: int) -> list[int]:
    """Select a deterministic set while protecting rare cellphone-positive coverage."""
    rng = random.Random(seed)
    phone_ids = [image_id for image_id, annotations in by_image.items() if any(int(a["category_id"]) == 77 for a in annotations)]
    person_ids = [image_id for image_id, annotations in by_image.items() if any(int(a["category_id"]) == 1 for a in annotations)]
    rng.shuffle(phone_ids)
    rng.shuffle(person_ids)
    desired_phone = min(len(phone_ids), max(1, round(limit * phone_fraction)))
    selected = phone_ids[:desired_phone]
    selected_set = set(selected)
    for image_id in person_ids + phone_ids:
        if len(selected) >= limit:
            break
        if image_id not in selected_set:
            selected.append(image_id)
            selected_set.add(image_id)
    if len(selected) < min(limit, len(by_image)):
        raise RuntimeError("Insufficient COCO images containing person or cell phone labels.")
    return selected


def download_with_retries(url: str, destination: Path, attempts: int = 3) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return True
    for attempt in range(1, attempts + 1):
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(destination)
            return True
        except Exception as exc:  # pragma: no cover - depends on remote transport
            temporary.unlink(missing_ok=True)
            print(f"Download attempt {attempt}/{attempts} failed for {url}: {exc}")
            time.sleep(attempt)
    return False


def yolo_lines(annotations: Iterable[dict[str, Any]], width: int, height: int) -> list[str]:
    lines: list[str] = []
    for annotation in annotations:
        x, y, box_w, box_h = (float(value) for value in annotation["bbox"])
        x = max(0.0, min(x, float(width)))
        y = max(0.0, min(y, float(height)))
        box_w = max(0.0, min(box_w, float(width) - x))
        box_h = max(0.0, min(box_h, float(height) - y))
        if box_w <= 1.0 or box_h <= 1.0:
            continue
        xc = (x + box_w / 2.0) / width
        yc = (y + box_h / 2.0) / height
        lines.append(f"{CLASS_MAP[int(annotation['category_id'])]} {xc:.6f} {yc:.6f} {box_w / width:.6f} {box_h / height:.6f}")
    return lines


def materialize_split(
    *,
    annotation_file: Path,
    split: str,
    output_root: Path,
    limit: int,
    phone_fraction: float,
    seed: int,
) -> dict[str, Any]:
    images, by_image = load_instances(annotation_file)
    selected = choose_ids(by_image, limit=limit, phone_fraction=phone_fraction, seed=seed)
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    manifest: list[dict[str, Any]] = []
    failures: list[int] = []
    class_box_counts = defaultdict(int)
    for position, image_id in enumerate(selected, start=1):
        image = images[image_id]
        image_name = str(image["file_name"])
        image_url = image.get("coco_url") or f"{COCO_IMAGE_BASE}/{split}2017/{image_name}"
        image_path = image_dir / image_name
        if not download_with_retries(image_url, image_path):
            failures.append(image_id)
            continue
        lines = yolo_lines(by_image[image_id], int(image["width"]), int(image["height"]))
        if not lines:
            image_path.unlink(missing_ok=True)
            continue
        (label_dir / f"{Path(image_name).stem}.txt").parent.mkdir(parents=True, exist_ok=True)
        (label_dir / f"{Path(image_name).stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        categories = sorted({CLASS_MAP[int(annotation["category_id"])] for annotation in by_image[image_id]})
        for line in lines:
            class_box_counts[int(line.split()[0])] += 1
        manifest.append({
            "coco_image_id": image_id,
            "file_name": image_name,
            "source_split": f"{split}2017",
            "url": image_url,
            "width": int(image["width"]),
            "height": int(image["height"]),
            "classes_present": categories,
            "boxes": len(lines),
        })
        if position % 100 == 0:
            print(f"{split}: materialized {position}/{len(selected)} selected images")
    report = {
        "split": split,
        "requested_images": limit,
        "selected_image_ids": len(selected),
        "downloaded_labeled_images": len(manifest),
        "download_failures": failures,
        "box_counts": {CLASS_NAMES[key]: value for key, value in sorted(class_box_counts.items())},
        "records": manifest,
    }
    write_json(output_root / "manifests" / f"{split}_manifest.json", report)
    return report


def write_dataset_yaml(dataset_root: Path) -> Path:
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
        "nc": 2,
    }
    target = dataset_root / "dataset.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def validate_dataset(dataset_root: Path, minimum_images: int = 100) -> dict[str, Any]:
    details: dict[str, Any] = {"checked_at": utc_now(), "splits": {}, "valid": True, "errors": []}
    for split in ("train", "val"):
        images = sorted((dataset_root / "images" / split).glob("*.jpg"))
        labels = dataset_root / "labels" / split
        missing = [path.name for path in images if not (labels / f"{path.stem}.txt").is_file()]
        invalid_lines: list[str] = []
        classes: set[int] = set()
        for image in images:
            for line_number, line in enumerate((labels / f"{image.stem}.txt").read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                if len(fields) != 5:
                    invalid_lines.append(f"{image.name}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError:
                    invalid_lines.append(f"{image.name}:{line_number}: nonnumeric label")
                    continue
                if class_id not in CLASS_NAMES or any(value < 0.0 or value > 1.0 for value in values):
                    invalid_lines.append(f"{image.name}:{line_number}: invalid class or normalized coordinate")
                classes.add(class_id)
        split_valid = len(images) >= minimum_images and not missing and not invalid_lines and classes == {0, 1}
        details["splits"][split] = {"images": len(images), "missing_labels": missing[:20], "invalid_lines": invalid_lines[:20], "classes": sorted(classes), "valid": split_valid}
        if not split_valid:
            details["valid"] = False
            details["errors"].append(f"{split} failed validation")
    write_json(dataset_root / "validation_report.json", details)
    if not details["valid"]:
        raise RuntimeError(f"Dataset preflight failed: {details['errors']}")
    return details


class CheckpointMirror:
    """Copies local run artifacts to Drive on a fixed interval without training on Drive."""

    def __init__(self, local_root: Path, drive_root: Path, interval_seconds: int = 120) -> None:
        self.local_root = local_root
        self.drive_root = drive_root
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sync_once(self) -> None:
        if not self.local_root.exists():
            return
        for source in self.local_root.rglob("*"):
            if not source.is_file():
                continue
            destination = self.drive_root / source.relative_to(self.local_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or source.stat().st_mtime > destination.stat().st_mtime:
                shutil.copy2(source, destination)

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.sync_once()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval_seconds + 5)
        self.sync_once()


def environment_snapshot(output_dir: Path) -> None:
    require_training_runtime()
    output_dir.mkdir(parents=True, exist_ok=True)
    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True).stdout
    nvidia_smi = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True).stdout
    write_json(output_dir / "environment.json", {
        "created_at": utc_now(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pip_freeze": pip_freeze.splitlines(),
        "nvidia_smi": nvidia_smi,
    })


def create_sanity_yaml(dataset_root: Path, output_dir: Path, count: int = 64) -> Path:
    image_paths = sorted((dataset_root / "images" / "train").glob("*.jpg"))[:count]
    if len(image_paths) < 8:
        raise RuntimeError("At least eight training images are required for the sanity gate.")
    list_path = output_dir / "sanity_train.txt"
    list_path.write_text("\n".join(str(path.resolve()) for path in image_paths) + "\n", encoding="utf-8")
    payload = {"path": str(dataset_root.resolve()), "train": str(list_path.resolve()), "val": "images/val", "names": CLASS_NAMES, "nc": 2}
    sanity_yaml = output_dir / "sanity_dataset.yaml"
    sanity_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return sanity_yaml


def scratch_model() -> tuple[YOLO, str]:
    require_training_runtime()
    model = YOLO("yolo11n.yaml")
    initial = parameter_hash(model)
    return model, initial


def train_kwargs(args: argparse.Namespace, data_yaml: Path, project: Path, name: str) -> dict[str, Any]:
    return {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "seed": args.seed,
        "deterministic": True,
        "amp": True,
        "save": True,
        "save_period": args.save_period,
        "patience": args.patience,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "pretrained": False,
        "plots": True,
    }


def command_prepare(args: argparse.Namespace) -> None:
    require_training_runtime(require_yolo=False)
    dataset_root = Path(args.dataset_root).resolve()
    annotation_dir = ensure_annotations(Path(args.annotation_cache).resolve())
    train = materialize_split(annotation_file=annotation_dir / "instances_train2017.json", split="train", output_root=dataset_root, limit=args.train_limit, phone_fraction=args.phone_fraction, seed=args.seed)
    val = materialize_split(annotation_file=annotation_dir / "instances_val2017.json", split="val", output_root=dataset_root, limit=args.val_limit, phone_fraction=args.phone_fraction, seed=args.seed + 1)
    data_yaml = write_dataset_yaml(dataset_root)
    report = validate_dataset(dataset_root, minimum_images=args.minimum_images)
    write_json(dataset_root / "preparation_summary.json", {"created_at": utc_now(), "data_yaml": str(data_yaml), "train": train, "val": val, "validation": report})
    print(f"Prepared validated dataset: {data_yaml}")


def command_sanity(args: argparse.Namespace) -> None:
    require_training_runtime()
    dataset_root = Path(args.dataset_root).resolve()
    validate_dataset(dataset_root, minimum_images=args.minimum_images)
    root = Path(args.local_root).resolve() / args.experiment_id / "sanity"
    root.mkdir(parents=True, exist_ok=True)
    sanity_yaml = create_sanity_yaml(dataset_root, root, count=args.sanity_images)
    model, initial_hash = scratch_model()
    proof = {"created_at": utc_now(), "model_source": "yolo11n.yaml", "pretrained": False, "initial_parameter_sha256": initial_hash, "purpose": "tiny-overfit preflight; this model is discarded before the full training run"}
    write_json(root / "initialization_proof.json", proof)
    sanity_args = argparse.Namespace(**vars(args))
    sanity_args.epochs = args.sanity_epochs
    sanity_args.imgsz = args.sanity_imgsz
    sanity_args.batch = args.sanity_batch
    results = model.train(**train_kwargs(sanity_args, sanity_yaml, root, "overfit"))
    write_json(root / "sanity_summary.json", {"created_at": utc_now(), "result": str(results), "initialization_proof": proof})
    print("Sanity gate completed. Review loss curves before invoking full training.")


def command_train(args: argparse.Namespace) -> None:
    require_training_runtime()
    dataset_root = Path(args.dataset_root).resolve()
    data_yaml = dataset_root / "dataset.yaml"
    validate_dataset(dataset_root, minimum_images=args.minimum_images)
    local_exp = Path(args.local_root).resolve() / args.experiment_id
    drive_exp = Path(args.drive_root).resolve() / "experiments" / args.experiment_id
    environment_snapshot(local_exp)
    (local_exp / "config_snapshot.yaml").write_text(yaml.safe_dump(vars(args), sort_keys=True), encoding="utf-8")
    mirror = CheckpointMirror(local_exp, drive_exp, interval_seconds=args.mirror_seconds)
    mirror.start()
    try:
        resume_candidate = local_exp / "train" / "weights" / "last.pt"
        if args.resume and not resume_candidate.exists():
            drive_candidate = drive_exp / "train" / "weights" / "last.pt"
            if drive_candidate.exists():
                resume_candidate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(drive_candidate, resume_candidate)
        if args.resume and resume_candidate.exists():
            print(f"Resuming from {resume_candidate}")
            model = YOLO(str(resume_candidate))
            model.train(resume=str(resume_candidate))
        else:
            model, initial_hash = scratch_model()
            proof = {"created_at": utc_now(), "model_source": "yolo11n.yaml", "pretrained": False, "initial_parameter_sha256": initial_hash, "seed": args.seed}
            write_json(local_exp / "initialization_proof.json", proof)
            model.train(**train_kwargs(args, data_yaml, local_exp, "train"))
    finally:
        mirror.stop()
    best = local_exp / "train" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training ended without best.pt at {best}")
    print(f"Training complete. Durable artifacts mirrored to {drive_exp}")


def command_verify(args: argparse.Namespace) -> None:
    require_training_runtime()
    dataset_root = Path(args.dataset_root).resolve()
    local_exp = Path(args.local_root).resolve() / args.experiment_id
    drive_exp = Path(args.drive_root).resolve() / "experiments" / args.experiment_id
    best = local_exp / "train" / "weights" / "best.pt"
    if not best.is_file():
        drive_best = drive_exp / "train" / "weights" / "best.pt"
        if not drive_best.is_file():
            raise RuntimeError("best.pt was not found locally or in the Drive backup.")
        best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_best, best)
    model = YOLO(str(best))
    metrics = model.val(data=str(dataset_root / "dataset.yaml"), imgsz=args.imgsz, device=args.device, workers=args.workers, plots=True)
    verification = {"verified_at": utc_now(), "checkpoint": str(best), "checkpoint_sha256": sha256_file(best), "checkpoint_bytes": best.stat().st_size, "metrics": str(metrics), "data_yaml": str(dataset_root / "dataset.yaml")}
    write_json(local_exp / "final_verification.json", verification)
    CheckpointMirror(local_exp, drive_exp, interval_seconds=1).sync_once()
    drive_root = Path(args.drive_root).resolve()
    drive_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, drive_root / "best.pt")
    print(json.dumps(verification, indent=2))


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", default="/content/coco2017_filtered")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/proctoring_cv")
    parser.add_argument("--local-root", default="/content/runs/experiments")
    parser.add_argument("--experiment-id", default="yolo11n_scratch_coco_v1_seed42")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--minimum-images", type=int, default=100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab YOLO11n scratch training on filtered COCO")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download and filter official COCO 2017 annotations/images")
    add_common_options(prepare)
    prepare.add_argument("--annotation-cache", default="/content/coco_annotations")
    prepare.add_argument("--train-limit", type=int, default=8000)
    prepare.add_argument("--val-limit", type=int, default=1200)
    prepare.add_argument("--phone-fraction", type=float, default=0.40)
    sanity = commands.add_parser("sanity", help="Run a small scratch-initialized overfit gate")
    add_common_options(sanity)
    sanity.add_argument("--sanity-images", type=int, default=64)
    sanity.add_argument("--sanity-epochs", type=int, default=5)
    sanity.add_argument("--sanity-imgsz", type=int, default=320)
    sanity.add_argument("--sanity-batch", type=int, default=8)
    train = commands.add_parser("train", help="Run full scratch training and mirror checkpoints to Drive")
    add_common_options(train)
    train.add_argument("--mirror-seconds", type=int, default=120)
    train.add_argument("--resume", action="store_true")
    verify = commands.add_parser("verify", help="Validate best.pt and write an integrity report")
    add_common_options(verify)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "sanity":
        command_sanity(args)
    elif args.command == "train":
        command_train(args)
    elif args.command == "verify":
        command_verify(args)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
