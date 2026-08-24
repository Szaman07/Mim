"""Google Drive synchronization manager for durable experiment artifacts."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from proctoring_cv.logging_utils import setup_logger
from proctoring_cv.reproducibility import compute_file_sha256


class DriveSyncManager:
    """Synchronizes experiment outputs, checkpoints, and logs to a durable Drive hierarchy."""

    def __init__(self, drive_root: Optional[Path | str] = None) -> None:
        self.logger = setup_logger("drive_sync")
        self.drive_root = Path(drive_root) if drive_root else Path("drive_root")
        self._init_drive_tree()

    def _init_drive_tree(self) -> None:
        """Ensures standard Drive tree subfolders exist."""
        subdirs = [
            "datasets",
            "experiments",
            "registry",
            "results",
            "exports",
            "backups",
            "environment_snapshots",
        ]
        for sub in subdirs:
            (self.drive_root / sub).mkdir(parents=True, exist_ok=True)

    def sync_experiment_directory(
        self,
        local_exp_dir: Path | str,
        experiment_id: str,
    ) -> Path:
        """Copies durable artifacts from local runtime scratch to Google Drive experiment folder."""
        local_dir = Path(local_exp_dir)
        target_dir = self.drive_root / "experiments" / experiment_id
        target_dir.mkdir(parents=True, exist_ok=True)

        if not local_dir.is_dir():
            self.logger.warning(f"Local experiment directory not found for sync: {local_dir}")
            return target_dir

        for item in local_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(local_dir)
                dest_file = target_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                # Copy file preserving timestamp
                shutil.copy2(item, dest_file)

        self.logger.info(f"[Drive Sync] Synchronized experiment '{experiment_id}' to: {target_dir.resolve()}")
        return target_dir

    def register_experiment(
        self,
        experiment_record: Dict[str, Any],
    ) -> None:
        """Appends experiment summary to Drive registry (CSV and JSONL)."""
        registry_dir = self.drive_root / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = registry_dir / "experiments.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_record) + "\n")

        csv_path = registry_dir / "experiments.csv"
        file_exists = csv_path.is_file()

        headers = [
            "experiment_id",
            "timestamp",
            "model_architecture",
            "initialization_method",
            "seed",
            "epochs",
            "mAP50",
            "mAP50_95",
            "person_precision",
            "cellphone_precision",
            "status",
        ]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(experiment_record)
