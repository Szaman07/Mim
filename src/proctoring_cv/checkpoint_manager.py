"""Checkpoint manager: atomic saves, integrity checksums, rotation, and metadata tracking."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from proctoring_cv.reproducibility import compute_file_sha256


class CheckpointManager:
    """Manages atomic checkpoint saving, rotation, checksum verification, and metadata."""

    def __init__(
        self,
        output_dir: Path | str,
        experiment_id: str,
        keep_periodic: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.keep_periodic = max(2, keep_periodic)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checksums_file = self.output_dir / "checksums.sha256"
        self._periodic_checkpoints: List[Path] = []

    def save_atomic_checkpoint(
        name: str,
        checkpoint_obj: Any,
        target_path: Path,
    ) -> Path:
        """Atomically saves checkpoint object to a temporary file, then renames."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = target_path.parent / ".tmp_checkpoints"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_fd, temp_file_path = tempfile.mkstemp(prefix="chkpt_", suffix=".pt", dir=str(temp_dir))
        os.close(temp_fd)

        try:
            if isinstance(checkpoint_obj, (str, Path)):
                # If path provided, copy atomically
                shutil.copy2(checkpoint_obj, temp_file_path)
            else:
                torch.save(checkpoint_obj, temp_file_path)

            # Atomic replace
            shutil.move(temp_file_path, target_path)
            return target_path
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

    def record_checkpoint(
        self,
        checkpoint_source: Path | str | Any,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        is_periodic: bool = False,
    ) -> Path:
        """Saves a checkpoint atomically, writes metadata, updates checksums, and manages rotation."""
        dest_file = self.output_dir / filename
        CheckpointManager.save_atomic_checkpoint(filename, checkpoint_source, dest_file)

        # Compute and record checksum
        sha = compute_file_sha256(dest_file)
        self._append_checksum(filename, sha)

        # Save metadata companion file
        if metadata:
            meta_path = self.output_dir / f"{filename}.json"
            meta_with_sha = dict(metadata)
            meta_with_sha["sha256"] = sha
            meta_with_sha["filename"] = filename
            meta_with_sha["experiment_id"] = self.experiment_id
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_with_sha, f, indent=2)

        # Handle periodic rotation
        if is_periodic:
            self._periodic_checkpoints.append(dest_file)
            if len(self._periodic_checkpoints) > self.keep_periodic:
                oldest = self._periodic_checkpoints.pop(0)
                if oldest.exists() and oldest.name not in ("best.pt", "last.pt"):
                    try:
                        oldest.unlink()
                        old_meta = oldest.with_suffix(".pt.json")
                        if old_meta.exists():
                            old_meta.unlink()
                    except OSError:
                        pass

        return dest_file

    def verify_checkpoint_integrity(self, filename: str) -> bool:
        """Verifies checkpoint file against recorded SHA-256 in checksums.sha256."""
        chkpt_path = self.output_dir / filename
        if not chkpt_path.is_file() or not self.checksums_file.is_file():
            return False

        current_sha = compute_file_sha256(chkpt_path)
        with open(self.checksums_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1] == filename:
                    return parts[0].lower() == current_sha.lower()
        return False

    def _append_checksum(self, filename: str, sha256_hash: str) -> None:
        """Appends or updates checksums.sha256."""
        lines: List[str] = []
        if self.checksums_file.is_file():
            with open(self.checksums_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1] != filename:
                        lines.append(line.strip())
        lines.append(f"{sha256_hash}  {filename}")
        with open(self.checksums_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
