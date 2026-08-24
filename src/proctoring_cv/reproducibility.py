"""Reproducibility utilities: seeding, git metadata, and model parameter hashing."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Sets random seeds across standard library, numpy, and PyTorch for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_git_metadata(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Extracts git commit SHA, branch, and dirty status without throwing exceptions."""
    cwd = Path(repo_root) if repo_root else Path.cwd()
    metadata: Dict[str, Any] = {
        "commit": "unknown",
        "branch": "unknown",
        "is_dirty": False,
        "git_available": False,
    }
    try:
        # Check if git is available and repo exists
        rev_parse = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if rev_parse.returncode != 0:
            return metadata

        metadata["git_available"] = True
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            metadata["commit"] = commit.stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if branch.returncode == 0:
            metadata["branch"] = branch.stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0:
            metadata["is_dirty"] = len(status.stdout.strip()) > 0
    except Exception:
        pass

    return metadata


def compute_file_sha256(file_path: Path | str) -> str:
    """Computes SHA-256 hash of a given file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for SHA256 computation: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_model_parameter_hash(model: torch.nn.Module, sample_k: int = 50) -> str:
    """Computes a deterministic hash over model parameter values to verify random initialization.
    
    Samples deterministic parameter slices to produce a compact fingerprint.
    """
    hasher = hashlib.sha256()
    count = 0
    for name, param in sorted(model.named_parameters()):
        if param.requires_grad:
            data = param.detach().cpu().numpy().flatten()
            if len(data) > 0:
                # Take deterministic sample
                step = max(1, len(data) // sample_k)
                sample = data[::step][:sample_k]
                hasher.update(sample.tobytes())
                count += 1
    hasher.update(str(count).encode("utf-8"))
    return hasher.hexdigest()
