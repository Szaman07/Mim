"""Dataset deduplication: exact SHA256 matching, perceptual hashing (dHash), and cross-split leakage prevention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from PIL import Image


def compute_dhash(image_path: Path | str, hash_size: int = 8) -> str:
    """Computes difference hash (dHash) for perceptual near-duplicate detection."""
    path = Path(image_path)
    with Image.open(path) as img:
        # Resize to (hash_size + 1, hash_size) in grayscale
        resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(resized)
        # Compare adjacent pixels
        diff = pixels[:, 1:] > pixels[:, :-1]
        # Flatten boolean array to hex string
        return hex(int("".join(["1" if v else "0" for v in diff.flatten()]), 2))[2:].zfill(hash_size * hash_size // 4)


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Computes Hamming distance between two hex hash strings."""
    val1 = int(hex_hash1, 16)
    val2 = int(hex_hash2, 16)
    xor = val1 ^ val2
    return bin(xor).count("1")


def find_duplicates(
    image_paths: List[Path | str],
    hamming_threshold: int = 3,
) -> Dict[str, Any]:
    """Finds exact and near duplicates among a list of image paths."""
    exact_hashes: Dict[str, str] = {}  # sha256 -> image_path
    dhashes: Dict[str, str] = {}       # image_path -> dhash
    duplicates: List[Dict[str, Any]] = []

    for p in image_paths:
        path = Path(p)
        if not path.is_file():
            continue

        # Exact SHA256
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        if file_hash in exact_hashes:
            duplicates.append({
                "type": "exact_sha256",
                "image1": exact_hashes[file_hash],
                "image2": str(path),
                "distance": 0,
            })
            continue
        exact_hashes[file_hash] = str(path)

        # Perceptual dHash
        try:
            dh = compute_dhash(path)
            # Check distance against previous
            for prev_path, prev_dh in dhashes.items():
                dist = hamming_distance(dh, prev_dh)
                if dist <= hamming_threshold:
                    duplicates.append({
                        "type": "near_perceptual",
                        "image1": prev_path,
                        "image2": str(path),
                        "distance": dist,
                    })
            dhashes[str(path)] = dh
        except Exception:
            pass

    return {
        "total_checked": len(image_paths),
        "total_duplicates_found": len(duplicates),
        "duplicates": duplicates,
    }


def check_cross_split_leakage(
    splits: Dict[str, List[Path | str]],
    hamming_threshold: int = 3,
) -> Dict[str, Any]:
    """Checks for exact and near duplicate image leakage across train, val, and test splits."""
    split_dhashes: Dict[str, Dict[str, str]] = {}
    split_sha256s: Dict[str, Dict[str, str]] = {}

    for split_name, paths in splits.items():
        split_dhashes[split_name] = {}
        split_sha256s[split_name] = {}
        for p in paths:
            path = Path(p)
            if not path.is_file():
                continue
            with open(path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            split_sha256s[split_name][str(path)] = sha
            try:
                split_dhashes[split_name][str(path)] = compute_dhash(path)
            except Exception:
                pass

    leakages: List[Dict[str, Any]] = []
    split_names = list(splits.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]

            # Cross-check SHA256
            s1_sha_inv = {v: k for k, v in split_sha256s[s1].items()}
            for path2, sha2 in split_sha256s[s2].items():
                if sha2 in s1_sha_inv:
                    leakages.append({
                        "split_1": s1,
                        "image_1": s1_sha_inv[sha2],
                        "split_2": s2,
                        "image_2": path2,
                        "type": "exact_sha256",
                        "distance": 0,
                    })

            # Cross-check perceptual dHash
            for path1, dh1 in split_dhashes[s1].items():
                for path2, dh2 in split_dhashes[s2].items():
                    dist = hamming_distance(dh1, dh2)
                    if dist <= hamming_threshold:
                        leakages.append({
                            "split_1": s1,
                            "image_1": path1,
                            "split_2": s2,
                            "image_2": path2,
                            "type": "near_perceptual",
                            "distance": dist,
                        })

    return {
        "cross_split_leakages_count": len(leakages),
        "leakages": leakages,
        "is_leak_free": len(leakages) == 0,
    }
