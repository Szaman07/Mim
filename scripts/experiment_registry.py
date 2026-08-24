"""Experiment registry inspector and table reporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import yaml


def scan_experiments(root_dir: Path | str) -> List[Dict[str, Any]]:
    """Scans and extracts status of all experiment directories under root."""
    root = Path(root_dir)
    exp_parent = root / "experiments" if (root / "experiments").is_dir() else root
    experiments: List[Dict[str, Any]] = []

    if not exp_parent.is_dir():
        return experiments

    for exp_dir in sorted(exp_parent.iterdir()):
        if not exp_dir.is_dir():
            continue

        cfg_file = exp_dir / "config.yaml"
        init_file = exp_dir / "initialization_proof.json"
        best_pt = exp_dir / "checkpoints" / "best.pt"
        last_pt = exp_dir / "checkpoints" / "last.pt"
        if not best_pt.exists() and (exp_dir / "best.pt").exists():
            best_pt = exp_dir / "best.pt"
        if not last_pt.exists() and (exp_dir / "last.pt").exists():
            last_pt = exp_dir / "last.pt"

        exp_data: Dict[str, Any] = {
            "experiment_id": exp_dir.name,
            "path": str(exp_dir),
            "has_config": cfg_file.is_file(),
            "has_init_proof": init_file.is_file(),
            "has_best_pt": best_pt.is_file(),
            "has_last_pt": last_pt.is_file(),
            "architecture": "unknown",
            "seed": "unknown",
            "epochs": "unknown",
        }

        if cfg_file.is_file():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    exp_data["architecture"] = cfg.get("model", {}).get("architecture", "unknown")
                    exp_data["seed"] = cfg.get("training", {}).get("seed", "unknown")
                    exp_data["epochs"] = cfg.get("training", {}).get("epochs", "unknown")
            except Exception:
                pass

        experiments.append(exp_data)

    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Proctoring CV Experiment Registry")
    parser.add_argument("--root", type=str, default="drive_root", help="Root directory or Drive root")
    parser.add_argument("--sort", type=str, default="latest", choices=["latest", "name"])
    parser.add_argument("--format", type=str, default="table", choices=["table", "json"])
    args = parser.parse_args()

    exps = scan_experiments(args.root)
    if args.sort == "latest":
        exps.reverse()

    if args.format == "json":
        print(json.dumps(exps, indent=2))
    else:
        print(f"{'EXPERIMENT ID':<35} {'ARCH':<16} {'SEED':<6} {'BEST.PT':<8} {'LAST.PT':<8} {'CONFIG':<8}")
        print("-" * 85)
        for e in exps:
            best_str = "YES" if e["has_best_pt"] else "NO"
            last_str = "YES" if e["has_last_pt"] else "NO"
            cfg_str = "YES" if e["has_config"] else "NO"
            print(f"{e['experiment_id']:<35} {str(e['architecture']):<16} {str(e['seed']):<6} {best_str:<8} {last_str:<8} {cfg_str:<8}")


if __name__ == "__main__":
    main()
