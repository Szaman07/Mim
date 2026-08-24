"""Configuration loader with schema validation, environment variable overrides, and SHA256 hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import yaml


class DatasetConfig(BaseModel):
    name: str = "coco_openimages_proctoring"
    root: str = "data/datasets"
    train_manifest: str = "data/manifests/train_manifest.json"
    val_manifest: str = "data/manifests/val_manifest.json"
    test_manifest: Optional[str] = "data/manifests/test_manifest.json"
    source_dataset_version: str = "v1"
    names: Dict[int, str] = Field(default_factory=lambda: {0: "person", 1: "cellphone"})
    nc: int = 2

    @field_validator("names")
    @classmethod
    def validate_classes(cls, v: Dict[int, str]) -> Dict[int, str]:
        if 0 not in v or 1 not in v:
            raise ValueError("Dataset names must map key 0 to 'person' and key 1 to 'cellphone'")
        if v[0] != "person" or v[1] != "cellphone":
            raise ValueError(f"Invalid class mapping: {v}. Must have 0: 'person', 1: 'cellphone'")
        return v


class ModelConfig(BaseModel):
    architecture: str = "yolo11n.yaml"
    initialization_method: str = "random_from_yaml"
    pretrained: bool = False
    model_type: str = "yolo11n"
    num_classes: int = 2

    @field_validator("pretrained")
    @classmethod
    def validate_pretrained(cls, v: bool, info: Any) -> bool:
        # Note: If initialization_method is random_from_yaml, pretrained MUST be False
        return v


class TrainingConfig(BaseModel):
    epochs: int = 50
    time_limit_hours: Optional[float] = 10.0
    imgsz: int = 640
    batch_size: int = 16
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    weight_decay: float = 0.0005
    seed: int = 42
    deterministic: bool = True
    amp: bool = True
    workers: int = 2
    checkpoint_interval: int = 5
    keep_periodic_checkpoints: int = 3
    patience: int = 15
    device: str = "0"
    augmentation: Dict[str, Any] = Field(default_factory=lambda: {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    })


class LookingAwayConfig(BaseModel):
    yaw_threshold_deg: float = 25.0
    pitch_threshold_deg: float = 20.0
    hysteresis_deg: float = 15.0
    calibration_frames: int = 30
    smoothing_ema_alpha: float = 0.3
    smoothing_median_window: int = 5
    min_valid_pose_ratio: float = 0.70


class EventEngineConfig(BaseModel):
    phone_confidence_threshold: float = 0.35
    person_confidence_threshold: float = 0.40
    phone_start_persistence_sec: float = 0.5
    phone_end_persistence_sec: float = 0.75
    multi_person_start_persistence_sec: float = 0.75
    multi_person_end_persistence_sec: float = 1.0
    looking_away_start_persistence_sec: float = 1.5
    looking_away_end_persistence_sec: float = 0.75
    gap_merge_tolerance_sec: float = 0.5
    event_cooldown_sec: float = 2.0


class RuntimeConfig(BaseModel):
    detector_checkpoint: Optional[str] = None
    webcam_index: int = 0
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    target_fps: float = 30.0
    show_overlay: bool = True
    save_raw_frames: bool = False
    log_file: str = "logs/events.jsonl"


class AppConfig(BaseModel):
    experiment_id: str = "yolo11n_scratch_coco_v1_seed42"
    drive_root: str = "drive_root"
    local_scratch_dir: str = "scratch"
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    looking_away: LookingAwayConfig = Field(default_factory=LookingAwayConfig)
    event_engine: EventEngineConfig = Field(default_factory=EventEngineConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def compute_sha256(self) -> str:
        """Computes deterministic SHA-256 hash of the configuration dictionary."""
        config_json = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def load_yaml(file_path: Path | str) -> Dict[str, Any]:
    """Loads a YAML file into a dictionary."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Applies environment variable overrides if present."""
    if "PROCTORING_DRIVE_ROOT" in os.environ:
        config_dict["drive_root"] = os.environ["PROCTORING_DRIVE_ROOT"]
    if "PROCTORING_EXPERIMENT_ID" in os.environ:
        config_dict["experiment_id"] = os.environ["PROCTORING_EXPERIMENT_ID"]
    if "PROCTORING_DATASET_ROOT" in os.environ:
        if "dataset" not in config_dict:
            config_dict["dataset"] = {}
        config_dict["dataset"]["root"] = os.environ["PROCTORING_DATASET_ROOT"]
    return config_dict


def load_config(
    yaml_path: Optional[Path | str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Loads, overrides, validates, and returns an AppConfig instance."""
    data: Dict[str, Any] = {}
    if yaml_path:
        data = load_yaml(yaml_path)

    data = apply_env_overrides(data)

    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                keys = k.split(".")
                curr = data
                for part in keys[:-1]:
                    if part not in curr or not isinstance(curr[part], dict):
                        curr[part] = {}
                    curr = curr[part]
                curr[keys[-1]] = v

    config = AppConfig.model_validate(data)
    return config


def save_config(config: AppConfig, output_path: Path | str) -> None:
    """Saves AppConfig to YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
