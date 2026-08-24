# RECOMMENDED_PROJECT_STRUCTURE.md

## Repository tree

```text
proctoring-cv/
├── configs/
│   ├── data.yaml
│   ├── runtime.yaml
│   └── experiments/
│       ├── yolo11n_scratch_coco.yaml
│       ├── yolov8n_scratch_coco.yaml
│       └── yolo26n_p2_scratch.yaml
├── src/proctoring_cv/
│   ├── __init__.py
│   ├── config.py
│   ├── environment.py
│   ├── reproducibility.py
│   ├── detector.py
│   ├── looking_away.py
│   ├── event_engine.py
│   ├── checkpoint_manager.py
│   ├── drive_sync.py
│   ├── logging_utils.py
│   └── schemas.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── split_dataset.py
│   ├── tiny_overfit.py
│   ├── train.py
│   ├── resume_training.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── infer_webcam.py
│   ├── run_event_replay.py
│   └── experiment_registry.py
├── dataset_tools/
│   ├── coco_to_yolo.py
│   ├── openimages_filter.py
│   ├── openimages_to_yolo.py
│   ├── deduplicate.py
│   └── manifest.py
├── evaluation/
│   ├── metrics.py
│   ├── event_metrics.py
│   ├── error_slices.py
│   └── scenario_labels.md
├── notebooks/
│   ├── 00_environment_smoke_test.ipynb
│   ├── 01_dataset_sanity.ipynb
│   ├── 02_train_scratch_colab.ipynb
│   └── 03_evaluate_and_replay.ipynb
├── tests/
│   ├── test_config.py
│   ├── test_dataset_validation.py
│   ├── test_conversion.py
│   ├── test_random_initialization.py
│   ├── test_checkpoint_manager.py
│   ├── test_event_engine.py
│   └── test_resume_metadata.py
├── docs/
│   ├── DATASET_SELECTION.md
│   ├── SYSTEM_ARCHITECTURE.md
│   └── COLAB_TRAINING_SURVIVAL_GUIDE.md
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Module responsibilities

| Area | Responsibility |
|---|---|
| `configs/` | Version-controlled, human-readable model/data/runtime/experiment settings. No secrets or absolute user-specific paths. |
| `dataset_tools/` | Acquisition manifests, source-license records, conversion, deduplication, and split preparation. |
| `src/.../detector.py` | Primary two-class model construction, prediction normalization, and explicit scratch/pretrained mode separation. |
| `src/.../looking_away.py` | Face landmarks, pose geometry, calibration, smoothing, and `UNAVAILABLE` diagnostics. |
| `src/.../event_engine.py` | Timestamp-based state machines, persistence, hysteresis, cooldowns, and structured event records. |
| `checkpoint_manager.py` | Atomic writes, integrity checks, rotation, `best.pt`/`last.pt`, and resume metadata. |
| `drive_sync.py` | Configurable Drive paths and safe artifact synchronization; never assumes `/content` persists. |
| `evaluation/` | Detector metrics, event metrics, latency, error slices, and locked-test controls. |
| `tests/` | Unit tests and smoke tests that must pass before full training. |

## Google Drive tree

```text
MyDrive/
└── proctoring_cv/
    ├── datasets/
    │   ├── coco2017_filtered_v1/
    │   ├── openimages_filtered_v1/
    │   └── webcam_test_restricted_v1/
    ├── experiments/
    │   └── yolo11n_scratch_coco_v1_seed42/
    │       ├── config.yaml
    │       ├── manifest.json
    │       ├── environment.json
    │       ├── checkpoints/
    │       │   ├── best.pt
    │       │   ├── last.pt
    │       │   ├── epoch_005.pt
    │       │   └── checksums.sha256
    │       ├── logs/
    │       ├── metrics/
    │       ├── plots/
    │       └── notes.md
    ├── registry/
    │   ├── experiments.csv
    │   └── experiments.jsonl
    ├── results/
    ├── exports/
    ├── backups/
    └── environment_snapshots/
```

## Git versus Drive

GitHub stores source code, configuration, tests, documentation, small manifests without sensitive data, and license/attribution files. Drive stores datasets, checkpoints, training logs, plots, evaluation outputs, exports, and restricted webcam data. Large artifacts are never committed blindly. `.gitignore` excludes dataset images, `.pt` files, runtime directories, caches, logs, and local secrets.

Every experiment records the Git commit and dataset manifest hash, so a Drive artifact remains traceable to version-controlled code without making GitHub the artifact store.
