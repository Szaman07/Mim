# Proctoring CV: Zero-Cost Reproducible Webcam Proctoring Research System

A reproducible, zero-cost computer vision proctoring research system built in Python and PyTorch.

---

## 1. Purpose and Architectural Boundary

The system detects and records **three observable visual events**:

1. `PHONE_DETECTED`: A mobile phone is visible in the webcam frame.
2. `MULTIPLE_PERSONS`: More than one person is visible in the webcam frame.
3. `LOOKING_AWAY`: The primary examinee's calibrated head orientation is turned away from the screen for a sustained interval.

> **Ethical & Technical Guardrail:**
> The system **never** outputs `CHEATING`, `CHEATING_TRUE`, `SUSPICIOUS_PERSON`, or claims about intent, identity, emotion, audio, or mental state. It outputs objective, uncertainty-aware temporal observations with diagnostics (`POSE_UNAVAILABLE`, `LOW_CONFIDENCE`, `CALIBRATION_PENDING`).

---

## 2. Non-Negotiable Primary Detector Rule

- The primary detector is Ultralytics **YOLO11n**, trained **strictly from random initialization** (`pretrained=False`) using YAML architecture configurations (`yolo11n.yaml`).
- Loading `.pt` weights in scratch mode is strictly prohibited and guarded by automated tests.
- Pretrained weights are permitted only for explicitly labeled transfer learning comparisons (`pretrained_comparison`) or the separate auxiliary MediaPipe/OpenCV head pose component.

---

## 3. Repository Architecture

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
│   ├── test_looking_away.py
│   ├── test_forbidden_outputs.py
│   └── test_resume_metadata.py
├── docs/
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

---

## 4. Quickstart & Installation

```bash
# Clone and install in editable mode with dependencies
git clone https://github.com/Szaman07/Mim.git
cd Mim
pip install -e .
```

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## 5. CLI Reference

### Dataset Tools & Validation
```bash
# Validate dataset integrity & check cross-split leakage
python scripts/validate_dataset.py --config configs/data.yaml --dry-run

# Prepare manifests and dataset structure
python scripts/prepare_dataset.py --config configs/data.yaml
```

### Preflight Gate & Training
```bash
# Run tiny overfit preflight gate
python scripts/tiny_overfit.py --config configs/experiments/yolo11n_scratch_coco.yaml

# Train scratch detector with Google Drive sync
python scripts/train.py \
  --config configs/experiments/yolo11n_scratch_coco.yaml \
  --experiment-id yolo11n_scratch_coco_v1_seed42 \
  --drive-root /content/drive/MyDrive/proctoring_cv \
  --mode full
```

### Resume / Branching
```bash
# Exact resume after timeout
python scripts/resume_training.py \
  --experiment-dir /content/drive/MyDrive/proctoring_cv/experiments/yolo11n_scratch_coco_v1_seed42 \
  --checkpoint last.pt \
  --resume

# Branching to a new experiment from best.pt
python scripts/resume_training.py \
  --experiment-dir /content/drive/MyDrive/proctoring_cv/experiments/yolo11n_scratch_coco_v1_seed42 \
  --checkpoint best.pt \
  --branch \
  --new-experiment-id yolo11n_coco_branch_v2
```

### Evaluation & Replay
```bash
# Evaluate checkpoint on validation or locked test set
python scripts/evaluate.py \
  --checkpoint runs/experiments/yolo11n_scratch_coco_v1_seed42/checkpoints/best.pt \
  --split val

# Replay deterministic event streams offline
python scripts/run_event_replay.py --config configs/runtime.yaml --scenario sustained_phone

# Realtime webcam inference with privacy HUD
python scripts/infer_webcam.py --config configs/runtime.yaml
```

---

## 6. Google Drive Persistence Hierarchy

```text
MyDrive/
└── proctoring_cv/
    ├── datasets/
    ├── experiments/<experiment_id>/
    │   ├── config.yaml
    │   ├── manifest.json
    │   ├── environment.json
    │   ├── checkpoints/ (best.pt, last.pt, checksums.sha256)
    │   ├── logs/
    │   ├── metrics/
    │   └── plots/
    ├── registry/ (experiments.csv, experiments.jsonl)
    ├── results/
    ├── exports/
    └── backups/
```

---

## 7. License & Attribution

- Source code licensed under the **MIT License**.
- COCO dataset annotations licensed under CC BY 4.0.
- Open Images V7 annotations licensed under CC BY 4.0, images under CC BY 2.0.
