# CODEX_IMPLEMENTATION_PROMPT.md

Copy the prompt below into Codex after creating or opening the repository.

---

## Codex task

Act as a senior Python computer-vision engineer. Implement a reproducible, zero-cost webcam proctoring research system in this repository. Do not make major architectural decisions that are already specified below. Implement incrementally in the ten phases and stop at the end of each phase to run its tests and report files changed, commands run, and remaining risks.

The system must report only three observable events:

1. `PHONE_DETECTED`: a mobile phone is visible in the webcam frame.
2. `MULTIPLE_PERSONS`: more than one person is visible in the webcam frame.
3. `LOOKING_AWAY`: the primary examinee’s calibrated head orientation is away from the screen for a sustained interval.

Never output `CHEATING`, `CHEATING_TRUE`, `SUSPICIOUS_PERSON`, intent, identity, emotion, speech, audio, mouth movement, or a gaze-based claim about mental state. Use `UNAVAILABLE` and `LOW_CONFIDENCE` diagnostics when the input is insufficient.

## Non-negotiable primary-detector rule

The primary detector must be the selected nano YOLO architecture, defaulting to Ultralytics YOLO11n, and the main experiment must be trained from random initialization. Construct the model from an architecture/configuration YAML and set `pretrained=False`. Do not load a `.pt` checkpoint in scratch mode. Do not silently download pretrained weights. Add an automated test that fails if scratch mode receives a `.pt` model path, if a download is attempted, or if initialization metadata does not say `random_from_yaml`.

Keep YOLOv8n from YAML as an established comparison configuration and YOLO26n/P2 as an optional controlled small-object configuration. Any experiment using pretrained weights must be explicitly labeled `pretrained_comparison` and must not overwrite or be reported as the primary scratch experiment.

Use MediaPipe Face Landmarker or another explicitly configured open-source auxiliary component for landmarks/head pose only. It is a separate pretrained auxiliary component and is not the primary detector. Do not train a temporal model unless a later experiment explicitly requests it.

## Repository to implement

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
├── requirements.txt
├── requirements-lock.txt
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Configuration requirements

Use YAML configuration with environment-variable and CLI overrides. No absolute paths or secrets may be hard-coded. Required settings include dataset root, manifest paths, Drive root, experiment ID, model YAML, initialization mode, class names, image size, batch mode, epochs/time limit, optimizer, learning rate, scheduler, augmentation, seed, deterministic mode, AMP, workers, checkpoint interval, checkpoint rotation, event thresholds, and logging paths.

The default classes are:

```yaml
names:
  0: person
  1: cellphone
```

The configuration must preserve `initialization_method: random_from_yaml`, `pretrained: false`, `source_dataset_version`, `manifest_sha256`, `git_commit`, and an environment snapshot.

## Dataset tooling

Implement the following behavior.

`dataset_tools/coco_to_yolo.py` must read COCO JSON, map `person` to project class 0 and `cell phone` to project class 1, validate image dimensions, convert boxes to normalized YOLO format, preserve source IDs, and write a conversion report.

`dataset_tools/openimages_filter.py` must read official Open Images metadata and box CSV files, filter exact `Person` and `Mobile phone` labels, create a capped download manifest, preserve image URLs and source terms, and never attempt the full approximately 561 GB dataset by default.

`dataset_tools/openimages_to_yolo.py` must convert filtered Open Images boxes to project classes 0/1 and preserve source annotation provenance.

`dataset_tools/deduplicate.py` must detect exact duplicates by source ID and near duplicates with a documented perceptual hash policy. It must prevent a cross-split duplicate.

`dataset_tools/manifest.py` must create immutable train/validation/test manifests with source, split, image path, dimensions, labels, checksums where feasible, license metadata, and a manifest hash.

## Dataset validation

`validate_dataset.py` must fail loudly for missing files, unreadable/corrupt images, missing required label files, malformed rows, invalid IDs, non-finite values, negative/zero boxes, boxes beyond permitted bounds, inconsistent stems, cross-split duplicates, and unknown classes. It must write JSON and CSV reports covering image counts, object counts, per-class counts, co-occurrence, box area/aspect ratio/size bins, split statistics, class imbalance, and dropped items.

Implement `--dry-run` and `--sample N` modes. Do not silently repair major annotation errors. Every exclusion must be logged.

## Detector module

`detector.py` must expose:

```python
build_detector(config, mode="scratch")
predict(frame, timestamp)
normalize_detections(raw_results)
verify_random_initialization(model, seed, known_checkpoint_paths=[])
```

Scratch mode must require YAML/configuration, disable pretrained loading, prevent automatic downloads where supported, and record parameter fingerprints before and after training. Prediction output must include class ID/name, confidence, pixel box, timestamp, and model ID. The module must support native checkpoint inference and documented export smoke tests.

## Training

`train.py` must:

1. Load and validate configuration.
2. Detect Python, PyTorch, CUDA, GPU name, VRAM, Ultralytics, OpenCV, MediaPipe, and OS versions.
3. Set seeds and deterministic flags.
4. Validate the dataset before training.
5. Run a short preflight batch.
6. Construct the detector from YAML with `pretrained=False` in scratch mode.
7. Save the complete configuration, environment, Git SHA, dataset manifest hash, initialization proof, and command line.
8. Use AMP where supported and choose/recommend a safe batch size from actual device capabilities.
9. Save `last.pt`, `best.pt`, periodic checkpoints, logs, metrics, plots, and checksums to the configured persistent directory.
10. Continue logging after interruption only in an exact resume workflow.

`tiny_overfit.py` must train a tiny fixed dataset and demonstrate loss reduction/overfit behavior. It must be a gate before full training.

## Checkpoint and resume infrastructure

`checkpoint_manager.py` must use temporary files plus atomic rename or verified copy. It must preserve model, optimizer, scheduler, scaler, epoch, best metric, configuration, Git SHA, manifest hash, environment, and checksum. Retain `best.pt`, `last.pt`, at least two periodic checkpoints, and a backup copy where storage permits.

`resume_training.py` must use `resume=True` only when the same experiment configuration, model architecture, manifest hash, and optimizer/scheduler plan are compatible. It must restore optimizer/scheduler/epoch state where supported. It must refuse to resume after a material configuration change and explain how to start a new branch from `best.pt` instead. A branch must receive a new experiment ID and must not overwrite the parent.

## Google Drive integration

`drive_sync.py` must accept a configurable Drive root such as `/content/drive/MyDrive/proctoring_cv`. Use the following layout:

```text
proctoring_cv/
├── datasets/
├── experiments/<experiment_id>/
│   ├── config.yaml
│   ├── manifest.json
│   ├── environment.json
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   ├── plots/
│   └── notes.md
├── registry/
├── results/
├── exports/
└── backups/
```

Never assume `/content` or `/kaggle/working` persists. Stage datasets locally when practical for I/O, but synchronize durable artifacts to Drive after checkpoint/metric writes. Print the exact destination path after every durable save.

## Looking-away module

`looking_away.py` must:

1. Accept a primary-person crop or frame and timestamps.
2. Use the configured face-landmark component in video/live-stream mode.
3. Return landmarks, pose validity, yaw/pitch/roll, confidence, calibration state, and diagnostics.
4. Support short frontal neutral-pose calibration.
5. Use documented canonical 3D points and 2D landmark correspondences with `solvePnP` or a documented transformation-matrix method.
6. Apply median/EMA smoothing.
7. Compute calibrated yaw/pitch deviations.
8. Return `POSE_UNAVAILABLE` rather than treating missing face evidence as looking away.

Initial validation settings are yaw 25°, pitch 20°, end hysteresis 15°, start persistence 1.5 s, end persistence 0.75 s, and valid-pose coverage 70%. Expose them in configuration and never claim they are universal.

## Event engine

`event_engine.py` must implement timestamp-based finite-state logic for `PHONE_DETECTED`, `MULTIPLE_PERSONS`, and `LOOKING_AWAY`. It must support candidate/active/ending states, persistence, hysteresis, cooldown, short-gap merging, and missing-input diagnostics. Use elapsed seconds as the source of truth; derive frame guards from timestamps.

Suggested starting rules are phone present for 3/5 frames or 0.5 s, multiple persons for 0.75 s or 5/7 frames, and looking away for 1.5 s with 70% valid pose. Tune only on validation data.

Event records must include event ID, event type, state, timestamp, monotonic time, duration, confidence summary, evidence counts, model ID, config hash, and diagnostics. Do not store raw frames by default.

## Evaluation

`evaluate.py` must report per-class detector precision, recall, F1, mAP50, mAP50-95, per-image errors, confusion analysis where applicable, latency, and FPS. It must report `person` and `cellphone` separately.

`evaluation/event_metrics.py` must report frame-level and event-level precision/recall/F1, false-positive and false-negative rates, event delay, interval overlap, fragmentation, duration bias, and temporal stability for looking-away and event signals.

`error_slices.py` must support small/medium/large objects, occlusion, low-resolution, indoor/webcam-like, phone-on-desk, phone-in-hand, two-person, and lighting/pose slices.

The proctoring-specific held-out set must include normal sitting, slight left/right, brief looking down, sustained looking away, visible/occluded/far phones, two people, brief frame entry, face unavailable, and phone-like distractors. The code must refuse to use its labels during training or tuning.

## Inference

`infer_webcam.py` must load a selected checkpoint and configuration, read webcam frames, timestamp them, run detector and looking-away components, feed the event engine, display optional privacy-safe overlays, and write structured event logs. It must not call a cloud API, save raw frames by default, or produce a cheating label.

`run_event_replay.py` must replay saved detection/pose streams through the event engine for deterministic unit and regression tests without requiring a webcam.

## Tests

Implement tests for configuration validation, class mapping, malformed labels, corrupt images, duplicate splits, scratch initialization, no-pretrained-download behavior, checkpoint atomicity, checksum failure, resume metadata compatibility, event persistence, hysteresis, cooldown, missing-face handling, FPS-independent timing, and prohibition of cheating/identity/audio outputs.

## CLI examples

```bash
python scripts/validate_dataset.py --config configs/data.yaml --dry-run
python scripts/tiny_overfit.py --config configs/experiments/yolo11n_scratch_coco.yaml
python scripts/train.py --config configs/experiments/yolo11n_scratch_coco.yaml --mode full --drive-root /content/drive/MyDrive/proctoring_cv
python scripts/resume_training.py --experiment-dir /content/drive/MyDrive/proctoring_cv/experiments/EXP_ID --checkpoint last.pt --resume
python scripts/evaluate.py --config configs/experiments/yolo11n_scratch_coco.yaml --checkpoint best.pt --split test
python scripts/infer_webcam.py --config configs/runtime.yaml --checkpoint /path/to/best.pt
pytest -q
```

## Ten-phase implementation plan

### Phase 1 — Repository and environment

Create the package structure, pinned requirements, configuration loader, environment detector, logging, Git metadata capture, `.gitignore`, README skeleton, and smoke tests. Do not train.

### Phase 2 — Dataset tooling

Implement COCO/Open Images conversion, filtered manifests, checksums, deduplication, class mapping, validation reports, and dry-run commands. Add tests using tiny fixtures.

### Phase 3 — Training pipeline

Implement model construction from YAML, scratch initialization proof, dataset preflight, tiny overfit, full training configuration, AMP/device detection, and metrics. Do not allow a pretrained `.pt` in scratch mode.

### Phase 4 — Checkpoint/resume infrastructure

Implement Drive paths, atomic checkpoints, rotation, checksums, configuration/environment snapshots, registry writes, resume compatibility checks, and recovery tests.

### Phase 5 — Evaluation

Implement detector metrics, per-class reports, latency/FPS, error slices, locked-test protection, and result manifests.

### Phase 6 — Looking-away module

Implement face landmarks, calibrated head pose, smoothing, diagnostics, and unit/replay tests. Do not train an additional model.

### Phase 7 — Event engine

Implement observable event state machines, persistence, hysteresis, cooldowns, timestamp handling, structured logs, and replay tests.

### Phase 8 — End-to-end inference

Integrate webcam capture, detector, primary-person selection, pose module, event engine, privacy-safe overlay, and structured event logging. Test failure states.

### Phase 9 — Tests

Run the full test suite, tiny overfit, dry-run, checkpoint corruption tests, and replay tests. Fix failures before documentation.

### Phase 10 — Documentation

Complete README, Colab notebook, Kaggle notes, CLI reference, recovery instructions, dataset attribution, limitations, and experiment examples. Ensure documentation never claims cheating detection.

## Definition of done

The implementation is complete only when all phase tests pass, scratch initialization is proven, no hidden pretrained checkpoint is loaded, paths are configurable, Drive persistence and exact resume work, the locked test set is protected, event outputs are observable and uncertainty-aware, and the README explains the ethical and technical limitations. Do not implement a giant one-shot change; complete and verify each phase in order.

---
