# EXPERIMENT_PLAN.md

## Principles

Every experiment must be reproducible, explicitly named, and separated from the locked test set. The primary scientific requirement is that the main detector starts from random initialization. Any run using pretrained weights is a separate transfer-learning comparison and must never be mislabeled as scratch training.

## Experiment registry

Record one row per run in CSV/JSON/YAML with experiment ID, UTC date, Git commit, dataset version and manifest hash, model architecture/config, initialization method, seed, image size, batch size, epochs or time budget, optimizer, learning rate, scheduler, augmentation, GPU/VRAM, software environment, duration, best epoch, mAP50, mAP50-95, precision, recall, per-class metrics, failure notes, and reviewer comments.

## Ordered experiments

| ID | Experiment | Change | Required outcome |
|---|---|---|---|
| E00 | Toolchain smoke test | Install pinned dependencies, detect device, validate paths, load YAML architecture, and run one batch. | Reproducible environment report; no pretrained download. |
| E01 | Tiny overfit test | Train scratch detector on a tiny fixed subset for enough iterations to overfit. | Loss decreases and near-memorization is demonstrated; failure blocks full training. |
| E02 | YOLO11n scratch baseline | COCO-filtered two-class data, YAML initialization, fixed seed, standard augmentation. | Primary reference model. |
| E03 | YOLOv8n scratch comparison | Same data, seed policy, image size, and budget; architecture/config only changes. | Educational baseline comparison. |
| E04 | YOLO26n/P2 scratch comparison | Same data and budget if package/config is stable and free compute permits. | Small-object trade-off, not assumed winner. |
| E05 | COCO + filtered Open Images | Add the versioned filtered Open Images training subset; keep validation source-aligned. | Measure whether diversity improves phone recall. |
| E06 | Input-size ablation | Compare 640 with a safe higher resolution. | Quantify phone recall versus VRAM/time. |
| E07 | Webcam augmentation ablation | Baseline versus conservative blur/downsample/compression/exposure/occlusion transforms. | Measure domain-gap mitigation without test tuning. |
| E08 | Event threshold sweep | Tune confidence, persistence, hysteresis, and cooldown on validation scenario data only. | Choose operating point by explicit precision/recall trade-off. |
| E09 | Locked final evaluation | Evaluate selected checkpoint once on the held-out source and consented webcam set. | Final report with detector and event metrics, slices, and limitations. |

## Scratch-initialization verification

The training script must require `model_config`/YAML rather than a `.pt` path for the primary mode, set `pretrained=False`, disable automatic model downloads where supported, and record the exact initialization method. Before training, hash or summarize a deterministic subset of model parameters; the same architecture instantiated with different seeds must differ, and no parameter should match a known pretrained checkpoint by accidental load. The log must show that no checkpoint was downloaded during setup.

## Training budget

Use a time-aware budget rather than a hard-coded GPU. Begin with a tiny overfit test, then run a short baseline, and expand only when validation metrics and checkpoint persistence work. Detect GPU model, VRAM, CUDA, PyTorch, and Ultralytics versions. Use AMP where supported, conservative workers, and a batch-size probe or safe default. Training from Google Drive directly is avoided when it causes I/O bottlenecks; data is staged locally and artifacts are synchronized back.

## Metrics and gates

A run is complete only if it reports per-class precision, recall, mAP50, mAP50-95, latency, FPS, and error slices. The phone class is a promotion gate: aggregate performance is insufficient if phone recall is poor. Event-engine promotion requires interval-level precision/recall, false positives per hour or session, false negatives on known scenarios, event delay, fragmentation, and stability near threshold.

No numeric performance target is invented before the data exists. The decision rule is evidence-based: select the simplest model that meets the project’s measured accuracy/latency and reproducibility requirements, then report failures and uncertainty.

## Test isolation

The final webcam scenarios are immutable after collection and labeling. The test manifest is not read by threshold-tuning or experiment-selection scripts. Model selection uses only validation data and a predefined decision rule. The final report includes the manifest hash, checkpoint hash, environment, and exact command used.

## References

[1]: https://docs.ultralytics.com/models/yolo11 "Ultralytics YOLO11"
[2]: https://docs.ultralytics.com/models/yolov8 "Ultralytics YOLOv8"
[3]: https://docs.ultralytics.com/models/yolo26 "Ultralytics YOLO26"
[4]: https://docs.ultralytics.com/datasets/detect/coco "COCO Dataset"
[5]: https://storage.googleapis.com/openimages/web/factsfigures_v7.html "Open Images V7 Facts and Figures"
[6]: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker "MediaPipe Face Landmarker"
[7]: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html "OpenCV Camera Calibration and 3D Reconstruction"
[8]: https://research.google.com/colaboratory/faq.html "Google Colab FAQ"
[9]: https://www.kaggle.com/docs/efficient-gpu-usage "Kaggle Efficient GPU Usage Tips"
[10]: https://docs.ultralytics.com/modes/train "Ultralytics Training and Resume"
