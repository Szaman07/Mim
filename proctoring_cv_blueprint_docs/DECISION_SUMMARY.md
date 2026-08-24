# DECISION_SUMMARY.md

## Final decisions

| Question | Decision |
|---|---|
| Which detector? | **YOLO11n** as the current default primary detector, with YOLOv8n as an established educational baseline and YOLO26n/P2 as a controlled small-object experiment. |
| Why? | YOLO11n is documented as stable, small, trainable/validatable/exportable, and operationally simple. YOLOv8 is retained for familiarity; YOLO26 is not adopted blindly because current benchmark advantages must be verified under equal scratch-training conditions. [1] [2] [3] |
| Scratch or pretrained? | **Scratch for the main detector:** construct from YAML/configuration with `pretrained=False`. Any pretrained auxiliary face-landmark model is explicitly separate. |
| Which datasets? | Filtered COCO 2017 for the baseline; capped filtered Open Images V7 for additional phone/person diversity; a consented webcam-style set for locked proctoring evaluation. [4] [5] |
| How many images? | COCO source split: 118,287 train and 5,000 validation images before filtering. Open Images source: 1,743,042 train and 41,620 validation box-annotated images, but begin with a capped 20,000 train/2,000 validation filtered subset if available. Actual filtered counts must be reported after acquisition. [4] [5] |
| Which classes? | Exactly `person` and `cellphone`. COCO `person` and `cell phone`, and Open Images `Person` and `Mobile phone`, map to IDs 0 and 1. |
| How split? | Preserve source train/validation splits, group self-created frames by subject/session, deduplicate by source ID/perceptual hash, and keep the webcam scenario set isolated until final evaluation. |
| How looking-away works? | Face landmarks → geometric head pose → neutral-session calibration → yaw/pitch smoothing → timestamp-based persistence and hysteresis → `LOOKING_AWAY` or `POSE_UNAVAILABLE`. [6] [7] |
| What is trained? | The two-class object detector. |
| What is not trained? | No audio, speech, mouth, emotion, identity, cheating classifier, or unnecessary temporal model. The face-landmark auxiliary model is used as a separately documented pretrained component. |
| How use Colab/Kaggle? | Use free GPU sessions as disposable compute, detect runtime capabilities, stage data locally, use AMP where supported, and save all durable artifacts to Drive. Colab and Kaggle resources/timeouts are variable. [8] [9] |
| How use Drive? | Store dataset versions, manifests, checkpoints, logs, metrics, plots, exports, registry, backups, and environment snapshots under `MyDrive/proctoring_cv/`. |
| How protect checkpoints? | Save `best.pt`, `last.pt`, periodic checkpoints, optimizer/scheduler state, checksums, configuration, Git SHA, and environment metadata. Use atomic/copy-safe writes and rotation. |
| How resume? | Use `resume=True` only for the same experiment with compatible state and `last.pt`; branch from `best.pt` in a new experiment when changing data, model, hyperparameters, or optimizer state. [10] |
| First experiments? | Smoke test, tiny overfit, YOLO11n scratch baseline, YOLOv8n scratch comparison, optional YOLO26n/P2, COCO versus COCO+Open Images, input-size, augmentation, and validation-only event threshold sweeps. |
| What measure? | Per-class precision, recall, F1, mAP50, mAP50-95, latency/FPS, size/occlusion slices, and looking-away/event precision, recall, false-positive rate, false-negative rate, delay, fragmentation, and temporal stability. |
| Major risks? | Generic-to-webcam domain gap, small-phone false negatives, free-GPU interruption, silent pretrained loading, pose instability, privacy, bias, false positives, and licensing/redistribution constraints. |

## One-sentence architecture

A scratch-trained two-class detector supplies phone and person observations; a separate lightweight landmark/head-pose module supplies calibrated head orientation; a deterministic timestamp-based event engine turns persistent observations into observable event records rather than cheating judgments.

## Sources

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
