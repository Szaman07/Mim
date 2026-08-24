# RESEARCH_REPORT.md

## Zero-Cost Computer-Vision Proctoring Blueprint

**Author:** Manus AI  
**Research date:** 25 August 2026  
**Status:** Research and specification only; no implementation or model training has been performed.

## Executive decision

The recommended system is a small, two-class object detector plus a separate, interpretable head-pose component and a deterministic temporal event engine. The detector should recognize only `person` and `cellphone`. It should be initialized from the selected architecture/configuration with random weights, not from a COCO-pretrained `.pt` file. The default primary candidate is **Ultralytics YOLO11n from YAML with `pretrained=False`**, with **YOLOv8n from YAML** retained as the established educational baseline and **YOLO26n/P2** considered as a controlled small-object experiment. The primary choice is not based on novelty: it balances current stability, small size, documentation, export support, and the project’s need to demonstrate reproducible scratch training.[1] [2] [3] [4]

Looking away should not be treated as ordinary object detection. The recommended path is face detection/landmarks, geometric head pose, session calibration, smoothing, and time-based hysteresis. MediaPipe Face Landmarker is suitable as a separately documented auxiliary component because it accepts live streams and produces 3D facial landmarks and transformation data; its pretrained status must remain explicit and separate from the detector’s random initialization.[5] [6] [7]

The data plan uses filtered COCO 2017 for a manageable baseline, a capped filtered subset of Open Images V7 for diversity and explicit `Mobile phone` annotations, and a small consented webcam-style test set reserved entirely for final evaluation. COCO provides 118,287 train and 5,000 validation images in the standard 2017 detection split and includes `person` and `cell phone`; Open Images is much larger, has explicit `Mobile phone` and `Person` labels, documented box attributes, and strong annotation coverage, but its full detection subset requires roughly 561 GB and is not practical for free student storage.[8] [9] [10]

## 1. Objective and non-goals

The system reports three observable signals: a visible phone, more than one visible person, and sustained head orientation away from the screen. It does not determine cheating, intent, identity, emotion, speech, audio, mouth movement, or gaze-based mental state. The output vocabulary is therefore `PHONE_DETECTED`, `MULTIPLE_PERSONS`, `LOOKING_AWAY`, `NO_EVENT`, and explicit diagnostic states such as `UNAVAILABLE` or `LOW_CONFIDENCE`.

This distinction matters scientifically and ethically. A model can miss a phone, fail to see a face, or misread a head pose. An event is evidence for human review, not a verdict. Research on online proctoring identifies privacy, fairness, transparency, autonomy, and accountability concerns, while a peer-reviewed study of one commercial system illustrates why face-detection and flagging behavior require subgroup and scenario testing rather than unsupported trust.[11] [12]

## 2. Decision criteria

| Criterion | Required interpretation |
|---|---|
| Scratch training | The detector is built from YAML/configuration and random initialization; no pretrained weights are loaded silently. |
| Free operation | Training uses free Colab/Kaggle GPU sessions, open-source libraries, public datasets, and Drive/GitHub persistence. |
| Educational reproducibility | The workflow is easy to understand, pin, resume, inspect, and repeat. |
| Runtime | A webcam-feasible nano model is preferred; latency must be measured on the actual runtime. |
| Small phone performance | Phone recall is a first-class metric, with size/occlusion slices and higher-resolution experiments. |
| Deployment | Export to ONNX and other practical formats is useful, but export is secondary to valid scratch training. |
| Governance | Logs report observable evidence and uncertainty; no automated adverse decision is made. |

## 3. Detector investigation

Ultralytics currently documents YOLO26, YOLO12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOv6, YOLOv5, RT-DETR, YOLO-NAS, and related families. The documentation assigns different maturity and use-case positions to these models rather than treating the newest model as automatically best.[1]

| Candidate | Evidence and strengths | Constraints | Decision |
|---|---|---|---|
| YOLOv8n | Established Python/CLI API, YAML and `.pt` workflows, validation/export, strong community familiarity; official docs report 3.2M parameters and 8.7 GFLOPs for the nano model.[3] | Older than YOLO11/26; Ultralytics AGPL-3.0/Enterprise licensing needs review for redistribution. | Required educational baseline and fallback. |
| YOLO11n | Official docs describe it as current stable, with detection training/validation/inference/export and 2.6M parameters/6.5 GFLOPs for nano.[2] | Same Ultralytics license family; fewer long-lived third-party examples than YOLOv8. | Recommended default primary candidate. |
| YOLO12n | Attention-centric, documented at 2.6M parameters; official docs warn about possible training instability, higher memory consumption, and slower CPU throughput.[13] | Attention blocks are a poor fit for first free-GPU baseline. | Research comparison only. |
| YOLO26n/P2 | Current model family documents end-to-end inference, small-target-aware label assignment, 2.4M parameters/5.4 GFLOPs for nano, and YAML-only P2 architectures for small objects.[4] | Newer API/configuration may be less established for a student’s reproducibility study; Ultralytics AGPL-3.0/Enterprise licensing remains relevant. | Controlled small-object experiment, not the first baseline. |
| YOLOX | Official Apache-2.0 repository; anchor-free design and ONNX, TensorRT, ncnn, and OpenVINO deployment paths.[14] | Separate training stack and less direct continuity with the requested Ultralytics workflow. | Permissive-license alternative or later comparison. |
| RT-DETR | Official Apache-2.0 repository; PyTorch/Paddle implementations, small R18 option, custom-data support, benchmark tables, sliced inference, and export discussions.[15] | More complex transformer training stack and heavier operational choices for free student sessions. | Alternative research track, not baseline. |

### Final detector recommendation

Use YOLO11n from its architecture YAML with pretrained loading disabled, and preserve a YOLOv8n-from-YAML experiment for comparison. The implementer must verify that no `.pt` checkpoint is fetched, that the model has non-identical random parameters before training, and that the experiment manifest records the initialization method. The current Ultralytics training documentation explicitly distinguishes YAML construction from pretrained `.pt` loading and exposes `pretrained=False`; it also documents automatic device selection, AMP, seeds, deterministic mode, save periods, and resume.[1] [16]

The recommendation is deliberately conservative. YOLO26n may be better on paper, especially for small objects, but its advantage should be measured under the same data, compute, and initialization rules rather than assumed from benchmark tables. The baseline must remain easy to explain and reproduce.

## 4. Dataset investigation and selection

COCO 2017 is the baseline source. Its documented YOLO mapping contains `person` at class 0 and `cell phone` at class 67, with 118,287 train images and 5,000 validation images. The documented training/validation image download is approximately 20.3 GB, which is large but manageable when staged selectively and cached carefully.[8]

Open Images V7 is the diversity source. Google’s official documentation describes about 9 million images, 1.74 million box-annotated training images, 41,620 validation images, 600 boxable classes, and 14.6 million training boxes. The official class list contains `Mobile phone` and `Person`; the documentation describes professional annotation for most boxes, `group-of` attributes, occlusion-related attributes, and different image/annotation licensing terms.[9] [10]

Do not download all Open Images. Use the official box CSV and image metadata to construct a filtered manifest containing only images with `Mobile phone` or `Person`, cap the download, preserve source IDs and URLs, and record the exact manifest hash. Start with up to 20,000 training images and 2,000 validation images if the filtered pool and storage permit. This is a practical experiment size, not a claim about the entire dataset.

| Source | Role | License/terms to verify | Conversion | Limitation |
|---|---|---|---|---|
| COCO 2017 | Main baseline, two classes | CC BY 4.0 as documented by the dataset page used for this blueprint; preserve attribution and re-check current terms.[8] | COCO JSON to YOLO, keep source IDs, retain only categories 0 and 67. | Generic image domain and possible lack of small desk phones. |
| Open Images V7 filtered subset | Diversity and explicit phone label | Images and annotations have different terms; Google documents images as CC BY 2.0 and annotations as CC BY 4.0; preserve notices and re-check current terms.[10] | CSV boxes to YOLO; map `Person` and `Mobile phone`. | Full dataset is roughly 561 GB through the Ultralytics route; filtered download requires robust URL handling and manifesting.[9] |
| Consented webcam set | Locked test/evaluation | Permission and institutional policy govern collection; never publish identifiable frames by default. | Direct YOLO labels plus interval labels for looking-away. | Small and not representative; use only for held-out scenario testing. |

## 5. Domain gap

Generic data contains varied internet photographs, camera angles, resolutions, and object contexts. The intended domain is a front-facing webcam, a student seated at a desk, a narrow field of view, indoor lighting, partial occlusion, and a small phone occupying few pixels. A detector can therefore obtain strong aggregate metrics while failing the actual phone event.

The mitigation is a layered design: filter both datasets for relevant classes; preserve hard examples; add modest downsampling, blur, compression, exposure, white-balance, and occlusion augmentation; report phone performance by size and occlusion; and lock a consented webcam-style test set. Collect supplementary images only with informed consent, minimal metadata, secure storage, and a clear deletion process. Do not use frames from the held-out test set to tune thresholds.

## 6. Looking-away decision

Head pose is the appropriate first abstraction because the requirement is sustained head orientation, not gaze intent. MediaPipe Face Landmarker supports image, video, and live-stream inputs, returns 3D face landmarks and transformation matrices, and uses a lightweight face detector in its bundle.[5] OpenCV documents the projective camera model and `solvePnP` methods for estimating pose from 3D/2D correspondences.[7] A peer-reviewed comparison describes HPE as face detection, landmark localization, and angle estimation, while noting errors under lighting, occlusion, and large poses.[6]

The preferred pipeline is:

```text
primary person
  → face landmarks
  → canonical 3D points + image points
  → solvePnP or documented facial transform
  → yaw/pitch/roll
  → neutral-pose calibration
  → median/EMA smoothing
  → timestamp-based persistence and hysteresis
  → LOOKING_AWAY or UNAVAILABLE
```

Gaze estimation and a learned temporal model are deferred. They add calibration, data, and interpretability costs and are not required to answer whether the head has been turned away for a sustained interval.

## 7. Event logic

Confidence and persistence are starting values, not universal thresholds. Use validation data and the locked scenario set only after model selection is complete.

| Event | Initial start rule | End rule |
|---|---|---|
| `PHONE_DETECTED` | Phone confidence above the validation-selected threshold in 3 of 5 frames or about 0.5 seconds. | No valid phone detection for about 0.75 seconds. |
| `MULTIPLE_PERSONS` | At least two person detections for about 0.75 seconds or 5 of 7 frames. | At most one for 1 second. |
| `LOOKING_AWAY` | Calibrated `abs(yaw-yaw0) ≥ 25°` or `abs(pitch-pitch0) ≥ 20°` for 1.5 seconds with pose valid in at least 70% of the window. | Return inside approximately 15° hysteresis boundary for 0.75 seconds. |

Use timestamps instead of hard-coded frame counts so the logic remains meaningful across webcam rates. Log event start/end time, confidence summary, evidence counts, model and configuration IDs, pose values where appropriate, and diagnostics. Raw frames are not retained by default.

## 8. Training and evaluation

The training pipeline is acquisition, validation, conversion, split, sanity checks, tiny overfit test, baseline training, validation, checkpointing, evaluation, error analysis, and controlled experiments. Every run stores the seed, Git SHA, dataset manifest hash, YAML/configuration, initialization mode, environment, GPU, image size, batch size, optimizer, learning rate, scheduler, augmentation, duration, and metrics.

Ultralytics documents `resume=True` as restoring model weights, optimizer, scheduler, and epoch from a partial checkpoint, while a new run from `best.pt` is a branch and does not mean exact continuation.[16] Use Drive for durable storage, stage data locally when possible, and save `best.pt`, `last.pt`, periodic checkpoints, logs, plots, manifests, and checksums.

Detector metrics are per-class precision, recall, F1, mAP50, mAP50-95, latency, and FPS under a declared runtime; validation documentation exposes the principal mAP and per-image metrics.[17] Looking-away metrics are frame/event precision, recall, F1, false-positive rate, false-negative rate, detection delay, fragmentation, duration bias, and temporal stability.

## 9. Cloud constraints

Colab’s free tier has variable resources, idle timeouts, restricted GPU access, and a maximum notebook runtime of up to 12 hours depending on availability and usage.[18] Kaggle documents free GPU access subject to queues and quota, versioned notebook environments, output persistence, and a 60-minute interactive idle timeout.[19] [20] The system must not assume that `/content`, `/kaggle/working`, or any runtime directory survives. Drive and GitHub are the persistence layers; the runtime is disposable.

## 10. Ethics and deployment boundary

Use the least invasive system that answers the stated research question. Process locally where possible, do not identify students, do not retain raw webcam streams by default, provide notice and consent, restrict access to event logs, set retention/deletion periods, and provide human review and appeal before any consequence. Test for subgroup and lighting/pose differences where ethically and legally appropriate. NIST and UNESCO both emphasize risk management, privacy, transparency, accountability, human oversight, and fairness.[21] [22]

Before real deployment, the institution must review applicable privacy, education, employment, accessibility, and data-protection obligations in the relevant jurisdiction. This report does not provide legal advice. The system’s scientifically defensible claim is only that an observable visual condition was detected with measured uncertainty.

## References

[1]: https://docs.ultralytics.com/models "Models Supported by Ultralytics"
[2]: https://docs.ultralytics.com/models/yolo11 "Ultralytics YOLO11"
[3]: https://docs.ultralytics.com/models/yolov8 "Explore Ultralytics YOLOv8"
[4]: https://docs.ultralytics.com/models/yolo26 "Ultralytics YOLO26"
[5]: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker "Google AI Edge MediaPipe Face Landmarker"
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9502716/ "Evaluation of Various State of the Art Head Pose Estimation Algorithms for Clinical Scenarios"
[7]: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html "OpenCV Camera Calibration and 3D Reconstruction"
[8]: https://docs.ultralytics.com/datasets/detect/coco "COCO Dataset"
[9]: https://docs.ultralytics.com/datasets/detect/open-images-v7 "Open Images V7 Dataset"
[10]: https://storage.googleapis.com/openimages/web/factsfigures_v7.html "Open Images V7 Facts and Figures"
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8407138/ "Good Proctor or ‘Big Brother’? Ethics of Online Exam Supervision Technologies"
[12]: https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2022.881449/full "Racial, skin tone, and sex disparities in automated proctoring software"
[13]: https://docs.ultralytics.com/models/yolo12 "YOLO12: Attention-Centric Object Detection"
[14]: https://github.com/Megvii-BaseDetection/YOLOX "Official YOLOX Repository"
[15]: https://github.com/lyuwenyu/RT-DETR "Official RT-DETR Repository"
[16]: https://docs.ultralytics.com/modes/train "Model Training with Ultralytics YOLO"
[17]: https://docs.ultralytics.com/modes/val "Model Validation with Ultralytics YOLO"
[18]: https://research.google.com/colaboratory/faq.html "Google Colab FAQ"
[19]: https://www.kaggle.com/docs/notebooks "Kaggle Notebooks Documentation"
[20]: https://www.kaggle.com/docs/efficient-gpu-usage "Kaggle Efficient GPU Usage Tips"
[21]: https://www.nist.gov/itl/ai-risk-management-framework "NIST AI Risk Management Framework"
[22]: https://www.unesco.org/en/artificial-intelligence/recommendation-ethics "UNESCO Recommendation on the Ethics of Artificial Intelligence"
