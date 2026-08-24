# Source Ledger — Initial Research

Access context: current research initiated Aug 25, 2026 (user timezone). URLs and claims must be rechecked at finalization because documentation, model catalogs, and licenses can change.

| ID | Source | Verified finding | Planned use |
|---|---|---|---|
| [1] | [Ultralytics Models](https://docs.ultralytics.com/models) | Current Ultralytics documentation lists YOLO26, YOLO12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOv6, YOLOv5, RT-DETR, YOLO-NAS, and other model families. The page distinguishes models by task and supported modes; it describes YOLO26 as new-project/edge-oriented, YOLO11 as a mature alternative, YOLOv8 as an established pipeline choice, and RT-DETR as a trainable real-time transformer detector. | Detector landscape and candidate comparison. |
| [2] | [Ultralytics Model Training](https://docs.ultralytics.com/modes/train) | Official training docs show that a model can be constructed from a YAML architecture/configuration, loaded from a `.pt` pretrained checkpoint, or built from YAML and transferred from a checkpoint. The docs explicitly list `pretrained=False` as random initialization while retaining the architecture. They document automatic device selection, configurable batch/image size, AMP, save periods, project/name directories, seeds, deterministic mode, and resume behavior. | From-scratch primary detector decision, free-GPU workflow, reproducibility, checkpointing, and Codex requirements. |
| [3] | [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8) | Official YOLOv8 docs describe YOLOv8 as released Jan. 10, 2023, with detection/segmentation/classification/pose/OBB variants, Python/CLI support, YAML or `.pt` model inputs, and benchmark figures for the n/s/m/l/x detection variants. The docs identify the implementation as AGPL-3.0 with Enterprise licensing also referenced. | YOLOv8 baseline and licensing caveat; comparison against newer alternatives. |
| [4] | [Ultralytics Validation](https://docs.ultralytics.com/modes/val) | Official validation docs expose mAP50, mAP50-95, mAP75, per-class metrics, per-image precision/recall/F1/TP/FP/FN, plots, JSON/text output, custom validation splits, image size, batch, confidence, IoU, device, and export/benchmark-related options. | Detector evaluation protocol and error-analysis outputs. |

## Initial decisions to challenge later

1. The main detector must be instantiated from an architecture YAML/configuration with pretrained loading disabled and must include a test that detects accidental checkpoint downloads or non-random initialization.
2. YOLOv8n remains a plausible educational baseline because it is established and documented, but current Ultralytics alternatives must be compared fairly rather than rejected solely for being newer.
3. Ultralytics licensing is a project-risk item: use of the package/model may carry AGPL-3.0 obligations, and the final documents must separate research use from redistribution/commercial deployment considerations without giving legal advice.
4. Validation must report phone and person classes separately, not only aggregate mAP.

## Dataset findings

| ID | Source | Verified finding | Planned use |
|---|---|---|---|
| [5] | [COCO official site](https://cocodataset.org/) | The official page identifies COCO as a large-scale common-objects-in-context dataset and links its terms of use, but the page extraction is sparse. Dataset counts and exact split details will be corroborated from the official release annotations/repository before final publication. | Candidate source for person and cell-phone classes, pending final license/count verification. |
| [6] | [Open Images V7 official site](https://storage.googleapis.com/openimages/web/index.html) | The official site reports 15,851,536 boxes on 600 classes, 2,785,498 instance segmentations on 350 classes, and substantial image-level/point-level annotations. | Candidate source for phone/person examples; likely requires filtered download because the full dataset is too large for a free student workflow. |

## Research caveat

The initial search surfaced several third-party dataset pages, but they will not be treated as authoritative for licensing or exact counts unless corroborated by official dataset sources. Small phone-specific datasets on hosting platforms will be evaluated only after their dataset-card license, provenance, annotation quality, and redistribution terms are verified.

## Looking-away component findings

| ID | Source | Verified finding | Planned use |
|---|---|---|---|
| [7] | [Google AI Edge MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | Face Landmarker accepts still images, decoded video, and live streams; outputs a face mesh with 3D landmarks, optional blendshapes, and facial transformation matrices. The documented bundle estimates 478 3D landmarks and uses a lightweight BlazeFace short-range detector. Live-stream mode and tracking-related confidence settings are documented. | Candidate low-compute auxiliary component. Supports a landmark-to-head-pose pipeline without training another model. |
| [8] | [Hammadi et al., Sensors 2022, “Evaluation of Various State of the Art Head Pose Estimation Algorithms for Clinical Scenarios”](https://pmc.ncbi.nlm.nih.gov/articles/PMC9502716/) | The paper describes HPE as face detection, facial-landmark localization, then 3D head-angle estimation. It compares OpenFace 2.0, 3DDFA_V2, and MediaPipe, notes challenges from lighting, occlusion, and large pose angles, and reports that MediaPipe is lightweight and uses a perspective-n-point approach while 3DDFA_V2 can provide a different accuracy/speed trade-off. The paper’s evaluation is informative but limited in generalizability to webcam proctoring. | Evidence for selecting an interpretable head-pose baseline, explicitly documenting failure modes and the need for scenario-specific validation. |
| [9] | [OpenCV calib3d documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) | OpenCV documents the pinhole projection model, camera intrinsics/extrinsics, `solvePnP` methods, and the use of 3D object points plus corresponding 2D image points to estimate pose. It also documents camera calibration and image-size-dependent intrinsic scaling. | Implementation specification for head-pose geometry, calibration assumptions, and reproducible yaw/pitch/roll computation. |

## Auxiliary-component decision direction

The default recommendation will be **face landmarks → geometric head pose → calibrated yaw/pitch thresholds → temporal smoothing and hysteresis**, not gaze estimation or a learned temporal model. This is a provisional direction to be challenged against evidence, because the requirement is sustained head orientation rather than gaze or intent. MediaPipe’s bundled model is auxiliary and pretrained; it is separate from the primary object detector, which must be trained from random initialization.

## Alternative-detector findings

| ID | Source | Verified finding | Planned use |
|---|---|---|---|
| [10] | [Official YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX) | YOLOX is an anchor-free YOLO implementation from Megvii. The repository documents PyTorch and MegEngine implementations and deployment paths including ONNX/ONNXRuntime, TensorRT, ncnn, and OpenVINO. GitHub identifies the repository as Apache-2.0 licensed. | Strong alternative for permissive licensing and deployment, but the final comparison must account for a separate configuration/training stack and lower pedagogical fit than the established Ultralytics workflow. |
| [11] | [Official RT-DETR repository](https://github.com/lyuwenyu/RT-DETR) | The official repository provides Paddle and PyTorch implementations, multiple RT-DETR/RT-DETRv2 variants, custom-data guidance, benchmark tables, sliced inference for small objects, and ONNXRuntime/TensorRT/OpenVINO deployment discussions. GitHub identifies the repository as Apache-2.0 licensed. The repository documents more complex distributed/training variants and relies heavily on published pretrained model tracks in its release history. | Accuracy/small-object and licensing alternative, but likely not the first recommendation for a zero-cost, reproducible student project because of implementation complexity, heavier resource needs, and less direct fit to the requested educational workflow. |

## Provisional detector recommendation

The recommendation remains **YOLOv8n via the Ultralytics stack as the main educational baseline**, trained from a YAML configuration with `pretrained=False`, because it is established, documented, has a simple Python/CLI workflow, supports validation/export/checkpointing, and is easy to explain. This is not yet final: current YOLO11/12/26 and licensing implications must be weighed explicitly. A permissively licensed YOLOX experiment may be offered as a secondary comparison or fallback, not silently substituted into the main design.

## Current Ultralytics-family findings

| ID | Source | Verified finding | Planned use |
|---|---|---|---|
| [12] | [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11) | Official docs call YOLO11 the current stable release/recommended choice, document n/s/m/l/x detection variants, training/validation/inference/export, and report 2.6M parameters and 6.5 GFLOPs for YOLO11n with published COCO and latency figures. The page states YOLO11 code/models are under AGPL-3.0 and Enterprise licenses. | Strong practical candidate; likely primary recommendation if the educational goal allows the current stable version. |
| [13] | [Ultralytics YOLO12](https://docs.ultralytics.com/models/yolo12) | Official docs describe an attention-centric, community-driven research release with possible training instability, higher memory use, and slower CPU throughput; they recommend YOLO11 or YOLO26 for most production workloads. YOLO12n is documented at 2.6M parameters and 7.6 GFLOPs, with detection `.pt` weights and YAML architectures for other tasks. | Include as a research alternative, but reject as baseline due to stability and resource trade-offs. |
| [14] | [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) | Official docs describe the Jan. 2026 release as smaller/faster with native end-to-end inference, a lighter head, STAL small-target-aware label assignment, and a YOLO26n model documented at 2.4M parameters, 5.4 GFLOPs, 40.9 COCO mAP, and 1.7 ms T4 TensorRT latency. They document YAML-only architecture variants for small-object/P2 heads and AGPL-3.0/Enterprise licensing. | Current performance candidate and possible primary detector, but must be weighed against newer-version reproducibility, API maturity, licensing, and the project’s explicit educational preference for a stable understandable baseline. |

## Updated detector decision direction

The final report will likely recommend **YOLO11n from YAML with `pretrained=False`** as the main current Ultralytics baseline, with **YOLOv8n** retained as the requested established educational baseline/ablation and **YOLO26n-P2 or YOLO26n** as a controlled small-object alternative if free compute permits. However, this must be stated as a trade-off: YOLO26 is objectively newer and has documented small-target-oriented features, while YOLO11 is documented as stable; YOLOv8 is more established but no longer current. If licensing permissiveness is a priority for redistribution, YOLOX/RT-DETR Apache-2.0 alternatives must be documented, but they are not automatically preferable for this student workflow.

## Concrete dataset selection direction

| Role | Selected source | Practical use | Why selected | Main limitation |
|---|---|---|---|---|
| Core detector training/validation | COCO 2017, filtered to `person` and `cell phone` | Use the official 118,287-image train2017 and 5,000-image val2017 splits; convert only classes 0 (`person`) and 67 (`cell phone`) to the project’s two-class YOLO dataset. | Standardized, well-documented, CC BY 4.0 as reported by the dataset documentation, existing person and phone boxes, manageable compared with Open Images, and useful for a reproducible baseline. | Generic internet-photo domain; phone examples may not represent small phones on desks or webcam framing. COCO test ground truth is not public, so it is not the project’s final held-out test. |
| Targeted diversity augmentation | Open Images V7 filtered to official `Mobile phone` and `Person` box labels | Do not download the full 561 GB detection subset. Build a filtered manifest from official box annotations and download only a capped subset of images containing relevant boxes, recording the exact manifest and count. A starting cap will be 20,000 train images and 2,000 validation images if the filtered pool supports it. | More varied scenes, explicit Mobile phone label, occlusion attributes, and strong box annotation documentation; official source reports 1.74M box-annotated train images, 41,620 validation images, and 600 boxable classes (the Ultralytics YAML exposes 601 entries). | The full resource is technically impractical on free storage; filtering and image URL availability must be handled carefully. Generic domain remains. Image and annotation licenses differ and require attribution/compliance. |
| Proctoring-specific held-out evaluation | Small consented, self-created scenario set | Capture separate subjects/scenarios after the training design is frozen: normal sitting, slight left/right, looking down, sustained away, phone visible/occluded/far, two people, and brief frame entry. Keep it out of training, threshold tuning, and model selection. | Directly tests intended webcam domain and event behavior. | Small, local, and not statistically representative; must not be treated as a universal benchmark. |

## Split and leakage policy

Use source-provided splits for COCO and Open Images as the starting point, then create project manifests by filtering within each source split. Never move an image from source validation/test into training. Deduplicate by source image ID and perceptual hash before combining sources. Keep all frames from the same self-created video/session and subject in exactly one of train, validation, or held-out test. Reserve the proctoring-specific set until all model and threshold choices are frozen.

## Dataset licensing caveat

The final documents will describe COCO as CC BY 4.0 based on the cited dataset documentation, Open Images images as CC BY 2.0 and annotations as CC BY 4.0 based on Google’s official facts-and-figures page, and will instruct the implementer to preserve attribution files and re-check the current terms before redistribution. These are source/license summaries, not legal advice; image-level third-party terms and takedown/availability issues remain relevant.

## Domain-gap mitigation

The recommended mitigation is not to pretend generic datasets are sufficient: use them for baseline learning, add filtered Open Images diversity, apply conservative webcam-like augmentation (resize/downsample, exposure/white-balance variation, moderate blur/compression, partial occlusion and scale variation), and evaluate on the isolated consented webcam set. If phone recall remains inadequate, add consented images to a separate adaptation experiment only after baseline evaluation, never by silently mixing the held-out test set.

## Free-cloud workflow findings

| ID | Source | Verified finding | Planned use |
|---|---|---|---|
| [15] | [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) | Google states that free Colab resources are not guaranteed or unlimited, GPU access is heavily restricted, idle runtimes time out, and free notebooks can run for at most 12 hours depending on availability and usage patterns. | Survival guide must assume interruption and variability; nothing important may live only in the runtime. |
| [16] | [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) | Kaggle documents versioned notebook environments, attached data sources, output persistence/chaining, changing default images, per-notebook package installation, and free GPU availability subject to queues and platform constraints. | Kaggle compatibility and environment snapshot guidance. |
| [17] | [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage) | Kaggle documents free NVIDIA Tesla P100 access, a weekly GPU quota that is commonly 30 hours or sometimes higher depending on demand/resources, a 60-minute interactive idle timeout, and monitoring/stop-session guidance. | Safe-stop, quota-aware, and monitoring sections of the survival guide. |

## Cloud/persistence decisions

1. Stage datasets from Google Drive or a dataset source onto local runtime storage before training when feasible, because direct Drive I/O can bottleneck training; persist all durable artifacts back to Drive after each checkpoint/epoch or at safe intervals.
2. Detect GPU name, CUDA availability/version, VRAM, PyTorch version, and package versions at startup. Choose batch size by a dry-run probe or conservative VRAM rule and expose an override.
3. Store each experiment in a unique Drive directory with immutable configuration and manifest snapshots. Save `last.pt`, `best.pt`, periodic checkpoints, metrics, plots, logs, environment JSON, Git SHA, and checksums.
4. Use `resume=True` only when continuing the same experiment from a compatible `last.pt` with optimizer/scheduler/epoch state. Start a new experiment from `best.pt` only when intentionally branching, changing data/model/hyperparameters, or recovering without the original optimizer state; never overwrite the source experiment.
5. Recovery always begins with a fresh runtime, Drive mount, pinned dependency installation, repository checkout, environment verification, experiment manifest loading, checkpoint integrity check, and then either exact resume or an explicitly named branch.

## Ethics and privacy findings

| ID | Source | Verified finding | Planned use |
|---|---|---|---|
| [18] | [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) | NIST presents AI RMF 1.0 as a voluntary framework for managing risks to individuals, organizations, and society, emphasizing trustworthy characteristics including fairness, transparency, accountability, privacy, human review, documentation, and monitoring. | Risk register, auditability, evaluation, and governance safeguards. |
| [19] | [UNESCO Recommendation on the Ethics of AI](https://www.unesco.org/en/artificial-intelligence/recommendation-ethics) | UNESCO emphasizes human rights and dignity, proportionality/do no harm, privacy and data protection throughout the lifecycle, responsibility/accountability, transparency/explainability, human oversight, and fairness/non-discrimination. | Privacy-by-design, consent, proportionality, human-review, and non-discrimination principles. |
| [20] | [Coghlan, Miller & Paterson, “Good Proctor or ‘Big Brother’?”](https://pmc.ncbi.nlm.nih.gov/articles/PMC8407138/) | Peer-reviewed ethical analysis identifies privacy, fairness, transparency, autonomy, accountability, and trust concerns in online proctoring and stresses that automated systems are not perfectly accurate and require responsible human intervention and governance. | Explain why anomaly flags must not determine cheating and why institutional review is required. |
| [21] | [Yoder-Himes et al., “Racial, skin tone, and sex disparities in automated proctoring software”](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2022.881449/full) | A peer-reviewed study reports disparities in flagging and face-detection-related outcomes across skin-tone/race/sex groups for one commercial system, illustrating the need for subgroup and intersectional testing; its results should not be generalized automatically to this proposed system. | Fairness-risk rationale and evaluation safeguards, with careful limits on inference. |

## Ethics decision direction

The design will use local processing where practical, no identity recognition, no audio, minimal event metadata, no raw video retention by default, explicit opt-in for evidence snapshots, access control, retention/deletion rules, consent and institutional review, human review before consequences, an appeals path, and subgroup/scenario evaluation where legally and ethically appropriate. It will state that jurisdictions differ and that institutional/legal review is required before deployment; it will not make universal legal claims.
