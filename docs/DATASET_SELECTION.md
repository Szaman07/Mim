# DATASET_SELECTION.md

## Final recommendation

Use a **three-part data strategy**. First, use a filtered COCO 2017 subset as the reproducible baseline. Second, add a capped filtered subset of Open Images V7 for phone/person diversity and hard examples. Third, maintain a small consented webcam-style set as a locked proctoring-specific evaluation set. Do not use commercial datasets, paid annotation services, or paid compute.

## Selected datasets

| Dataset | Role | Source size and relevant classes | Annotation/licensing | Planned project use |
|---|---|---|---|---|
| COCO 2017 | Baseline train/validation | 118,287 train images and 5,000 validation images in the standard detection split; `person` is source class 0 and `cell phone` is source class 67.[1] | COCO documentation used here reports CC BY 4.0; preserve attribution and re-check terms before redistribution.[1] | Filter to `person` and `cell phone`, convert to project IDs 0/1, and retain source train/val separation. Approximate full train/val image storage is 20.3 GB through the documented workflow; filtered storage depends on selected files.[1] |
| Open Images V7 | Diversity and hard-phone augmentation | Google reports about 9M total images, 1.74M box-annotated train images, 41,620 validation images, and 600 boxable classes. The official class list includes `Mobile phone` and `Person`.[2] | Google documents images as CC BY 2.0 and annotations as CC BY 4.0; keep notices and attribution separate from project code.[2] | Build an official-annotation filtered manifest and download up to 20,000 train and 2,000 validation images initially. Do not download the full approximately 561 GB Ultralytics route.[3] |
| Consented webcam set | Locked domain test | Project-created, scenario-labeled images/videos rather than a public dataset. | Consent and institutional policy govern collection; no raw frames are redistributed by default. | Capture multiple sessions and scenarios after model/threshold decisions are frozen. Store restricted metadata and interval labels; do not use for tuning. |

## Alternatives considered

A full Open Images download is rejected for the baseline because the documented route requires roughly 561 GB, which conflicts with free storage and fragile notebook runtimes.[3] A phone-specific dataset hosted on an arbitrary platform is not selected without a verifiable dataset card, provenance, license, and annotation policy. Third-party mirrors may be used only as a convenience after comparing their content and license to the official source.

COCO is not sufficient by itself for proctoring because its broad internet-image domain does not guarantee webcam-like framing, small desk phones, partial occlusion, or low-resolution appearance. Open Images is therefore an augmentation source, not proof that the domain gap is solved.

## Exact project class map

| Project ID | Project class | COCO source | Open Images source | Purpose |
|---:|---|---|---|---|
| 0 | `person` | `person` (source ID 0) | `Person` | Count visible people. |
| 1 | `cellphone` | `cell phone` (source ID 67) | `Mobile phone` | Detect a visible mobile phone. |

No `looking_away` label is included in this detector dataset. Looking away is computed from face landmarks and head pose. No `cheating`, `student`, `face`, `hand`, or `desk` class is added.

## Acquisition procedure

The preparation script should download only files named in a versioned manifest. For COCO, use the official 2017 image archives and annotation JSON, then retain only images containing at least one selected class plus a controlled sample of negative images. For Open Images, download official box annotations and image metadata first, filter by exact source label, then fetch only the selected image URLs. Each manifest row records source, source split, source image ID, URL, local path, image dimensions, selected labels, license metadata, and a checksum when feasible.

The exact filtered image count must be reported after acquisition, not guessed from the source total. The starting Open Images cap is a resource guardrail, not a fixed scientific requirement. If the filtered pool is smaller or download failures occur, the report must state the actual count and failed URLs.

## Conversion

COCO JSON boxes are converted from pixel coordinates to normalized YOLO boxes. Open Images CSV boxes are converted from normalized source coordinates to pixel coordinates using verified image width/height and then normalized to the final YOLO representation. The converter must preserve original annotation rows and write `source_class_map.json`, `conversion_report.json`, and dropped-class counts.

Every conversion must verify that the image dimensions used in the box transform match the downloaded image. Invalid or missing dimensions, malformed boxes, non-finite values, and boxes outside allowed bounds cause a loud failure or an explicitly logged exclusion. Boxes are clipped only under a documented tolerance policy; silently repairing major errors is prohibited.

## Split policy

Use COCO’s official train2017 and val2017 as baseline train and validation. Use Open Images’ official train and validation splits as source-aligned train and validation. Do not use COCO test-dev as a local test because ground truth is withheld. The project-level test consists of a separately held-out permitted source subset, if available, plus the locked consented webcam set.

When combining sources, deduplicate by source ID and perceptual hash before creating manifests. Keep all frames from a recording session and all images of a self-created subject in one split. Freeze test manifests before any threshold or hyperparameter tuning. Store a SHA-256 hash of each manifest in the experiment configuration.

## Storage planning

| Artifact | Storage policy |
|---|---|
| Public source archives | Do not keep unnecessary archives after verified extraction; record source URL and checksum. |
| Filtered images | Store in Drive under a versioned dataset directory; stage a working copy on runtime local disk for training where feasible. |
| Labels/manifests | Keep in Git when they contain no restricted personal data and in Drive backups as well. |
| Consented webcam data | Restricted Drive location, not GitHub; publish only aggregate metrics and de-identified metadata. |
| License/attribution files | Keep with each dataset version and include in release documentation. |

## Legal and technical suitability

The selected public datasets are technically usable for a student research project subject to the cited terms, attribution, and current-source verification. This document is not legal advice. Before redistribution or commercial use, inspect the current license, image-level terms, source URLs, institutional policy, and any takedown or third-party restrictions. The self-created set requires informed consent and a documented retention/deletion policy.

## References

[1]: https://docs.ultralytics.com/datasets/detect/coco "COCO Dataset documentation"
[2]: https://storage.googleapis.com/openimages/web/factsfigures_v7.html "Open Images V7 Facts and Figures"
[3]: https://docs.ultralytics.com/datasets/detect/open-images-v7 "Open Images V7 Dataset documentation"
