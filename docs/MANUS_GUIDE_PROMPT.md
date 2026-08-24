# Build a Project Guide Website for "Proctoring-CV"

Build a single-page, vertically-scrolling **project guide and roadmap website** for **"Proctoring-CV"** — a zero-cost webcam proctoring computer vision system.

The purpose of this website is to clearly show:
1. **What the project is** (brief intro, not a sales pitch)
2. **What has been built** (completed phases with details)
3. **What still needs to be done** (upcoming milestones)
4. **How to do each step** (concrete commands and workflows)

GitHub: `https://github.com/Szaman07/Mim`

---

## CRITICAL DESIGN RULES

**DO NOT** make an Apple-style marketing page. No hero gradients, no floating product mockups, no "Get Started Free" CTAs, no glassmorphism cards, no animated counters.

**DO** match this aesthetic exactly:
- Warm off-white paper background (`#fbfbfa`)
- Slate-black text (`#1a1a2e`)
- Thin dividers (`#e2e8f0`)
- Callout boxes: light gray background (`#f1f5f9`) with a 3px left border in dark slate (`#1e293b`) or steel blue (`#3b82f6`)
- Section labels: UPPERCASE LETTERSPACED MONOSPACE (e.g. `PHASE 3 · PRIMARY DETECTOR`)
- Body font: `Charter`, `Lora`, or `Georgia` (serif, editorial)
- Headings: `Inter` or `Plus Jakarta Sans` (clean sans-serif)
- Layout: two-column prose blocks where appropriate, styled data tables with monospace headers, collapsible accordions for how-to steps
- Feels like reading a well-typeset technical document, not browsing a product website

**Single continuous scroll.** No routing, no multi-page. One `index.html` file.

---

## PAGE CONTENT (in scroll order)

### 1. Header Bar
Sticky minimal top bar. Left: monospace brand `PROCTORING · CV` with a small geometric eye icon. Right: two links — "GitHub" and "What's Next" (anchor scroll).

### 2. Project Introduction
**Section label**: `PROJECT OVERVIEW`
**Heading**: `Zero-Cost Reproducible Webcam Proctoring System`

One short paragraph — not an abstract, just a plain description:

> Proctoring-CV is an open-source computer vision system that monitors webcam video during online exams and reports three observable visual events: **phone visible in frame**, **multiple people in frame**, and **head turned away from screen**. It does not claim to detect cheating — it records timestamped visual observations with confidence scores and uncertainty diagnostics. The primary detector (YOLO11n) is trained entirely from scratch with no pretrained weights.

Below that, a **horizontal stat bar** (simple number + label pairs, not animated):

| 75 | 41 | 10 | 3 | 2 | $0 |
|---|---|---|---|---|---|
| Source Files | Passing Tests | Completed Phases | Observable Events | Target Classes | Cloud API Cost |

---

### 3. System Architecture Diagram
**Section label**: `HOW IT WORKS`

A clean **static flow diagram** (CSS/SVG boxes and arrows, not an image) showing:

```
Webcam Frame
    │
    ├──► YOLO11n Detector (person + cellphone)
    │       ├── person count → MULTIPLE_PERSONS candidate
    │       ├── phone detection → PHONE_DETECTED candidate
    │       └── primary person box ─┐
    │                               │
    └──► Face Landmarks ────────────┘
            └── solvePnP head pose
                └── calibration + smoothing
                    └── LOOKING_AWAY candidate

All candidates ──► Event Engine (4-state FSM) ──► Event Log + Django API
```

Below the diagram, a small **callout box**:
> The system never outputs `CHEATING`, `SUSPICIOUS_PERSON`, emotion, intent, identity, or audio analysis. When evidence is missing, it reports `POSE_UNAVAILABLE` or `LOW_CONFIDENCE` instead of guessing.

---

### 4. What's Been Built (Completed Phases)
**Section label**: `COMPLETED WORK`
**Heading**: `10 implementation phases are done.`

Display as a **vertical timeline or numbered card stack**, each phase showing: phase number, title, what was built, and key files. Mark all as ✅ complete.

**Phase 1 — Repository & Environment Layer**
Core Python package with config loader (YAML + env overrides + SHA-256 hashing), hardware diagnostics probe, reproducibility seeding, and structured JSONL logging.
Files: `src/proctoring_cv/config.py`, `environment.py`, `reproducibility.py`, `logging_utils.py`, `schemas.py`

**Phase 2 — Dataset Tooling**
COCO and Open Images annotation converters (→ YOLO format), cross-split duplicate detection (SHA-256 + perceptual dHash), immutable dataset manifests, and automated download script.
Files: `dataset_tools/coco_to_yolo.py`, `openimages_filter.py`, `deduplicate.py`, `manifest.py`, `scripts/download_dataset.py`

**Phase 3 — Primary Detector**
YOLO11n loaded strictly from YAML (`pretrained=False`). Automated guard rejects `.pt` weights in scratch mode. Parameter hash verification across random seeds. Prediction normalization into standard Detection schemas.
Files: `src/proctoring_cv/detector.py`, `scripts/tiny_overfit.py`, `scripts/train.py`

**Phase 4 — Checkpoint Manager & Drive Sync**
Atomic checkpoint writes (tempfile → rename), SHA-256 integrity verification, periodic rotation, Google Drive folder tree synchronization, and exact-resume vs. branching workflow.
Files: `src/proctoring_cv/checkpoint_manager.py`, `drive_sync.py`, `scripts/resume_training.py`

**Phase 5 — Evaluation & Error Slicing**
Precision, Recall, F1, mAP50, mAP50-95, latency/FPS benchmarking. Temporal event metrics (onset delay, duration bias, interval IoU). Size-based and context-based error slice breakdowns.
Files: `evaluation/metrics.py`, `event_metrics.py`, `error_slices.py`, `scripts/evaluate.py`, `scripts/benchmark.py`

**Phase 6 — Head Pose & Looking-Away**
3D–2D facial geometry via OpenCV `solvePnP`. 30-frame frontal calibration for baseline yaw/pitch. Median + EMA smoothing. Hysteresis thresholds (start ≥25°, end ≤15°). Returns `POSE_UNAVAILABLE` when face is missing.
Files: `src/proctoring_cv/looking_away.py`

**Phase 7 — Event Engine & Replay**
Deterministic timestamp-driven finite state machine: `INACTIVE → CANDIDATE → ACTIVE → ENDING`. Configurable persistence (0.5–1.5s), short-gap merging (0.5s), cooldowns (2.0s). Offline replay for reproducible testing.
Files: `src/proctoring_cv/event_engine.py`, `scripts/run_event_replay.py`

**Phase 8 — Webcam Inference**
Live webcam runner with primary person tracking, real-time event evaluation, privacy-safe HUD overlay, and structured JSONL session logging.
Files: `scripts/infer_webcam.py`

**Phase 9 — Test Suite**
41 automated pytest tests covering: config loading, dataset conversion & validation, random initialization verification, checkpoint atomicity, resume compatibility, event engine logic, head-pose geometry, and forbidden output rejection.
Files: `tests/test_*.py` (9 test modules)

**Phase 10 — Django Integration & Documentation**
Production-ready Django integration: ORM models (`ExamSession`, `ProctoringEvent`), thread-safe `ProctoringService` singleton, Django Channels WebSocket consumer, REST API endpoints, browser client (3 FPS canvas → WebSocket). Plus 4 Jupyter notebooks and full documentation.
Files: `examples/django_integration/`, `notebooks/`, `docs/`, `README.md`

---

### 5. What Needs to Be Done (Upcoming Milestones)
**Section label**: `WHAT'S NEXT`
**Heading**: `6 milestones remain before the system is production-ready.`

Display as a **table or card stack**, each marked with a status icon (⬜ not started).

| # | Milestone | What It Involves | Estimated Effort |
|---|---|---|---|
| M1 | **Train on Real COCO Data** | Download filtered COCO 2017 (`python scripts/download_dataset.py --source coco_val2017`). Run preflight gate (`tiny_overfit.py`). Launch full scratch training on Colab T4 GPU for ~50 epochs. | 4–8 hours (Colab GPU time) |
| M2 | **Run Experiment Comparisons** | Train YOLOv8n scratch baseline (E03). Optionally YOLO26n/P2 (E04). Compare per-class metrics under identical conditions. | 8–12 hours |
| M3 | **Tune Event Thresholds** | Sweep confidence, persistence, hysteresis, and cooldown on validation scenario data (E08). Choose operating point by precision/recall trade-off. | 2–3 hours |
| M4 | **Locked Final Evaluation** | Run selected checkpoint once on held-out test set (E09). Report detector + event metrics, error slices, and limitations. This evaluation is one-shot — no re-tuning after. | 1–2 hours |
| M5 | **Deploy Django Application** | Integrate trained `best.pt` checkpoint into Django backend. Configure ASGI server (Daphne), database (PostgreSQL), and channel layer (Redis). Test end-to-end student → proctor WebSocket flow. | 1–2 days |
| M6 | **Collect Webcam Test Scenarios** | Record consented webcam clips covering: normal exam, phone in hand, phone on desk, second person entering, head turns, lighting changes, face partially occluded. Label with locked scenario protocol. | 1–2 days |

---

### 6. How-To Guides (Collapsible Accordions)
**Section label**: `STEP-BY-STEP GUIDES`
**Heading**: `How to execute each milestone.`

Use **collapsible accordions** (collapsed by default, expand on click). Each accordion contains the exact terminal commands and explanation.

**Accordion 1: "How to Set Up the Environment"**
```bash
git clone https://github.com/Szaman07/Mim.git
cd Mim
pip install -e .
python -m pytest tests/ -v   # Should show: 41 passed
```

**Accordion 2: "How to Get the Dataset"**
```bash
# Quick synthetic dev dataset (instant, for testing)
python scripts/download_dataset.py --source sample --output data/datasets/coco2017_filtered_v1

# Real filtered COCO 2017 (downloads ~1GB)
python scripts/download_dataset.py --source coco_val2017 --output data/datasets/coco2017_filtered_v1

# Validate dataset integrity
python scripts/validate_dataset.py --root data/datasets/coco2017_filtered_v1
```

**Accordion 3: "How to Train on Google Colab"**
```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone and install
!git clone https://github.com/Szaman07/Mim.git /content/Mim
%cd /content/Mim
!pip install -e .

# 3. Preflight gate (must show decreasing loss)
!python scripts/tiny_overfit.py --config configs/experiments/yolo11n_scratch_coco.yaml --epochs 5 --device 0

# 4. Full training
!python scripts/train.py \
    --config configs/experiments/yolo11n_scratch_coco.yaml \
    --experiment-id yolo11n_scratch_coco_v1_seed42 \
    --drive-root /content/drive/MyDrive/proctoring_cv \
    --mode full
```

**Accordion 4: "How to Resume After Colab Disconnects"**
```bash
python scripts/resume_training.py \
    --experiment-dir /content/drive/MyDrive/proctoring_cv/experiments/yolo11n_scratch_coco_v1_seed42 \
    --checkpoint last.pt \
    --resume
```

**Accordion 5: "How to Evaluate a Trained Checkpoint"**
```bash
python scripts/evaluate.py \
    --checkpoint experiments/yolo11n_scratch_coco_v1_seed42/checkpoints/best.pt \
    --split val
```

**Accordion 6: "How to Deploy the Django App"**
```bash
pip install django channels daphne
pip install -e /path/to/Mim

# Copy examples/django_integration/ files into your Django app
# Configure settings.py with ASGI_APPLICATION and CHANNEL_LAYERS
# Place best.pt in weights/ directory

python manage.py makemigrations
python manage.py migrate
python -m daphne -p 8000 myproject.asgi:application
```

---

### 7. Event Engine State Diagram
**Section label**: `EVENT ENGINE`

A clean **static state-flow diagram** (CSS boxes with arrows):

```
INACTIVE ──[condition met]──► CANDIDATE ──[persistence met]──► ACTIVE ──[condition lost]──► ENDING ──[timeout]──► INACTIVE
                                  │                                                             │
                                  └──[lost early]──► INACTIVE                     └──[returns]──► ACTIVE
```

Below it, a small table:

| Event | Start Persistence | End Persistence | Gap Merge |
|---|---|---|---|
| `PHONE_DETECTED` | 0.5 s | 0.75 s | 0.5 s |
| `MULTIPLE_PERSONS` | 0.75 s | 1.0 s | 0.5 s |
| `LOOKING_AWAY` | 1.5 s (≥70% valid pose) | 0.75 s | 0.5 s |

---

### 8. Repository Map
**Section label**: `REPOSITORY STRUCTURE`

Show the folder tree in a **monospaced code block** with brief inline annotations:

```
proctoring-cv/
├── src/proctoring_cv/      # Core library (detector, pose, events, config)
├── scripts/                 # CLI tools (train, evaluate, download, replay)
├── dataset_tools/           # COCO/OpenImages converters, deduplication
├── evaluation/              # Metrics, event metrics, error slices
├── examples/django_integration/  # Django models, views, WebSocket consumer
├── notebooks/               # 4 Jupyter notebooks (smoke test → evaluation)
├── configs/                 # YAML configs for data, runtime, experiments
├── tests/                   # 41 automated tests (9 modules)
├── docs/                    # Architecture, training guide, Django guide
└── README.md
```

---

### 9. Footer
- GitHub: `https://github.com/Szaman07/Mim`
- License: MIT
- Small text: "Built with Python, PyTorch, Ultralytics, OpenCV, and Django."

---

## IMPLEMENTATION CONSTRAINTS
1. **Single `index.html` file** with inline CSS and minimal JS (accordion toggles only).
2. **No frameworks** — no React, no Vue, no Tailwind. Vanilla HTML/CSS/JS.
3. **No animated counters, parallax scrolling, or marketing animations.** Hover effects on table rows and cards are fine.
4. **Two-column layouts collapse to single column on mobile.**
5. **The page should feel like reading a well-typeset project notebook — calm, structured, informative.**
