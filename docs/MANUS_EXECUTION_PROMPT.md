# Manus Autonomous Execution Prompt: End-to-End Proctoring CV Implementation & Deployment

Copy the prompt below directly into **Manus**:

```markdown
# TASK: Execute End-to-End Proctoring CV Training, Web App Deployment, and Visual Guide

You are an autonomous AI engineering agent. Your objective is to clone the **Proctoring-CV** repository, run its full test and evaluation suite, launch a complete working Django web application with live WebSocket webcam proctoring, test it in your browser environment, and build the documentation guide website.

Repository URL: https://github.com/Szaman07/Mim.git

---

## EXECUTION WORKFLOW

Execute the following 5 phases sequentially, verifying each before proceeding:

---

### PHASE 1: Clone Repository, Environment Setup & Unit Tests

1. Clone the repository and install it in editable mode:
   ```bash
   git clone https://github.com/Szaman07/Mim.git
   cd Mim
   pip install -e .
   pip install django channels daphne
   ```
2. Run the test suite and ensure all 41 unit tests pass:
   ```bash
   python -m pytest tests/ -v
   ```
3. Run the environment diagnostic probe:
   ```bash
   python -c "from proctoring_cv.environment import probe_environment; print(probe_environment().model_dump_json(indent=2))"
   ```

---

### PHASE 2: Dataset Acquisition, Preflight Gate & Baseline Training

1. Generate / acquire the dataset:
   ```bash
   # Download/generate dataset with manifests and dataset.yaml
   python scripts/download_dataset.py --source sample --output data/datasets/coco2017_filtered_v1
   python scripts/validate_dataset.py --root data/datasets/coco2017_filtered_v1
   ```
2. Run the preflight overfit gate (verifying loss decreases from random scratch weights):
   ```bash
   python scripts/tiny_overfit.py --config configs/experiments/yolo11n_scratch_coco.yaml --epochs 5 --device cpu
   ```
3. Run evaluation & benchmark scripts:
   ```bash
   # Benchmark FPS throughput and latency
   python scripts/benchmark.py --device cpu --num-warmup 5 --num-benchmark 20

   # Run deterministic offline event replay scenarios
   python scripts/run_event_replay.py --scenario sustained_phone
   python scripts/run_event_replay.py --scenario multiple_persons_entry
   python scripts/run_event_replay.py --scenario persistent_looking_away
   ```

---

### PHASE 3: Build & Launch Complete Django Web Application

Build and run a complete, functional Django web app integrating the `examples/django_integration/` module.

1. **Create Django Project Structure**:
   ```bash
   django-admin startproject exam_server .
   mkdir -p proctoring_app templates static/js weights
   ```

2. **Integrate Module Files**:
   - Copy `examples/django_integration/models.py` → `proctoring_app/models.py`
   - Copy `examples/django_integration/services.py` → `proctoring_app/services.py`
   - Copy `examples/django_integration/consumers.py` → `proctoring_app/consumers.py`
   - Copy `examples/django_integration/views.py` → `proctoring_app/views.py`
   - Copy `examples/django_integration/urls.py` → `proctoring_app/urls.py`
   - Copy `examples/django_integration/frontend_webrtc_client.js` → `static/js/frontend_webrtc_client.js`

3. **Create UI Templates**:
   - `templates/student_exam.html`: Modern exam UI showing questions, examinee webcam preview box with real-time HUD status badge (OK / Calibration / Looking Away / Multiple Persons / Phone).
   - `templates/proctor_dashboard.html`: Live proctor monitoring dashboard showing active exam sessions, live event stream table, and real-time alert notifications over WebSockets.

4. **Configure Settings & ASGI**:
   - In `exam_server/settings.py`, add `'daphne'`, `'channels'`, `'proctoring_app'` to `INSTALLED_APPS`.
   - Set `ASGI_APPLICATION = "exam_server.asgi.application"`.
   - In `exam_server/asgi.py`, route HTTP to Django and `websocket` to `proctoring_app.urls.websocket_urlpatterns`.

5. **Run Migrations & Start Server**:
   ```bash
   python manage.py makemigrations proctoring_app
   python manage.py migrate
   python -m daphne -b 0.0.0.0 -p 8000 exam_server.asgi:application &
   ```

---

### PHASE 4: Browser Automation & Verification

Use your browser capabilities to verify the web application:
1. Navigate to `http://localhost:8000/` or the exam page.
2. Verify that the student webcam feed initializes and frames are sent via WebSocket.
3. Open `http://localhost:8000/proctor/` to verify the live proctor dashboard receives events.
4. Capture a screenshot of the student exam view and proctor dashboard as verification artifacts.

---

### PHASE 5: Build the Project Guide Website

Build a single-page static website (`guide/index.html`) following the TransferGrid scientific styling specified in `docs/MANUS_GUIDE_PROMPT.md`:
- Warm paper background (`#fbfbfa`), slate typography, monospace labels.
- Section 1: Overview & stat bar (75 files, 41 tests, 10 phases, 3 events, $0 cost).
- Section 2: Architecture diagram (CSS/SVG boxes & arrows).
- Section 3: Completed work (10 phases with checkmarks).
- Section 4: Upcoming milestones (M1–M6 roadmap table).
- Section 5: Step-by-step how-to accordions (setup, dataset, Colab training, resume, evaluate, deploy).
- Section 6: State machine diagram & JSON event schema.
- Serve or render the page and provide a preview screenshot.

---

## OUTPUT REQUIREMENTS

Deliver a final summary containing:
1. **Test Results**: Output of the 41 passing pytest unit tests and tiny overfit test.
2. **Benchmark Table**: Latency and FPS throughput from `scripts/benchmark.py`.
3. **Django App Status**: Proof of running Daphne ASGI server, WebSocket connectivity, and screenshot of the exam UI.
4. **Project Guide Page**: Path/URL to the rendered TransferGrid-style visual guide.
```
