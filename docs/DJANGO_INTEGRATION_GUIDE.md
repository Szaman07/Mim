# Django Web Application Integration Guide

This guide describes how to integrate the **Proctoring CV** system with a Django web application for real-time exam monitoring, event logging, and proctor dashboards.

---

## 1. System Architecture

```text
[ Browser (Student Exam UI) ]
   │
   │ 1. Capture webcam frame via HTML5 Canvas (2-3 FPS)
   │ 2. Send frame over WebSocket (Django Channels) or REST API
   ▼
[ Django Backend (ASGI / Channels / REST) ]
   │
   │ 3. ProctoringService (Singleton Model Runner)
   │    ├── Primary Detector (YOLO11n: Person + Cellphone)
   │    ├── Head Pose & Looking-Away Estimator (solvePnP)
   │    └── Deterministic Event Engine (FSM with persistence)
   │
   │ 4. Persist observable events to Django Database (PostgreSQL / SQLite)
   │ 5. Emit real-time alerts back via WebSocket
   ▼
[ Database & Proctor Dashboard ]
   ├── ExamSession Model
   ├── ProctoringEvent Model (PHONE_DETECTED, MULTIPLE_PERSONS, LOOKING_AWAY)
   └── Real-time Proctor HUD / Review Stream
```

---

## 2. Integration Files Provided

Reference implementation files are available in [`examples/django_integration/`](file:///e:/study/Projects/mim/examples/django_integration/):

| File | Purpose |
|---|---|
| [`models.py`](file:///e:/study/Projects/mim/examples/django_integration/models.py) | Django models for `ExamSession`, `ProctoringEvent`, and `ProctoringDiagnostic`. |
| [`services.py`](file:///e:/study/Projects/mim/examples/django_integration/services.py) | Thread-safe `ProctoringService` singleton managing detector inference, session FSMs, and database writes. |
| [`consumers.py`](file:///e:/study/Projects/mim/examples/django_integration/consumers.py) | Django Channels WebSocket consumer for live frame ingestion and alert broadcasts. |
| [`views.py`](file:///e:/study/Projects/mim/examples/django_integration/views.py) | REST API endpoints for periodic frame upload and audit event queries. |
| [`urls.py`](file:///e:/study/Projects/mim/examples/django_integration/urls.py) | URL routing patterns for HTTP and WebSocket paths. |
| [`frontend_webrtc_client.js`](file:///e:/study/Projects/mim/examples/django_integration/frontend_webrtc_client.js) | Browser client for HTML5 webcam capture and WebSocket streaming. |

---

## 3. Step-by-Step Django Setup

### Step 3.1: Install Dependencies in Django Virtual Environment
```bash
pip install django channels daphne
pip install -e /path/to/Mim  # Installs proctoring-cv
```

### Step 3.2: Configure Django `settings.py`
Add `daphne` and `channels` to `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    "daphne",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "channels",
    "proctoring_app",  # Your Django app
]

ASGI_APPLICATION = "myproject.asgi.application"

# Optional Redis channel layer for scaling
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer"
    }
}
```

### Step 3.3: Configure `asgi.py`
```python
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import proctoring_app.urls

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(proctoring_app.urls.websocket_urlpatterns)
    ),
})
```

### Step 3.4: Configure Model Checkpoint
Place your trained `best.pt` in `myproject/weights/best.pt` and set in `runtime.yaml` or pass to `ProctoringService.get_instance(checkpoint_path="weights/best.pt")`.

### Step 3.5: Run Migrations and Start ASGI Server
```bash
python manage.py makemigrations
python manage.py migrate
python -m daphne -p 8000 myproject.asgi:application
```

---

## 4. Frontend Integration in Exam Template

Include `frontend_webrtc_client.js` in your exam template HTML:

```html
<script src="/static/js/frontend_webrtc_client.js"></script>
<script>
    const sessionId = "{{ exam_session.session_id }}";
    
    const client = new ProctoringClient(sessionId, {
        fps: 3, // 3 frames per second
        onEvent: (event) => {
            // Display non-intrusive banner to examinee
            if (event.state === "started") {
                console.warn("Event started:", event.event_type);
            }
        }
    });

    // Start streaming when exam starts
    client.start().catch(err => {
        alert("Please allow camera access to begin the exam.");
    });
</script>
```
