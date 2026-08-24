"""REST API Views for Django Proctoring Integration."""

import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View

from .models import ExamSession, ProctoringEvent
from .services import ProctoringService


@method_decorator(csrf_exempt, name="dispatch")
class AnalyzeFrameAPIView(View):
    """HTTP POST endpoint for periodic single-frame analysis (e.g. 1 frame every 1-2 seconds)."""

    def post(self, request, session_id):
        try:
            body = json.loads(request.body.decode("utf-8"))
            image_b64 = body.get("image")
        except Exception:
            # Check multipart form-data
            image_file = request.FILES.get("image")
            image_b64 = image_file.read() if image_file else None

        if not image_b64:
            return HttpResponseBadRequest("Missing image data in request.")

        service = ProctoringService.get_instance()
        result = service.process_frame(session_id=str(session_id), image_data=image_b64, persist_to_db=True)
        return JsonResponse(result)


class SessionEventsListView(View):
    """Returns structured event audit log for a completed or active exam session."""

    def get(self, request, session_id):
        events = ProctoringEvent.objects.filter(session_id=session_id).values(
            "event_id", "event_type", "state", "timestamp", "duration_seconds",
            "confidence_max", "evidence", "diagnostics"
        )
        return JsonResponse({
            "session_id": str(session_id),
            "total_events": len(events),
            "events": list(events),
        })
