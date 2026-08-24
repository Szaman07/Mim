"""Django URL routing for proctoring REST endpoints and WebSocket routes."""

from django.urls import path, re_path
from .views import AnalyzeFrameAPIView, SessionEventsListView
from .consumers import ProctoringLiveConsumer

urlpatterns = [
    # REST API endpoints
    path("api/proctoring/sessions/<uuid:session_id>/analyze-frame/", AnalyzeFrameAPIView.as_view(), name="analyze_frame"),
    path("api/proctoring/sessions/<uuid:session_id>/events/", SessionEventsListView.as_view(), name="session_events"),
]

# Django Channels ASGI WebSocket routing
websocket_urlpatterns = [
    re_path(r"^ws/proctoring/(?P<session_id>[0-9a-f-]+)/$", ProctoringLiveConsumer.as_asgi()),
]
