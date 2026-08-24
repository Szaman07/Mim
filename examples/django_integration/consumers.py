"""Django Channels WebSocket Consumer for live video frame streaming and real-time alerts."""

import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .services import ProctoringService


class ProctoringLiveConsumer(AsyncWebsocketConsumer):
    """Handles bidirectional WebSocket connection for live examinee video analysis."""

    async def connect(self):
        self.session_id = self.scope["url_route"]["kwargs"]["session_id"]
        self.room_group_name = f"proctoring_{self.session_id}"

        # Join session group
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

        await self.send(text_data=json.dumps({
            "type": "connection_established",
            "session_id": self.session_id,
            "message": "Connected to real-time proctoring stream."
        }))

    async def disconnect(self, close_code):
        # Leave session group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        # Cleanup in-memory state
        service = ProctoringService.get_instance()
        service.cleanup_session(self.session_id)

    async def receive(self, text_data=None, bytes_data=None):
        """Receives frame payload from browser (base64 in JSON or raw binary bytes)."""
        image_data = None
        if bytes_data:
            image_data = bytes_data
        elif text_data:
            try:
                data = json.loads(text_data)
                image_data = data.get("frame")
            except Exception:
                pass

        if not image_data:
            return

        service = ProctoringService.get_instance()
        # Process frame asynchronously via threadpool
        analysis_result = await sync_to_async(service.process_frame)(
            session_id=self.session_id,
            image_data=image_data,
            persist_to_db=True,
        )

        # Send response back to student client
        await self.send(text_data=json.dumps({
            "type": "frame_analysis",
            "data": analysis_result,
        }))

        # Broadcast critical started/updated events to proctor dashboard channel
        if analysis_result.get("emitted_events"):
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "proctor_alert",
                    "payload": analysis_result,
                }
            )

    async def proctor_alert(self, event):
        """Handler for group alert broadcasts."""
        await self.send(text_data=json.dumps({
            "type": "proctor_alert",
            "data": event["payload"],
        }))
