/**
 * Frontend Webcam Capture & Streaming Client for Django Proctoring
 * 
 * Captures examinee video stream using HTML5 getUserMedia,
 * extracts frames onto an offscreen canvas at a controlled frame rate (2-4 FPS),
 * and transmits frames over WebSockets (Django Channels) or REST API.
 */

class ProctoringClient {
    constructor(sessionId, options = {}) {
        this.sessionId = sessionId;
        this.fps = options.fps || 3; // 3 frames per second is optimal for low bandwidth & high responsiveness
        this.wsEndpoint = options.wsEndpoint || `ws://${window.location.host}/ws/proctoring/${sessionId}/`;
        this.onEventCallback = options.onEvent || console.log;
        
        this.videoElement = document.createElement("video");
        this.videoElement.autoplay = true;
        this.videoElement.playsInline = true;
        this.videoElement.muted = true;

        this.canvasElement = document.createElement("canvas");
        this.canvasCtx = this.canvasElement.getContext("2d");
        
        this.socket = null;
        this.intervalId = null;
        this.stream = null;
    }

    async start() {
        console.log(`Starting proctoring session: ${this.sessionId}`);

        // 1. Request Webcam Permission
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                audio: false,
            });
            this.videoElement.srcObject = this.stream;
            await this.videoElement.play();
        } catch (err) {
            console.error("Camera access denied or unavailable:", err);
            throw new Error(`Webcam permission required for proctored exam: ${err.message}`);
        }

        // 2. Connect WebSocket to Django Channels
        this.socket = new WebSocket(this.wsEndpoint);

        this.socket.onopen = () => {
            console.log("WebSocket connected to Django proctoring backend.");
            this._startFrameCapture();
        };

        this.socket.onmessage = (event) => {
            const response = JSON.parse(event.data);
            if (response.type === "frame_analysis") {
                this._handleAnalysisResult(response.data);
            }
        };

        this.socket.onerror = (error) => {
            console.error("WebSocket proctoring error:", error);
        };

        this.socket.onclose = () => {
            console.warn("WebSocket proctoring connection closed.");
            this.stop();
        };
    }

    _startFrameCapture() {
        const intervalMs = 1000 / this.fps;
        this.intervalId = setInterval(() => {
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                this._captureAndSendFrame();
            }
        }, intervalMs);
    }

    _captureAndSendFrame() {
        if (!this.videoElement.videoWidth || !this.videoElement.videoHeight) return;

        this.canvasElement.width = 640;
        this.canvasElement.height = 480;
        this.canvasCtx.drawImage(this.videoElement, 0, 0, 640, 480);

        // Convert canvas image to compressed JPEG data URL (quality=0.75 for low bandwidth)
        const frameDataUrl = this.canvasElement.toDataURL("image/jpeg", 0.75);

        this.socket.send(JSON.stringify({
            session_id: this.sessionId,
            timestamp: Date.now() / 1000,
            frame: frameDataUrl,
        }));
    }

    _handleAnalysisResult(result) {
        if (result.emitted_events && result.emitted_events.length > 0) {
            for (const ev of result.emitted_events) {
                console.warn(`[ALERT] Proctoring Event: ${ev.event_type} (${ev.state})`);
                this.onEventCallback(ev);
            }
        }
    }

    stop() {
        if (this.intervalId) clearInterval(this.intervalId);
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        if (this.socket) this.socket.close();
        console.log("Proctoring session stopped.");
    }
}

// Export for module systems or global window
if (typeof module !== "undefined" && module.exports) {
    module.exports = ProctoringClient;
} else {
    window.ProctoringClient = ProctoringClient;
}
