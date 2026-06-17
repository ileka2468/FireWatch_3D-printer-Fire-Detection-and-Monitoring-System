"""
Stream Manager — manages OpenCV video capture threads for multiple network streams.

Each stream runs in its own daemon thread, continuously reading frames and storing
only the latest one for the detection engine and GUI to consume.
"""

import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)


class StreamWorker:
    """Background thread worker that continuously reads frames from a single video source."""

    def __init__(self, name: str, url: str, zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0, brightness: int = 0, contrast: float = 1.0):
        self.name = name
        self.url = url
        self.zoom = max(1.0, zoom)
        self.pan_x = max(-1.0, min(1.0, pan_x))
        self.pan_y = max(-1.0, min(1.0, pan_y))
        self.brightness = max(-100, min(100, int(brightness)))
        self.contrast = max(0.1, min(3.0, float(contrast)))
        
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self):
        """Start the capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"stream-{self.name}")
        self._thread.start()
        logger.info(f"[{self.name}] Stream worker started for {self.url}")

    def stop(self):
        """Stop the capture thread and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._connected = False
        logger.info(f"[{self.name}] Stream worker stopped")

    def get_latest_frame(self):
        """Return the most recent frame (numpy array) or None if not available."""
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            
        # Apply crop/pan logic if zoomed in
        if frame is not None and self.zoom > 1.01:  # Allow tiny floating point tolerance
            h, w = frame.shape[:2]
            
            # Target size of the crop box
            crop_w = int(w / self.zoom)
            crop_h = int(h / self.zoom)
            
            # Max distance we can shift the center without going out of bounds
            max_offset_x = (w - crop_w) / 2
            max_offset_y = (h - crop_h) / 2
            
            # Calculate new center based on pan (-1.0 to 1.0)
            center_x = (w / 2) + (self.pan_x * max_offset_x)
            center_y = (h / 2) + (self.pan_y * max_offset_y)
            
            # Bounding box
            x1 = max(0, int(center_x - crop_w / 2))
            y1 = max(0, int(center_y - crop_h / 2))
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)
            
            # Safety bounds adjustment
            if x2 - x1 < crop_w:
                if x1 == 0: x2 = min(w, crop_w)
                else: x1 = max(0, w - crop_w)
            if y2 - y1 < crop_h:
                if y1 == 0: y2 = min(h, crop_h)
                else: y1 = max(0, h - crop_h)
                
            frame = frame[y1:y2, x1:x2]
            
        # Apply brightness and contrast if adjusted
        if frame is not None and (self.brightness != 0 or self.contrast != 1.0):
            frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
            
        return frame

    def _run(self):
        """Main capture loop with automatic reconnection."""
        backoff = 1  # seconds
        max_backoff = 30

        while self._running:
            try:
                # Attempt to open the stream
                # Try to parse as integer for local camera index
                try:
                    source = int(self.url)
                except ValueError:
                    source = self.url
                # Prevent silent hanging on RTSP streams by forcing a 5-second timeout
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000|stimeout;5000000|rtsp_transport;tcp"

                self._cap = cv2.VideoCapture(source)

                # Set buffer size to 1 to always get latest frame
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not self._cap.isOpened():
                    raise ConnectionError(f"Cannot open stream: {self.url}")

                self._connected = True
                backoff = 1  # Reset backoff on successful connection
                logger.info(f"[{self.name}] Connected to {self.url}")

                # Read loop
                consecutive_failures = 0
                while self._running and consecutive_failures < 30:
                    ret, frame = self._cap.read()
                    if ret:
                        with self._lock:
                            self._frame = frame
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        time.sleep(0.05)

                if consecutive_failures >= 30:
                    logger.warning(f"[{self.name}] Too many consecutive read failures, reconnecting...")

            except Exception as e:
                logger.error(f"[{self.name}] Stream error: {e}")

            finally:
                self._connected = False
                if self._cap and self._cap.isOpened():
                    self._cap.release()

            # Reconnect with backoff
            if self._running:
                logger.info(f"[{self.name}] Reconnecting in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)


class StreamManager:
    """Manages all stream workers — add, remove, and query streams."""

    def __init__(self):
        self._workers: dict[str, StreamWorker] = {}
        self._lock = threading.Lock()

    def add_stream(self, name: str, url: str, zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0, brightness: int = 0, contrast: float = 1.0) -> bool:
        """Add and start a new stream. Returns False if name already exists."""
        with self._lock:
            if name in self._workers:
                logger.warning(f"Stream '{name}' already exists")
                return False
            worker = StreamWorker(name, url, zoom, pan_x, pan_y, brightness, contrast)
            worker.start()
            self._workers[name] = worker
            return True

    def remove_stream(self, name: str) -> bool:
        """Stop and remove a stream. Returns False if name not found."""
        with self._lock:
            worker = self._workers.pop(name, None)
            if worker:
                worker.stop()
                return True
            return False

    def get_stream_names(self) -> list[str]:
        """Return list of all stream names."""
        with self._lock:
            return list(self._workers.keys())

    def get_frame(self, name: str):
        """Get the latest frame for a specific stream."""
        with self._lock:
            worker = self._workers.get(name)
        if worker:
            return worker.get_latest_frame()
        return None

    def get_all_frames(self) -> dict:
        """Get latest frames from all streams. Returns {name: frame_or_None}."""
        with self._lock:
            names = list(self._workers.keys())
            workers = {n: self._workers[n] for n in names}
        return {name: worker.get_latest_frame() for name, worker in workers.items()}

    def get_stream_status(self, name: str) -> bool:
        """Check if a stream is connected."""
        with self._lock:
            worker = self._workers.get(name)
        return worker.connected if worker else False
        
    def get_stream_settings(self, name: str) -> dict:
        """Get the current zoom, pan, and lighting settings for a stream."""
        with self._lock:
            worker = self._workers.get(name)
            if worker:
                return {
                    "zoom": worker.zoom, 
                    "pan_x": worker.pan_x, 
                    "pan_y": worker.pan_y,
                    "brightness": worker.brightness,
                    "contrast": worker.contrast
                }
            return {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "brightness": 0, "contrast": 1.0}
            
    def update_stream_settings(self, name: str, zoom: float, pan_x: float, pan_y: float, brightness: int = 0, contrast: float = 1.0):
        """Update the settings for a stream on the fly."""
        with self._lock:
            worker = self._workers.get(name)
            if worker:
                worker.zoom = max(1.0, zoom)
                worker.pan_x = max(-1.0, min(1.0, pan_x))
                worker.pan_y = max(-1.0, min(1.0, pan_y))
                worker.brightness = max(-100, min(100, int(brightness)))
                worker.contrast = max(0.1, min(3.0, float(contrast)))

    def get_all_status(self) -> dict[str, bool]:
        """Get connection status for all streams."""
        with self._lock:
            return {name: worker.connected for name, worker in self._workers.items()}

    def stop_all(self):
        """Stop all stream workers."""
        with self._lock:
            for worker in self._workers.values():
                worker.stop()
            self._workers.clear()
        logger.info("All stream workers stopped")
