"""
Detection Engine — periodically queries Gemma 4 12B via Ollama for fire/smoke detection.

Runs in a background thread, grabbing the latest frame from each stream,
encoding it to base64, and sending it to the local vision model.
"""

import base64
import cv2
import json
import threading
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

import ollama

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of a single fire detection query."""
    stream_name: str
    timestamp: datetime
    status: str  # "PRINTING", "SAFE", "WARNING", "FIRE", "FAILED", "DONE"
    confidence: float
    description: str
    raw_response: str = ""

    @property
    def is_alert(self) -> bool:
        return self.status in ("WARNING", "FIRE")


DETECTION_PROMPT = (
    "You are an AI monitoring system for 3D printers. "
    "Analyze this image from a camera monitoring a 3D printer and determine its status. "
    "Look for fire hazards, print failures, or completed jobs. "
    "IMPORTANT: Do NOT trigger a warning, fire, or failure for normal 3D printer lighting, LED strips, reflections, or black cables/wires hanging behind the printer. "
    "CRITICAL: Due to camera lighting and shadows, normal 3D prints (especially large white plastic structures) may look like shapeless blobs or extrusion failures. Assume these are the intended model, NOT a failure. "
    "Respond ONLY with a JSON object in this exact format: "
    '{"status": "PRINTING", "confidence": 0.95, "description": "brief explanation"} '
    "where status is one of: PRINTING, FAILED, DONE, WARNING, or FIRE. "
    "PRINTING = Normal operation, active printing, no hazards, or printer is safely idle. "
    "FAILED = Print failure, massive stringing, spaghetti monster, or detached model (ONLY IF THERE IS NO SMOKE/FIRE). "
    "DONE = The 3D print is successfully completed and sitting fully printed on the bed with the toolhead parked. "
    "WARNING = Ambiguous conditions (e.g. slight smoke, melting plastic, concerning thermal anomaly). "
    "FIRE = Unambiguous active fire, distinct flames, or heavy thick smoke visible. "
    "CRITICAL HIERARCHY: FIRE > WARNING > FAILED > PRINTING. If you see ANY smoke, sparks, or fire, you MUST output FIRE or WARNING, even if the print has completely failed. A failed print must NEVER overshadow a safety hazard. "
    "confidence is a float from 0.0 to 1.0 indicating how certain you are. "
    "CRITICAL: The 'description' MUST be extremely brief (1 sentence max). "
    "ONLY comment on print quality, safety, and hazards. DO NOT describe the object being printed or yap about irrelevant details. "
    "Respond ONLY with the JSON, no other text."
)


class DetectionEngine:
    """
    Background engine that periodically queries the vision model
    for fire detection on all active streams.
    """

    def __init__(self, stream_manager, model: str = "gemma4:12b", interval: float = 5.0):
        self._stream_manager = stream_manager
        self._model = model
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._cv_thread: threading.Thread | None = None
        self._callbacks: list = []
        self._ollama_available = False
        self._history = {} # stream_name -> [(timestamp, grayscale_frame)]
        self._static_start = {} # stream_name -> timestamp of when it became static
        self._static_durations = {} # stream_name -> current static duration in seconds

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, value: float):
        self._interval = max(1.0, min(30.0, value))

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    @property
    def ollama_available(self) -> bool:
        return self._ollama_available

    def register_callback(self, callback):
        """Register a callback function that receives DetectionResult objects."""
        self._callbacks.append(callback)

    def check_ollama(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models]
            # Check if our target model is available (flexible matching)
            for name in model_names:
                if self._model.replace(":", "") in name.replace(":", "") or \
                   name.replace(":", "").startswith(self._model.split(":")[0]):
                    self._ollama_available = True
                    logger.info(f"Ollama is available, found model: {name}")
                    return True
            logger.warning(f"Ollama is running but model '{self._model}' not found. Available: {model_names}")
            self._ollama_available = False
            return False
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            self._ollama_available = False
            return False

    def start(self):
        """Start the detection loops."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="detection-engine")
        self._cv_thread = threading.Thread(target=self._run_cv, daemon=True, name="cv-engine")
        self._thread.start()
        self._cv_thread.start()
        logger.info(f"Detection engine started (interval={self._interval}s, model={self._model})")

    def stop(self):
        """Stop the detection loops."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        if self._cv_thread and self._cv_thread.is_alive():
            self._cv_thread.join(timeout=10)
        logger.info("Detection engine stopped")

    def _run_cv(self):
        """Dedicated high-frequency loop for tracking motion."""
        while self._running:
            try:
                frames = self._stream_manager.get_all_frames()
                for stream_name, frame in frames.items():
                    if not self._running:
                        break
                    if frame is not None:
                        self._check_motion(stream_name, frame)
            except Exception as e:
                logger.error(f"CV cycle error: {e}")
            
            # Run motion tracking exactly every 1 second like debug_motion.py
            time.sleep(1.0)

    def _run(self):
        """Main detection loop."""
        while self._running:
            cycle_start = time.time()

            try:
                frames = self._stream_manager.get_all_frames()
                for stream_name, frame in frames.items():
                    if not self._running:
                        break
                    if frame is None:
                        continue

                    result = self._analyze_frame(stream_name, frame)
                    if result:
                        self._dispatch_result(result)

            except Exception as e:
                logger.error(f"Detection cycle error: {e}")

            # Sleep for remaining interval time
            elapsed = time.time() - cycle_start
            sleep_time = max(0.1, self._interval - elapsed)
            # Use small sleep increments so we can stop quickly
            end_time = time.time() + sleep_time
            while self._running and time.time() < end_time:
                time.sleep(0.2)

    def _check_motion(self, stream_name: str, current_frame) -> float:
        """
        Check if the print bed/toolhead is moving using MSE.
        Returns the duration (in seconds) that the stream has been continuously static.
        """
        now = time.time()
        
        # Convert to grayscale and resize to 512x512 to preserve small toolhead details
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (512, 512))
        # Minimal blur to reduce sensor noise without destroying motion
        small_gray = cv2.GaussianBlur(small_gray, (3, 3), 0)
        
        if stream_name not in self._history:
            self._history[stream_name] = []
            self._static_start[stream_name] = now
            
        history = self._history[stream_name]
        
        # Prune frames older than 15 seconds to avoid slow lighting drift triggering motion
        history = [h for h in history if now - h[0] < 15]
        
        # Compare current frame against ALL frames in the recent history buffer
        # This prevents the "unlucky sync" problem where the toolhead returns to the 
        # exact same spot every 5 seconds.
        motion_detected = False
        max_changed = 0
        
        for _, old_frame in history:
            diff = cv2.absdiff(small_gray, old_frame)
            _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            changed_pixels = cv2.countNonZero(thresh)
            
            if changed_pixels > max_changed:
                max_changed = changed_pixels
                
            # If > 500 pixels (out of 262,144) changed, it's definitively motion
            # (Timestamps and text overlays usually change 100-200 pixels)
            if changed_pixels > 500:
                motion_detected = True
                break
                
        if motion_detected:
            # Motion detected! Reset the static streak
            self._static_start[stream_name] = now
            logger.debug(f"[{stream_name}] Motion detected (max changed pixels: {max_changed}). Resetting streak.")
        else:
            logger.debug(f"[{stream_name}] Scene static (max changed pixels: {max_changed})")
            
        # Store current frame
        history.append((now, small_gray))
        self._history[stream_name] = history
        
        static_duration = now - self._static_start[stream_name]
        self._static_durations[stream_name] = static_duration

    def _analyze_frame(self, stream_name: str, frame) -> DetectionResult | None:
        """Encode a frame and query the vision model."""
        try:
            # Read the latest static streak duration from the dedicated CV thread
            static_duration = self._static_durations.get(stream_name, 0.0)

            # Resize frame for faster inference (keep aspect ratio, max 768px wide)
            h, w = frame.shape[:2]
            if w > 768:
                scale = 768 / w
                frame = cv2.resize(frame, (768, int(h * scale)))

            # Encode to JPEG then base64
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_b64 = base64.b64encode(buffer).decode("utf-8")

            # Query Ollama
            response = ollama.generate(
                model=self._model,
                prompt=DETECTION_PROMPT,
                images=[image_b64],
            )

            raw_text = response.get("response", "").strip()
            result = self._parse_response(stream_name, raw_text)
            
            # CV Override Logic
            if static_duration >= 30.0 and result.status == "PRINTING":
                logger.info(f"[{stream_name}] CV OVERRIDE: Printer idle for {static_duration:.0f}s. Changing PRINTING to DONE.")
                result.status = "DONE"
                result.description = f"CV Override: Printer has been perfectly static for {int(static_duration)} seconds."
            elif result.status == "DONE" and static_duration < 30.0:
                logger.info(f"[{stream_name}] LLM returned DONE, but idle time is only {static_duration:.0f}s. Downgrading to PRINTING.")
                result.status = "PRINTING"
                
            return result

        except Exception as e:
            logger.error(f"[{stream_name}] Analysis error: {e}")
            return DetectionResult(
                stream_name=stream_name,
                timestamp=datetime.now(),
                status="SAFE",
                confidence=0.0,
                description=f"Analysis error: {str(e)[:100]}",
                raw_response=""
            )

    def _parse_response(self, stream_name: str, raw_text: str) -> DetectionResult:
        """Parse the model's JSON response into a DetectionResult."""
        try:
            # Try to extract JSON from the response (model might add extra text)
            json_start = raw_text.find("{")
            json_end = raw_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON object found in response")

            status = data.get("status", "PRINTING").upper()
            if status == "SAFE":
                status = "PRINTING"  # Map legacy SAFE to PRINTING
            if status not in ("PRINTING", "WARNING", "FIRE", "FAILED", "DONE"):
                status = "PRINTING"

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            description = data.get("description", "No description")

            return DetectionResult(
                stream_name=stream_name,
                timestamp=datetime.now(),
                status=status,
                confidence=confidence,
                description=description,
                raw_response=raw_text
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"[{stream_name}] Failed to parse response: {e}\nRaw: {raw_text[:200]}")
            return DetectionResult(
                stream_name=stream_name,
                timestamp=datetime.now(),
                status="SAFE",
                confidence=0.0,
                description=f"Parse error — raw: {raw_text[:80]}",
                raw_response=raw_text
            )

    def _dispatch_result(self, result: DetectionResult):
        """Send result to all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
