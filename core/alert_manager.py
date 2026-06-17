"""
Alert Manager — handles fire detection alerts with sound, visual indicators, and frame saving.

The alarm is CONTINUOUS and will not stop until manually acknowledged by the user.
Provides cooldown logic to prevent alert spam and manages an alert history log.
"""

import os
import cv2
import threading
import time
import winsound
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AlertEntry:
    """A single alert log entry."""
    timestamp: datetime
    stream_name: str
    status: str
    confidence: float
    description: str


class AlertManager:
    """
    Manages fire detection alerts — plays sounds, saves frames,
    and maintains an alert history with cooldown logic.

    The alarm siren loops CONTINUOUSLY until acknowledge() is called.
    """

    def __init__(self, alert_cooldown: float = 30.0, save_frames: bool = True,
                 alerts_dir: str = "alerts"):
        self._cooldown = alert_cooldown
        self._save_frames = save_frames
        self._alerts_dir = Path(alerts_dir)
        self._alert_history: list[AlertEntry] = []
        self._last_alert_time: dict[str, float] = {}  # stream_name -> timestamp
        self._lock = threading.Lock()
        self._gui_callback = None  # Set by GUI to receive visual alerts
        self._sound_enabled = True

        # Alarm state — continuous until acknowledged
        self._alarm_active = False
        self._alarm_level = None  # "WARNING" or "FIRE"
        self._alarm_stream = None  # Which stream triggered it
        self._alarm_thread: threading.Thread | None = None
        self._acknowledge_callback = None  # GUI callback when alarm state changes

        # Create alerts directory if saving frames
        if self._save_frames:
            self._alerts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def alarm_active(self) -> bool:
        return self._alarm_active

    @property
    def alarm_level(self) -> str | None:
        return self._alarm_level

    @property
    def cooldown(self) -> float:
        return self._cooldown

    @cooldown.setter
    def cooldown(self, value: float):
        self._cooldown = max(5.0, min(300.0, value))

    @property
    def sound_enabled(self) -> bool:
        return self._sound_enabled

    @sound_enabled.setter
    def sound_enabled(self, value: bool):
        self._sound_enabled = value

    @property
    def alert_history(self) -> list[AlertEntry]:
        with self._lock:
            return list(self._alert_history)

    def set_gui_callback(self, callback):
        """Set callback for GUI visual alerts. Callback receives (stream_name, status, description)."""
        self._gui_callback = callback

    def set_acknowledge_callback(self, callback):
        """Set callback for alarm state changes. Callback receives (active: bool, level: str|None)."""
        self._acknowledge_callback = callback

    def process_detection(self, result, frame=None):
        """
        Process a detection result. Triggers alerts for WARNING/FIRE status
        if cooldown has elapsed for that stream.

        Args:
            result: DetectionResult from the detection engine
            frame: Optional numpy array of the frame that triggered the alert
        """
        # Always log to history
        entry = AlertEntry(
            timestamp=result.timestamp,
            stream_name=result.stream_name,
            status=result.status,
            confidence=result.confidence,
            description=result.description
        )

        with self._lock:
            self._alert_history.append(entry)
            # Keep history to last 500 entries
            if len(self._alert_history) > 500:
                self._alert_history = self._alert_history[-500:]

        # Check cooldown for non-PRINTING states
        if result.status != "PRINTING" and not self._alarm_active:
            now = time.time()
            with self._lock:
                last_alert = self._last_alert_time.get(result.stream_name, 0)
                if now - last_alert < self._cooldown:
                    logger.debug(f"[{result.stream_name}] Event suppressed (cooldown)")
                    return
                self._last_alert_time[result.stream_name] = now

        if result.status != "PRINTING":
            logger.info(
                f"EVENT [{result.status}] on '{result.stream_name}': "
                f"{result.description} (confidence: {result.confidence:.0%})"
            )

        # Save frame for significant events
        if self._save_frames and frame is not None and result.status != "PRINTING":
            self._save_alert_frame(result, frame)

        # Notify GUI (this handles visual badge updates)
        if self._gui_callback:
            try:
                self._gui_callback(result.stream_name, result.status, result.description)
            except Exception as e:
                logger.error(f"GUI alert callback error: {e}")

        # Handle Audio and Escalation based on severity
        if result.status in ("WARNING", "FIRE"):
            # Start or escalate the aggressive continuous alarm
            self._trigger_alarm(result.status, result.stream_name)
        elif result.status == "FAILED" and self._sound_enabled:
            # Play a soft error sound (one-off)
            threading.Thread(target=winsound.MessageBeep, args=(winsound.MB_ICONHAND,), daemon=True).start()
        elif result.status == "DONE" and self._sound_enabled:
            # Play a pleasant success chime (one-off)
            threading.Thread(target=winsound.PlaySound, args=("SystemAsterisk", winsound.SND_ALIAS), daemon=True).start()

    def _trigger_alarm(self, level: str, stream_name: str):
        """Start or escalate the continuous alarm siren."""
        # Escalate WARNING -> FIRE if already alarming at WARNING
        if self._alarm_active and level == "FIRE" and self._alarm_level == "WARNING":
            self._alarm_level = "FIRE"
            logger.warning("ALARM ESCALATED to FIRE")
            return

        # Don't restart if already alarming at same or higher level
        if self._alarm_active:
            return

        self._alarm_active = True
        self._alarm_level = level
        self._alarm_stream = stream_name

        # Notify GUI to show acknowledge button
        if self._acknowledge_callback:
            try:
                self._acknowledge_callback(True, level)
            except Exception as e:
                logger.error(f"Acknowledge callback error: {e}")

        # Start the siren loop
        if self._sound_enabled:
            self._alarm_thread = threading.Thread(
                target=self._siren_loop, daemon=True, name="alarm-siren"
            )
            self._alarm_thread.start()

        logger.warning(f"ALARM ACTIVATED [{level}] — will not stop until acknowledged!")

    def _siren_loop(self):
        """
        Continuous alarm siren that loops until acknowledged.
        FIRE = aggressive high-low siren pattern
        WARNING = pulsing alert tone
        """
        while self._alarm_active and self._sound_enabled:
            try:
                if self._alarm_level == "FIRE":
                    # Aggressive two-tone siren (like a fire alarm)
                    # High tone
                    winsound.Beep(2500, 400)
                    if not self._alarm_active:
                        break
                    # Low tone
                    winsound.Beep(1500, 400)
                    if not self._alarm_active:
                        break
                    # Rapid burst
                    winsound.Beep(3000, 150)
                    if not self._alarm_active:
                        break
                    winsound.Beep(3000, 150)
                    if not self._alarm_active:
                        break
                    time.sleep(0.15)

                elif self._alarm_level == "WARNING":
                    # Pulsing warning tone — less aggressive but still persistent
                    winsound.Beep(1800, 300)
                    if not self._alarm_active:
                        break
                    time.sleep(0.5)
                    winsound.Beep(1800, 300)
                    if not self._alarm_active:
                        break
                    time.sleep(1.0)

            except Exception as e:
                logger.error(f"Siren error: {e}")
                time.sleep(0.5)

    def acknowledge(self):
        """
        Acknowledge and silence the alarm.
        This is the ONLY way to stop the siren.
        """
        was_active = self._alarm_active
        self._alarm_active = False
        self._alarm_level = None
        self._alarm_stream = None

        if was_active:
            logger.info("ALARM ACKNOWLEDGED — siren silenced")

        # Wait for siren thread to finish current beep
        if self._alarm_thread and self._alarm_thread.is_alive():
            self._alarm_thread.join(timeout=2)

        # Notify GUI to hide acknowledge button
        if self._acknowledge_callback:
            try:
                self._acknowledge_callback(False, None)
            except Exception as e:
                logger.error(f"Acknowledge callback error: {e}")

    def _save_alert_frame(self, result, frame):
        """Save the triggering frame to disk."""
        try:
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            safe_name = result.stream_name.replace(" ", "_").replace("/", "_")
            filename = f"{timestamp_str}_{safe_name}_{result.status}.jpg"
            filepath = self._alerts_dir / filename
            cv2.imwrite(str(filepath), frame)
            logger.info(f"Alert frame saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save alert frame: {e}")

    def clear_history(self):
        """Clear the alert history."""
        with self._lock:
            self._alert_history.clear()
            self._last_alert_time.clear()
