"""
Stream Panel — individual stream display widget showing live video feed,
connection status, and detection results.
"""

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np


class StreamPanel(ctk.CTkFrame):
    """
    A single stream display panel containing:
    - Stream name header
    - Live video feed
    - Connection status indicator
    - Last detection result
    - Visual alert (red pulse) on fire detection
    """

    # Status colors
    COLORS = {
        "SAFE": "#0cca4a",       # Legacy mapping
        "PRINTING": "#4a5568",   # Muted grey-blue for normal operation
        "WARNING": "#f59e0b",    # Amber
        "FIRE": "#e94560",       # Red
        "FAILED": "#f97316",     # Orange
        "DONE": "#0cca4a",       # Green
        "DISCONNECTED": "#6b7280",
        "CONNECTED": "#0cca4a",
    }

    def __init__(self, master, stream_name: str, stream_manager=None, on_settings_changed=None, **kwargs):
        super().__init__(
            master,
            corner_radius=12,
            fg_color="#1e1e2e",
            border_width=2,
            border_color="#2a2a3e",
            **kwargs
        )

        self.stream_name = stream_name
        self._stream_manager = stream_manager
        self._on_settings_changed = on_settings_changed
        
        self._alert_active = False
        self._alert_flash_state = False
        self._current_photo = None  # Keep reference to prevent GC
        self._last_status = "SAFE"
        self._settings_popup = None

        self._build_ui()

    def _build_ui(self):
        """Construct the panel UI."""
        # Header row
        header_frame = ctk.CTkFrame(self, fg_color="transparent", height=32)
        header_frame.pack(fill="x", padx=10, pady=(8, 4))
        header_frame.pack_propagate(False)

        # Status dot
        self._status_dot = ctk.CTkLabel(
            header_frame,
            text="●",
            font=("", 14),
            text_color=self.COLORS["DISCONNECTED"],
            width=20
        )
        self._status_dot.pack(side="left")

        # Stream name
        self._name_label = ctk.CTkLabel(
            header_frame,
            text=self.stream_name,
            font=("Segoe UI Semibold", 13),
            text_color="#e0e0e0",
            anchor="w"
        )
        self._name_label.pack(side="left", padx=(4, 0), fill="x", expand=True)

        # Detection status badge
        self._status_badge = ctk.CTkLabel(
            header_frame,
            text="  IDLE  ",
            font=("Segoe UI Bold", 10),
            text_color="#1a1a2e",
            fg_color="#6b7280",
            corner_radius=6,
            height=22
        )
        self._status_badge.pack(side="right")
        
        # Settings button
        self._settings_btn = ctk.CTkButton(
            header_frame,
            text="⚙️",
            width=30,
            height=22,
            font=("", 14),
            fg_color="transparent",
            hover_color="#3a3a4e",
            command=self._open_settings_popup
        )
        self._settings_btn.pack(side="right", padx=(0, 6))

        # Video display area
        self._video_frame = ctk.CTkFrame(self, fg_color="#0d0d15", corner_radius=8)
        self._video_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self._video_label = ctk.CTkLabel(
            self._video_frame,
            text="Connecting...",
            font=("Segoe UI", 12),
            text_color="#6b7280"
        )
        self._video_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Detection info bar
        self._info_label = ctk.CTkLabel(
            self,
            text="Waiting for detection...",
            font=("Segoe UI", 10),
            text_color="#8888aa",
            anchor="w",
            wraplength=350
        )
        self._info_label.pack(fill="x", padx=12, pady=(2, 8))

    def update_frame(self, frame: np.ndarray | None, panel_width: int = 380, panel_height: int = 260):
        """
        Update the video display with a new frame.
        Called from the main thread via .after().
        """
        if frame is None:
            self._video_label.configure(image=None, text="No signal")
            self._status_dot.configure(text_color=self.COLORS["DISCONNECTED"])
            return

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Calculate display size maintaining aspect ratio
            h, w = frame_rgb.shape[:2]
            display_w = max(panel_width - 28, 200)
            display_h = max(panel_height - 90, 150)
            scale = min(display_w / w, display_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            img = Image.fromarray(frame_resized)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))

            self._video_label.configure(image=photo, text="")
            self._current_photo = photo  # Keep reference

            self._status_dot.configure(text_color=self.COLORS["CONNECTED"])

        except Exception as e:
            self._video_label.configure(image=None, text=f"Error: {str(e)[:40]}")

    def update_detection(self, status: str, confidence: float, description: str):
        """Update the detection result display."""
        self._last_status = status

        # Update badge
        color = self.COLORS.get(status, "#6b7280")
        self._status_badge.configure(
            text=f"  {status}  ",
            fg_color=color,
            text_color="#ffffff" if status == "FIRE" else "#1a1a2e"
        )

        # Update info text
        conf_str = f"{confidence:.0%}"
        self._info_label.configure(
            text=f"[{conf_str}] {description}",
            text_color=color
        )

        # Trigger visual alert on FIRE/WARNING or soft borders for others
        if status == "FIRE":
            self._start_alert_flash()
        elif status == "WARNING":
            self.configure(border_color="#f59e0b")
            self._stop_alert_flash()
        elif status == "FAILED":
            self.configure(border_color="#f97316")
            self._stop_alert_flash()
        elif status == "DONE":
            self.configure(border_color="#0cca4a")
            self._stop_alert_flash()
        else:
            self.configure(border_color="#2a2a3e")
            self._stop_alert_flash()

    def set_alert_status(self, active: bool, level: str | None = None):
        """Enable or disable visual alert states."""
        self._alert_active = active
        if active and level:
            self._last_status = level
            self._start_alert_flash()
        else:
            self._stop_alert_flash()

    def _open_settings_popup(self):
        """Open a popup to adjust zoom and pan settings for this camera."""
        if self._settings_popup is not None and self._settings_popup.winfo_exists():
            self._settings_popup.focus()
            return
            
        self._settings_popup = ctk.CTkToplevel(self)
        self._settings_popup.title(f"Settings: {self.stream_name}")
        self._settings_popup.geometry("320x360")
        self._settings_popup.resizable(False, False)
        self._settings_popup.transient(self.winfo_toplevel())
        self._settings_popup.attributes('-topmost', True)
        
        # Get current settings
        settings = {"zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "brightness": 0, "contrast": 1.0}
        if self._stream_manager:
            settings = self._stream_manager.get_stream_settings(self.stream_name)
            
        # UI Elements
        frame = ctk.CTkFrame(self._settings_popup, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Zoom Slider
        ctk.CTkLabel(frame, text="Zoom Level", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self._zoom_slider = ctk.CTkSlider(frame, from_=1.0, to=5.0, number_of_steps=40, command=self._on_settings_slide)
        self._zoom_slider.set(settings["zoom"])
        self._zoom_slider.pack(fill="x", pady=(0, 15))
        
        # Pan X Slider
        ctk.CTkLabel(frame, text="Pan X (Left/Right)", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self._pan_x_slider = ctk.CTkSlider(frame, from_=-1.0, to=1.0, number_of_steps=40, command=self._on_settings_slide)
        self._pan_x_slider.set(settings["pan_x"])
        self._pan_x_slider.pack(fill="x", pady=(0, 15))
        
        # Pan Y Slider
        ctk.CTkLabel(frame, text="Pan Y (Up/Down)", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self._pan_y_slider = ctk.CTkSlider(frame, from_=-1.0, to=1.0, number_of_steps=40, command=self._on_settings_slide)
        self._pan_y_slider.set(settings["pan_y"])
        self._pan_y_slider.pack(fill="x", pady=(0, 15))
        
        # Brightness Slider
        ctk.CTkLabel(frame, text="Brightness", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self._brightness_slider = ctk.CTkSlider(frame, from_=-100, to=100, number_of_steps=100, command=self._on_settings_slide)
        self._brightness_slider.set(settings["brightness"])
        self._brightness_slider.pack(fill="x", pady=(0, 15))

        # Contrast Slider
        ctk.CTkLabel(frame, text="Contrast", font=("Segoe UI Semibold", 12)).pack(anchor="w")
        self._contrast_slider = ctk.CTkSlider(frame, from_=0.1, to=3.0, number_of_steps=58, command=self._on_settings_slide)
        self._contrast_slider.set(settings["contrast"])
        self._contrast_slider.pack(fill="x", pady=(0, 5))
        
        # Bind close event to trigger a save
        self._settings_popup.protocol("WM_DELETE_WINDOW", self._on_settings_close)
        
    def _on_settings_slide(self, _):
        """Called repeatedly while sliders are moving."""
        if self._stream_manager:
            z = self._zoom_slider.get()
            px = self._pan_x_slider.get()
            py = self._pan_y_slider.get()
            b = self._brightness_slider.get()
            c = self._contrast_slider.get()
            self._stream_manager.update_stream_settings(self.stream_name, z, px, py, int(b), float(c))
            
    def _on_settings_close(self):
        """Called when settings popup is closed."""
        if self._on_settings_changed:
            self._on_settings_changed()
        self._settings_popup.destroy()
        self._settings_popup = None

    def _start_alert_flash(self):
        """Start red flashing border animation."""
        if self._alert_active:
            return
        self._alert_active = True
        self._flash_cycle()

    def _stop_alert_flash(self):
        """Stop the flashing animation."""
        self._alert_active = False
        self.configure(border_color="#2a2a3e")

    def _flash_cycle(self):
        """Alternate border color for flash effect."""
        if not self._alert_active:
            return
        self._alert_flash_state = not self._alert_flash_state
        color = "#e94560" if self._alert_flash_state else "#1a1a2e"
        self.configure(border_color=color, border_width=3 if self._alert_flash_state else 2)
        self.after(500, self._flash_cycle)

    def set_disconnected(self):
        """Mark stream as disconnected."""
        self._status_dot.configure(text_color=self.COLORS["DISCONNECTED"])
        self._video_label.configure(image=None, text="Disconnected")
        self._current_photo = None
