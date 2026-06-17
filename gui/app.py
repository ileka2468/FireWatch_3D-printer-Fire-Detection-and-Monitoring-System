"""
Main Application Window — the primary GUI for the 3D Print Fire Detection System.

Manages stream panels in a dynamic grid, provides controls for adding/removing
streams, configuring detection settings, and displays the alert log.
"""

import customtkinter as ctk
import cv2
import math
import logging
import threading
import os
from pathlib import Path
from tkinter import messagebox, filedialog

from gui.stream_panel import StreamPanel
from gui.alert_log import AlertLog
from core.stream_manager import StreamManager
from core.detection_engine import DetectionEngine, DetectionResult
from core.alert_manager import AlertManager
from core import config_manager
from core.config_manager import get_app_dir, get_resource_path
from web import server as web_server
import socket

logger = logging.getLogger(__name__)


class FireDetectionApp(ctk.CTk):
    """Main application window for the Fire Detection System."""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("🔥 3D Print Fire Detection System")
        self.geometry("1280x800")
        self.minsize(900, 600)

        # Set window icon
        try:
            self.iconbitmap(str(get_resource_path("icon.ico")))
        except Exception as e:
            logger.warning(f"Could not load icon: {e}")

        # Set dark theme colors
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Configure window background
        self.configure(fg_color="#0f0f1a")

        # Load config
        self._config = self._load_config()

        # Initialize core components
        self._stream_manager = StreamManager()
        self._alert_manager = AlertManager(
            alert_cooldown=self._config.get("alert_cooldown", 30),
            save_frames=self._config.get("save_alert_frames", True)
        )
        self._detection_engine = DetectionEngine(
            stream_manager=self._stream_manager,
            model=self._config.get("model", "gemma4:12b"),
            interval=self._config.get("detection_interval", 5)
        )

        # Register detection callback
        self._detection_engine.register_callback(self._on_detection_result)

        # Web dashboard state
        self._web_server_thread = None
        self._dashboard_enabled = self._config.get("enable_dashboard", False)
        self._dashboard_port = self._config.get("dashboard_port", 5050)
        
        # Start web dashboard if enabled
        if self._dashboard_enabled:
            self._start_web_server()

        # Stream panels dict
        self._stream_panels: dict[str, StreamPanel] = {}
        self._monitoring = False
        self._topbar_flash_active = False

        # Resize lag mitigation
        self._is_resizing = False
        self._resize_timer = None
        self.bind("<Configure>", self._on_configure)

        # Build UI
        self._build_ui()

        # Register alarm state callback (after UI is built so button exists)
        self._alert_manager.set_acknowledge_callback(self._on_alarm_state_change)

        # Check Ollama status
        self._check_ollama_status()

        # Load saved streams
        self._load_saved_streams()

        # Start frame update loop
        self._update_frames()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_configure(self, event):
        """Detect window resize/drag to pause heavy video updates."""
        if str(getattr(event, "widget", "")) == str(self):
            self._is_resizing = True
            if self._resize_timer is not None:
                self.after_cancel(self._resize_timer)
            # Resume updates 150ms after the last configure event
            self._resize_timer = self.after(150, self._on_resize_end)

    def _on_resize_end(self):
        self._is_resizing = False

    def _load_config(self) -> dict:
        """Load configuration from INI file."""
        return config_manager.load_config()

    def _save_config(self):
        """Save current configuration to INI file."""
        try:
            streams = []
            for name in self._stream_manager.get_stream_names():
                worker = self._stream_manager._workers.get(name)
                if worker:
                    streams.append({
                        "name": name, 
                        "url": worker.url,
                        "zoom": worker.zoom,
                        "pan_x": worker.pan_x,
                        "pan_y": worker.pan_y,
                        "brightness": worker.brightness,
                        "contrast": worker.contrast
                    })

            # Also include test panels' source info? No — test panels are ephemeral.
            # Filter out test panels
            streams = [s for s in streams if not s["name"].startswith("[TEST] ")]

            config_manager.save_config(
                streams=streams,
                detection_interval=int(self._detection_engine.interval),
                alert_cooldown=int(self._alert_manager.cooldown),
                model=self._detection_engine.model,
                save_alert_frames=self._config.get("save_alert_frames", True),
                enable_dashboard=self._dashboard_enabled,
                dashboard_port=self._dashboard_port
            )
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _build_ui(self):
        """Construct the main application layout."""
        # ─── Top Bar ─────────────────────────────────────────
        self._build_top_bar()

        # ─── Main Content Area ───────────────────────────────
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Left panel — stream grid (70%)
        self._grid_container = ctk.CTkFrame(content_frame, fg_color="transparent")
        self._grid_container.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Placeholder when no streams
        self._empty_state = ctk.CTkFrame(self._grid_container, fg_color="#13131d", corner_radius=16)
        self._empty_state.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._grid_container.rowconfigure(0, weight=1)
        self._grid_container.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self._empty_state,
            text="📷",
            font=("", 48),
        ).pack(pady=(80, 10))

        ctk.CTkLabel(
            self._empty_state,
            text="No Camera Streams",
            font=("Segoe UI Semibold", 20),
            text_color="#6b7280"
        ).pack()

        ctk.CTkLabel(
            self._empty_state,
            text="Add a camera stream using the panel on the right\nto start monitoring your 3D printers.",
            font=("Segoe UI", 13),
            text_color="#4a4a5e",
            justify="center"
        ).pack(pady=(8, 0))

        # Right panel — controls & log (30%, fixed width)
        right_panel = ctk.CTkFrame(content_frame, fg_color="transparent", width=340)
        right_panel.pack(side="right", fill="y", padx=(6, 0))
        right_panel.pack_propagate(False)

        self._build_stream_controls(right_panel)
        self._build_settings_panel(right_panel)

        # Alert log (fills remaining space)
        self._alert_log = AlertLog(right_panel)
        self._alert_log.pack(fill="both", expand=True, pady=(8, 0))

    def _build_top_bar(self):
        """Build the top navigation/status bar."""
        top_bar = ctk.CTkFrame(self, fg_color="#13131d", height=56, corner_radius=0)
        top_bar.pack(fill="x", padx=0, pady=0)
        top_bar.pack_propagate(False)

        # App title
        title_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        title_frame.pack(side="left", padx=16)

        ctk.CTkLabel(
            title_frame,
            text="🔥 FireWatch",
            font=("Segoe UI Bold", 20),
            text_color="#ffffff"
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame,
            text="  3D Print Fire Detection",
            font=("Segoe UI", 13),
            text_color="#6b7280"
        ).pack(side="left", pady=(3, 0))

        # Right side — status and controls
        right_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        right_frame.pack(side="right", padx=16)

        # Ollama status
        self._ollama_status_frame = ctk.CTkFrame(right_frame, fg_color="#1a1a2e", corner_radius=8)
        self._ollama_status_frame.pack(side="left", padx=(0, 12))

        self._ollama_dot = ctk.CTkLabel(
            self._ollama_status_frame,
            text="●",
            font=("", 12),
            text_color="#6b7280",
            width=20
        )
        self._ollama_dot.pack(side="left", padx=(8, 2))

        self._ollama_label = ctk.CTkLabel(
            self._ollama_status_frame,
            text="Checking Ollama...",
            font=("Segoe UI", 11),
            text_color="#8888aa"
        )
        self._ollama_label.pack(side="left", padx=(0, 8), pady=4)

        # Test Image button
        self._test_btn = ctk.CTkButton(
            right_frame,
            text="\U0001f9ea  Test Image",
            font=("Segoe UI Semibold", 13),
            fg_color="#7c3aed",
            hover_color="#6d28d9",
            height=36,
            width=140,
            corner_radius=8,
            command=self._test_image
        )
        self._test_btn.pack(side="left", padx=(0, 8))

        # Start/Stop monitoring button
        self._monitor_btn = ctk.CTkButton(
            right_frame,
            text="\u25b6  Start Monitoring",
            font=("Segoe UI Semibold", 13),
            fg_color="#0f3460",
            hover_color="#1a4a80",
            height=36,
            width=180,
            corner_radius=8,
            command=self._toggle_monitoring
        )
        self._monitor_btn.pack(side="left")

        # Acknowledge alarm button — hidden until alarm fires
        self._ack_btn = ctk.CTkButton(
            right_frame,
            text="\U0001f6a8  ACKNOWLEDGE ALARM",
            font=("Segoe UI Bold", 14),
            fg_color="#dc2626",
            hover_color="#b91c1c",
            text_color="#ffffff",
            height=36,
            width=220,
            corner_radius=8,
            command=self._acknowledge_alarm
        )
        # Hidden by default — pack_forget'd
        self._ack_btn.pack_forget()

    def _build_stream_controls(self, parent):
        """Build the stream add/remove controls."""
        stream_frame = ctk.CTkFrame(parent, fg_color="#13131d", corner_radius=12)
        stream_frame.pack(fill="x")

        ctk.CTkLabel(
            stream_frame,
            text="Camera Streams",
            font=("Segoe UI Semibold", 14),
            text_color="#c0c0d0"
        ).pack(anchor="w", padx=12, pady=(10, 6))

        # Stream name input
        ctk.CTkLabel(
            stream_frame,
            text="Name",
            font=("Segoe UI", 10),
            text_color="#6b7280"
        ).pack(anchor="w", padx=12, pady=(0, 2))

        self._name_entry = ctk.CTkEntry(
            stream_frame,
            placeholder_text="e.g. Printer 1",
            font=("Segoe UI", 12),
            height=32,
            fg_color="#1a1a2e",
            border_color="#2a2a3e",
            corner_radius=6
        )
        self._name_entry.pack(fill="x", padx=12, pady=(0, 6))

        # Stream URL input
        ctk.CTkLabel(
            stream_frame,
            text="URL (RTSP, HTTP/MJPEG, or device index)",
            font=("Segoe UI", 10),
            text_color="#6b7280"
        ).pack(anchor="w", padx=12, pady=(0, 2))

        self._url_entry = ctk.CTkEntry(
            stream_frame,
            placeholder_text="rtsp://192.168.1.100:554/stream",
            font=("Segoe UI", 12),
            height=32,
            fg_color="#1a1a2e",
            border_color="#2a2a3e",
            corner_radius=6
        )
        self._url_entry.pack(fill="x", padx=12, pady=(0, 8))

        # Buttons row
        btn_row = ctk.CTkFrame(stream_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkButton(
            btn_row,
            text="+ Add Stream",
            font=("Segoe UI Semibold", 12),
            fg_color="#0cca4a",
            hover_color="#0aa03a",
            text_color="#0f0f1a",
            height=32,
            corner_radius=6,
            command=self._add_stream
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        ctk.CTkButton(
            btn_row,
            text="− Remove",
            font=("Segoe UI Semibold", 12),
            fg_color="#e94560",
            hover_color="#c9354e",
            text_color="#ffffff",
            height=32,
            corner_radius=6,
            command=self._remove_stream_dialog
        ).pack(side="right", expand=True, fill="x", padx=(4, 0))

        # Active streams list
        self._streams_list_frame = ctk.CTkFrame(stream_frame, fg_color="transparent")
        self._streams_list_frame.pack(fill="x", padx=12, pady=(0, 10))

    def _build_settings_panel(self, parent):
        """Build the detection settings controls."""
        settings_frame = ctk.CTkFrame(parent, fg_color="#13131d", corner_radius=12)
        settings_frame.pack(fill="x", pady=(8, 0))

        ctk.CTkLabel(
            settings_frame,
            text="Detection Settings",
            font=("Segoe UI Semibold", 14),
            text_color="#c0c0d0"
        ).pack(anchor="w", padx=12, pady=(10, 6))

        # Interval slider
        interval_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        interval_row.pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(
            interval_row,
            text="Check Interval",
            font=("Segoe UI", 11),
            text_color="#8888aa"
        ).pack(side="left")

        self._interval_value_label = ctk.CTkLabel(
            interval_row,
            text=f"{self._config.get('detection_interval', 5)}s",
            font=("Segoe UI Bold", 11),
            text_color="#0cca4a"
        )
        self._interval_value_label.pack(side="right")

        self._interval_slider = ctk.CTkSlider(
            settings_frame,
            from_=2,
            to=15,
            number_of_steps=13,
            fg_color="#1a1a2e",
            progress_color="#0f3460",
            button_color="#0cca4a",
            button_hover_color="#0aa03a",
            command=self._on_interval_change
        )
        self._interval_slider.set(self._config.get("detection_interval", 5))
        self._interval_slider.pack(fill="x", padx=12, pady=(0, 6))

        # Alert cooldown slider
        cooldown_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        cooldown_row.pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(
            cooldown_row,
            text="Alert Cooldown",
            font=("Segoe UI", 11),
            text_color="#8888aa"
        ).pack(side="left")

        self._cooldown_value_label = ctk.CTkLabel(
            cooldown_row,
            text=f"{self._config.get('alert_cooldown', 30)}s",
            font=("Segoe UI Bold", 11),
            text_color="#0cca4a"
        )
        self._cooldown_value_label.pack(side="right")

        self._cooldown_slider = ctk.CTkSlider(
            settings_frame,
            from_=5,
            to=120,
            number_of_steps=23,
            fg_color="#1a1a2e",
            progress_color="#0f3460",
            button_color="#0cca4a",
            button_hover_color="#0aa03a",
            command=self._on_cooldown_change
        )
        self._cooldown_slider.set(self._config.get("alert_cooldown", 30))
        self._cooldown_slider.pack(fill="x", padx=12, pady=(0, 6))

        # Sound toggle
        sound_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        sound_row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(
            sound_row,
            text="Sound Alerts",
            font=("Segoe UI", 11),
            text_color="#8888aa"
        ).pack(side="left")

        self._sound_switch = ctk.CTkSwitch(
            sound_row,
            text="",
            width=44,
            fg_color="#2a2a3e",
            progress_color="#0cca4a",
            button_color="#e0e0e0",
            command=self._on_sound_toggle
        )
        self._sound_switch.select()  # On by default
        self._sound_switch.pack(side="right")
        
        # Dashboard toggle
        dash_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
        dash_row.pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(
            dash_row,
            text="Web Dashboard",
            font=("Segoe UI", 11),
            text_color="#8888aa"
        ).pack(side="left")

        self._dash_switch = ctk.CTkSwitch(
            dash_row,
            text="",
            width=44,
            fg_color="#2a2a3e",
            progress_color="#0cca4a",
            button_color="#e0e0e0",
            command=self._on_dash_toggle
        )
        if self._dashboard_enabled:
            self._dash_switch.select()
        else:
            self._dash_switch.deselect()
        self._dash_switch.pack(side="right")
        
        # Dashboard Info Label
        self._dash_info_label = ctk.CTkLabel(
            settings_frame,
            text="",
            font=("Segoe UI", 10),
            text_color="#0cca4a"
        )
        self._dash_info_label.pack(fill="x", padx=12, pady=(0, 5))
        self._update_dash_info()

    def _update_dash_info(self):
        if self._dashboard_enabled:
            try:
                ip = socket.gethostbyname(socket.gethostname())
                self._dash_info_label.configure(text=f"Live at: http://{ip}:{self._dashboard_port}")
            except Exception:
                self._dash_info_label.configure(text=f"Live at: port {self._dashboard_port}")
        else:
            self._dash_info_label.configure(text="")
            
    def _on_dash_toggle(self):
        self._dashboard_enabled = self._dash_switch.get() == 1
        if self._dashboard_enabled:
            self._start_web_server()
        self._update_dash_info()
        self._save_config()
        
    def _start_web_server(self):
        if not self._web_server_thread or not self._web_server_thread.is_alive():
            try:
                self._web_server_thread = web_server.start_server(
                    self._stream_manager, 
                    self._alert_manager, 
                    self._dashboard_port
                )
            except Exception as e:
                logger.error(f"Failed to start web server: {e}")

    # ─── Test Mode ────────────────────────────────────────────

    def _test_image(self):
        """Open a file dialog, pick an image, show it in a test panel, and run detection."""
        test_images_dir = str(get_resource_path("test_images"))
        filepath = filedialog.askopenfilename(
            title="Select Test Image",
            initialdir=test_images_dir,
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        if not filepath:
            return

        # Read the image with OpenCV
        frame = cv2.imread(filepath)
        if frame is None:
            messagebox.showerror("Error", f"Could not read image:\n{filepath}")
            return

        # Create or reuse the test panel
        test_name = "[TEST] " + Path(filepath).stem

        # Remove previous test panel if it exists
        for name in list(self._stream_panels.keys()):
            if name.startswith("[TEST] "):
                old_panel = self._stream_panels.pop(name)
                old_panel.destroy()

        # Create test panel and add to grid
        panel = StreamPanel(
            self._grid_container, 
            stream_name=test_name, 
            stream_manager=self._stream_manager, 
            on_settings_changed=self._save_config
        )
        self._stream_panels[test_name] = panel
        self._relayout_grid()

        # Show the image in the panel
        try:
            pw = panel.winfo_reqwidth() or 500
            ph = panel.winfo_reqheight() or 400
        except Exception:
            pw, ph = 500, 400
        panel.update_frame(frame, panel_width=pw, panel_height=ph)

        # Disable button while analyzing
        self._test_btn.configure(state="disabled", text="Analyzing...")

        # Run detection in background thread
        def run_analysis():
            result = self._detection_engine._analyze_frame(test_name, frame)
            # Back to main thread
            self.after(0, lambda: self._on_test_result(result, frame, test_name))

        threading.Thread(target=run_analysis, daemon=True).start()

    def _on_test_result(self, result, frame, test_name: str):
        """Handle test image detection result on the main thread."""
        # Re-enable button
        self._test_btn.configure(state="normal", text="\U0001f9ea  Test Image")

        if result is None:
            messagebox.showerror(
                "Analysis Failed",
                "Could not get a response from Ollama.\n"
                "Make sure Ollama is running with the model loaded."
            )
            return

        # Update the test panel with detection result
        panel = self._stream_panels.get(test_name)
        if panel:
            panel.update_detection(result.status, result.confidence, result.description)

        # Process through alert manager (triggers sound + saves frame)
        self._alert_manager.process_detection(result, frame)

        # Log to alert log
        self._alert_log.add_entry(
            stream_name=result.stream_name,
            status=result.status,
            confidence=result.confidence,
            description=result.description,
            timestamp=result.timestamp
        )

        # Show result dialog
        status_emoji = {"SAFE": "\u2705", "WARNING": "\u26a0\ufe0f", "FIRE": "\U0001f525"}
        icon = status_emoji.get(result.status, "?")
        msg = (
            f"Status: {icon} {result.status}\n"
            f"Confidence: {result.confidence:.0%}\n\n"
            f"{result.description}"
        )

        if result.status == "FIRE":
            messagebox.showwarning("FIRE DETECTED", msg)
        elif result.status == "WARNING":
            messagebox.showwarning("WARNING", msg)
        else:
            messagebox.showinfo("All Clear", msg)

    # ─── Alarm Acknowledge ────────────────────────────────────

    def _on_alarm_state_change(self, active: bool, level: str | None):
        """Called from AlertManager when alarm state changes. May be called from background thread."""
        self.after(0, lambda: self._update_alarm_ui(active, level))

    def _update_alarm_ui(self, active: bool, level: str | None):
        """Update UI elements based on alarm state (runs on main thread)."""
        if active:
            # Show the acknowledge button
            self._ack_btn.pack(side="left", padx=(12, 0))
            # Start top bar flashing
            self._topbar_flash_active = True
            self._flash_topbar()
        else:
            # Hide the acknowledge button
            self._ack_btn.pack_forget()
            # Stop top bar flashing
            self._topbar_flash_active = False

    def _acknowledge_alarm(self):
        """User clicked the acknowledge button — silence the alarm."""
        self._alert_manager.acknowledge()
        # Reset all stream panel flash states
        for panel in self._stream_panels.values():
            panel._stop_alert_flash()

    def _flash_topbar(self):
        """Flash the top bar red during active alarm."""
        if not self._topbar_flash_active:
            # Restore normal color
            # Find the top bar widget (first child frame)
            for child in self.winfo_children():
                if isinstance(child, ctk.CTkFrame) and child.cget("height") == 56:
                    child.configure(fg_color="#13131d")
                    break
            return

        # Toggle between red and dark
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkFrame) and child.cget("height") == 56:
                current = child.cget("fg_color")
                # Toggle
                if current == "#13131d" or current == ("#13131d", "#13131d"):
                    child.configure(fg_color="#5c0a0a")
                else:
                    child.configure(fg_color="#13131d")
                break

        self.after(600, self._flash_topbar)

    # ─── Actions ─────────────────────────────────────────────

    def _add_stream(self):
        """Add a new camera stream."""
        name = self._name_entry.get().strip()
        url = self._url_entry.get().strip()

        if not name:
            messagebox.showwarning("Missing Name", "Please enter a name for the stream.")
            return
        if not url:
            messagebox.showwarning("Missing URL", "Please enter a stream URL or device index.")
            return

        if not self._stream_manager.add_stream(name, url):
            messagebox.showwarning("Duplicate", f"A stream named '{name}' already exists.")
            return

        # Create stream panel
        panel = StreamPanel(
            self._grid_container, 
            stream_name=name, 
            stream_manager=self._stream_manager, 
            on_settings_changed=self._save_config
        )
        self._stream_panels[name] = panel
        self._relayout_grid()

        # Update streams list
        self._update_streams_list()

        # Clear inputs
        self._name_entry.delete(0, "end")
        self._url_entry.delete(0, "end")

        # Save config
        self._save_config()

        logger.info(f"Added stream: {name} -> {url}")

    def _remove_stream_dialog(self):
        """Show dialog to select and remove a stream."""
        names = self._stream_manager.get_stream_names()
        if not names:
            messagebox.showinfo("No Streams", "No streams to remove.")
            return

        # Create a simple removal dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Remove Stream")
        dialog.geometry("300x200")
        dialog.configure(fg_color="#0f0f1a")
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(
            dialog,
            text="Select stream to remove:",
            font=("Segoe UI", 13),
            text_color="#c0c0d0"
        ).pack(pady=(16, 8))

        selected = ctk.StringVar(value=names[0])
        dropdown = ctk.CTkOptionMenu(
            dialog,
            values=names,
            variable=selected,
            fg_color="#1a1a2e",
            button_color="#2a2a3e",
            button_hover_color="#3a3a4e",
            dropdown_fg_color="#1a1a2e",
            dropdown_hover_color="#2a2a3e",
            font=("Segoe UI", 12)
        )
        dropdown.pack(padx=20, fill="x")

        def do_remove():
            name = selected.get()
            self._stream_manager.remove_stream(name)
            panel = self._stream_panels.pop(name, None)
            if panel:
                panel.destroy()
            self._relayout_grid()
            self._update_streams_list()
            self._save_config()
            dialog.destroy()
            logger.info(f"Removed stream: {name}")

        ctk.CTkButton(
            dialog,
            text="Remove",
            font=("Segoe UI Semibold", 13),
            fg_color="#e94560",
            hover_color="#c9354e",
            height=36,
            corner_radius=8,
            command=do_remove
        ).pack(pady=16)

    def _toggle_monitoring(self):
        """Start or stop the detection engine."""
        if self._monitoring:
            self._detection_engine.stop()
            self._monitoring = False
            self._monitor_btn.configure(
                text="▶  Start Monitoring",
                fg_color="#0f3460",
                hover_color="#1a4a80"
            )
            logger.info("Monitoring stopped")
        else:
            if not self._detection_engine.ollama_available:
                if not self._detection_engine.check_ollama():
                    messagebox.showerror(
                        "Ollama Unavailable",
                        f"Cannot connect to Ollama or model '{self._detection_engine.model}' "
                        "is not available.\n\n"
                        "Make sure Ollama is running and the model is pulled:\n"
                        f"  ollama pull {self._detection_engine.model}"
                    )
                    return

            names = self._stream_manager.get_stream_names()
            if not names:
                messagebox.showwarning(
                    "No Streams",
                    "Add at least one camera stream before starting monitoring."
                )
                return

            self._detection_engine.start()
            self._monitoring = True
            self._monitor_btn.configure(
                text="⏹  Stop Monitoring",
                fg_color="#e94560",
                hover_color="#c9354e"
            )
            logger.info("Monitoring started")

    def _on_interval_change(self, value):
        """Handle interval slider change."""
        val = int(round(value))
        self._interval_value_label.configure(text=f"{val}s")
        self._detection_engine.interval = val

    def _on_cooldown_change(self, value):
        """Handle cooldown slider change."""
        val = int(round(value))
        self._cooldown_value_label.configure(text=f"{val}s")
        self._alert_manager.cooldown = val

    def _on_sound_toggle(self):
        """Handle sound toggle."""
        self._alert_manager.sound_enabled = self._sound_switch.get() == 1

    def _check_ollama_status(self):
        """Check Ollama connectivity and update status indicator."""
        import threading

        def check():
            available = self._detection_engine.check_ollama()
            # Update UI from main thread
            self.after(0, lambda: self._update_ollama_ui(available))

        threading.Thread(target=check, daemon=True).start()

    def _update_ollama_ui(self, available: bool):
        """Update the Ollama status indicator."""
        if available:
            self._ollama_dot.configure(text_color="#0cca4a")
            self._ollama_label.configure(
                text=f"Ollama ✓ ({self._detection_engine.model})",
                text_color="#0cca4a"
            )
        else:
            self._ollama_dot.configure(text_color="#e94560")
            self._ollama_label.configure(
                text="Ollama ✗ — Not connected",
                text_color="#e94560"
            )

    def _load_saved_streams(self):
        """Load streams from config."""
        for stream_info in self._config.get("streams", []):
            name = stream_info.get("name", "")
            url = stream_info.get("url", "")
            zoom = stream_info.get("zoom", 1.0)
            pan_x = stream_info.get("pan_x", 0.0)
            pan_y = stream_info.get("pan_y", 0.0)
            brightness = stream_info.get("brightness", 0)
            contrast = stream_info.get("contrast", 1.0)
            if name and url:
                self._stream_manager.add_stream(name, url, zoom, pan_x, pan_y, brightness, contrast)
                panel = StreamPanel(
                    self._grid_container, 
                    stream_name=name, 
                    stream_manager=self._stream_manager, 
                    on_settings_changed=self._save_config
                )
                self._stream_panels[name] = panel

        if self._stream_panels:
            self._relayout_grid()
            self._update_streams_list()

    def _relayout_grid(self):
        """Re-layout stream panels in a responsive grid."""
        panels = list(self._stream_panels.values())
        n = len(panels)

        # Hide/show empty state
        if n == 0:
            self._empty_state.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            self._grid_container.rowconfigure(0, weight=1)
            self._grid_container.columnconfigure(0, weight=1)
            return
        else:
            self._empty_state.grid_forget()

        # Remove all panels from grid first
        for panel in panels:
            panel.grid_forget()

        # Calculate grid dimensions
        if n == 1:
            cols = 1
        elif n <= 4:
            cols = 2
        elif n <= 9:
            cols = 3
        else:
            cols = 4
        rows = math.ceil(n / cols)

        # Configure grid weights
        for i in range(cols):
            self._grid_container.columnconfigure(i, weight=1)
        for i in range(rows):
            self._grid_container.rowconfigure(i, weight=1)

        # Place panels
        for idx, panel in enumerate(panels):
            r = idx // cols
            c = idx % cols
            panel.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

    def _update_streams_list(self):
        """Update the active streams list display."""
        for child in self._streams_list_frame.winfo_children():
            child.destroy()

        names = self._stream_manager.get_stream_names()
        for name in names:
            connected = self._stream_manager.get_stream_status(name)
            dot_color = "#0cca4a" if connected else "#6b7280"

            row = ctk.CTkFrame(self._streams_list_frame, fg_color="transparent", height=22)
            row.pack(fill="x", pady=1)
            row.pack_propagate(False)

            ctk.CTkLabel(
                row,
                text="●",
                font=("", 10),
                text_color=dot_color,
                width=16
            ).pack(side="left")

            ctk.CTkLabel(
                row,
                text=name,
                font=("Segoe UI", 11),
                text_color="#b0b0c0",
                anchor="w"
            ).pack(side="left", padx=(2, 0))

    # ─── Frame Update Loop ───────────────────────────────────

    def _update_frames(self):
        """Periodically update all stream panels with latest frames."""
        # Prevent infinite mirror / parallax lag by pausing updates during drag/resize
        if getattr(self, '_is_resizing', False):
            self.after(66, self._update_frames)
            return

        for name, panel in self._stream_panels.items():
            frame = self._stream_manager.get_frame(name)
            # Calculate panel size based on grid
            try:
                pw = panel.winfo_width()
                ph = panel.winfo_height()
            except Exception:
                pw, ph = 380, 260
            panel.update_frame(frame, panel_width=pw, panel_height=ph)

        # Update connection status indicators periodically
        self._update_streams_list()

        # Schedule next update (~15 FPS for display)
        self.after(66, self._update_frames)

    # ─── Detection Callback ──────────────────────────────────

    def _on_detection_result(self, result: DetectionResult):
        """Handle detection result from the engine (called from detection thread)."""
        # Get the frame for potential alert saving
        frame = self._stream_manager.get_frame(result.stream_name)

        # Process through alert manager
        self._alert_manager.process_detection(result, frame)

        # Update GUI (schedule on main thread)
        self.after(0, lambda r=result: self._update_gui_detection(r))

    def _update_gui_detection(self, result: DetectionResult):
        """Update GUI elements with detection result (runs on main thread)."""
        # Update stream panel
        panel = self._stream_panels.get(result.stream_name)
        if panel:
            panel.update_detection(result.status, result.confidence, result.description)

        # Add to alert log
        self._alert_log.add_entry(
            stream_name=result.stream_name,
            status=result.status,
            confidence=result.confidence,
            description=result.description,
            timestamp=result.timestamp
        )

    # ─── Cleanup ─────────────────────────────────────────────

    def _on_closing(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self._topbar_flash_active = False
        self._alert_manager.acknowledge()  # Kill any active alarm siren
        self._save_config()  # Must save config BEFORE stopping streams because stop_all clears the worker list
        self._detection_engine.stop()
        self._stream_manager.stop_all()
        self.destroy()
