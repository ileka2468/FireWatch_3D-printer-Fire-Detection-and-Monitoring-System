"""
Alert Log — scrollable log panel showing detection events with timestamps and color coding.
"""

import customtkinter as ctk
from datetime import datetime


class AlertLog(ctk.CTkFrame):
    """
    Scrollable alert/event log panel.
    Color-coded entries: green for SAFE, yellow for WARNING, red for FIRE.
    """

    STATUS_COLORS = {
        "SAFE": "#0cca4a",
        "PRINTING": "#4a5568",
        "WARNING": "#f59e0b",
        "FIRE": "#e94560",
        "FAILED": "#f97316",
        "DONE": "#0cca4a",
    }

    STATUS_ICONS = {
        "SAFE": "✓",
        "PRINTING": "ℹ",
        "WARNING": "⚠",
        "FIRE": "🔥",
        "FAILED": "✗",
        "DONE": "★",
    }

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#13131d", corner_radius=12, **kwargs)

        self._build_ui()
        self._entry_count = 0
        self._max_entries = 200

    def _build_ui(self):
        """Construct the log panel UI."""
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent", height=36)
        header_frame.pack(fill="x", padx=12, pady=(10, 4))
        header_frame.pack_propagate(False)

        ctk.CTkLabel(
            header_frame,
            text="Detection Log",
            font=("Segoe UI Semibold", 14),
            text_color="#c0c0d0"
        ).pack(side="left")

        # Clear button
        self._clear_btn = ctk.CTkButton(
            header_frame,
            text="Clear",
            font=("Segoe UI", 11),
            width=60,
            height=26,
            fg_color="#2a2a3e",
            hover_color="#3a3a4e",
            corner_radius=6,
            command=self.clear_log
        )
        self._clear_btn.pack(side="right")

        # Scrollable log area
        self._scroll_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="#0d0d15",
            corner_radius=8,
            scrollbar_button_color="#2a2a3e",
            scrollbar_button_hover_color="#3a3a4e"
        )
        self._scroll_frame.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        # Empty state
        self._empty_label = ctk.CTkLabel(
            self._scroll_frame,
            text="No detection events yet.\nStart monitoring to see results.",
            font=("Segoe UI", 11),
            text_color="#555566",
            justify="center"
        )
        self._empty_label.pack(pady=30)

    def add_entry(self, stream_name: str, status: str, confidence: float, description: str,
                  timestamp: datetime | None = None):
        """Add a new log entry."""
        # Remove empty state label on first entry
        if self._entry_count == 0:
            self._empty_label.destroy()

        # Limit entries
        self._entry_count += 1
        if self._entry_count > self._max_entries:
            # Remove oldest entry
            children = self._scroll_frame.winfo_children()
            if children:
                children[0].destroy()

        ts = timestamp or datetime.now()
        time_str = ts.strftime("%H:%M:%S")
        color = self.STATUS_COLORS.get(status, "#6b7280")
        icon = self.STATUS_ICONS.get(status, "?")

        bg_color = "#1a1a28"  # Default for SAFE/PRINTING/DONE
        if status == "WARNING":
            bg_color = "#2a1a0a"
        elif status == "FIRE":
            bg_color = "#2a0a0a"
        elif status == "FAILED":
            bg_color = "#2a160a"

        # Entry frame
        entry_frame = ctk.CTkFrame(
            self._scroll_frame,
            fg_color=bg_color,
            corner_radius=6,
            height=48
        )
        entry_frame.pack(fill="x", pady=2)
        entry_frame.pack_propagate(False)

        # Top row: time, icon, stream name, status
        top_row = ctk.CTkFrame(entry_frame, fg_color="transparent")
        top_row.pack(fill="x", padx=8, pady=(4, 0))

        ctk.CTkLabel(
            top_row,
            text=time_str,
            font=("Consolas", 10),
            text_color="#6b7280",
            width=60,
            anchor="w"
        ).pack(side="left")

        ctk.CTkLabel(
            top_row,
            text=f"{icon} {stream_name}",
            font=("Segoe UI Semibold", 11),
            text_color=color,
            anchor="w"
        ).pack(side="left", padx=(4, 0))

        ctk.CTkLabel(
            top_row,
            text=f"{confidence:.0%}",
            font=("Segoe UI Bold", 10),
            text_color=color,
            anchor="e"
        ).pack(side="right")

        # Bottom row: description
        ctk.CTkLabel(
            entry_frame,
            text=description[:80],
            font=("Segoe UI", 9),
            text_color="#8888aa",
            anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 4))

        # Auto-scroll to bottom
        self._scroll_frame._parent_canvas.yview_moveto(1.0)

    def clear_log(self):
        """Clear all log entries."""
        for child in self._scroll_frame.winfo_children():
            child.destroy()
        self._entry_count = 0

        # Re-add empty state
        self._empty_label = ctk.CTkLabel(
            self._scroll_frame,
            text="No detection events yet.\nStart monitoring to see results.",
            font=("Segoe UI", 11),
            text_color="#555566",
            justify="center"
        )
        self._empty_label.pack(pady=30)
