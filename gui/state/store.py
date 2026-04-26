from __future__ import annotations

from datetime import datetime
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppStore:
    # Pipeline runtime state
    running: bool = False
    stop_requested: bool = False
    total: int = 0
    done: int = 0
    stop_event: threading.Event | None = None
    logs: list[str] = field(default_factory=list)

    # Job/metrics state
    planned_counts: dict[str, int] = field(
        default_factory=lambda: {"image": 0, "video": 0}
    )
    completed_counts: dict[str, int] = field(
        default_factory=lambda: {"image": 0, "video": 0}
    )
    failed_counts: dict[str, int] = field(
        default_factory=lambda: {"image": 0, "video": 0}
    )
    progress_units_done: int = 0
    progress_units_total: int = 0
    progress_units_label: str = "work"
    progress_last_percent: int = 0
    latest_outqueue_write_fps: float = 0.0
    # Runtime/preview timing and metrics (SSOT fields)
    run_started_at: float = 0.0
    processing_started_at: float = 0.0
    last_progress_done: int = 0
    last_progress_ts: float = 0.0
    last_pipeline_metrics: dict[str, Any] = field(default_factory=dict)
    last_preview_render_ts: float = 0.0
    last_preview_signature: str = ""
    preview_active_layer: str = "a"
    preview_paused: bool = False

    def append_log(self, message: str) -> str:
        stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self.logs.append(stamped)
        if len(self.logs) > 1200:
            self.logs = self.logs[-1200:]
        return stamped

    def clear_logs(self) -> None:
        self.logs.clear()

    def reset_for_run(self) -> None:
        # Keep lifecycle flags (running/stop_event) intact.
        # These are owned by PipelineController.start/request_stop/finish.
        self.completed_counts = {"image": 0, "video": 0}
        self.failed_counts = {"image": 0, "video": 0}
        self.progress_units_done = 0
        self.progress_units_total = 0
        self.progress_units_label = "work"
        self.progress_last_percent = 0
        self.latest_outqueue_write_fps = 0.0
        # reset runtime/preview fields
        self.run_started_at = 0.0
        self.processing_started_at = 0.0
        self.last_progress_done = 0
        self.last_progress_ts = 0.0
        self.last_pipeline_metrics = {}
        self.last_preview_render_ts = 0.0
        self.last_preview_signature = ""
        self.preview_active_layer = "a"
        self.preview_paused = False
