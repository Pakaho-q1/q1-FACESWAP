from __future__ import annotations

import threading
from typing import Any, Callable


class AutosaveCoordinator:
    """Debounced settings autosave helper for GUI controls."""

    def __init__(
        self,
        project_root: str,
        collect_values: Callable[[], dict[str, Any]],
        save_project_settings: Callable[[str, dict[str, Any]], Any],
        default_delay_s: float = 0.6,
    ) -> None:
        self._project_root = project_root
        self._collect_values = collect_values
        self._save_project_settings = save_project_settings
        self._default_delay_s = max(0.05, float(default_delay_s))
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _autosave_now(self) -> None:
        try:
            settings = self._collect_values()
            self._save_project_settings(self._project_root, settings)
        except Exception:
            # Autosave must be best-effort and never break GUI runtime.
            return

    def schedule(self, delay_s: float | None = None) -> None:
        delay = self._default_delay_s if delay_s is None else max(0.05, float(delay_s))
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(delay, self._autosave_now)
            self._timer.daemon = True
            self._timer.start()

    def bind(self, control: Any) -> None:
        on_change = getattr(control, "on_value_change", None)
        if callable(on_change):
            on_change(lambda _e: self.schedule())

    def shutdown(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
