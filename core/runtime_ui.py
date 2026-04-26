import threading
import time

from core.ui_log import ui_print


class RuntimeUI:
    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = False
        self._live = None
        self._status = "GPU: -- | MODE: normal | HOT: - | Q[detect:00 swap:00] | P[detect:00 swap:00]"
        self._task_id = None
        self._progress = None
        self._renderable = None
        self._fps_window_start_ts = None
        self._fps_window_frames = 0
        self._last_write_fps = 0.0

    def _build_renderable(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        tuner_text = Text(self._status, no_wrap=True, overflow="ellipsis")
        return Group(
            Panel(tuner_text, title="Tuner", border_style="green"),
            Panel(self._progress, title="Pipeline", border_style="blue"),
        )

    def start(self):
        try:
            from rich.live import Live
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
        except Exception:
            self._enabled = False
            return

        with self._lock:
            self._enabled = True
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("| [yellow]{task.completed:,.0f}/{task.total:,.0f} fr"),
                TextColumn("| [magenta]{task.fields[write_fps]:5.1f} write fps"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self._renderable = self._build_renderable()
            self._live = Live(self._renderable, refresh_per_second=8, transient=False)
            self._live.start()

    def stop(self):
        with self._lock:
            if self._enabled and self._live is not None:
                self._live.stop()
                self._live = None

    def _refresh(self):
        if self._enabled and self._live is not None and self._renderable is not None:
            self._live.update(self._renderable, refresh=True)

    def _find_task(self, task_id):
        if self._progress is None or task_id is None:
            return None
        for task in self._progress.tasks:
            if task.id == task_id:
                return task
        return None

    def set_progress(self, total, description):
        with self._lock:
            if self._enabled and self._progress is not None:
                # Keep only one visible progress row (current file/job).
                for task in list(self._progress.tasks):
                    self._progress.remove_task(task.id)
                self._fps_window_start_ts = None
                self._fps_window_frames = 0
                self._last_write_fps = 0.0
                self._task_id = self._progress.add_task(
                    description,
                    total=total,
                    write_fps=0.0,
                )
                self._refresh()

    def set_progress_description(self, description):
        with self._lock:
            if self._enabled and self._progress is not None and self._task_id is not None:
                task = self._find_task(self._task_id)
                if task is not None:
                    self._progress.update(self._task_id, description=description)
                    self._refresh()

    def advance_progress(self, amount=1):
        with self._lock:
            if self._enabled and self._progress is not None and self._task_id is not None:
                task = self._find_task(self._task_id)
                if task is None:
                    return
                now_ts = time.perf_counter()
                if self._fps_window_start_ts is None:
                    self._fps_window_start_ts = now_ts

                self._fps_window_frames += int(amount)
                elapsed = now_ts - self._fps_window_start_ts

                # Windowed output FPS from actual written/advanced frames.
                if elapsed >= 0.5:
                    self._last_write_fps = float(self._fps_window_frames) / max(1e-6, elapsed)
                    self._fps_window_start_ts = now_ts
                    self._fps_window_frames = 0

                self._progress.update(
                    self._task_id,
                    advance=amount,
                    write_fps=self._last_write_fps,
                )

    def finish_progress(self):
        with self._lock:
            if self._enabled and self._progress is not None and self._task_id is not None:
                task = self._find_task(self._task_id)
                if task is None:
                    self._task_id = None
                    return
                remaining = max(0, int(task.total) - int(task.completed))
                if remaining > 0:
                    self._progress.update(
                        self._task_id, advance=remaining, write_fps=self._last_write_fps
                    )
                self._refresh()
                self._task_id = None

    def update_tuner(self, gpu_util, mode_name, hot_stage, sizes, permits, ordered_stages):
        queue_text = " ".join(f"{stage}:{sizes.get(stage, 0):02d}" for stage in ordered_stages)
        permit_text = " ".join(f"{stage}:{permits.get(stage, 0):02d}" for stage in ordered_stages)
        line = (
            f"GPU: {int(gpu_util):3d}% | MODE: {mode_name:<6} | HOT: {hot_stage:<7} "
            f"| Q[{queue_text}] | P[{permit_text}]"
        )
        with self._lock:
            if self._enabled:
                self._status = line
                try:
                    self._renderable = self._build_renderable()
                    self._refresh()
                except Exception:
                    pass

    def print_message(self, message, fallback=None):
        ui_print(message, fallback)

