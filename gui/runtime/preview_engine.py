from __future__ import annotations

import base64
from collections import deque
import queue
import threading
import time
from typing import Any


class PreviewEngine:
    def __init__(self, base_fps: float, ring_size: int = 2, render_queue_size: int = 2) -> None:
        self._base_fps = max(0.5, min(30.0, float(base_fps)))
        self._gpu_util = 0
        self._ingress_dropped = 0
        self._worker_dropped = 0
        self._rendered = 0
        self._ring: deque[dict[str, Any]] = deque(maxlen=max(1, ring_size))
        self._ring_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop_event = threading.Event()
        self._render_q: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=max(1, render_queue_size))
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="gui-preview-worker")
        self._worker.start()

    @property
    def ingress_dropped(self) -> int:
        return self._ingress_dropped

    @property
    def worker_dropped(self) -> int:
        return self._worker_dropped

    @property
    def rendered(self) -> int:
        return self._rendered

    def mark_rendered(self) -> None:
        self._rendered += 1

    def set_gpu_util(self, gpu_util: int) -> None:
        self._gpu_util = max(0, min(100, int(gpu_util)))

    def set_base_fps(self, base_fps: float) -> None:
        self._base_fps = max(0.5, min(30.0, float(base_fps)))
        self._wakeup.set()

    def adaptive_fps(self) -> float:
        gpu = self._gpu_util
        if gpu >= 97:
            factor = 0.25
        elif gpu >= 94:
            factor = 0.35
        elif gpu >= 90:
            factor = 0.50
        elif gpu >= 80:
            factor = 0.75
        else:
            factor = 1.0
        return max(0.5, min(30.0, self._base_fps * factor))

    def ingest(self, payload: dict[str, Any]) -> None:
        with self._ring_lock:
            if len(self._ring) >= self._ring.maxlen:
                self._ingress_dropped += 1
            self._ring.append(payload)
        self._wakeup.set()

    def pop_latest_render(self) -> dict[str, Any] | None:
        latest: dict[str, Any] | None = None
        while True:
            try:
                latest = self._render_q.get_nowait()
            except queue.Empty:
                break
        return latest

    def reset(self) -> None:
        with self._ring_lock:
            self._ring.clear()
        while True:
            try:
                self._render_q.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        self._stop_event.set()
        self._wakeup.set()

    def _prepare(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        raw_data_url = str(payload.get("data_url", ""))
        if "," not in raw_data_url:
            return None
        prefix, encoded = raw_data_url.split(",", 1)
        try:
            raw_bytes = base64.b64decode(encoded, validate=True)
            normalized = base64.b64encode(raw_bytes).decode("ascii")
        except Exception:  # noqa: BLE001
            return None
        kind = str(payload.get("kind", "")).lower()
        output_name = str(payload.get("output_name", ""))
        item_id = str(payload.get("item_id", ""))
        frame_id = int(payload.get("frame_id", 0) or 0)
        signature = f"{kind}:{output_name}:{item_id}:{frame_id}:{len(raw_bytes)}"
        return {
            "kind": kind,
            "output_name": output_name,
            "item_id": item_id,
            "frame_id": frame_id,
            "data_url": f"{prefix},{normalized}",
            "signature": signature,
        }

    def _worker_loop(self) -> None:
        last_emit_ts = 0.0
        while not self._stop_event.is_set():
            self._wakeup.wait(timeout=0.25)
            self._wakeup.clear()
            while not self._stop_event.is_set():
                with self._ring_lock:
                    if not self._ring:
                        break
                    latest_payload = self._ring.pop()
                    self._ring.clear()
                prepared = self._prepare(latest_payload)
                if prepared is None:
                    continue
                min_interval = 1.0 / max(0.5, self.adaptive_fps())
                now = time.perf_counter()
                wait_s = min_interval - (now - last_emit_ts)
                if wait_s > 0:
                    time.sleep(min(wait_s, 0.2))
                last_emit_ts = time.perf_counter()
                if self._render_q.full():
                    try:
                        self._render_q.get_nowait()
                        self._worker_dropped += 1
                    except queue.Empty:
                        pass
                try:
                    self._render_q.put_nowait(prepared)
                except queue.Full:
                    self._worker_dropped += 1
