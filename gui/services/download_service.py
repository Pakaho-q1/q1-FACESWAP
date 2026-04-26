from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib import request as urllib_request
from typing import Any, Callable

from nicegui import ui


class ModelDownloadService:
    def __init__(
        self,
        project_root: str,
        model_fallback_urls: dict[str, str],
        read_model_manifest_status: Callable[[str], list[dict[str, Any]]],
        check_tensorrt_status: Callable[[str], dict[str, Any]],
    ) -> None:
        self.project_root = project_root
        self.model_fallback_urls = model_fallback_urls
        self.read_model_manifest_status = read_model_manifest_status
        self.check_tensorrt_status = check_tensorrt_status
        self.download_state_lock = threading.Lock()
        self.download_state: dict[str, dict[str, Any]] = {}
        self.download_pause_event = threading.Event()
        self.prebuild_state_lock = threading.Lock()
        self.prebuild_state: dict[str, str] = {}
        self.checksum_state_lock = threading.Lock()
        self.checksum_state: dict[str, str] = {}

    def _set_checksum_state(self, filename: str, state: str) -> None:
        with self.checksum_state_lock:
            self.checksum_state[str(filename)] = str(state)

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def start_verify_checksum(self, filename: str, model_path: str, manifest_sha256: str) -> None:
        file_name = str(filename).strip()
        target_path = Path(str(model_path))
        expected_sha = str(manifest_sha256 or "").strip().lower()
        if not file_name:
            return
        if not expected_sha:
            self._set_checksum_state(file_name, "missing_manifest_sha256")
            return
        if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
            self._set_checksum_state(file_name, "invalid_manifest_sha256")
            return
        if not target_path.is_file():
            self._set_checksum_state(file_name, "missing_local_file")
            return

        self._set_checksum_state(file_name, "verifying")
        try:
            actual_sha = self._sha256_file(target_path)
        except Exception:  # noqa: BLE001
            self._set_checksum_state(file_name, "verify_error")
            return

        if actual_sha == expected_sha:
            self._set_checksum_state(file_name, "ok")
        else:
            self._set_checksum_state(file_name, "mismatch")

    def snapshot_download_state(self) -> dict[str, dict[str, Any]]:
        with self.download_state_lock:
            return {k: dict(v) for k, v in self.download_state.items()}

    def has_active_downloads(self) -> bool:
        with self.download_state_lock:
            return any(
                str(v.get("status", "")) in {"queued", "downloading"}
                for v in self.download_state.values()
            )

    def _download_worker(
        self, filename: str, url: str, target_path: str, cancel_event: threading.Event
    ) -> None:
        tmp_path: Path | None = None
        try:
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=f"{filename}.", suffix=".part", dir=str(target.parent))
            os.close(fd)
            tmp_path = Path(tmp_name)
            with urllib_request.urlopen(url, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", "0") or "0")
                read_size = 0
                with tmp_path.open("wb") as f:
                    while True:
                        if cancel_event.is_set():
                            with self.download_state_lock:
                                state = self.download_state.get(filename, {})
                                state["status"] = "canceled"
                                state["detail"] = "Canceled by user"
                                self.download_state[filename] = state
                            return
                        while self.download_pause_event.is_set():
                            with self.download_state_lock:
                                state = self.download_state.get(filename, {})
                                state["status"] = "paused"
                                state["detail"] = "Paused"
                                self.download_state[filename] = state
                            time.sleep(0.2)
                            if cancel_event.is_set():
                                with self.download_state_lock:
                                    state = self.download_state.get(filename, {})
                                    state["status"] = "canceled"
                                    state["detail"] = "Canceled by user"
                                    self.download_state[filename] = state
                                return
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        read_size += len(chunk)
                        progress = 0.0
                        if total > 0:
                            progress = max(0.0, min(1.0, read_size / total))
                        with self.download_state_lock:
                            state = self.download_state.get(filename, {})
                            state["status"] = "downloading"
                            state["progress"] = progress
                            state["detail"] = f"{read_size}/{total} bytes" if total > 0 else f"{read_size} bytes"
                            self.download_state[filename] = state
            os.replace(str(tmp_path), str(target))
            with self.download_state_lock:
                state = self.download_state.get(filename, {})
                state["status"] = "done"
                state["progress"] = 1.0
                state["detail"] = "Completed"
                self.download_state[filename] = state
        except Exception as exc:  # noqa: BLE001
            with self.download_state_lock:
                state = self.download_state.get(filename, {})
                state["status"] = "error"
                state["progress"] = 0.0
                state["detail"] = str(exc)
                self.download_state[filename] = state
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def start_model_download(self, filename: str, url: str, target_path: str, force: bool = False) -> None:
        if not filename or not url or not target_path:
            return
        with self.download_state_lock:
            state = self.download_state.get(filename, {})
            if (not force) and state.get("status") in {"downloading", "queued", "paused"}:
                return
            cancel_event = threading.Event()
            self.download_state[filename] = {
                "status": "queued",
                "progress": 0.0,
                "detail": "Queued",
                "url": url,
                "path": target_path,
                "cancel_event": cancel_event,
            }
        thread = threading.Thread(
            target=self._download_worker,
            args=(filename, url, target_path, cancel_event),
            daemon=True,
            name=f"model-download-{filename}",
        )
        thread.start()

    def download_all_missing_models(self, latest_health_report: dict[str, Any]) -> int:
        rows = list(latest_health_report.get("model_status", []))
        if not rows:
            rows = self.read_model_manifest_status(self.project_root)
        queued = 0
        for row in rows:
            filename = str(row.get("filename", "")).strip()
            present = bool(row.get("present", False))
            url = str(row.get("url", "")).strip() or self.model_fallback_urls.get(filename, "")
            path = str(row.get("path", "")).strip()
            if present or not url or not path:
                continue
            self.start_model_download(filename, url, path)
            queued += 1
        return queued

    def toggle_pause_all_downloads(self, pause_resume_downloads_btn: Any) -> None:
        if self.download_pause_event.is_set():
            self.download_pause_event.clear()
            pause_resume_downloads_btn.set_text("Pause All")
        else:
            self.download_pause_event.set()
            pause_resume_downloads_btn.set_text("Resume All")

    def cancel_model_download(self, filename: str) -> None:
        with self.download_state_lock:
            state = self.download_state.get(filename, {})
            cancel_event = state.get("cancel_event")
            if isinstance(cancel_event, threading.Event):
                cancel_event.set()
            state["status"] = "canceled"
            state["detail"] = "Canceled by user"
            self.download_state[filename] = state

    def retry_model_download(self, filename: str) -> None:
        with self.download_state_lock:
            state = dict(self.download_state.get(filename, {}))
        url = str(state.get("url", ""))
        path = str(state.get("path", ""))
        if not url or not path:
            rows = self.read_model_manifest_status(self.project_root)
            for row in rows:
                if str(row.get("filename", "")) == filename:
                    url = str(row.get("url", "")).strip() or self.model_fallback_urls.get(filename, "")
                    path = str(row.get("path", ""))
                    break
        if url and path:
            self.start_model_download(filename, url, path, force=True)

    def clear_finished_downloads(self) -> None:
        with self.download_state_lock:
            keys = list(self.download_state.keys())
            for key in keys:
                status = str(self.download_state[key].get("status", ""))
                if status in {"done", "error", "canceled"}:
                    self.download_state.pop(key, None)

    def render_download_center(self, download_center_list: Any, download_center_summary: Any) -> None:
        clear_items = getattr(download_center_list, "clear", None)
        if callable(clear_items):
            clear_items()
        with self.download_state_lock:
            rows = [(k, dict(v)) for k, v in self.download_state.items()]
        if not rows:
            download_center_summary.set_text("No downloads yet")
            return
        download_center_summary.set_text(f"Total downloads: {len(rows)}")
        for filename, state in sorted(rows, key=lambda x: x[0].casefold()):
            status = str(state.get("status", "queued"))
            progress = float(state.get("progress", 0.0) or 0.0)
            detail = str(state.get("detail", ""))
            with download_center_list:
                with ui.card().classes("w-full border border-slate-100"):
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label(filename).classes("text-sm text-slate-800 grow")
                        ui.badge(status).classes("text-slate-700")
                        ui.button(
                            "Retry",
                            on_click=lambda _e=None, fn=filename: self.retry_model_download(fn),
                            color="secondary",
                        ).props("flat dense")
                        ui.button(
                            "Cancel",
                            on_click=lambda _e=None, fn=filename: self.cancel_model_download(fn),
                            color="negative",
                        ).props("flat dense")
                    ui.linear_progress(value=progress).classes("w-full")
                    ui.label(detail).classes("text-[11px] text-slate-500")

    def open_tensorrt_dialog(self, tensorrt_dialog: Any, trt_missing_label: Any, trt_target_label: Any) -> bool:
        status = self.check_tensorrt_status(self.project_root)
        if bool(status.get("ok", False)):
            return False
        missing = list(status.get("missing", []))
        trt_missing_label.set_text(f"Missing files: {', '.join(missing)}")
        trt_target_label.set_text(f"Place TensorRT files at: {status.get('bin', '')}")
        tensorrt_dialog.open()
        return True

    def _prebuild_trt_worker(self, filename: str, model_path: str) -> None:
        try:
            import numpy as np
            import onnxruntime as ort

            cache_dir = Path(self.project_root) / "assets" / "trt_cache" / "gui_prebuild"
            cache_dir.mkdir(parents=True, exist_ok=True)
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache_dir),
                        "trt_fp16_enable": True,
                        "trt_engine_cache_prefix": f"gui_{Path(filename).stem}",
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            with self.prebuild_state_lock:
                self.prebuild_state[filename] = "building"
            session = ort.InferenceSession(model_path, providers=providers)
            feed: dict[str, Any] = {}
            for inp in session.get_inputs():
                shape = []
                for dim in inp.shape:
                    if isinstance(dim, int) and dim > 0:
                        shape.append(dim)
                    else:
                        shape.append(1)
                dtype = np.float32
                if "int64" in str(inp.type):
                    dtype = np.int64
                elif "int32" in str(inp.type):
                    dtype = np.int32
                elif "float16" in str(inp.type):
                    dtype = np.float16
                feed[inp.name] = np.zeros(shape, dtype=dtype)
            outputs = [session.get_outputs()[0].name] if session.get_outputs() else None
            session.run(outputs, feed)
            with self.prebuild_state_lock:
                self.prebuild_state[filename] = "done"
        except Exception as exc:  # noqa: BLE001
            with self.prebuild_state_lock:
                self.prebuild_state[filename] = f"error: {exc}"

    def start_prebuild_trt(
        self,
        filename: str,
        model_path: str,
        tensorrt_dialog: Any,
        trt_missing_label: Any,
        trt_target_label: Any,
    ) -> None:
        if self.open_tensorrt_dialog(tensorrt_dialog, trt_missing_label, trt_target_label):
            return
        if not str(filename).lower().endswith(".onnx"):
            return
        with self.prebuild_state_lock:
            status = self.prebuild_state.get(filename, "")
            if status == "building":
                return
            self.prebuild_state[filename] = "queued"
        thread = threading.Thread(
            target=self._prebuild_trt_worker,
            args=(filename, model_path),
            daemon=True,
            name=f"trt-prebuild-{filename}",
        )
        thread.start()

    def render_model_status_dialog(
        self,
        model_status_list: Any,
        model_status_summary: Any,
        tensorrt_dialog: Any,
        trt_missing_label: Any,
        trt_target_label: Any,
    ) -> None:
        rows = self.read_model_manifest_status(self.project_root)
        clear_items = getattr(model_status_list, "clear", None)
        if callable(clear_items):
            clear_items()
        total = len(rows)
        found = sum(1 for r in rows if bool(r.get("present", False)))
        model_status_summary.set_text(f"Found {found}/{total} models")
        for row in rows:
            filename = str(row.get("filename", ""))
            present = bool(row.get("present", False))
            url = str(row.get("url", "")).strip() or self.model_fallback_urls.get(filename, "")
            path = str(row.get("path", "")).strip()
            with self.download_state_lock:
                state = dict(self.download_state.get(filename, {}))
            status_name = str(state.get("status", "ready"))
            progress_value = float(state.get("progress", 0.0) or 0.0)
            detail = str(state.get("detail", ""))
            with self.prebuild_state_lock:
                prebuild_name = str(self.prebuild_state.get(filename, "ready"))
            with model_status_list:
                with ui.row().classes("w-full items-center gap-2 border-b border-slate-100 py-1"):
                    ui.label(filename).classes("text-sm text-slate-800 grow")
                    ui.badge("OK" if present else "Missing").classes("text-emerald-700" if present else "text-rose-700")
                    if present:
                        ui.label("Installed").classes("text-xs text-emerald-700")
                    elif url:
                        ui.button(
                            "Download",
                            on_click=lambda _e=None, fn=filename, u=url, p=path: self.start_model_download(fn, u, p),
                            color="primary",
                        ).props("flat dense")
                    else:
                        ui.label("No URL").classes("text-xs text-rose-700")
                if not present:
                    with ui.row().classes("w-full items-center gap-2 pb-2"):
                        ui.label(f"Status: {status_name}").classes("text-[11px] text-slate-600")
                        ui.linear_progress(value=progress_value).classes("grow")
                        ui.label(detail).classes("text-[11px] text-slate-500")
                elif filename.lower().endswith(".onnx"):
                    with ui.row().classes("w-full items-center gap-2 pb-2"):
                        ui.label(f"TRT prebuild: {prebuild_name}").classes("text-[11px] text-slate-600 grow")
                        ui.button(
                            "Prebuild TRT",
                            on_click=lambda _e=None, fn=filename, p=path: self.start_prebuild_trt(
                                fn, p, tensorrt_dialog, trt_missing_label, trt_target_label
                            ),
                            color="warning",
                        ).props("flat dense")
