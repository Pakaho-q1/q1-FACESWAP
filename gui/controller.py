from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import json
import time
from typing import Any
from pathlib import Path

try:
    from gui.state import AppStore
except ImportError:
    from state import AppStore  # type: ignore


class PipelineController:
    def __init__(self, state: AppStore | None = None) -> None:
        # allow injection of AppStore (SSOT)
        self.state = state or AppStore()
        self.event_q: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=5000)
        self._worker_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._dropped_log_events = 0
        self._dropped_nonlog_events = 0
        self._last_metrics_emit = 0.0
        self._project_root = Path(__file__).resolve().parents[1]
        self._tuner_re = re.compile(
            r"GPU:\s*(?P<gpu>\d+)%\s*\|\s*MODE:\s*(?P<mode>[a-zA-Z_]+)\s*\|\s*HOT:\s*(?P<hot>[a-zA-Z_]+)"
        )
        self._q_re = re.compile(
            r"Q\[detect:(?P<d>\d+)\s+swap:(?P<s>\d+)\s+restore:(?P<r>\d+)\s+parse:(?P<p>\d+)\]"
        )
        self._p_re = re.compile(
            r"P\[detect:(?P<d>\d+)\s+swap:(?P<s>\d+)\s+restore:(?P<r>\d+)\s+parse:(?P<p>\d+)\]"
        )
        self._progress_re = re.compile(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)")
        self._ansi_re = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
        self._gui_prefix = "__Q1_GUI__"

    def _push_event(self, kind: str, payload: Any) -> None:
        try:
            self.event_q.put_nowait((kind, payload))
            return
        except queue.Full:
            pass
        if kind == "log":
            self._dropped_log_events += 1
            return
        try:
            _ = self.event_q.get_nowait()
            self._dropped_nonlog_events += 1
        except queue.Empty:
            pass
        try:
            self.event_q.put_nowait((kind, payload))
        except queue.Full:
            self._dropped_nonlog_events += 1

    def _build_cli_argv(self, values: dict[str, Any]) -> list[str]:
        def b(flag: str, value: bool) -> list[str]:
            return [flag, "true" if value else "false"]

        argv: list[str] = [
            "--project-path",
            values["project_path"].strip(),
            "--face-model-name",
            values["face_name"].strip(),
            "--format",
            values["format"],
            "--input-path",
            values["input_path"].strip(),
            "--output-path",
            values["output_path"].strip(),
            "--workers-per-stage",
            str(values["workers_per_stage"]),
            "--worker-queue-size",
            str(values["worker_queue_size"]),
            "--out-queue-size",
            str(values["out_queue_size"]),
            "--tuner-mode",
            values["tuner_mode"],
            "--gpu-target-util",
            str(values["gpu_target_util"]),
            "--high-watermark",
            str(values["high_watermark"]),
            "--low-watermark",
            str(values["low_watermark"]),
            "--switch-cooldown-s",
            str(values["switch_cooldown_s"]),
            "--restore-choice",
            values["restore_choice"],
            "--parser-choice",
            values["parser_choice"],
            "--parser-mask-blur",
            str(values["parser_mask_blur"]),
            "--provider-all",
            values["provider_all"],
            "--max-frames",
            str(values["max_frames"]),
            "--max-retries",
            str(values["max_retries"]),
        ]
        argv.extend(b("--use-swaper", values["use_swaper"]))
        argv.extend(b("--use-restore", values["use_restore"]))
        argv.extend(b("--use-parser", values["use_parser"]))
        argv.extend(b("--dry-run", values["dry_run"]))
        argv.extend(b("--preserve-swap-eyes", values["preserve_swap_eyes"]))
        argv.extend(["--swaper-weigh", str(values["swapper_blend"])])
        argv.extend(["--restore-weigh", str(values["restore_weight"])])
        argv.extend(["--restore-blend", str(values["restore_blend"])])

        return argv

    def _normalize_line(self, line: str) -> str:
        return self._ansi_re.sub("", line).strip()

    def _emit_tuner_status_from_line(self, line: str) -> None:
        if "TUNER" not in line and "GPU:" not in line:
            return
        tuner = self._tuner_re.search(line)
        if tuner is None:
            return
        sizes = {}
        permits = {}
        q_match = self._q_re.search(line)
        p_match = self._p_re.search(line)
        if q_match is not None:
            sizes = {
                "detect": int(q_match.group("d")),
                "swap": int(q_match.group("s")),
                "restore": int(q_match.group("r")),
                "parse": int(q_match.group("p")),
            }
        if p_match is not None:
            permits = {
                "detect": int(p_match.group("d")),
                "swap": int(p_match.group("s")),
                "restore": int(p_match.group("r")),
                "parse": int(p_match.group("p")),
            }
        self._push_event(
            "event",
            {
                "name": "tuner_status",
                "payload": {
                    "gpu_util": int(tuner.group("gpu")),
                    "mode_name": tuner.group("mode"),
                    "hot_stage": tuner.group("hot"),
                    "sizes": sizes,
                    "permits": permits,
                },
            },
        )

    def _emit_progress_from_line(self, line: str) -> None:
        match = self._progress_re.search(line)
        if match is None:
            return
        done = int(match.group("done"))
        total = int(match.group("total"))
        if total <= 0:
            return
        label = "work"
        if "Rendering" in line:
            label = "video"
        elif "Processing Images" in line:
            label = "image"
        self._push_event(
            "progress",
            {"label": label, "completed": done, "total": total},
        )

    def _stream_reader(self, stream, source: str) -> None:
        if stream is None:
            return
        buffer: list[str] = []

        def handle_gui_line(line: str) -> bool:
            if not line.startswith(self._gui_prefix):
                return False
            raw_payload = line[len(self._gui_prefix) :].strip()
            if not raw_payload:
                return True
            try:
                payload = json.loads(raw_payload)
            except Exception:
                return True
            event_type = str(payload.get("type", "")).strip().lower()
            if event_type == "event":
                self._push_event(
                    "event",
                    {
                        "name": str(payload.get("name", "event")),
                        "payload": dict(payload.get("payload", {})),
                    },
                )
                return True
            if event_type == "progress":
                self._push_event(
                    "progress",
                    {
                        "label": str(payload.get("label", "work")),
                        "completed": int(payload.get("completed", 0)),
                        "total": int(payload.get("total", 0)),
                    },
                )
                return True
            if event_type == "log":
                self._push_event("log", str(payload.get("message", "")))
                return True
            return True

        def flush_buffer() -> None:
            if not buffer:
                return
            raw_line = "".join(buffer)
            buffer.clear()
            line = self._normalize_line(raw_line)
            if not line:
                return
            if handle_gui_line(line):
                return
            self._push_event("log", line)
            self._emit_tuner_status_from_line(line)
            self._emit_progress_from_line(line)

        try:
            while True:
                ch = stream.read(1)
                if ch == "":
                    flush_buffer()
                    break
                if ch in ("\n", "\r"):
                    flush_buffer()
                    continue
                buffer.append(ch)
        except Exception as exc:  # noqa: BLE001
            self._push_event("log", f"{source}_stream_error: {exc}")
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _stop_process_tree(
        self, proc: subprocess.Popen[str], grace_seconds: float = 8.0
    ) -> None:
        """Try graceful stop first (terminal-like), then force kill as fallback."""
        if proc.poll() is not None:
            return

        pid = proc.pid
        if os.name == "nt":
            # Terminal-close-like path for console process groups.
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                proc.wait(timeout=grace_seconds)
                return
            except Exception:  # noqa: BLE001
                pass

            # Soft taskkill (without /F) before hard kill.
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                proc.wait(timeout=max(1.0, grace_seconds / 2.0))
                return
            except Exception:  # noqa: BLE001
                pass

            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        try:
            os.killpg(os.getpgid(pid), signal.SIGINT)
            proc.wait(timeout=grace_seconds)
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            proc.wait(timeout=max(1.0, grace_seconds / 2.0))
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass

    def _worker(self, values: dict[str, Any], stop_event: threading.Event) -> None:
        try:
            cmd = [
                sys.executable,
                "-u",
                "face_swap_unified.py",
                *self._build_cli_argv(values),
            ]
            creationflags = 0
            popen_kwargs: dict[str, Any] = {}
            child_env = dict(os.environ)
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["Q1_GUI_EVENTS"] = "1"
            child_env["Q1_GUI_PREVIEW_ENABLED"] = (
                "1" if bool(values.get("preview_enabled", True)) else "0"
            )
            preview_fps = float(values.get("preview_fps_limit", 2.5) or 2.5)
            preview_fps = max(0.5, min(30.0, preview_fps))
            child_env["Q1_GUI_PREVIEW_FPS"] = f"{preview_fps:.3f}"
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True

            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self._project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
                creationflags=creationflags,
                **popen_kwargs,
            )
            self._push_event("log", f"Spawned CLI process pid={self._proc.pid}")

            self._stdout_thread = threading.Thread(
                target=self._stream_reader,
                args=(self._proc.stdout, "stdout"),
                daemon=True,
                name="gui-cli-stdout",
            )
            self._stderr_thread = threading.Thread(
                target=self._stream_reader,
                args=(self._proc.stderr, "stderr"),
                daemon=True,
                name="gui-cli-stderr",
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

            return_code = self._proc.wait()
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout=2.0)
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2.0)

            if stop_event.is_set() or self.state.stop_requested:
                self._push_event("stopped", {"reason": "user_stop", "return_code": return_code})
            elif return_code == 0:
                self._push_event("done", {"return_code": return_code})
            else:
                self._push_event("error", f"CLI exited with code {return_code}")
        except BaseException as exc:  # noqa: BLE001
            if stop_event.is_set():
                self._push_event("stopped", {"reason": "user_stop", "details": str(exc)})
            else:
                self._push_event("error", str(exc))
        finally:
            self._proc = None
            self._stdout_thread = None
            self._stderr_thread = None

    def start(self, values: dict[str, Any]) -> bool:
        # Recover from stale running flag if worker died unexpectedly.
        if (
            self.state.running
            and self._worker_thread is not None
            and not self._worker_thread.is_alive()
        ):
            self.finish()
        if self.state.running:
            return False
        self.state.running = True
        self.state.stop_requested = False
        self.state.total = 0
        self.state.done = 0
        self.state.stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker,
            args=(values, self.state.stop_event),
            daemon=True,
            name="gui-pipeline-worker",
        )
        self._worker_thread.start()
        return True

    def request_stop(self) -> bool:
        if not self.state.running or self.state.stop_event is None:
            return False
        self.state.stop_requested = True
        self.state.stop_event.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._push_event("log", f"Stopping process pid={proc.pid} (graceful first)...")
            self._stop_process_tree(proc)
        return True

    def finish(self) -> None:
        self.state.running = False
        self.state.stop_requested = False
        self.state.stop_event = None
        self._worker_thread = None
        self._proc = None
        self._stdout_thread = None
        self._stderr_thread = None

    def poll_events(self) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        queue_size = self.event_q.qsize()
        while True:
            try:
                events.append(self.event_q.get_nowait())
            except queue.Empty:
                break
        # If worker died without terminal event, surface a synthetic error so UI unlocks.
        if (
            not events
            and self.state.running
            and self._worker_thread is not None
            and not self._worker_thread.is_alive()
        ):
            if self.state.stop_requested:
                events.append(
                    ("stopped", {"reason": "user_stop", "details": "Worker stopped"})
                )
            else:
                events.append(("error", "Worker exited unexpectedly"))
        now = time.monotonic()
        if (now - self._last_metrics_emit) >= 1.0:
            self._last_metrics_emit = now
            events.append(
                (
                    "event",
                    {
                        "name": "controller_metrics",
                        "payload": {
                            "queue_depth": queue_size,
                            "dropped_logs": self._dropped_log_events,
                            "dropped_events": self._dropped_nonlog_events,
                        },
                    },
                )
            )
        return events
