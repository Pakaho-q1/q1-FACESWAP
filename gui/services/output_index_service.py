from __future__ import annotations

import concurrent.futures
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


LOGGER = logging.getLogger(__name__)

_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov"}


def _scan_output_snapshot(output_path: str) -> list[tuple[str, str, float, int, str]]:
    rows: list[tuple[str, str, float, int, str]] = []
    if not output_path or not os.path.isdir(output_path):
        return rows
    with os.scandir(output_path) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            _, ext = os.path.splitext(entry.name)
            ext_l = ext.lower()
            kind = ""
            if ext_l in _IMAGE_EXT:
                kind = "image"
            elif ext_l in _VIDEO_EXT:
                kind = "video"
            if not kind:
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            rows.append((entry.name, os.path.abspath(entry.path), float(stat.st_mtime), int(stat.st_size), kind))
    return rows


class OutputIndexService:
    """Persistent output index with async scanner process and paged reads."""

    def __init__(self, db_path: str, stale_seconds: float = 3.0, max_scan_seconds: float = 90.0) -> None:
        self._db_path = str(db_path)
        self._stale_seconds = max(1.0, float(stale_seconds))
        self._max_scan_seconds = max(5.0, float(max_scan_seconds))
        self._lock = threading.Lock()
        self._executor: concurrent.futures.Executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="output-index-scan")
        self._future: concurrent.futures.Future[list[tuple[str, str, float, int, str]]] | None = None
        self._scan_path = ""
        self._scan_started_at = 0.0
        self._scan_seq = 0
        self._scheduled_force = False
        self._scheduled_path = ""
        self._scan_failures = 0
        self._last_scan_duration_ms = 0.0
        self._last_scan_files = 0
        self._last_scan_finished_at = 0.0
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _db(self) -> Any:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_db(self) -> None:
        db_parent = Path(self._db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS output_files (
                  output_path TEXT NOT NULL,
                  rel_name TEXT NOT NULL,
                  abs_path TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  mtime REAL NOT NULL,
                  size INTEGER NOT NULL,
                  PRIMARY KEY (output_path, rel_name)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS output_index_meta (
                  output_path TEXT PRIMARY KEY,
                  last_scan_finished_at REAL NOT NULL DEFAULT 0,
                  last_scan_duration_ms REAL NOT NULL DEFAULT 0,
                  last_scan_files INTEGER NOT NULL DEFAULT 0,
                  scan_failures INTEGER NOT NULL DEFAULT 0,
                  last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_output_files_sort ON output_files(output_path, mtime DESC, rel_name ASC)"
            )

    def shutdown(self) -> None:
        with self._lock:
            future = self._future
            self._future = None
        if future is not None:
            try:
                future.cancel()
            except Exception:
                pass
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _restart_executor(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="output-index-scan")

    def _index_stats(self, output_path: str) -> tuple[float, float]:
        with self._db() as conn:
            row = conn.execute(
                "SELECT last_scan_finished_at, last_scan_duration_ms FROM output_index_meta WHERE output_path = ?",
                (output_path,),
            ).fetchone()
        if row is None:
            return 0.0, 0.0
        return float(row["last_scan_finished_at"]), float(row["last_scan_duration_ms"])

    def request_scan(self, output_path: str, force: bool = False) -> bool:
        norm_path = os.path.abspath(os.path.expanduser(str(output_path or "").strip()))
        if not norm_path:
            return False
        if not os.path.isdir(norm_path):
            return False
        with self._lock:
            if self._future is not None and not self._future.done():
                self._scheduled_path = norm_path
                self._scheduled_force = self._scheduled_force or bool(force)
                return False
            last_ts, _ = self._index_stats(norm_path)
            is_stale = (time.time() - last_ts) >= self._stale_seconds
            if not force and not is_stale:
                return False
            self._scan_path = norm_path
            self._scan_started_at = time.perf_counter()
            self._scan_seq += 1
            self._future = self._executor.submit(_scan_output_snapshot, norm_path)
            return True

    def poll(self) -> None:
        completed_future: concurrent.futures.Future[list[tuple[str, str, float, int, str]]] | None = None
        target_path = ""
        started_at = 0.0
        with self._lock:
            if self._future is not None and not self._future.done():
                if (time.perf_counter() - self._scan_started_at) > self._max_scan_seconds:
                    timeout_s = time.perf_counter() - self._scan_started_at
                    self._scan_failures += 1
                    self._upsert_meta(
                        output_path=self._scan_path,
                        duration_ms=timeout_s * 1000.0,
                        files=0,
                        failures=self._scan_failures,
                        last_error=f"scan_timeout:{timeout_s:.1f}s",
                    )
                    self._future = None
                    self._scan_path = ""
                    self._scan_started_at = 0.0
                    self._restart_executor()
                    self._maybe_schedule_pending()
                return
            if self._future is None or not self._future.done():
                return
            completed_future = self._future
            target_path = self._scan_path
            started_at = self._scan_started_at
            self._future = None
            self._scan_path = ""
            self._scan_started_at = 0.0

        assert completed_future is not None
        try:
            rows = completed_future.result()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("output index scan failed: %s", exc)
            self._scan_failures += 1
            self._upsert_meta(
                output_path=target_path,
                duration_ms=max(0.0, (time.perf_counter() - started_at) * 1000.0),
                files=0,
                failures=self._scan_failures,
                last_error=str(exc),
            )
            self._maybe_schedule_pending()
            return

        duration_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
        self._apply_snapshot(output_path=target_path, rows=rows, duration_ms=duration_ms)
        self._scan_failures = 0
        self._last_scan_duration_ms = duration_ms
        self._last_scan_files = len(rows)
        self._last_scan_finished_at = time.time()
        self._maybe_schedule_pending()

    def _maybe_schedule_pending(self) -> None:
        with self._lock:
            scheduled_path = self._scheduled_path
            scheduled_force = self._scheduled_force
            self._scheduled_path = ""
            self._scheduled_force = False
        if scheduled_path:
            self.request_scan(scheduled_path, force=scheduled_force)

    def _upsert_meta(
        self,
        *,
        output_path: str,
        duration_ms: float,
        files: int,
        failures: int,
        last_error: str,
    ) -> None:
        with self._db() as conn:
            conn.execute(
                """
                INSERT INTO output_index_meta(output_path, last_scan_finished_at, last_scan_duration_ms, last_scan_files, scan_failures, last_error)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(output_path) DO UPDATE SET
                  last_scan_finished_at=excluded.last_scan_finished_at,
                  last_scan_duration_ms=excluded.last_scan_duration_ms,
                  last_scan_files=excluded.last_scan_files,
                  scan_failures=excluded.scan_failures,
                  last_error=excluded.last_error
                """,
                (output_path, time.time(), float(duration_ms), int(files), int(failures), str(last_error or "")),
            )

    def _apply_snapshot(
        self,
        *,
        output_path: str,
        rows: list[tuple[str, str, float, int, str]],
        duration_ms: float,
    ) -> None:
        seen = {name for name, _, _, _, _ in rows}
        with self._db() as conn:
            conn.execute("BEGIN")
            if rows:
                conn.executemany(
                    """
                    INSERT INTO output_files(output_path, rel_name, abs_path, kind, mtime, size)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(output_path, rel_name) DO UPDATE SET
                      abs_path=excluded.abs_path,
                      kind=excluded.kind,
                      mtime=excluded.mtime,
                      size=excluded.size
                    """,
                    [(output_path, name, abs_path, kind, mtime, size) for (name, abs_path, mtime, size, kind) in rows],
                )
            stale_rows = conn.execute(
                "SELECT rel_name FROM output_files WHERE output_path = ?",
                (output_path,),
            ).fetchall()
            stale_names = [str(r["rel_name"]) for r in stale_rows if str(r["rel_name"]) not in seen]
            if stale_names:
                conn.executemany(
                    "DELETE FROM output_files WHERE output_path = ? AND rel_name = ?",
                    [(output_path, stale) for stale in stale_names],
                )
            conn.execute("COMMIT")
        self._upsert_meta(
            output_path=output_path,
            duration_ms=duration_ms,
            files=len(rows),
            failures=0,
            last_error="",
        )

    def get_page(
        self,
        output_path: str,
        page: int,
        page_size: int,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        norm_path = os.path.abspath(os.path.expanduser(str(output_path or "").strip()))
        requested_page = max(1, int(page))
        requested_size = max(1, min(500, int(page_size)))

        self.poll()
        self.request_scan(norm_path, force=force_refresh)

        with self._db() as conn:
            count_row = conn.execute(
                "SELECT COUNT(*) AS c FROM output_files WHERE output_path = ?",
                (norm_path,),
            ).fetchone()
            total = int(count_row["c"]) if count_row else 0
            total_pages = max(1, (total + requested_size - 1) // requested_size) if total > 0 else 1
            safe_page = min(requested_page, total_pages)
            offset = (safe_page - 1) * requested_size
            row_items = conn.execute(
                """
                SELECT rel_name, abs_path, kind, mtime FROM output_files
                WHERE output_path = ?
                ORDER BY mtime DESC, rel_name ASC
                LIMIT ? OFFSET ?
                """,
                (norm_path, requested_size, offset),
            ).fetchall()
            meta = conn.execute(
                "SELECT last_scan_finished_at, last_scan_duration_ms, last_scan_files, scan_failures, last_error FROM output_index_meta WHERE output_path = ?",
                (norm_path,),
            ).fetchone()

        rows = [
            {
                "name": str(r["rel_name"]),
                "path": str(r["abs_path"]),
                "uri": Path(str(r["abs_path"])).resolve().as_uri(),
                "kind": str(r["kind"]),
                "mtime": str(float(r["mtime"])),
            }
            for r in row_items
        ]
        with self._lock:
            inflight = self._future is not None and not self._future.done()
            scheduled = bool(self._scheduled_path)
        stats = {
            "scan_inflight": inflight,
            "scan_scheduled": scheduled,
            "last_scan_duration_ms": float(meta["last_scan_duration_ms"]) if meta else 0.0,
            "last_scan_files": int(meta["last_scan_files"]) if meta else 0,
            "scan_failures": int(meta["scan_failures"]) if meta else 0,
            "last_scan_finished_at": float(meta["last_scan_finished_at"]) if meta else 0.0,
            "last_error": str(meta["last_error"]) if meta else "",
            "queue_depth": int(inflight) + int(scheduled),
        }
        return {
            "rows": rows,
            "total": total,
            "page": safe_page,
            "page_size": requested_size,
            "total_pages": total_pages if total > 0 else 0,
            "stats": stats,
        }
