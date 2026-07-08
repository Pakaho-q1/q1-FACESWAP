from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable


_STATE_VERSION = 1


@dataclass(frozen=True)
class ItemState:
    status: str
    attempts: int
    updated_at: float
    last_error: str = ""


class JobStateManager:
    def __init__(self, path: str, job_key: str, plan_snapshot: dict):
        self._path = path
        self._lock = threading.Lock()
        self._dirty = False
        self._last_save_ts = 0.0
        self._save_interval_sec = 0.5
        self._data = {
            "version": _STATE_VERSION,
            "job_key": job_key,
            "plan": plan_snapshot,
            "updated_at": time.time(),
            "items": {},
        }
        self._load_existing()

    @property
    def path(self) -> str:
        return self._path

    def _load_existing(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded.get("version") != _STATE_VERSION:
                return
            if loaded.get("job_key") != self._data["job_key"]:
                return
            if isinstance(loaded.get("items"), dict):
                self._data = loaded
        except Exception:
            # Corrupted/partial state file should not block pipeline execution.
            return

    def _mark_dirty(self):
        self._dirty = True
        self._data["updated_at"] = time.time()

    def _save_locked(self, force: bool = False) -> None:
        now = time.time()
        if not self._dirty and not force:
            return
        if not force and (now - self._last_save_ts) < self._save_interval_sec:
            return

        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        temp_path = f"{self._path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=True, indent=2)
        os.replace(temp_path, self._path)
        self._last_save_ts = now
        self._dirty = False

    def save(self, force: bool = False) -> None:
        with self._lock:
            self._save_locked(force=force)

    def mark_planned(self, item_ids: Iterable[str]) -> None:
        with self._lock:
            items = self._data["items"]
            now = time.time()
            for item_id in item_ids:
                entry = items.get(item_id)
                if entry is None:
                    items[item_id] = {
                        "status": "planned",
                        "attempts": 0,
                        "updated_at": now,
                        "last_error": "",
                    }
            self._mark_dirty()
            self._save_locked(force=False)

    def compact_to_items(self, valid_item_ids: Iterable[str]) -> None:
        valid_set = set(valid_item_ids)
        with self._lock:
            items = self._data["items"]
            to_remove = [item_id for item_id in items.keys() if item_id not in valid_set]
            for item_id in to_remove:
                del items[item_id]
            if to_remove:
                self._mark_dirty()
                self._save_locked(force=False)

    def reset_in_progress(self, error: str = "interrupted_previous_run") -> None:
        with self._lock:
            items = self._data["items"]
            now = time.time()
            changed = False
            for item_id, entry in items.items():
                if entry.get("status") == "in_progress":
                    entry["status"] = "failed"
                    entry["updated_at"] = now
                    entry["last_error"] = error[:500]
                    changed = True
            if changed:
                self._mark_dirty()
                self._save_locked(force=False)

    def mark_started(self, item_id: str) -> None:
        with self._lock:
            items = self._data["items"]
            now = time.time()
            prev = items.get(item_id, {})
            items[item_id] = {
                "status": "in_progress",
                "attempts": int(prev.get("attempts", 0)) + 1,
                "updated_at": now,
                "last_error": "",
            }
            self._mark_dirty()
            # Persist start transition immediately to improve crash resume accuracy.
            self._save_locked(force=True)

    def mark_completed(self, item_id: str) -> None:
        with self._lock:
            items = self._data["items"]
            prev = items.get(item_id, {})
            items[item_id] = {
                "status": "completed",
                "attempts": int(prev.get("attempts", 0)),
                "updated_at": time.time(),
                "last_error": "",
            }
            self._mark_dirty()
            self._save_locked(force=True)

    def mark_failed(self, item_id: str, error: str = "") -> None:
        with self._lock:
            items = self._data["items"]
            prev = items.get(item_id, {})
            items[item_id] = {
                "status": "failed",
                "attempts": int(prev.get("attempts", 1)),
                "updated_at": time.time(),
                "last_error": (error or "")[:500],
            }
            self._mark_dirty()
            self._save_locked(force=True)

    def pending_items(self, item_ids: Iterable[str], max_attempts: int | None = None) -> list[str]:
        with self._lock:
            items = self._data["items"]
            pending = []
            for item_id in item_ids:
                entry = items.get(item_id, {})
                status = entry.get("status", "planned")
                attempts = int(entry.get("attempts", 0))
                if status != "completed":
                    if max_attempts is not None and attempts >= max_attempts:
                        continue
                    pending.append(item_id)
            return pending

    def get_item(self, item_id: str) -> ItemState:
        with self._lock:
            entry = copy.deepcopy(self._data["items"].get(item_id, {}))
        return ItemState(
            status=str(entry.get("status", "planned")),
            attempts=int(entry.get("attempts", 0)),
            updated_at=float(entry.get("updated_at", 0.0)),
            last_error=str(entry.get("last_error", "")),
        )

    def snapshot_counts(self) -> Dict[str, int]:
        with self._lock:
            counts = {"planned": 0, "in_progress": 0, "completed": 0, "failed": 0}
            for entry in self._data["items"].values():
                status = entry.get("status", "planned")
                counts[status] = counts.get(status, 0) + 1
            return counts
