from __future__ import annotations

from typing import Optional

from .store import AppStore

_GLOBAL_STORE: Optional[AppStore] = None


def get_store() -> AppStore:
    """Return the global AppStore singleton.

    This provides a single source-of-truth (SSOT) entrypoint for UI code.
    """
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = AppStore()
    return _GLOBAL_STORE


def reset_store() -> AppStore:
    global _GLOBAL_STORE
    _GLOBAL_STORE = AppStore()
    return _GLOBAL_STORE


def get_ui_state() -> AppStore:
    # Backward-compatible alias while migrating older call sites.
    return get_store()


def reset_ui_state() -> AppStore:
    # Backward-compatible alias while migrating older call sites.
    return reset_store()


__all__ = [
    "AppStore",
    "get_store",
    "reset_store",
    "get_ui_state",
    "reset_ui_state",
]
