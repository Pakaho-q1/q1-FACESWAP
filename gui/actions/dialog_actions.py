from __future__ import annotations

from typing import Callable, Any


def open_dialog(dialog: Any) -> None:
    dialog.open()


def close_then(dialog: Any, callback: Callable[[], None]) -> None:
    dialog.close()
    callback()
