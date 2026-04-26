from __future__ import annotations

import threading
from typing import Callable, Optional


_UI_PRINT_CALLBACK: Optional[Callable[[str], None]] = None
_UI_PRINT_LOCK = threading.Lock()


def set_ui_print_callback(callback: Optional[Callable[[str], None]]) -> None:
    """Register or clear a callback that receives every ui_print line."""
    global _UI_PRINT_CALLBACK
    with _UI_PRINT_LOCK:
        _UI_PRINT_CALLBACK = callback


def _emit_ui_callback(text: str) -> None:
    callback = None
    with _UI_PRINT_LOCK:
        callback = _UI_PRINT_CALLBACK
    if callback is not None:
        try:
            callback(str(text))
        except Exception:
            # Never let UI callback failures break pipeline logging.
            pass


def ui_print(message, fallback=None):
    """Print user-facing status messages with Unicode-safe fallback."""
    output_text = str(message)
    try:
        print(output_text)
    except UnicodeEncodeError:
        if fallback is not None:
            output_text = str(fallback)
            print(output_text)
        else:
            output_text = output_text.encode("ascii", "ignore").decode("ascii").strip()
            print(output_text)
    finally:
        _emit_ui_callback(output_text)
