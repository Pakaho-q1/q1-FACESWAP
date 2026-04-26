from __future__ import annotations

import time
from typing import Any, Callable


def sync_preview_card(*, preview_enabled: Any, store: Any, preview_expansion: Any, pause_preview_btn: Any) -> None:
    enabled = bool(preview_enabled.value)
    should_open = enabled and not store.preview_paused
    preview_expansion.set_value(should_open)
    pause_preview_btn.set_text("Pause Preview" if should_open else "Resume Preview")


def flush_preview_render(
    *,
    preview_engine: Any,
    store: Any,
    preview_enabled: Any,
    preview_meta: Any,
    swap_preview: Callable[[str], None],
) -> None:
    latest = preview_engine.pop_latest_render()
    if latest is None:
        return
    if store.preview_paused or not bool(preview_enabled.value):
        return
    signature = str(latest.get("signature", ""))
    if signature and signature == store.last_preview_signature:
        return
    data_url = str(latest.get("data_url", ""))
    if not data_url:
        return

    swap_preview(data_url)
    store.preview_active_layer = "b" if store.preview_active_layer == "a" else "a"

    kind = str(latest.get("kind", ""))
    adaptive_fps = preview_engine.adaptive_fps()
    if kind == "video":
        preview_meta.set_text(
            f"Preview [video]: {latest.get('item_id', '')} frame {latest.get('frame_id', 0)} | "
            f"fps={adaptive_fps:.1f} | drop(in/wk)={preview_engine.ingress_dropped}/{preview_engine.worker_dropped}"
        )
    else:
        preview_meta.set_text(
            f"Preview [image]: {latest.get('output_name', '')} | "
            f"fps={adaptive_fps:.1f} | drop(in/wk)={preview_engine.ingress_dropped}/{preview_engine.worker_dropped}"
        )

    store.last_preview_signature = signature
    store.last_preview_render_ts = time.perf_counter()
    preview_engine.mark_rendered()


def toggle_preview_pause(*, store: Any, preview_enabled: Any, preview_expansion: Any, pause_preview_btn: Any) -> None:
    store.preview_paused = not store.preview_paused
    sync_preview_card(
        preview_enabled=preview_enabled,
        store=store,
        preview_expansion=preview_expansion,
        pause_preview_btn=pause_preview_btn,
    )
