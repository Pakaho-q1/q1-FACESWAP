from __future__ import annotations

from typing import Any, Literal

from nicegui import ui


def metric_card(
    title: str,
    value: str = "-",
    *,
    size: Literal["compact", "normal", "large"] = "normal",
    title_classes: str | None = None,
    value_classes: str | None = None,
    card_classes: str | None = None,
) -> Any:
    presets = {
        "compact": {
            "title": "text-[8px] uppercase text-slate-500 font-bold",
            "value": "text-base font-bold text-slate-800",
            "card": "w-full border border-slate-200 bg-white shadow-sm rounded-md q-pa-xs",
        },
        "normal": {
            "title": "text-[10px] uppercase text-slate-500 font-bold",
            "value": "text-xl font-bold text-slate-800",
            "card": "w-full border border-slate-200 bg-white shadow-sm rounded-lg q-pa-sm",
        },
        "large": {
            "title": "text-xs uppercase text-slate-500 font-bold",
            "value": "text-2xl font-bold text-slate-800",
            "card": "w-full border border-slate-200 bg-white shadow-md rounded-xl q-pa-md",
        },
    }
    selected = presets[size]
    resolved_title_classes = title_classes or selected["title"]
    resolved_value_classes = value_classes or selected["value"]
    resolved_card_classes = card_classes or selected["card"]

    with ui.card().classes(resolved_card_classes).style("line-height: 1.2;"):
        # Keep metric cards dense without affecting regular form cards.
        with ui.column().classes("gap-0"):
            ui.label(title).classes(resolved_title_classes)
            value_label = ui.label(value).classes(resolved_value_classes)
    return value_label


def attach_card_tooltip(target: Any, message: str) -> Any:
    """Attach tooltip to a card-like UI container and return the same target."""
    text = str(message or "").strip()
    if text:
        target.tooltip(text)
    return target


def attach_ui_tooltip(target: Any, message: str) -> Any:
    """Attach tooltip to any UI element and return the same target."""
    text = str(message or "").strip()
    if text:
        target.tooltip(text)
    return target
