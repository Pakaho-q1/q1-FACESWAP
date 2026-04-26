from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from gui.runtime.events import DomainEvent, map_controller_event
except ImportError:
    from runtime.events import DomainEvent, map_controller_event  # type: ignore

@dataclass(frozen=True)
class UiEvent:
    kind: str
    name: str
    payload: dict[str, Any]


def normalize_controller_event(kind: str, payload: Any) -> UiEvent:
    # Backward-compatible adapter for existing UI code.
    event: DomainEvent = map_controller_event(kind, payload)
    return UiEvent(kind=event.kind, name=event.name, payload=event.payload)
