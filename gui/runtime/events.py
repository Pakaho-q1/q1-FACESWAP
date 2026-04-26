from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DomainEvent:
    kind: str
    name: str
    payload: dict[str, Any]


def map_controller_event(kind: str, payload: Any) -> DomainEvent:
    if kind == "event":
        data = dict(payload or {})
        return DomainEvent(
            kind="event",
            name=str(data.get("name", "event")),
            payload=dict(data.get("payload", {})),
        )
    if kind == "progress":
        data = dict(payload or {})
        return DomainEvent(kind="progress", name="progress", payload=data)
    if kind == "log":
        return DomainEvent(kind="log", name="log", payload={"message": str(payload)})
    if kind in {"done", "stopped", "error"}:
        data = payload if isinstance(payload, dict) else {"value": payload}
        return DomainEvent(kind=kind, name=kind, payload=dict(data))
    return DomainEvent(kind=kind, name=kind, payload={"value": payload})
