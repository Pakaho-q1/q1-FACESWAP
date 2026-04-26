from __future__ import annotations

from typing import Any

from gui.controller import PipelineController


def start_pipeline(controller: PipelineController, values: dict[str, Any]) -> bool:
    return controller.start(values)


def stop_pipeline(controller: PipelineController) -> bool:
    return controller.request_stop()
