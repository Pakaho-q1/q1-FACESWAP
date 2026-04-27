from __future__ import annotations

import unittest

from gui.controller import PipelineController
from gui.state.store import AppStore


class ControllerBackpressureTests(unittest.TestCase):
    def test_event_queue_is_bounded_and_drops_logs(self) -> None:
        controller = PipelineController(state=AppStore())
        for i in range(8000):
            controller._push_event("log", f"log {i}")  # noqa: SLF001
        self.assertLessEqual(controller.event_q.qsize(), 5000)
        self.assertGreater(controller._dropped_log_events, 0)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
