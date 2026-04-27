from __future__ import annotations

import threading
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

    def test_terminal_event_survives_when_queue_is_full(self) -> None:
        controller = PipelineController(state=AppStore())
        max_size = int(controller.event_q.maxsize)
        for i in range(max_size):
            controller._push_event(
                "event", {"name": "noop", "payload": {"idx": i}}
            )  # noqa: SLF001
        self.assertEqual(controller.event_q.qsize(), max_size)

        controller._push_event("done", {"return_code": 0})  # noqa: SLF001
        events = controller.poll_events()
        self.assertTrue(any(kind == "done" for kind, _ in events))
        self.assertGreaterEqual(controller._dropped_nonlog_events, 1)  # noqa: SLF001

    def test_all_terminal_kinds_survive_when_queue_is_full(self) -> None:
        terminal_samples = [
            ("done", {"return_code": 0}),
            ("stopped", {"reason": "user_stop"}),
            ("error", "CLI exited with code 1"),
        ]
        for terminal_kind, terminal_payload in terminal_samples:
            controller = PipelineController(state=AppStore())
            max_size = int(controller.event_q.maxsize)
            for i in range(max_size):
                controller._push_event(  # noqa: SLF001
                    "event", {"name": "warmup", "payload": {"idx": i}}
                )
            controller._push_event(terminal_kind, terminal_payload)  # noqa: SLF001
            events = controller.poll_events()
            self.assertTrue(
                any(kind == terminal_kind for kind, _ in events),
                msg=f"terminal kind {terminal_kind} should survive full queue",
            )

    def test_synthetic_terminal_emits_even_when_queue_has_non_terminal_events(self) -> None:
        controller = PipelineController(state=AppStore())
        controller.state.running = True
        controller.state.stop_requested = False
        dead_worker = threading.Thread(target=lambda: None)
        dead_worker.start()
        dead_worker.join(timeout=1.0)
        controller._worker_thread = dead_worker  # noqa: SLF001
        controller._push_event("event", {"name": "noop", "payload": {}})  # noqa: SLF001

        events = controller.poll_events()
        self.assertTrue(
            any(
                kind == "error" and str(payload) == "Worker exited unexpectedly"
                for kind, payload in events
            )
        )

        events_again = controller.poll_events()
        self.assertFalse(
            any(kind == "error" and str(payload) == "Worker exited unexpectedly" for kind, payload in events_again)
        )

    def test_synthetic_stopped_emits_for_user_stop_even_with_pending_events(self) -> None:
        controller = PipelineController(state=AppStore())
        controller.state.running = True
        controller.state.stop_requested = True
        dead_worker = threading.Thread(target=lambda: None)
        dead_worker.start()
        dead_worker.join(timeout=1.0)
        controller._worker_thread = dead_worker  # noqa: SLF001
        controller._push_event("event", {"name": "noop", "payload": {}})  # noqa: SLF001

        events = controller.poll_events()
        self.assertTrue(
            any(kind == "stopped" for kind, _ in events),
            msg="should emit synthetic stopped event when user requested stop",
        )


if __name__ == "__main__":
    unittest.main()
