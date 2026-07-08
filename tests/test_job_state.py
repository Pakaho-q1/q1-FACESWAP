import os
import tempfile
import unittest

from core.job_state import JobStateManager


class JobStateManagerTests(unittest.TestCase):
    def test_completed_items_are_filtered_from_pending(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, ".job_state_image.json")
            manager = JobStateManager(path=path, job_key="k1", plan_snapshot={"kind": "image"})
            manager.mark_planned(["a.jpg", "b.jpg"])
            manager.mark_started("a.jpg")
            manager.mark_completed("a.jpg")

            pending = manager.pending_items(["a.jpg", "b.jpg"], max_attempts=2)
            self.assertEqual(pending, ["b.jpg"])

    def test_max_attempts_limits_retry(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, ".job_state_image.json")
            manager = JobStateManager(path=path, job_key="k2", plan_snapshot={"kind": "image"})
            manager.mark_planned(["x.jpg"])
            manager.mark_started("x.jpg")
            manager.mark_failed("x.jpg", "first")
            manager.mark_started("x.jpg")
            manager.mark_failed("x.jpg", "second")

            pending = manager.pending_items(["x.jpg"], max_attempts=2)
            self.assertEqual(pending, [])
            item = manager.get_item("x.jpg")
            self.assertEqual(item.attempts, 2)
            self.assertEqual(item.status, "failed")

    def test_resume_marks_stale_in_progress_as_failed(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, ".job_state_image.json")
            manager = JobStateManager(path=path, job_key="k3", plan_snapshot={"kind": "image"})
            manager.mark_planned(["z.jpg"])
            manager.mark_started("z.jpg")

            reloaded = JobStateManager(path=path, job_key="k3", plan_snapshot={"kind": "image"})
            reloaded.reset_in_progress()
            item = reloaded.get_item("z.jpg")

            self.assertEqual(item.status, "failed")
            self.assertIn("interrupted", item.last_error)

    def test_compact_removes_items_not_in_latest_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, ".job_state_image.json")
            manager = JobStateManager(path=path, job_key="k4", plan_snapshot={"kind": "image"})
            manager.mark_planned(["old.jpg", "new.jpg"])
            manager.mark_completed("old.jpg")
            manager.compact_to_items(["new.jpg"])

            counts = manager.snapshot_counts()
            self.assertEqual(sum(counts.values()), 1)
            pending = manager.pending_items(["new.jpg"], max_attempts=3)
            self.assertEqual(pending, ["new.jpg"])


if __name__ == "__main__":
    unittest.main()
