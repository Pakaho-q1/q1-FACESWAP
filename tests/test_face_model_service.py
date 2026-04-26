import os
import tempfile
import time
import unittest

from core.face_model_builder import FaceModelBuildError, FaceModelBuildResult
from gui.services.face_model_service import FaceModelService


def _wait_until(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class FaceModelServiceTests(unittest.TestCase):
    def test_build_state_done(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = os.path.join(td, "q1-FACESWAP")
            os.makedirs(project_root, exist_ok=True)

            def fake_build(**kwargs):
                progress = kwargs.get("progress_callback")
                if callable(progress):
                    from core.face_model_builder import FaceModelBuildStats

                    progress(FaceModelBuildStats(scanned_images=1, accepted_images=1))
                return FaceModelBuildResult(
                    face_name=str(kwargs["face_name"]),
                    output_path=os.path.join(str(kwargs["faces_dir"]), f"{kwargs['face_name']}.safetensors"),
                    scanned_images=1,
                    accepted_images=1,
                    skipped_no_face=0,
                    skipped_multi_face=0,
                    skipped_invalid_embedding=0,
                    outlier_filtered=0,
                )

            service = FaceModelService(project_root=project_root, build_face_model_fn=fake_build)
            started = service.start_build(face_name="alice", input_dir=project_root)
            self.assertTrue(started)

            ok = _wait_until(
                lambda: service.snapshot_build_state().get("alice", {}).get("status") == "done"
            )
            self.assertTrue(ok, "expected build status to become done")
            state = service.snapshot_build_state().get("alice", {})
            self.assertEqual(str(state.get("status")), "done")
            self.assertIn("output_path", state)

    def test_build_state_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = os.path.join(td, "q1-FACESWAP")
            os.makedirs(project_root, exist_ok=True)

            def fake_build(**_kwargs):
                raise FaceModelBuildError("boom")

            service = FaceModelService(project_root=project_root, build_face_model_fn=fake_build)
            started = service.start_build(face_name="bob", input_dir=project_root)
            self.assertTrue(started)
            ok = _wait_until(
                lambda: service.snapshot_build_state().get("bob", {}).get("status") == "error"
            )
            self.assertTrue(ok, "expected build status to become error")
            state = service.snapshot_build_state().get("bob", {})
            self.assertIn("boom", str(state.get("detail", "")))

    def test_clear_finished_keeps_active(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = os.path.join(td, "q1-FACESWAP")
            os.makedirs(project_root, exist_ok=True)

            def fake_build(**kwargs):
                # Slow build so state remains running briefly.
                time.sleep(0.2)
                return FaceModelBuildResult(
                    face_name=str(kwargs["face_name"]),
                    output_path=os.path.join(str(kwargs["faces_dir"]), f"{kwargs['face_name']}.safetensors"),
                    scanned_images=1,
                    accepted_images=1,
                    skipped_no_face=0,
                    skipped_multi_face=0,
                    skipped_invalid_embedding=0,
                    outlier_filtered=0,
                )

            service = FaceModelService(project_root=project_root, build_face_model_fn=fake_build)
            service.start_build(face_name="carol", input_dir=project_root)
            time.sleep(0.05)
            service.clear_finished()
            # Active build should still exist.
            self.assertIn("carol", service.snapshot_build_state())
            _wait_until(
                lambda: service.snapshot_build_state().get("carol", {}).get("status") == "done"
            )
            service.clear_finished()
            self.assertNotIn("carol", service.snapshot_build_state())


if __name__ == "__main__":
    unittest.main()
