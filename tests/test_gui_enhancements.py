from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gui.config.loader import load_gui_defaults
from gui.services.download_service import ModelDownloadService


class GuiEnhancementTests(unittest.TestCase):
    def test_loader_migrates_legacy_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "q1-FACESWAP"
            assets = project_root / "assets"
            assets.mkdir(parents=True, exist_ok=True)
            (assets / ".env").write_text("INPUT_DIR=D:/input\n", encoding="utf-8")
            (assets / ".env_user").write_text("OUTPUT_DIR=D:/output\n", encoding="utf-8")

            def _load_settings(_project_root: str) -> dict[str, str]:
                return {
                    "FACE_MODEL_NAME": "spy",
                    "worker-queue": "96",
                }

            defaults = load_gui_defaults(str(project_root), _load_settings)
            self.assertEqual(defaults["input_path"], "D:/input")
            self.assertEqual(defaults["output_path"], "D:/output")
            self.assertEqual(defaults["face_name"], "spy")
            self.assertEqual(int(defaults["worker_queue_size"]), 96)
            notes = list(defaults.get("_migration_notes", []))
            self.assertTrue(any("FACE_MODEL_NAME" in note for note in notes))

    def test_download_service_checksum_state_when_manifest_missing_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "q1-FACESWAP"
            assets = project_root / "assets"
            models = assets / "models"
            models.mkdir(parents=True, exist_ok=True)
            model_path = models / "a.onnx"
            model_path.write_bytes(b"abc")

            def _status(_project_root: str) -> list[dict[str, str]]:
                return [{"filename": "a.onnx", "present": True, "path": str(model_path), "url": "", "sha256": ""}]

            def _trt(_project_root: str) -> dict[str, str]:
                return {"ok": True}

            service = ModelDownloadService(
                project_root=str(project_root),
                model_fallback_urls={},
                read_model_manifest_status=_status,
                check_tensorrt_status=_trt,
            )
            service.start_verify_checksum("a.onnx", str(model_path), "")
            with service.checksum_state_lock:
                state = service.checksum_state.get("a.onnx", "")
            self.assertEqual(state, "missing_manifest_sha256")


if __name__ == "__main__":
    unittest.main()
