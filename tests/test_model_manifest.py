import json
import os
import tempfile
import unittest
from pathlib import Path

from core.config import _load_model_manifest, _sync_models_from_manifest


class ModelManifestTests(unittest.TestCase):
    def test_load_model_manifest_requires_models(self):
        with tempfile.TemporaryDirectory() as td:
            manifest_path = os.path.join(td, "model_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"version": 1, "models": []}, f)
            with self.assertRaises(Exception):
                _load_model_manifest(manifest_path)

    def test_sync_models_downloads_required_file(self):
        with tempfile.TemporaryDirectory() as td:
            source_file = os.path.join(td, "source.bin")
            with open(source_file, "wb") as f:
                f.write(b"hello-model")

            models_dir = os.path.join(td, "models")
            os.makedirs(models_dir, exist_ok=True)

            manifest = {
                "version": 1,
                "models": [
                    {
                        "filename": "demo.bin",
                        "url": Path(source_file).as_uri(),
                        "sha256": "",
                    }
                ],
            }

            _sync_models_from_manifest(
                models_dir=models_dir,
                manifest=manifest,
                preload_models=True,
                required_filenames={"demo.bin"},
            )

            self.assertTrue(os.path.isfile(os.path.join(models_dir, "demo.bin")))


if __name__ == "__main__":
    unittest.main()
